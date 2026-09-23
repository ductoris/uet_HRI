#include <memory>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <cmath>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

class CircleTrajectoryNode : public rclcpp::Node
{
public:
  CircleTrajectoryNode() : Node("circle_trajectory_node")
  {
    declare_parameter<std::string>("planning_group", "ur_manipulator");
    declare_parameter<std::string>("base_frame", "base_link");
    // New parameter to select tool orientation handling
    declare_parameter<std::string>("orientation_mode", "down");

    // VÙNG LÀM VIỆC AN TOÀN CHO UR3/UR3e (< 0.50m từ gốc robot, tránh lệch sâu sang trái):
    // Đặt tâm tại (X=0.11, Y=0.35, Z=0.15), bán kính 0.07m (đường kính 14cm).
    // Dải tọa độ: X thuộc [0.04, 0.18] (hoàn toàn dương, không bị lệch sâu sang trái x <= 0)
    //             Y thuộc [0.28, 0.42] (trước mặt robot)
    // Khoảng cách tới gốc: từ 0.29m đến 0.43m -> vùng với linh hoạt và an toàn nhất của UR3e.
    declare_parameter<double>("center_x", 0.11);       // Tọa độ tâm X (m)
    declare_parameter<double>("center_y", 0.35);      // Tọa độ tâm Y (m)
    declare_parameter<double>("center_z", 0.15);      // Độ cao mặt phẳng vẽ Z (m)
    declare_parameter<double>("radius", 0.07);        // Bán kính đường tròn (m) - 7cm (đường kính 14cm)
    declare_parameter<double>("start_angle", 1.5707963); // Góc xuất phát: PI/2 (đỉnh trên của đường tròn, ngay trước mặt robot)
    declare_parameter<int>("num_points", 72);         // Số điểm chia đường tròn (5 độ / điểm)

    // Khoảng nhấc bút lên cao theo phương thẳng đứng (+Z)
    declare_parameter<double>("pen_lift", 0.04);      // Nhấc 4cm

    declare_parameter<double>("vel_scale", 0.15);
    declare_parameter<double>("acc_scale", 0.15);
    declare_parameter<int>("num_cycles", 2);          // Số chu trình lặp lại (mặc định 2 vòng)

    // Publish trên cả 2 topic để RViz luôn hiển thị (dù đang subscribe topic nào):
    circle_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/drawn_circle_marker", 10);
    letter_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/drawn_letter_marker", 10);
  }

  void init()
  {
    auto node = shared_from_this();
    group_name_ = get_parameter("planning_group").as_string();
    base_frame_ = get_parameter("base_frame").as_string();

    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node, group_name_);
    move_group_->setPoseReferenceFrame(base_frame_);
    move_group_->setMaxVelocityScalingFactor(get_parameter("vel_scale").as_double());
    move_group_->setMaxAccelerationScalingFactor(get_parameter("acc_scale").as_double());
    move_group_->setPlanningTime(10.0);
    move_group_->setNumPlanningAttempts(15);
    move_group_->setGoalPositionTolerance(0.005);
    move_group_->setGoalOrientationTolerance(0.02);

    // Thêm vật cản sàn bảo vệ (floor_table) tại z <= -0.01 vào Planning Scene
    addFloorCollisionObject();

    // Hiển thị ngay đường tròn mẫu (reference circle) lên RViz để người dùng quan sát trước
    publishReferenceMarker();

    // Tạo timer định kỳ phát marker đường tròn tham chiếu để RViz không bị mất
    marker_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(1000),
      [this]() { publishReferenceMarker(); });

    RCLCPP_INFO(get_logger(), "=== CircleTrajectoryNode khởi tạo thành công ===");
    RCLCPP_INFO(get_logger(), "Planning group: %s | Base frame: %s | End-effector: %s",
                group_name_.c_str(), base_frame_.c_str(), move_group_->getEndEffectorLink().c_str());
  }

  void run()
  {
    const int total_cycles = get_parameter("num_cycles").as_int();
    const double cx = get_parameter("center_x").as_double();
    const double cy = get_parameter("center_y").as_double();
    const double cz = get_parameter("center_z").as_double();
    const double r = get_parameter("radius").as_double();
    const int n_pts = get_parameter("num_points").as_int();

    RCLCPP_INFO(get_logger(), "==========================================");
    RCLCPP_INFO(get_logger(), ">>> THÔNG SỐ QUỸ ĐẠO HÌNH TRÒN <<<");
    RCLCPP_INFO(get_logger(), "  Tâm: (X=%.3f, Y=%.3f, Z=%.3f) m", cx, cy, cz);
    RCLCPP_INFO(get_logger(), "  Bán kính: %.3f m (Đường kính: %.1f cm)", r, r * 200.0);
    RCLCPP_INFO(get_logger(), "  Vùng quét X: [%.3f, %.3f] m (Dương hoàn toàn, tránh lệch trái)", cx - r, cx + r);
    RCLCPP_INFO(get_logger(), "  Vùng quét Y: [%.3f, %.3f] m", cy - r, cy + r);
    RCLCPP_INFO(get_logger(), "  Số điểm nội suy: %d điểm (%.1f độ/bước)", n_pts, 360.0 / n_pts);
    RCLCPP_INFO(get_logger(), "  Số chu trình lặp lại: %d vòng", total_cycles);
    RCLCPP_INFO(get_logger(), "==========================================");

    // Bước 0: Đưa robot về trạng thái ban đầu (home)
    RCLCPP_INFO(get_logger(), "Đang đưa robot về trạng thái ban đầu (home)...");
    move_group_->setStartStateToCurrentState();
    move_group_->setNamedTarget("home");
    planAndExecute();

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // Sinh tập waypoints hình tròn
    auto circle_stroke = buildCircleWaypoints();
    if (circle_stroke.size() < 3) {
      RCLCPP_ERROR(get_logger(), "Lỗi sinh waypoints đường tròn!");
      return;
    }

    // Điểm tiếp cận trên không đầu tiên của đường tròn
    const auto & initial_approach = circle_stroke.front();
    RCLCPP_INFO(get_logger(), "Di chuyển tới vị trí đón bút trên không tại (X=%.3f, Y=%.3f, Z=%.3f)...",
                initial_approach.position.x, initial_approach.position.y, initial_approach.position.z);

    // Cơ chế thử lại (Retry) tối đa 3 lần tiếp cận
    bool approach_ok = false;
    for (int attempt = 1; attempt <= 3; ++attempt) {
      RCLCPP_INFO(get_logger(), "Tiếp cận vị trí bắt đầu vẽ đường tròn (thử lần %d/3)...", attempt);
      move_group_->setStartStateToCurrentState();
      move_group_->setPoseTarget(initial_approach);
      if (planAndExecute()) {
        approach_ok = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    if (!approach_ok) {
      // Fallback: adjust start angle by +90 degrees and retry once
      double fallback_angle = start_angle + M_PI_2; // 90 degrees
      RCLCPP_WARN(get_logger(), "Thử góc bắt đầu dự phòng (%.3f rad) vì tiếp cận thất bại.", fallback_angle);
      double x_f = cx + r * std::cos(fallback_angle);
      double y_f = cy + r * std::sin(fallback_angle);
      geometry_msgs::msg::Pose fallback_pose = initial_approach;
      fallback_pose.position.x = x_f;
      fallback_pose.position.y = y_f;
      // attempt once with fallback
      move_group_->setStartStateToCurrentState();
      move_group_->setPoseTarget(fallback_pose);
      if (planAndExecute()) {
        approach_ok = true;
      } else {
        RCLCPP_ERROR(get_logger(), "Không thể tiếp cận vị trí bắt đầu vẽ đường tròn kể cả góc dự phòng!");
        return;
      }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // VÒNG LẶP CHU TRÌNH VẼ ĐƯỜNG TRÒN (Mặc định 2 vòng liên tiếp)
    for (int cycle = 1; cycle <= total_cycles; ++cycle) {
      RCLCPP_INFO(get_logger(), "==========================================");
      RCLCPP_INFO(get_logger(), ">>> BẮT ĐẦU CHU TRÌNH VẼ ĐƯỜNG TRÒN LẦN %d/%d <<<", cycle, total_cycles);
      RCLCPP_INFO(get_logger(), "==========================================");

      // Thực thi nét vẽ đường tròn hoàn chỉnh
      drawCircleStroke(circle_stroke, cycle);

      RCLCPP_INFO(get_logger(), ">>> HOÀN THÀNH CHU TRÌNH %d/%d <<<", cycle, total_cycles);

      if (cycle < total_cycles) {
        RCLCPP_INFO(get_logger(), "Chuẩn bị bắt đầu vòng vẽ tiếp theo...");
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      }
    }

    // Kết thúc: Nâng bút lên cao an toàn rồi trở về home
    RCLCPP_INFO(get_logger(), "==========================================");
    RCLCPP_INFO(get_logger(), ">>> HOÀN THÀNH TẤT CẢ %d CHU TRÌNH VẼ ĐƯỜNG TRÒN! <<<", total_cycles);
    RCLCPP_INFO(get_logger(), "==========================================");

    geometry_msgs::msg::Pose safe_lift = circle_stroke.back();
    safe_lift.position.z += 0.05; // Cao thêm 5cm
    executeCartesianPath({ safe_lift });

    RCLCPP_INFO(get_logger(), "Đưa robot về trạng thái ban đầu (home)...");
    move_group_->setStartStateToCurrentState();
    move_group_->setNamedTarget("home");
    planAndExecute();

    RCLCPP_INFO(get_logger(), ">>> TOÀN BỘ QUÁ TRÌNH VẼ ĐƯỜNG TRÒN KẾT THÚC THÀNH CÔNG! <<<");
  }

private:
  std::string group_name_, base_frame_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr circle_marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr letter_marker_pub_;
  rclcpp::TimerBase::SharedPtr marker_timer_;

  void publishMarkerToAll(const visualization_msgs::msg::Marker & marker)
  {
    if (circle_marker_pub_) circle_marker_pub_->publish(marker);
    if (letter_marker_pub_) letter_marker_pub_->publish(marker);
  }

  // Thêm mặt sàn va chạm vào Planning Scene để bảo vệ an toàn (z <= -0.01)
  void addFloorCollisionObject()
  {
    moveit_msgs::msg::CollisionObject floor_obj;
    floor_obj.header.frame_id = base_frame_;
    floor_obj.id = "floor_table";

    shape_msgs::msg::SolidPrimitive primitive;
    primitive.type = primitive.BOX;
    primitive.dimensions.resize(3);
    primitive.dimensions[primitive.BOX_X] = 2.0;
    primitive.dimensions[primitive.BOX_Y] = 2.0;
    primitive.dimensions[primitive.BOX_Z] = 0.04;

    geometry_msgs::msg::Pose box_pose;
    box_pose.orientation.w = 1.0;
    box_pose.position.x = 0.0;
    box_pose.position.y = 0.0;
    box_pose.position.z = -0.03; // Mặt trên tại z = -0.01 (tránh va chạm ảo với đế robot)

    floor_obj.primitives.push_back(primitive);
    floor_obj.primitive_poses.push_back(box_pose);
    floor_obj.operation = floor_obj.ADD;

    planning_scene_interface_.applyCollisionObject(floor_obj);
    RCLCPP_INFO(get_logger(), "Đã kích hoạt vật cản sàn bảo vệ (floor_table) tại z <= -0.01m trong Planning Scene.");
  }

  // Phát Marker đường tròn tham chiếu (hiển thị ngay khi khởi động node)
  void publishReferenceMarker()
  {
    const double cx = get_parameter("center_x").as_double();
    const double cy = get_parameter("center_y").as_double();
    const double cz = get_parameter("center_z").as_double();
    const double r = get_parameter("radius").as_double();
    const int n_pts = 72;

    // 1. Đường tròn tham chiếu màu xanh nhạt
    visualization_msgs::msg::Marker ref_marker;
    ref_marker.header.frame_id = base_frame_;
    ref_marker.header.stamp = this->now();
    ref_marker.ns = "circle_reference";
    ref_marker.id = 0;
    ref_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    ref_marker.action = visualization_msgs::msg::Marker::ADD;
    ref_marker.scale.x = 0.003; // Nét mảnh 3mm
    ref_marker.color.r = 0.3f;
    ref_marker.color.g = 0.8f;
    ref_marker.color.b = 1.0f;
    ref_marker.color.a = 0.6f;

    for (int i = 0; i <= n_pts; ++i) {
      double th = 2.0 * M_PI * i / n_pts;
      geometry_msgs::msg::Point pt;
      pt.x = cx + r * std::cos(th);
      pt.y = cy + r * std::sin(th);
      pt.z = cz;
      ref_marker.points.push_back(pt);
    }
    publishMarkerToAll(ref_marker);

    // 2. Điểm xuất phát (Start Point) hình cầu màu cam
    double start_angle = get_parameter("start_angle").as_double();
    visualization_msgs::msg::Marker start_marker;
    start_marker.header.frame_id = base_frame_;
    start_marker.header.stamp = this->now();
    start_marker.ns = "circle_start_point";
    start_marker.id = 1;
    start_marker.type = visualization_msgs::msg::Marker::SPHERE;
    start_marker.action = visualization_msgs::msg::Marker::ADD;
    start_marker.scale.x = 0.015;
    start_marker.scale.y = 0.015;
    start_marker.scale.z = 0.015;
    start_marker.pose.position.x = cx + r * std::cos(start_angle);
    start_marker.pose.position.y = cy + r * std::sin(start_angle);
    start_marker.pose.position.z = cz;
    start_marker.pose.orientation.w = 1.0;
    start_marker.color.r = 1.0f;
    start_marker.color.g = 0.6f;
    start_marker.color.b = 0.0f;
    start_marker.color.a = 0.9f;
    publishMarkerToAll(start_marker);
  }

  // Sinh tập waypoints tham số hóa đường tròn 360 độ:
  // Xuất phát từ start_angle (mặc định PI/2, đỉnh trên đường tròn, ngay trước mặt robot)
  std::vector<geometry_msgs::msg::Pose> buildCircleWaypoints()
  {
    const double cx = get_parameter("center_x").as_double();
    const double cy = get_parameter("center_y").as_double();
    const double cz = get_parameter("center_z").as_double();
    const double r = get_parameter("radius").as_double();
    const double start_angle = get_parameter("start_angle").as_double();
    const int n_pts = get_parameter("num_points").as_int();
    const double lift = get_parameter("pen_lift").as_double();

    tf2::Quaternion q;
    // Point the tool straight down (roll=0, pitch=M_PI, yaw=0)
    q.setRPY(0.0, M_PI, 0.0);
    auto orientation_down = tf2::toMsg(q);

    std::vector<geometry_msgs::msg::Pose> waypoints;

    // Tọa độ điểm bắt đầu
    double x_start = cx + r * std::cos(start_angle);
    double y_start = cy + r * std::sin(start_angle);

    // 1. Điểm trên không đầu cung tròn (Pen up)
    geometry_msgs::msg::Pose p_up_start;
    p_up_start.position.x = x_start;
    p_up_start.position.y = y_start;
    p_up_start.position.z = cz + lift;
    p_up_start.orientation = orientation_down;
    waypoints.push_back(p_up_start);

    // 2. Điểm chạm mặt phẳng bắt đầu vẽ (Pen down)
    geometry_msgs::msg::Pose p_down_start;
    p_down_start.position.x = x_start;
    p_down_start.position.y = y_start;
    p_down_start.position.z = cz;
    p_down_start.orientation = orientation_down;
    waypoints.push_back(p_down_start);

    // 3. Các điểm dọc theo cung tròn từ start_angle đến start_angle + 2*PI
    for (int i = 1; i <= n_pts; ++i) {
      double theta = start_angle + 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(n_pts);
      geometry_msgs::msg::Pose p;
      p.position.x = cx + r * std::cos(theta);
      p.position.y = cy + r * std::sin(theta);
      p.position.z = cz;
      p.orientation = orientation_down;
      waypoints.push_back(p);
    }

    // 4. Điểm trên không kết thúc cung tròn (Pen up tại điểm kết thúc)
    geometry_msgs::msg::Pose p_up_end;
    p_up_end.position.x = x_start;
    p_up_end.position.y = y_start;
    p_up_end.position.z = cz + lift;
    p_up_end.orientation = orientation_down;
    waypoints.push_back(p_up_end);

    return waypoints;
  }

  bool planAndExecute()
  {
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool ok = (move_group_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
    if (ok) {
      ok = (move_group_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
    }
    return ok;
  }

  bool executeCartesianPath(const std::vector<geometry_msgs::msg::Pose> & waypoints)
  {
    if (waypoints.empty()) return true;

    moveit_msgs::msg::RobotTrajectory trajectory;
    const double eef_step = 0.005; // 5mm bước nội suy
    const double jump_threshold = 4.0; // Chặn bước nhảy khớp đột ngột

    move_group_->setStartStateToCurrentState();
    double fraction = move_group_->computeCartesianPath(
      waypoints, eef_step, jump_threshold, trajectory, true);

    RCLCPP_INFO(get_logger(), "  Cartesian path fraction = %.2f%% (%zu waypoints)",
                fraction * 100.0, waypoints.size());

    if (fraction < 0.90) {
      RCLCPP_ERROR(get_logger(),
        "Cartesian path không đạt yêu cầu (fraction = %.2f%% < 90%%). Kiểm tra lại tầm với robot.",
        fraction * 100.0);
      return false;
    }

    auto result = move_group_->execute(trajectory);
    return (result == moveit::core::MoveItErrorCode::SUCCESS);
  }

  void drawCircleStroke(const std::vector<geometry_msgs::msg::Pose> & stroke_pts, int cycle_index)
  {
    if (stroke_pts.size() < 3) return;

    // Thực thi Cartesian path từ pen_down_start tới pen_up_end
    std::vector<geometry_msgs::msg::Pose> cart_pts(stroke_pts.begin() + 1, stroke_pts.end());

    if (executeCartesianPath(cart_pts)) {
      // Publish Marker vệt vẽ thực tế
      std::vector<geometry_msgs::msg::Pose> draw_pts(cart_pts.begin(), cart_pts.end() - 1);
      publishDrawnMarker(draw_pts, cycle_index);
    } else {
      RCLCPP_WARN(get_logger(), "Thực thi nét vẽ đường tròn thất bại.");
    }
  }

  void publishDrawnMarker(const std::vector<geometry_msgs::msg::Pose>& poses, int cycle_index)
  {
    static int marker_id = 10;
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = base_frame_;
    marker.header.stamp = this->now();
    marker.ns = "circle_drawn_path";
    marker.id = marker_id++;
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.008; // Nét dày 8mm

    // Màu sắc phân biệt theo chu trình:
    if (cycle_index % 2 == 1) {
      // Vòng 1: Xanh ngọc Neon Cyan (#00f5d4)
      marker.color.r = 0.0f;
      marker.color.g = 0.96f;
      marker.color.b = 0.83f;
    } else {
      // Vòng 2: Vàng ánh kim (#ffd166)
      marker.color.r = 1.0f;
      marker.color.g = 0.82f;
      marker.color.b = 0.40f;
    }
    marker.color.a = 1.0f;

    for (const auto& p : poses) {
      marker.points.push_back(p.position);
    }
    publishMarkerToAll(marker);
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CircleTrajectoryNode>();
  node->init();

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread spin_thread([&executor]() { executor.spin(); });

  node->run();

  rclcpp::shutdown();
  spin_thread.join();
  return 0;
}
