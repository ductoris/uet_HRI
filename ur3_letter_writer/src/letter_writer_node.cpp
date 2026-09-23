#include <memory>
#include <vector>
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

class LetterWriterNode : public rclcpp::Node
{
public:
  LetterWriterNode() : Node("letter_writer_node")
  {
    declare_parameter<std::string>("planning_group", "ur_manipulator");
    declare_parameter<std::string>("base_frame", "base_link");

    // Tọa độ gốc vẽ chữ T (mặt phẳng ngang XY)
    // Chú ý: tất cả waypoint phải nằm trong tầm với UR3e (~0.50m từ gốc robot).
    // Với chữ rộng 12cm, cao 15cm:
    //   Điểm xa nhất ≈ sqrt((0.05+0.12)^2 + (0.28+0.15)^2) = sqrt(0.029+0.185) ≈ 0.462m < 0.50m ✓
    declare_parameter<double>("origin_x", 0.05); // Góc trái của chữ: 5cm sang phải
    declare_parameter<double>("origin_y", 0.28); // 28cm về phía trước robot
    declare_parameter<double>("origin_z", 0.15); // Cao 15cm so với mặt bàn (z=0)

    // Kích thước chữ T
    declare_parameter<double>("letter_width", 0.12);  // Rộng nét ngang 12cm
    declare_parameter<double>("letter_height", 0.15); // Dài nét dọc 15cm

    // Khoảng nhấc bút lên cao theo phương thẳng đứng (+Z)
    declare_parameter<double>("pen_lift", 0.04);      // Nhấc 4cm

    declare_parameter<double>("vel_scale", 0.15);
    declare_parameter<double>("acc_scale", 0.15);
    declare_parameter<int>("num_cycles", 2);          // Số chu trình lặp lại (mặc định 2 lần)

    // Publisher để hiển thị vệt nét vẽ trên RViz
    marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/drawn_letter_marker", 10);
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
    move_group_->setNumPlanningAttempts(10);
    move_group_->setGoalPositionTolerance(0.002);
    move_group_->setGoalOrientationTolerance(0.01);

    // Thêm vật cản sàn bảo vệ (floor_table) vào Planning Scene
    addFloorCollisionObject();

    RCLCPP_INFO(get_logger(), "Planning group: %s | Base frame: %s | End-effector: %s",
                group_name_.c_str(), base_frame_.c_str(), move_group_->getEndEffectorLink().c_str());
  }

  void run()
  {
    const int total_cycles = get_parameter("num_cycles").as_int();

    // Bước 0: Đưa robot về trạng thái ban đầu (home)
    RCLCPP_INFO(get_logger(), "==========================================");
    RCLCPP_INFO(get_logger(), "Đang đưa robot về trạng thái ban đầu (home)...");
    move_group_->setStartStateToCurrentState();
    move_group_->setNamedTarget("home");
    planAndExecute();

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    auto strokes = buildLetterT();
    if (strokes.empty()) {
      RCLCPP_ERROR(get_logger(), "Tập nét vẽ trống, dừng thực thi.");
      return;
    }

    // Điểm tiếp cận trên không đầu tiên của chữ T (đầu Nét 1)
    const auto & initial_approach_pose = strokes[0].front();

    RCLCPP_INFO(get_logger(), "Di chuyển tới vị trí tiếp cận trên không của chữ T...");
    move_group_->setStartStateToCurrentState();
    move_group_->setPoseTarget(initial_approach_pose);
    if (!planAndExecute()) {
      RCLCPP_ERROR(get_logger(), "Không thể tiếp cận vị trí bắt đầu vẽ!");
      return;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // VÒNG LẶP CHU TRÌNH (Lặp lại 2 lần liên tiếp tạo thành chu trình hoàn chỉnh)
    for (int cycle = 1; cycle <= total_cycles; ++cycle) {
      RCLCPP_INFO(get_logger(), "==========================================");
      RCLCPP_INFO(get_logger(), ">>> BẮT ĐẦU CHU TRÌNH VẼ LẦN %d/%d <<<", cycle, total_cycles);
      RCLCPP_INFO(get_logger(), "==========================================");

      for (size_t i = 0; i < strokes.size(); ++i) {
        RCLCPP_INFO(get_logger(), "--- [Chu trình %d/%d] Đang vẽ Nét %zu/%zu ---",
                    cycle, total_cycles, i + 1, strokes.size());
        drawStroke(strokes[i], cycle);

        // Nếu còn nét tiếp theo trong cùng chu trình: chuyển tiếp trên không bằng Cartesian path
        if (i + 1 < strokes.size()) {
          RCLCPP_INFO(get_logger(), "Chuyển tiếp trên không sang Nét %zu...", i + 2);
          std::vector<geometry_msgs::msg::Pose> transition_to_next = { strokes[i + 1].front() };
          executeCartesianPath(transition_to_next);
          std::this_thread::sleep_for(std::chrono::milliseconds(300));
        }
      }

      RCLCPP_INFO(get_logger(), ">>> HOÀN THÀNH CHU TRÌNH %d/%d <<<", cycle, total_cycles);

      // Nếu còn chu trình tiếp theo: chuyển tiếp trên không quay trở lại đầu Nét 1
      if (cycle < total_cycles) {
        RCLCPP_INFO(get_logger(), "Chuyển tiếp trên không quay lại đầu Nét 1 cho chu trình tiếp theo...");
        std::vector<geometry_msgs::msg::Pose> transition_to_cycle_start = { strokes[0].front() };
        executeCartesianPath(transition_to_cycle_start);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      }
    }

    // Kết thúc tất cả chu trình: nhấc bút lên cao an toàn rồi trở về home
    RCLCPP_INFO(get_logger(), "==========================================");
    RCLCPP_INFO(get_logger(), ">>> HOÀN THÀNH TẤT CẢ %d CHU TRÌNH VẼ CHỮ T! <<<", total_cycles);
    RCLCPP_INFO(get_logger(), "==========================================");

    // Nhấc bút lên cao thêm 5cm theo phương thẳng đứng
    geometry_msgs::msg::Pose safe_lift = strokes.back().back();
    safe_lift.position.z += 0.05;
    executeCartesianPath({ safe_lift });

    RCLCPP_INFO(get_logger(), "Đưa robot về trạng thái ban đầu (home)...");
    move_group_->setStartStateToCurrentState();
    move_group_->setNamedTarget("home");
    planAndExecute();

    RCLCPP_INFO(get_logger(), ">>> TOÀN BỘ QUÁ TRÌNH KẾT THÚC THÀNH CÔNG! <<<");
  }

private:
  std::string group_name_, base_frame_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;

  // Thêm mặt sàn va chạm vào Planning Scene để bảo vệ an toàn (z <= 0)
  void addFloorCollisionObject()
  {
    moveit_msgs::msg::CollisionObject floor_obj;
    floor_obj.header.frame_id = base_frame_;
    floor_obj.id = "floor_table";

    shape_msgs::msg::SolidPrimitive primitive;
    primitive.type = primitive.BOX;
    primitive.dimensions.resize(3);
    primitive.dimensions[primitive.BOX_X] = 2.0; // 2m rộng
    primitive.dimensions[primitive.BOX_Y] = 2.0; // 2m dài
    primitive.dimensions[primitive.BOX_Z] = 0.04; // 4cm dày

    geometry_msgs::msg::Pose box_pose;
    box_pose.orientation.w = 1.0;
    box_pose.position.x = 0.0;
    box_pose.position.y = 0.0;
    box_pose.position.z = -0.02; // Mặt trên của sàn phẳng đúng tại z = 0.0 (mặt đế robot)

    floor_obj.primitives.push_back(primitive);
    floor_obj.primitive_poses.push_back(box_pose);
    floor_obj.operation = floor_obj.ADD;

    planning_scene_interface_.applyCollisionObject(floor_obj);
    RCLCPP_INFO(get_logger(), "Đã kích hoạt vật cản sàn bảo vệ (floor_table) tại z <= 0 trong MoveIt Planning Scene.");
  }

  // Chuyển đổi tọa độ 2D của chữ cái sang Pose 3D trên mặt phẳng ngang XY
  // u: hoành độ chữ (theo trục X)
  // v: tung độ chữ (theo trục Y)
  // pen_down: true -> chạm mặt vẽ (Z = origin_z), false -> nhấc bút (Z = origin_z + pen_lift)
  geometry_msgs::msg::Pose toPose(double u, double v, bool pen_down)
  {
    const double ox = get_parameter("origin_x").as_double();
    const double oy = get_parameter("origin_y").as_double();
    const double oz = get_parameter("origin_z").as_double();
    const double lift = get_parameter("pen_lift").as_double();

    geometry_msgs::msg::Pose p;
    p.position.x = ox + u;
    p.position.y = oy + v;
    p.position.z = oz + (pen_down ? 0.0 : lift);

    // Hướng mũi công tác vuông góc chúc thẳng xuống sàn (Roll = 180 độ quanh trục X)
    tf2::Quaternion q;
    q.setRPY(M_PI, 0.0, 0.0);
    p.orientation = tf2::toMsg(q);

    return p;
  }

  // Tạo tập waypoints cho chữ 'T' trên mặt phẳng ngang XY
  std::vector<std::vector<geometry_msgs::msg::Pose>> buildLetterT()
  {
    const double W = get_parameter("letter_width").as_double();
    const double H = get_parameter("letter_height").as_double();

    std::vector<std::vector<geometry_msgs::msg::Pose>> strokes;

    // ==========================================
    // NÉT 1: Thanh ngang trên đỉnh (Từ Trái sang Phải)
    // Tại v = H, chạy từ u = 0 đến u = W
    // ==========================================
    {
      std::vector<geometry_msgs::msg::Pose> s;
      s.push_back(toPose(0.0, H, false)); // 1. Trên không điểm đầu nét ngang (pen up)
      s.push_back(toPose(0.0, H, true));  // 2. Hạ bút chạm mặt phẳng (pen down)
      s.push_back(toPose(W, H, true));    // 3. Kẻ nét ngang sang phải (pen down)
      s.push_back(toPose(W, H, false));   // 4. Nhấc bút lên (pen up)
      strokes.push_back(s);
    }

    // ==========================================
    // NÉT 2: Thanh dọc chính giữa (Từ Đỉnh xuống Chân)
    // Tại u = W/2, chạy từ v = H xuống v = 0
    // ==========================================
    {
      std::vector<geometry_msgs::msg::Pose> s;
      s.push_back(toPose(W / 2.0, H, false)); // 5. Trên không giữa nét ngang (pen up)
      s.push_back(toPose(W / 2.0, H, true));  // 6. Hạ bút chạm mặt phẳng (pen down)
      s.push_back(toPose(W / 2.0, 0.0, true));// 7. Kẻ dọc thẳng xuống chân (pen down)
      s.push_back(toPose(W / 2.0, 0.0, false));// 8. Nhấc bút lên kết thúc (pen up)
      strokes.push_back(s);
    }

    return strokes;
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

  // Thực thi quỹ đạo Cartesian path xác định, mượt mà và an toàn
  bool executeCartesianPath(const std::vector<geometry_msgs::msg::Pose> & waypoints)
  {
    if (waypoints.empty()) return true;

    moveit_msgs::msg::RobotTrajectory trajectory;
    const double eef_step = 0.005; // 5mm bước nội suy
    const double jump_threshold = 4.0; // Phát hiện và ngăn chặn nhảy khớp đột ngột

    move_group_->setStartStateToCurrentState();
    double fraction = move_group_->computeCartesianPath(
      waypoints, eef_step, jump_threshold, trajectory, true);

    RCLCPP_INFO(get_logger(), "  computeCartesianPath fraction = %.2f%% (%zu waypoints)",
                fraction * 100.0, waypoints.size());

    if (fraction < 0.85) {
      RCLCPP_ERROR(get_logger(),
        "Cartesian path không đạt yêu cầu (fraction = %.2f%% < 85%%). Kiểm tra lại tọa độ vùng vẽ.",
        fraction * 100.0);
      return false;
    }

    auto result = move_group_->execute(trajectory);
    return (result == moveit::core::MoveItErrorCode::SUCCESS);
  }

  // Vẽ một nét chữ hoàn chỉnh: robot đã ở vị trí pen_up đầu nét
  void drawStroke(const std::vector<geometry_msgs::msg::Pose> & stroke_pts, int cycle_index)
  {
    // stroke_pts format: [pen_up_start, pen_down_start, ..., pen_down_end, pen_up_end]
    if (stroke_pts.size() < 3) {
      RCLCPP_WARN(get_logger(), "Nét vẽ quá ít điểm (%zu), bỏ qua.", stroke_pts.size());
      return;
    }

    // Robot đã ở trên không điểm đầu nét vẽ (pen_up_start).
    // Thực thi Cartesian path từ pen_down_start tới pen_up_end:
    std::vector<geometry_msgs::msg::Pose> cart_pts(stroke_pts.begin() + 1, stroke_pts.end());

    if (executeCartesianPath(cart_pts)) {
      // Publish Marker cho các điểm chạm mặt vẽ (bỏ điểm nhấc bút cuối cùng)
      std::vector<geometry_msgs::msg::Pose> draw_pts(cart_pts.begin(), cart_pts.end() - 1);
      publishMarker(draw_pts, cycle_index);
    } else {
      RCLCPP_WARN(get_logger(), "Thực thi nét vẽ thất bại.");
    }
  }

  void publishMarker(const std::vector<geometry_msgs::msg::Pose>& poses, int cycle_index)
  {
    static int marker_id = 0;
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = base_frame_;
    marker.header.stamp = this->now();
    marker.ns = "letter_t";
    marker.id = marker_id++;
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.008; // Nét dày 8mm

    // Màu sắc phân biệt giữa các chu trình:
    if (cycle_index % 2 == 1) {
      // Chu trình 1 (lẻ): Màu xanh lá tươi
      marker.color.r = 0.1f;
      marker.color.g = 0.9f;
      marker.color.b = 0.2f;
    } else {
      // Chu trình 2 (chẵn): Màu xanh cyan tươi sáng
      marker.color.r = 0.0f;
      marker.color.g = 0.85f;
      marker.color.b = 1.0f;
    }
    marker.color.a = 1.0f;

    for (const auto& p : poses) {
      marker.points.push_back(p.position);
    }
    marker_pub_->publish(marker);
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<LetterWriterNode>();
  node->init();

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread spin_thread([&executor]() { executor.spin(); });

  node->run();

  rclcpp::shutdown();
  spin_thread.join();
  return 0;
}
