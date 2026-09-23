#include <memory>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <algorithm>

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

#include <opencv2/opencv.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

namespace fs = std::filesystem;

class ImageLetterWriterNode : public rclcpp::Node
{
public:
  ImageLetterWriterNode() : Node("image_letter_writer_node")
  {
    declare_parameter<std::string>("planning_group", "ur_manipulator");
    declare_parameter<std::string>("base_frame", "base_link");

    // Đường dẫn ảnh đầu vào. Nếu để trống sẽ tự động tìm ảnh mẫu hoặc sinh ảnh chữ cái
    declare_parameter<std::string>("image_path", "");
    declare_parameter<std::string>("letter_text", "T");

    // Vùng vẽ an toàn trong tầm với của UR3/UR3e (<0.50m từ gốc robot)
    declare_parameter<double>("origin_x", 0.05);  // Cách tâm robot 5cm sang phải
    declare_parameter<double>("origin_y", 0.28);  // Cách tâm robot 28cm về phía trước
    declare_parameter<double>("origin_z", 0.15);  // Độ cao mặt phẳng vẽ 15cm
    declare_parameter<double>("letter_width", 0.12);  // Chiều rộng chữ (12cm)
    declare_parameter<double>("letter_height", 0.15); // Chiều dài chữ (15cm)
    declare_parameter<double>("pen_lift", 0.04);      // Nhấc bút cao 4cm khi chuyển nét

    // Tham số xử lý ảnh và trích xuất đường bao (OpenCV)
    declare_parameter<std::string>("threshold_method", "otsu"); // "otsu" (ảnh đồ họa/scan) hoặc "adaptive" (ảnh chụp camera/chữ viết tay)
    declare_parameter<double>("epsilon_approx", 2.0);      // Sai số Douglas-Peucker (pixel)
    declare_parameter<double>("min_contour_area", 100.0);  // Diện tích contour tối thiểu bỏ nhiễu
    declare_parameter<double>("max_cartesian_step", 0.01); // 10mm bước nội suy tối đa giữa các điểm

    declare_parameter<double>("vel_scale", 0.15);
    declare_parameter<double>("acc_scale", 0.15);
    declare_parameter<int>("num_cycles", 2);

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

    // Thêm vật cản bảo vệ mặt sàn (z <= 0) để tuyệt đối không va chạm sàn
    addFloorCollisionObject();

    RCLCPP_INFO(get_logger(), "=== ImageLetterWriterNode khởi tạo thành công ===");
    RCLCPP_INFO(get_logger(), "Planning group: %s | Base frame: %s | End-effector: %s",
                group_name_.c_str(), base_frame_.c_str(), move_group_->getEndEffectorLink().c_str());
  }

  void run()
  {
    const int total_cycles = get_parameter("num_cycles").as_int();

    // ----------------------------------------------------
    // BƯỚC 1: Xử lý ảnh và trích xuất quỹ đạo (OpenCV Pipeline)
    // ----------------------------------------------------
    RCLCPP_INFO(get_logger(), "==========================================");
    RCLCPP_INFO(get_logger(), ">>> BẮT ĐẦU PIPELINE XỬ LÝ ẢNH CHỮ CÁI <<<");
    RCLCPP_INFO(get_logger(), "==========================================");

    cv::Mat input_img = loadOrGenerateImage();
    if (input_img.empty()) {
      RCLCPP_ERROR(get_logger(), "Không thể tải hoặc tạo ảnh chữ cái!");
      return;
    }

    auto strokes = extractStrokesFromImage(input_img);
    if (strokes.empty()) {
      RCLCPP_ERROR(get_logger(), "Không trích xuất được nét vẽ nào từ ảnh!");
      return;
    }

    RCLCPP_INFO(get_logger(), "Đã trích xuất thành công %zu đường nét từ ảnh.", strokes.size());

    // ----------------------------------------------------
    // BƯỚC 2: Điều khiển robot UR3/UR3e thực thi qua MoveIt 2
    // ----------------------------------------------------
    RCLCPP_INFO(get_logger(), "==========================================");
    RCLCPP_INFO(get_logger(), ">>> BẮT ĐẦU ĐIỀU KHIỂN ROBOT VẼ CHỮ <<<");
    RCLCPP_INFO(get_logger(), "==========================================");

    // Đưa robot về home
    RCLCPP_INFO(get_logger(), "Đang đưa robot về trạng thái ban đầu (home)...");
    move_group_->setStartStateToCurrentState();
    move_group_->setNamedTarget("home");
    planAndExecute();

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // Điểm tiếp cận trên không của nét vẽ đầu tiên
    const auto & initial_approach = strokes[0].front();
    RCLCPP_INFO(get_logger(), "Di chuyển tới vị trí đón bút trên không...");
    move_group_->setStartStateToCurrentState();
    move_group_->setPoseTarget(initial_approach);
    if (!planAndExecute()) {
      RCLCPP_ERROR(get_logger(), "Không thể tiếp cận điểm bắt đầu vẽ!");
      return;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Vòng lặp chu trình vẽ (mặc định 2 lần liên tiếp)
    for (int cycle = 1; cycle <= total_cycles; ++cycle) {
      RCLCPP_INFO(get_logger(), "==========================================");
      RCLCPP_INFO(get_logger(), ">>> BẮT ĐẦU CHU TRÌNH VẼ LẦN %d/%d <<<", cycle, total_cycles);
      RCLCPP_INFO(get_logger(), "==========================================");

      for (size_t s = 0; s < strokes.size(); ++s) {
        RCLCPP_INFO(get_logger(), "--- [Chu trình %d/%d] Đang vẽ Contour %zu/%zu (%zu điểm) ---",
                    cycle, total_cycles, s + 1, strokes.size(), strokes[s].size());
        drawStroke(strokes[s], cycle);

        // Chuyển tiếp trên không sang nét tiếp theo trong cùng chu trình
        if (s + 1 < strokes.size()) {
          RCLCPP_INFO(get_logger(), "Chuyển tiếp trên không sang Contour %zu...", s + 2);
          std::vector<geometry_msgs::msg::Pose> transition = { strokes[s + 1].front() };
          executeCartesianPath(transition);
          std::this_thread::sleep_for(std::chrono::milliseconds(300));
        }
      }

      RCLCPP_INFO(get_logger(), ">>> HOÀN THÀNH CHU TRÌNH %d/%d <<<", cycle, total_cycles);

      // Chuyển tiếp trên không quay lại đầu nét 1 cho chu trình tiếp theo
      if (cycle < total_cycles) {
        RCLCPP_INFO(get_logger(), "Chuyển tiếp trên không quay lại đầu nét vẽ cho chu trình tiếp theo...");
        std::vector<geometry_msgs::msg::Pose> loop_transition = { strokes[0].front() };
        executeCartesianPath(loop_transition);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      }
    }

    // Kết thúc: Nâng bút cao an toàn rồi trở về home
    RCLCPP_INFO(get_logger(), "==========================================");
    RCLCPP_INFO(get_logger(), ">>> HOÀN THÀNH TẤT CẢ %d CHU TRÌNH VẼ TỪ ẢNH! <<<", total_cycles);
    RCLCPP_INFO(get_logger(), "==========================================");

    geometry_msgs::msg::Pose safe_lift = strokes.back().back();
    safe_lift.position.z += 0.05; // Nâng thêm 5cm
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

  // ----------------------------------------------------
  // HÀM XỬ LÝ ẢNH (OPENCV PIPELINE)
  // ----------------------------------------------------

  // Nạp ảnh từ file hoặc tự động tạo ảnh chữ cái sắc nét
  cv::Mat loadOrGenerateImage()
  {
    std::string path = get_parameter("image_path").as_string();

    // 1. Nếu người dùng chỉ định đường dẫn hợp lệ
    if (!path.empty() && fs::exists(path)) {
      RCLCPP_INFO(get_logger(), "Đang đọc ảnh từ đường dẫn: %s", path.c_str());
      cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
      if (!img.empty()) {
        cv::imwrite("/tmp/image_letter_input.png", img);
        return img;
      }
      RCLCPP_WARN(get_logger(), "Đọc file ảnh thất bại, thử tìm ảnh mẫu mặc định...");
    }

    // 2. Thử tìm trong thư mục sample_images của package
    try {
      std::string pkg_share = ament_index_cpp::get_package_share_directory("ur3_letter_writer");
      std::string sample_path = pkg_share + "/sample_images/letter_T.png";
      if (fs::exists(sample_path)) {
        RCLCPP_INFO(get_logger(), "Đang dùng ảnh mẫu có sẵn: %s", sample_path.c_str());
        cv::Mat img = cv::imread(sample_path, cv::IMREAD_COLOR);
        if (!img.empty()) {
          cv::imwrite("/tmp/image_letter_input.png", img);
          return img;
        }
      }
    } catch (...) {
      // Bỏ qua lỗi tìm package
    }

    // 3. Tự động sinh ảnh chữ cái chất lượng cao bằng OpenCV
    std::string letter = get_parameter("letter_text").as_string();
    if (letter.empty()) letter = "T";

    RCLCPP_INFO(get_logger(), "Tự động sinh ảnh chữ cái '%s' bằng OpenCV...", letter.c_str());
    cv::Mat img = cv::Mat::ones(500, 500, CV_8UC3) * 255; // Nền trắng

    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 11.0;
    int thickness = 26;
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(letter, font_face, font_scale, thickness, &baseline);

    cv::Point text_org((500 - text_size.width) / 2, (500 + text_size.height) / 2);
    cv::putText(img, letter, text_org, font_face, font_scale, cv::Scalar(0, 0, 0), thickness, cv::LINE_AA);

    cv::imwrite("/tmp/image_letter_input.png", img);
    RCLCPP_INFO(get_logger(), "Đã lưu ảnh chữ cái đầu vào tại: /tmp/image_letter_input.png");
    return img;
  }

  // Toàn bộ pipeline: Threshold -> Contours -> Simplify -> Scale -> Waypoints
  std::vector<std::vector<geometry_msgs::msg::Pose>> extractStrokesFromImage(const cv::Mat & img)
  {
    // Bước 1: Grayscale + Lọc nhiễu Gaussian Blur
    cv::Mat gray, blurred;
    if (img.channels() == 3) {
      cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
      gray = img.clone();
    }

    // Lọc nhiễu mịn ảnh (khử hạt nhiễu camera, gân giấy)
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    // Kiểm tra độ sáng góc ảnh để tự động xác định nền sáng hay nền tối
    int corner_sum = blurred.at<uint8_t>(0, 0) + blurred.at<uint8_t>(0, blurred.cols - 1) +
                     blurred.at<uint8_t>(blurred.rows - 1, 0) + blurred.at<uint8_t>(blurred.rows - 1, blurred.cols - 1);
    bool is_white_background = (corner_sum / 4) > 128;

    cv::Mat bin_img;
    std::string method = get_parameter("threshold_method").as_string();

    if (method == "adaptive") {
      // Ngưỡng thích nghi cục bộ: Xử lý xuất sắc ảnh chụp thực tế từ camera bị bóng đổ, ánh sáng chênh lệch
      int block_size = 31;
      double c_val = 15.0;
      int thresh_type = is_white_background ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY;
      cv::adaptiveThreshold(blurred, bin_img, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type, block_size, c_val);
      RCLCPP_INFO(get_logger(), "Áp dụng Adaptive Thresholding (khử bóng đổ & chênh sáng)");
    } else {
      // Ngưỡng Otsu: Tự động tính ngưỡng toàn cục tối ưu cho ảnh đồ họa, scan rõ nét
      int thresh_type = (is_white_background ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY) | cv::THRESH_OTSU;
      cv::threshold(blurred, bin_img, 0, 255, thresh_type);
      RCLCPP_INFO(get_logger(), "Áp dụng Otsu Thresholding tự động");
    }

    // Xử lý hình thái học (Morphological Operations) để hàn gắn nét đứt và tẩy đốm mực li ti
    cv::Mat morph_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(bin_img, bin_img, cv::MORPH_OPEN, morph_kernel);  // Xóa đốm nhiễu li ti
    cv::morphologyEx(bin_img, bin_img, cv::MORPH_CLOSE, morph_kernel); // Hàn gắn các nét bút bị xước/đứt

    cv::imwrite("/tmp/image_letter_binary.png", bin_img);
    RCLCPP_INFO(get_logger(), "1. Binarization & Lọc nhiễu hoàn tất. Đã lưu: /tmp/image_letter_binary.png");

    // Bước 2: Contour Extraction (Trích xuất đường bao với cấu trúc cây)
    std::vector<std::vector<cv::Point>> raw_contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(bin_img, raw_contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

    double min_area = get_parameter("min_contour_area").as_double();
    double epsilon = get_parameter("epsilon_approx").as_double();

    std::vector<std::vector<cv::Point>> valid_contours;
    std::vector<cv::Point> all_points;

    // Bước 3: Simplify / Resample (Douglas-Peucker approximation)
    for (size_t i = 0; i < raw_contours.size(); ++i) {
      double area = cv::contourArea(raw_contours[i]);
      if (area < min_area) continue;

      std::vector<cv::Point> approx;
      cv::approxPolyDP(raw_contours[i], approx, epsilon, true);

      if (approx.size() >= 3) {
        valid_contours.push_back(approx);
        for (const auto & p : approx) {
          all_points.push_back(p);
        }
        RCLCPP_INFO(get_logger(), "  Contour %zu: diện tích = %.1f px^2, thô = %zu điểm -> tối ưu = %zu điểm",
                    valid_contours.size(), area, raw_contours[i].size(), approx.size());
      }
    }

    if (valid_contours.empty() || all_points.empty()) {
      RCLCPP_ERROR(get_logger(), "Không tìm thấy contour hợp lệ nào lớn hơn %.1f px^2!", min_area);
      return {};
    }

    // Vẽ ảnh debug các contour tìm được
    cv::Mat debug_contours = img.clone();
    cv::drawContours(debug_contours, valid_contours, -1, cv::Scalar(0, 0, 255), 2);
    cv::imwrite("/tmp/image_letter_contours.png", debug_contours);
    RCLCPP_INFO(get_logger(), "2. Contour Extraction & Simplify hoàn tất. Đã lưu: /tmp/image_letter_contours.png");

    // Bước 4: Scale + Translate (Pixel -> Không gian Cartesian của UR3)
    cv::Rect bbox = cv::boundingRect(all_points);
    RCLCPP_INFO(get_logger(), "Bounding Box ảnh: [x=%d, y=%d, w=%d, h=%d]",
                bbox.x, bbox.y, bbox.width, bbox.height);

    const double ox = get_parameter("origin_x").as_double();
    const double oy = get_parameter("origin_y").as_double();
    const double oz = get_parameter("origin_z").as_double();
    const double target_w = get_parameter("letter_width").as_double();
    const double target_h = get_parameter("letter_height").as_double();
    const double lift = get_parameter("pen_lift").as_double();
    const double max_step = get_parameter("max_cartesian_step").as_double();

    // Giữ nguyên tỉ lệ hình dạng chữ (Aspect Ratio)
    double scale = std::min(target_w / std::max(1, bbox.width), target_h / std::max(1, bbox.height));
    double real_w = bbox.width * scale;
    double real_h = bbox.height * scale;

    // Căn giữa chữ trong khung cho trước
    double x_offset = ox + (target_w - real_w) / 2.0;
    double y_offset = oy + (target_h - real_h) / 2.0;

    RCLCPP_INFO(get_logger(), "Tỉ lệ scale: %.6f m/px | Kích thước thực tế: %.3fm x %.3fm",
                scale, real_w, real_h);

    tf2::Quaternion q_down;
    q_down.setRPY(M_PI, 0.0, 0.0); // Mũi công tác chúc thẳng vuông góc mặt phẳng vẽ
    auto orientation_down = tf2::toMsg(q_down);

    // Bước 5: Chuyển đổi sang tập Cartesian Waypoints
    std::vector<std::vector<geometry_msgs::msg::Pose>> all_strokes;

    for (const auto & cnt : valid_contours) {
      std::vector<geometry_msgs::msg::Point> cart_2d_pts;

      // Đưa từng điểm pixel (u, v) sang tọa độ Cartesian (x, y)
      // Chú ý: Trục V trong ảnh hướng xuống, trong khi hệ robot trục Y hướng về phía trước
      for (const auto & pt : cnt) {
        geometry_msgs::msg::Point p;
        p.x = x_offset + (pt.x - bbox.x) * scale;
        p.y = y_offset + (bbox.height - (pt.y - bbox.y)) * scale;
        p.z = oz;
        cart_2d_pts.push_back(p);
      }

      // Đóng kín vòng contour (quay lại điểm đầu)
      cart_2d_pts.push_back(cart_2d_pts.front());

      // Nội suy thêm điểm nếu khoảng cách giữa 2 điểm liên tiếp lớn hơn max_step
      std::vector<geometry_msgs::msg::Point> dense_pts;
      for (size_t j = 0; j < cart_2d_pts.size() - 1; ++j) {
        dense_pts.push_back(cart_2d_pts[j]);
        double dx = cart_2d_pts[j + 1].x - cart_2d_pts[j].x;
        double dy = cart_2d_pts[j + 1].y - cart_2d_pts[j].y;
        double dist = std::hypot(dx, dy);

        if (dist > max_step) {
          int steps = static_cast<int>(std::ceil(dist / max_step));
          for (int s = 1; s < steps; ++s) {
            geometry_msgs::msg::Point interp;
            interp.x = cart_2d_pts[j].x + (dx * s) / steps;
            interp.y = cart_2d_pts[j].y + (dy * s) / steps;
            interp.z = oz;
            dense_pts.push_back(interp);
          }
        }
      }
      dense_pts.push_back(cart_2d_pts.back());

      // Xây dựng Stroke hoàn chỉnh: Pen_Up_Start -> Pen_Down_Path -> Pen_Up_End
      std::vector<geometry_msgs::msg::Pose> stroke;

      // 1. Điểm trên không đầu nét (Pen up)
      geometry_msgs::msg::Pose p_up_start;
      p_up_start.position = dense_pts.front();
      p_up_start.position.z = oz + lift;
      p_up_start.orientation = orientation_down;
      stroke.push_back(p_up_start);

      // 2. Toàn bộ quỹ đạo chạm mặt vẽ (Pen down)
      for (const auto & pt : dense_pts) {
        geometry_msgs::msg::Pose p_down;
        p_down.position = pt;
        p_down.position.z = oz;
        p_down.orientation = orientation_down;
        stroke.push_back(p_down);
      }

      // 3. Điểm trên không cuối nét (Pen up)
      geometry_msgs::msg::Pose p_up_end;
      p_up_end.position = dense_pts.back();
      p_up_end.position.z = oz + lift;
      p_up_end.orientation = orientation_down;
      stroke.push_back(p_up_end);

      all_strokes.push_back(stroke);
    }

    return all_strokes;
  }

  // ----------------------------------------------------
  // HÀM ĐIỀU KHIỂN MOVEIT 2
  // ----------------------------------------------------

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
    box_pose.position.z = -0.02; // Mặt trên tại z = 0.0

    floor_obj.primitives.push_back(primitive);
    floor_obj.primitive_poses.push_back(box_pose);
    floor_obj.operation = floor_obj.ADD;

    planning_scene_interface_.applyCollisionObject(floor_obj);
    RCLCPP_INFO(get_logger(), "Đã kích hoạt vật cản sàn bảo vệ (floor_table) tại z <= 0 trong Planning Scene.");
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
    const double jump_threshold = 4.0; // Ngăn chặn nhảy khớp

    move_group_->setStartStateToCurrentState();
    double fraction = move_group_->computeCartesianPath(
      waypoints, eef_step, jump_threshold, trajectory, true);

    RCLCPP_INFO(get_logger(), "  Cartesian path fraction = %.2f%% (%zu waypoints)",
                fraction * 100.0, waypoints.size());

    if (fraction < 0.85) {
      RCLCPP_ERROR(get_logger(),
        "Cartesian path không đạt yêu cầu (fraction = %.2f%% < 85%%). Kiểm tra lại tầm với robot.",
        fraction * 100.0);
      return false;
    }

    auto result = move_group_->execute(trajectory);
    return (result == moveit::core::MoveItErrorCode::SUCCESS);
  }

  void drawStroke(const std::vector<geometry_msgs::msg::Pose> & stroke_pts, int cycle_index)
  {
    if (stroke_pts.size() < 3) return;

    // Thực thi Cartesian path từ pen_down_start tới pen_up_end
    std::vector<geometry_msgs::msg::Pose> cart_pts(stroke_pts.begin() + 1, stroke_pts.end());

    if (executeCartesianPath(cart_pts)) {
      // Publish vệt vẽ cho các điểm chạm mặt vẽ
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
    marker.ns = "image_letter";
    marker.id = marker_id++;
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.006; // Nét dày 6mm

    // Màu sắc phân biệt theo chu trình
    if (cycle_index % 2 == 1) {
      marker.color.r = 0.1f;
      marker.color.g = 0.9f;
      marker.color.b = 0.2f; // Xanh lá
    } else {
      marker.color.r = 0.0f;
      marker.color.g = 0.85f;
      marker.color.b = 1.0f; // Xanh cyan
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
  auto node = std::make_shared<ImageLetterWriterNode>();
  node->init();

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread spin_thread([&executor]() { executor.spin(); });

  node->run();

  rclcpp::shutdown();
  spin_thread.join();
  return 0;
}
