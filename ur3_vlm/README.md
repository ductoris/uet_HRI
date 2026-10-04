# UR3/UR3e Vision-Language Model (VLM) & Computer Vision Control (Bài Thực Hành Tuần 3)

[![ROS 2](https://img.shields.io/badge/ROS%202-Humble-blue.svg)](https://docs.ros.org/en/humble/)
[![Gazebo](https://img.shields.io/badge/Gazebo-Ignition%20Fortress-orange.svg)](https://gazebosim.org/)
[![MoveIt 2](https://img.shields.io/badge/MoveIt%202-Humble-green.svg)](https://moveit.picknik.ai/humble/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-yellow.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)


---

## 🌟 Giới thiệu

Package `ur3_vlm` hiện thực hóa hệ thống điều khiển tay máy cộng tác **Universal Robots UR3/UR3e** bằng **Vision-Language Model (VLM)** và **Thị giác máy tính (Computer Vision Pipeline)** trong môi trường bàn thao tác tương tác (tabletop manipulation) trên nền tảng **ROS 2 Humble**, **Gazebo Ignition**, và **MoveIt 2**.

### Các tính năng cốt lõi:
1. **Pipeline Thị giác Máy tính (Camera Perception):**
   - Thu nhận luồng ảnh RGB thời gian thực từ camera quan sát trên cao (Overhead Camera qua `ros_gz_bridge`).
   - Phân đoạn màu sắc HSV không chồng lấn cho **5 khối lập phương động** (Red, Yellow, Blue, Green, Purple) và **4 khay vùng mục tiêu** (Zone A, Zone B, Zone C, Staging Area / Zone T).
   - Mô hình quang học **Pinhole Camera** ước lượng tọa độ 3D Metric trong hệ quy chiếu `base_link` của robot với sai số $< 0.5\,\text{mm}$.
   - Thuật toán **Dynamic Occupancy Check** tự động xác định khối nào đang chiếm chỗ trong khay nào, xuất trạng thái `WorldState` JSON chuẩn hóa.

2. **VLM Task Planner Đa Nền Tảng (Multi-Backend Planner):**
   - Hỗ trợ **Google Gemini API** (`gemini-1.5-flash`, `gemini-2.0-flash`).
   - Hỗ trợ **OpenAI-compatible / 9Router API** (`gpt-4o-mini`, vLLM, Ollama).
   - Bộ lập kế hoạch dự phòng **Offline Heuristic Planner** chạy độc lập 100% không cần mạng internet.
   - Hỗ trợ câu lệnh ngôn ngữ tự nhiên song ngữ (Tiếng Việt và Tiếng Anh).

3. **Bộ Kỹ Năng Thao Tác Mở Rộng (Extended Skill Set):**
   - `pick(object)`: Tiếp cận, hạ gắp và kẹp khóa vật thể.
   - `place(object, target)`: Di chuyển, hạ đặt vào khay mục tiêu.
   - `stack(bottom_object, top_object)`: Xếp chồng khối lên khối khác.
   - `unstack(top_object, target)`: Dỡ khối trên đỉnh tháp và dời sang khay.
   - `move_to(target)`: Di chuyển đầu công tác đến vị trí chỉ định ở cao độ an toàn.
   - `home()`: Đưa cánh tay về tư thế khớp an toàn mặc định.

4. **Tối Ưu Hóa Động Học & Chống Vung Khớp:**
   - Module `ur3_kinematics.py` giải động học nghịch (IK) bằng thuật toán tối ưu hóa phi tuyến đa mầm (Multi-Seed L-BFGS-B), giữ đầu kẹp luôn chúc thẳng xuống mặt bàn và triệt tiêu hoàn toàn hiện tượng nhảy nghiệm / lật khuỷu.

---

## 📁 Cấu trúc Thư mục Package

```
ur3_vlm/
├── .env                              # Khai báo API keys (GEMINI_API_KEY, OPENAI_API_KEY)
├── package.xml                       # Định nghĩa dependencies (rclpy, cv_bridge, moveit_msgs, ...)
├── setup.py                          # Cấu hình entry points và share data files
├── setup.cfg                         # Cấu hình cài đặt
├── test_percep.py                    # Script kiểm thử độc lập pipeline thị giác
├── ur3_vlm_report.tex                # Báo cáo học phần Tuần 3 định dạng LaTeX chi tiết
├── README.md                         # Tài liệu hướng dẫn sử dụng
├── config/
│   └── scene.yaml                    # Thông số hình học môi trường, camera, khay & khối
├── launch/
│   └── vlm_robot.launch.py           # Launch file tự động khởi động toàn bộ hệ thống
├── worlds/
│   └── vlm_world.sdf                 # Môi trường thế giới mô phỏng trong Gazebo Ignition
└── ur3_vlm/
    ├── __init__.py
    ├── camera_perception.py          # Pipeline Computer Vision & Pinhole 3D Projection
    ├── llm_planner.py                # VLM Task Planner đa backend & Offline Heuristic
    ├── robot_skills.py               # 6 kỹ năng MoveIt 2 & đồng bộ vật lý Gazebo
    ├── scene_spawner.py              # Sinh bàn, giá camera, 5 khối màu & khay zone bitmap
    ├── ur3_kinematics.py             # Bộ giải FK/IK tối ưu hóa liên tục góc khớp
    ├── gripper_joint_state_publisher.py # Khử cảnh báo MoveGroup cho khớp kẹp ảo
    └── vlm_console.py                # Giao diện dòng lệnh tương tác End-to-End
```

---

## 🚀 Hướng dẫn Cài đặt & Chạy Thực nghiệm

### 1. Biên dịch Package
```bash
cd ~/workspaces/ur_gz
source /opt/ros/humble/setup.bash
colcon build --packages-select ur3_vlm
source install/setup.bash
```

### 2. Thiết lập Biến Môi trường (.env)
Tạo hoặc chỉnh sửa file `src/ur3_vlm/.env`:
```bash
GEMINI_API_KEY="your_gemini_api_key_here"

# Tùy chọn 9Router / OpenAI
OPENAI_API_KEY="your_openai_or_9router_key"
OPENAI_BASE_URL="https://api.9router.com/v1"
OPENAI_MODEL="gpt-4o-mini"
```

### 3. Khởi động Toàn Bộ Hệ Thống
Hệ thống cung cấp file launch 1 chạm tự động cấu hình Static TF, camera bridge, spawn bàn ghế/khối màu, MoveIt 2 và RViz:
```bash
# Terminal 1: Khởi chạy mô phỏng và Vision Pipeline
source install/setup.bash
ros2 launch ur3_vlm vlm_robot.launch.py
```

### 4. Khởi chạy Giao diện Điều khiển VLM Console
```bash
# Terminal 2: Khởi chạy Console tương tác
source install/setup.bash
ros2 run ur3_vlm vlm_console
```

### 5. Kiểm thử Pipeline Thị giác Độc lập
```bash
# Kiểm tra độ chính xác nhận diện và tọa độ 3D từ ảnh camera
python3 src/ur3_vlm/test_percep.py
```

---

## 💬 Kịch bản Câu lệnh Mẫu

| Loại tác vụ | Câu lệnh tiếng Việt | English Command |
| :--- | :--- | :--- |
| **Gắp đặt đơn** | `Gắp khối màu đỏ đặt vào vùng A` | `Please put the red cube in zone A.` |
| **Xếp chồng (Stack)** | `Xếp khối màu xanh lá lên trên khối màu xanh dương` | `Stack the green cube on top of the blue cube.` |
| **Dỡ khối (Unstack)** | `Dỡ khối màu xanh lá chuyển sang vùng B` | `Unstack the green cube and place it into zone B.` |
| **Sắp xếp theo MSSV** | `Sắp xếp các khối theo mã số sinh viên của tôi` | `Arrange all objects according to my student ID.` |
| **Tư thế an toàn** | `Đưa robot về vị trí an toàn` | `Go back to home position.` |

---
