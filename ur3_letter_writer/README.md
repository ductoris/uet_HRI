# ur3_letter_writer

Package ROS 2 điều khiển tay máy UR3/UR3e vẽ chữ cái bằng MoveIt 2 Cartesian path.

Package cung cấp **2 Node** giải quyết 2 bài toán:
1. **`letter_writer_node`**: Tự thiết kế waypoints hình học (chữ cái "T", lặp lại 2 chu trình liên tiếp, chống đập sàn).
2. **`image_letter_writer_node`**: Tạo trajectory tự động từ ảnh chữ cái bất kỳ bằng OpenCV (Threshold $\to$ Contour Extraction $\to$ Simplify $\to$ Scale $\to$ Cartesian Waypoints $\to$ MoveIt).

## Cấu trúc Package

```
ur3_letter_writer/
├── CMakeLists.txt
├── package.xml
├── src/
│   ├── letter_writer_node.cpp          # Node 1: vẽ chữ T theo waypoint hình học
│   └── image_letter_writer_node.cpp    # Node 2: trích xuất contour từ ảnh bằng OpenCV
├── launch/
│   ├── bringup_ur3e.launch.py          # File 1: Khởi động UR3e + Gazebo + MoveIt 2
│   ├── write_letter.launch.py          # File 2A: Chạy Node 1 vẽ chữ T
│   └── write_image_letter.launch.py    # File 2B: Chạy Node 2 vẽ chữ từ ảnh
└── sample_images/                      # Thư mục ảnh mẫu kiểm thử
    ├── letter_T.png
    ├── letter_M.png
    └── letter_A.png
```

---

## Biên dịch (Build)

```bash
cd ~/workspaces/ur_gz
source /opt/ros/humble/setup.bash
colcon build --packages-select ur3_letter_writer --symlink-install
source install/setup.bash
```

---

## Hướng dẫn chạy

Chạy theo 2 bước ở 2 terminal:

### **Terminal 1: Khởi động mô phỏng Gazebo & MoveIt 2**
```bash
source /opt/ros/humble/setup.bash
source ~/workspaces/ur_gz/install/setup.bash
ros2 launch ur3_letter_writer bringup_ur3e.launch.py ur_type:=ur3e
```
*Đợi đến khi Gazebo và RViz hiển thị robot đứng yên ổn định.*

---

### **Cách 1: Chạy Node 1 - Vẽ chữ T theo Waypoints (Lặp lại 2 chu trình)**
```bash
source /opt/ros/humble/setup.bash
source ~/workspaces/ur_gz/install/setup.bash
ros2 launch ur3_letter_writer write_letter.launch.py
```

---

### **Cách 2: Chạy Node 2 - Vẽ chữ tự động từ Ảnh (Image Letter Writer)**

#### 2.1. Chạy với ảnh mẫu chữ T mặc định (hoặc tự động sinh):
```bash
source /opt/ros/humble/setup.bash
source ~/workspaces/ur_gz/install/setup.bash
ros2 launch ur3_letter_writer write_image_letter.launch.py
```

#### 2.2. Chạy với ảnh mẫu chữ khác (ví dụ: chữ M hoặc chữ A):
```bash
ros2 launch ur3_letter_writer write_image_letter.launch.py \
  image_path:=/home/ductri/workspaces/ur_gz/src/ur3_letter_writer/sample_images/letter_M.png
```

```bash
ros2 launch ur3_letter_writer write_image_letter.launch.py \
  image_path:=/home/ductri/workspaces/ur_gz/src/ur3_letter_writer/sample_images/letter_A.png
```

#### 2.3. Tự động sinh và vẽ chữ cái bất kỳ (không cần file ảnh):
```bash
ros2 launch ur3_letter_writer write_image_letter.launch.py letter_text:=B
```

---

### **Cách 3: Chạy Node 3 - Quỹ đạo hình tròn (`draw_circle.launch.py`)**

Launch file riêng biệt thực hiện quỹ đạo đường tròn $360^\circ$ hoàn chỉnh:

```bash
source /opt/ros/humble/setup.bash
source ~/workspaces/ur_gz/install/setup.bash
ros2 launch ur3_letter_writer draw_circle.launch.py
```

- **Tùy chỉnh bán kính hoặc vị trí tâm:**
  ```bash
  ros2 launch ur3_letter_writer draw_circle.launch.py radius:=0.10 center_y:=0.35 num_cycles:=2
  ```

---

## Hướng dẫn quay video nộp bài (Upload Google Drive chia sẻ công khai)

1. **Chuẩn bị màn hình quan sát**:
   - Mở đồng thời cả 2 cửa sổ: **Gazebo** (quan sát robot 3D) và **RViz** (quan sát vệt Marker quỹ đạo).
   - Trong RViz, topic `/drawn_circle_marker` (hoặc `/drawn_letter_marker`) sẽ tự động hiển thị vệt vẽ màu neon rất nổi bật.
2. **Quay video màn hình trên Ubuntu**:
   - Nhấn tổ hợp phím `Ctrl + Alt + Shift + R` (hoặc dùng công cụ `SimpleScreenRecorder` / `OBS Studio` / phím chụp màn hình `PrintScreen` chọn chế độ Video).
   - Chạy lệnh ở Terminal 2 để robot thực hiện chu trình vẽ.
   - Khi robot hoàn thành và quay về `home`, dừng quay video.
3. **Upload Google Drive & Chia sẻ công khai**:
   - Tải file video (`.mp4` / `.webm`) lên Google Drive của bạn.
   - Nhấp chuột phải vào video $\to$ chọn **Chia sẻ (Share)** $\to$ chuyển từ **Bị hạn chế (Restricted)** sang **Bất kỳ ai có đường liên kết (Anyone with the link)** với quyền **Người xem (Viewer)**.
   - Sao chép đường liên kết và nộp bài.

---

## Pipeline xử lý ảnh trong `image_letter_writer_node`

1. **Load/Generate Image**: Đọc file PNG/JPG hoặc tự động sinh ảnh chữ cái 500x500.
2. **Threshold / Binarization**: Chuyển Grayscale, tự động nhận diện màu nền và áp dụng Otsu Thresholding $\to$ lưu tại `/tmp/image_letter_binary.png`.
3. **Contour Extraction**: Sử dụng `cv::findContours` với cờ `cv::RETR_TREE` để lấy cả đường bao ngoài lẫn lỗ rỗng bên trong (như chữ A, B, O).
4. **Simplify / Resample**: Áp dụng Douglas-Peucker `cv::approxPolyDP` để tối ưu hóa số lượng điểm, khử răng cưa $\to$ lưu ảnh debug tại `/tmp/image_letter_contours.png`.
5. **Scale & Translate**: Chuyển đổi tọa độ pixel $(u, v)$ sang mặt phẳng làm việc Cartesian $(x, y)$ của robot UR3e (bảo toàn tỉ lệ Aspect Ratio, an toàn trong bán kính $< 0.46\text{m}$).
6. **Cartesian Waypoints & MoveIt**: Thêm tiếp cận trên không (pen-up), chạm mặt vẽ (pen-down), nhấc bút an toàn và thực thi qua `computeCartesianPath` với vật cản sàn bảo vệ chống đập sàn.

---

## Bảng tham số chính

| Tham số | Ý nghĩa | Mặc định |
|---|---|---|
| `image_path` | Đường dẫn file ảnh đầu vào (PNG/JPG) | `""` (dùng ảnh mẫu hoặc tự sinh) |
| `letter_text` | Ký tự cần sinh tự động nếu không truyền ảnh | `'T'` |
| `origin_x, origin_y, origin_z` | Gốc tọa độ của chữ trong khung `base_link` | `0.05, 0.28, 0.15` |
| `letter_width, letter_height` | Kích thước vùng vẽ chữ tối đa (m) | `0.12, 0.15` |
| `pen_lift` | Khoảng nhấc bút lên khi chuyển nét (m) | `0.04` |
| `num_cycles` | Số chu trình lặp lại | `2` |
| `epsilon_approx` | Sai số xấp xỉ contour của Douglas-Peucker (pixel) | `2.0` |
| `min_contour_area` | Diện tích contour tối thiểu để lọc bỏ nhiễu | `100.0` |
| `vel_scale, acc_scale` | Hệ số giới hạn vận tốc/gia tốc | `0.15, 0.15` |
