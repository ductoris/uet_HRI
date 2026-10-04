#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
camera_perception.py — Pipeline thị giác máy tính cho UR3e VLM (Bài 03).

Chức năng:
  1. Nhận luồng ảnh từ overhead camera (/camera/image_raw).
  2. Phân đoạn màu sắc (HSV segmentation) + phân tích contour để nhận diện:
     - 5 khối lập phương (Red, Yellow, Blue, Green, Purple)
     - 3 khay Zone (Zone A, Zone B, Zone C)
  3. Chuyển đổi tọa độ từ pixel (u, v) sang tọa độ thế giới thực (x, y, z)
     trong hệ base_link của robot UR3e bằng mô hình Pinhole Camera.
  4. Xác định trạng thái chiếm dụng (Occupancy Check): khối nào đang nằm trong zone nào.
  5. Xuất cấu trúc WorldState (JSON) chuẩn phục vụ LLM Planner.
  6. Vẽ bounding box, tâm tọa độ, nhãn nhận diện lên ảnh debug (/tmp/perception_debug.jpg)
     và publish lên topic /camera/perception_annotated.

Sinh viên: Mai Duc Tri | MSSV: 23020776
"""

import os
import math
import json
import yaml
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from ament_index_python.packages import get_package_share_directory

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError


# ===========================================================================
#  Dải màu HSV chuẩn cho 5 khối vật thể và 3 khay Zone (Hoàn toàn không chồng lấn)
# ===========================================================================
HSV_RANGES = {
    # 5 Khối vật thể động (4cm x 4cm)
    "red_cube": [
        (np.array([0, 100, 60]), np.array([9, 255, 255])),
        (np.array([170, 100, 60]), np.array([180, 255, 255])),
    ],
    "yellow_cube": [
        (np.array([21, 100, 80]), np.array([38, 255, 255])),
    ],
    "green_cube": [
        (np.array([40, 80, 50]), np.array([80, 255, 255])),
    ],
    "blue_cube": [
        (np.array([98, 60, 50]), np.array([126, 255, 255])),
    ],
    "purple_cube": [
        (np.array([128, 60, 50]), np.array([152, 255, 255])),
    ],
    # 3 Khay Zone (7cm x 7cm)
    "zone_a": [  # Cam (Orange)
        (np.array([10, 100, 70]), np.array([20, 255, 255])),
    ],
    "zone_b": [  # Teal / Cyan
        (np.array([82, 70, 60]), np.array([96, 255, 255])),
    ],
    "zone_c": [  # Hồng / Mauve
        (np.array([154, 30, 70]), np.array([169, 255, 255])),
    ],
}

# Màu vẽ BGR phục vụ hiển thị debug
DISPLAY_COLORS = {
    "red_cube": (0, 0, 255),
    "yellow_cube": (0, 220, 255),
    "blue_cube": (255, 50, 0),
    "green_cube": (0, 220, 0),
    "purple_cube": (220, 0, 180),
    "zone_a": (0, 140, 255),
    "zone_b": (200, 200, 0),
    "zone_c": (200, 100, 200),
    "staging_area": (180, 180, 180),
    "zone_t": (180, 180, 180),
}


# ===========================================================================
#  Lớp CameraPerception
# ===========================================================================
class CameraPerception:
    """Core logic xử lý ảnh và trích xuất WorldState không phụ thuộc ROS."""

    def __init__(self, scene_config: dict = None):
        self.config = scene_config or {}
        
        # Thông số camera (treo tại Z = 0.83m)
        cam_cfg = self.config.get("camera", {})
        cam_pos = cam_cfg.get("position", {"x": 0.40, "y": 0.0, "z": 0.83})
        self.cam_x = float(cam_pos.get("x", 0.40))
        self.cam_y = float(cam_pos.get("y", 0.0))
        self.cam_z = float(cam_pos.get("z", 0.83))
        
        self.img_w = int(cam_cfg.get("image_width", 640))
        self.img_h = int(cam_cfg.get("image_height", 480))
        self.hfov = float(cam_cfg.get("horizontal_fov", 1.04719755))  # 60 deg
        
        # Tiêu cự tính từ HFov
        self.fx = (self.img_w / 2.0) / math.tan(self.hfov / 2.0)
        self.fy = self.fx  # Square pixels
        self.cx = self.img_w / 2.0
        self.cy = self.img_h / 2.0
        
        # Chiều cao mặt bàn và kích thước khối
        heights = self.config.get("heights", {})
        self.table_surface_z = float(heights.get("table_surface", 0.0))
        self.cube_size = float(heights.get("cube_size", 0.04))
        
        # Bán kính kiểm tra chiếm dụng
        zone_det = self.config.get("zone_detection", {})
        self.occupancy_radius = float(zone_det.get("occupancy_radius", 0.04))

    def pixel_to_robot(self, u: float, v: float, target_z: float = 0.04) -> tuple:
        """
        Chuyển đổi từ tọa độ pixel (u, v) trên ảnh overhead sang tọa độ (x, y, z)
        trong hệ base_link của robot UR3e.
        
        Quy ước hình học:
          - Camera nhìn thẳng xuống tâm bàn (pitch = 90 deg).
          - Trục +X của robot hướng từ dưới lên trên trong ảnh (tương ứng v giảm).
          - Trục +Y của robot hướng từ phải sang trái trong ảnh (tương ứng u giảm).
        """
        dz = self.cam_z - target_z
        # Pinhole projection
        dx_robot = (self.cy - v) * (dz / self.fy)
        dy_robot = (self.cx - u) * (dz / self.fx)
        
        x_robot = self.cam_x + dx_robot
        y_robot = self.cam_y + dy_robot
        return round(x_robot, 4), round(y_robot, 4), round(target_z, 4)

    def process_image(self, cv_image: np.ndarray) -> tuple:
        """
        Phân tích ảnh RGB:
          Returns (world_state_dict, annotated_image)
        """
        if cv_image is None:
            return {}, None

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        annotated = cv_image.copy()

        detected_objects = {}
        detected_zones = {}

        # ── 1. Phát hiện 5 khối lập phương ──
        for obj_name in ["red_cube", "yellow_cube", "blue_cube", "green_cube", "purple_cube"]:
            ranges = HSV_RANGES[obj_name]
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for (lower, upper) in ranges:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

            # Lọc nhiễu
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                valid_cnts = [c for c in contours if cv2.contourArea(c) > 35]
                if valid_cnts:
                    c = max(valid_cnts, key=cv2.contourArea)
                    rect = cv2.minAreaRect(c)
                    (u, v), (rw, rh), angle_deg = rect
                    
                    # Chiếu tâm khối tại Z = 0.02m (độ cao tâm thực tế của khối 4x4cm trên bàn)
                    rx, ry, _ = self.pixel_to_robot(u, v, target_z=0.02)
                    rz = 0.02
                    
                    # Tính góc quay yaw của khối trong hệ base_link
                    box_pts = cv2.boxPoints(rect)
                    edge_vec = box_pts[1] - box_pts[0]
                    yaw_robot = math.atan2(-edge_vec[0], -edge_vec[1])
                    yaw_norm = ((yaw_robot + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0
                    
                    color_name = obj_name.replace("_cube", "")
                    detected_objects[obj_name] = {
                        "name": obj_name,
                        "color": color_name,
                        "pixel": [round(u, 1), round(v, 1)],
                        "position": [rx, ry, rz],
                        "yaw": round(float(yaw_norm), 4),
                        "in_zone": None,
                    }

                    # Vẽ Oriented Bounding Box và tâm sub-pixel
                    color_bgr = DISPLAY_COLORS.get(obj_name, (255, 255, 255))
                    int_pts = np.int0(box_pts)
                    cv2.drawContours(annotated, [int_pts], 0, color_bgr, 2)
                    cv2.circle(annotated, (int(u), int(v)), 3, (0, 0, 255), -1)
                    label_str = f"{color_name.upper()} ({rx:.2f},{ry:.2f})"
                    bx, by, bw, bh = cv2.boundingRect(c)
                    cv2.putText(annotated, label_str, (bx, max(15, by - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1, cv2.LINE_AA)

        # ── 2. Chuẩn hóa & Render các khay Zone tại vị trí tâm hình học tuyệt đối ──
        cfg_zones = self.config.get("zones", {})
        for zone_name, zcfg in cfg_zones.items():
            label_cfg = zcfg.get("label", zone_name)
            char_key = 'T' if ("staging" in zone_name.lower() or "t" in label_cfg.lower()) else zone_name[-1].upper()
            if char_key not in ['A', 'B', 'C', 'T']:
                for k in ['A', 'B', 'C', 'T']:
                    if k in label_cfg.upper() or k in zone_name.upper():
                        char_key = k
                        break
            label_display = f"Zone {char_key}" if char_key in ['A', 'B', 'C', 'T'] else label_cfg
            
            # Lấy tọa độ chuẩn tâm khay từ cấu hình
            zp = zcfg.get("position", {})
            std_x = float(zp.get("x", 0.45))
            std_y = float(zp.get("y", 0.0))
            std_z = float(zp.get("z", 0.0025))
            
            # Tính tọa độ pixel chuẩn của tâm khay theo pinhole projection
            u_std = int(self.cx - (std_y - self.cam_y) * (self.fx / (self.cam_z - std_z)))
            v_std = int(self.cy - (std_x - self.cam_x) * (self.fy / (self.cam_z - std_z)))
            
            detected_zones[zone_name] = {
                "name": zone_name,
                "label": label_display,
                "pixel": [u_std, v_std],
                "position": [std_x, std_y, std_z],
                "occupant": None,
            }

            # Vẽ viền khay zone chuẩn xác bao quanh tâm
            half_tray_px = int((0.10 / (self.cam_z - std_z)) * self.fx / 2.0)
            color_bgr = DISPLAY_COLORS.get(zone_name, (180, 180, 180))
            cv2.rectangle(annotated, 
                          (u_std - half_tray_px, v_std - half_tray_px), 
                          (u_std + half_tray_px, v_std + half_tray_px), 
                          color_bgr, 2)
            cv2.circle(annotated, (u_std, v_std), 4, color_bgr, -1)
            cv2.putText(annotated, f"[{char_key}] {label_display}", 
                        (u_std - half_tray_px, v_std + half_tray_px + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color_bgr, 2, cv2.LINE_AA)

        # ── 3. Kiểm tra chiếm dụng (Occupancy Check) ──
        for obj_name, obj_data in detected_objects.items():
            ox, oy, _ = obj_data["position"]
            for zone_name, zone_data in detected_zones.items():
                zx, zy, _ = zone_data["position"]
                dist = math.hypot(ox - zx, oy - zy)
                if dist <= self.occupancy_radius:
                    obj_data["in_zone"] = zone_name
                    zone_data["occupant"] = obj_name
                    break

        # ── 4. Tổng hợp WorldState Dictionary ──
        world_state = {
            "timestamp": None,
            "detected_objects": detected_objects,
            "detected_zones": detected_zones,
            "summary": {
                "num_objects": len(detected_objects),
                "num_zones": len(detected_zones),
                "occupied_zones": [zn for zn, zd in detected_zones.items() if zd["occupant"] is not None],
                "free_zones": [zn for zn, zd in detected_zones.items() if zd["occupant"] is None],
            },
        }

        # Vẽ bảng trạng thái tóm tắt lên góc ảnh
        cv2.rectangle(annotated, (10, 10), (310, 85), (20, 20, 20), -1)
        cv2.rectangle(annotated, (10, 10), (310, 85), (100, 100, 100), 1)
        cv2.putText(annotated, "UR3e VLM Perception (Phase 2)", (18, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"Blocks: {len(detected_objects)}/5 | Zones: {len(detected_zones)}", (18, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        occ_str = ", ".join([f"{zd['label'].split()[-1]}:{zd['occupant'][:3]}" for zn, zd in detected_zones.items() if zd['occupant']]) or "None"
        cv2.putText(annotated, f"Occupied: {occ_str}", (18, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 255, 180), 1, cv2.LINE_AA)

        return world_state, annotated


# ===========================================================================
#  Node ROS 2 — CameraPerceptionNode
# ===========================================================================
class CameraPerceptionNode(Node):
    """ROS 2 Node nhận diện hình ảnh và publish WorldState."""

    def __init__(self):
        super().__init__("camera_perception_node")
        self.get_logger().info("═" * 55)
        self.get_logger().info("  CameraPerceptionNode (ur3_vlm — Bài 03) đang khởi tạo...")
        self.get_logger().info("═" * 55)

        # Load scene config
        pkg_dir = get_package_share_directory("ur3_vlm")
        config_path = os.path.join(pkg_dir, "config", "scene.yaml")
        with open(config_path, "r") as f:
            self.scene_config = yaml.safe_load(f)

        self.perception = CameraPerception(self.scene_config)
        self.bridge = CvBridge()
        self.latest_cv_image = None
        self.latest_world_state = None

        # QoS Profiles
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        qos_state = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10,
        )

        # Subscriber: Camera image topic
        image_topic = self.scene_config.get("camera", {}).get("image_topic", "/camera/image_raw")
        self.image_sub = self.create_subscription(
            Image, image_topic, self._image_callback, qos_sensor
        )

        # Publishers
        self.annotated_pub = self.create_publisher(Image, "/camera/perception_annotated", 10)
        self.world_state_pub = self.create_publisher(String, "/world_state_json", qos_state)

        # Timer định kỳ lưu snapshot và publish status (1 Hz)
        self.timer = self.create_timer(1.0, self._periodic_publish)

        self.get_logger().info(f"✓ Đã subscribe: {image_topic}")
        self.get_logger().info("✓ Publishers sẵn sàng: /camera/perception_annotated, /world_state_json")
        self.get_logger().info("═" * 55)

    def _image_callback(self, msg: Image):
        """Nhận và xử lý ảnh từ camera."""
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_cv_image = cv_img

            world_state, annotated = self.perception.process_image(cv_img)
            world_state["timestamp"] = self.get_clock().now().nanoseconds / 1e9
            self.latest_world_state = world_state

            # Publish WorldState JSON ngay khi có frame ảnh mới
            json_str = json.dumps(world_state)
            ws_msg = String()
            ws_msg.data = json_str
            self.world_state_pub.publish(ws_msg)

            # Publish annotated image
            if annotated is not None:
                if self.annotated_pub.get_subscription_count() > 0:
                    ann_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
                    ann_msg.header = msg.header
                    self.annotated_pub.publish(ann_msg)

                # Hiển thị trực tiếp cửa sổ ảnh Live Camera với overlay nhận diện nếu có GUI
                try:
                    if os.environ.get("DISPLAY"):
                        cv2.imshow("UR3e VLM — Overhead Camera Perception", annotated)
                        cv2.waitKey(1)
                except Exception:
                    pass

        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
        except Exception as e:
            self.get_logger().error(f"Lỗi xử lý ảnh perception: {e}")

    def _periodic_publish(self):
        """Định kỳ publish WorldState JSON và lưu ảnh snapshot debug."""
        if self.latest_world_state is not None:
            # Publish JSON
            json_str = json.dumps(self.latest_world_state, indent=2)
            msg = String()
            msg.data = json_str
            self.world_state_pub.publish(msg)

            # Lưu ảnh snapshot ra /tmp/perception_debug.jpg
            if self.latest_cv_image is not None:
                _, annotated = self.perception.process_image(self.latest_cv_image)
                if annotated is not None:
                    cv2.imwrite("/tmp/perception_debug.jpg", annotated)

            # Log tóm tắt
            summary = self.latest_world_state.get("summary", {})
            objs = self.latest_world_state.get("detected_objects", {})
            self.get_logger().info(
                f"[Perception] Phát hiện {len(objs)}/5 khối | "
                f"Zones chiếm dụng: {summary.get('occupied_zones', [])} | "
                f"Zones trống: {summary.get('free_zones', [])}"
            )


# ===========================================================================
#  Entrypoint
# ===========================================================================
def main(args=None):
    rclpy.init(args=args)
    node = CameraPerceptionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Dừng CameraPerceptionNode.")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
