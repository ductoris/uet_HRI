#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scene_spawner.py — Spawn toàn bộ môi trường Bài 03 vào Ignition Gazebo.

Bao gồm:
  1. Bàn làm việc 4 chân (work_table)
  2. Đế robot hình trụ (robot_pedestal)
  3. Khung giá đỡ camera (camera_stand_rig) + overhead camera
  4. 5 khối lập phương động (red/yellow/blue/green/purple_cube)
  5. 3 khay zone (zone_a/zone_b/zone_c) với viền màu trung tính
  6. ros_gz_bridge cho camera topics
  7. Collision objects + RViz markers cho MoveIt Planning Scene

Mọi tọa độ trong scene.yaml theo hệ base_link.
Khi spawn vào Gazebo (world frame), cộng thêm robot_base_height vào Z.

Sinh viên: Mai Duc Tri | MSSV: 23020776
"""

import os
import time
import yaml
import subprocess

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from ament_index_python.packages import get_package_share_directory

from visualization_msgs.msg import Marker, MarkerArray
from moveit_msgs.msg import PlanningScene, CollisionObject, ObjectColor
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose, Point
from std_msgs.msg import ColorRGBA, Header


# ===========================================================================
#  SDF Templates — Khối lập phương (dynamic, ma sát cao)
# ===========================================================================
def _cube_sdf(name: str, size: float,
              r: float, g: float, b: float, a: float = 1.0,
              mass: float = 0.03) -> str:
    """Tạo SDF cho khối lập phương ĐỘNG (static=false) với ma sát cao."""
    inertia = (1.0 / 6.0) * mass * size * size
    # Ma sát cao để gripper kẹp giữ chắc chắn
    return f"""<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="{name}">
    <static>false</static>
    <link name="link">
      <inertial>
        <mass>{mass}</mass>
        <inertia>
          <ixx>{inertia:.8f}</ixx><iyy>{inertia:.8f}</iyy><izz>{inertia:.8f}</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry><box><size>{size} {size} {size}</size></box></geometry>
        <surface>
          <friction>
            <ode>
              <mu>50.0</mu>
              <mu2>50.0</mu2>
              <fdir1>0 0 0</fdir1>
            </ode>
          </friction>
          <contact>
            <ode>
              <kp>1000000.0</kp>
              <kd>100.0</kd>
              <min_depth>0.001</min_depth>
              <max_vel>0.01</max_vel>
            </ode>
          </contact>
        </surface>
      </collision>
      <visual name="visual">
        <geometry><box><size>{size} {size} {size}</size></box></geometry>
        <material>
          <ambient>{r} {g} {b} {a}</ambient>
          <diffuse>{r} {g} {b} {a}</diffuse>
          <specular>0.4 0.4 0.4 1</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""


# ===========================================================================
#  Bitmap mẫu chữ cho Zone trên mặt sàn Gazebo ([A], [B], [C], [T])
# ===========================================================================
ZONE_LETTER_BITMAPS = {
    'A': [
        "##  ###  ##",
        " # #   # # ",
        " # ##### # ",
        " # #   # # ",
        "## #   # ##",
    ],
    'B': [
        "## ####  ##",
        " # #   # # ",
        " # ####  # ",
        " # #   # # ",
        "## ####  ##",
    ],
    'C': [
        "##  #### ##",
        " # #     # ",
        " # #     # ",
        " # #     # ",
        "##  #### ##",
    ],
    'T': [
        "## ##### ##",
        " # ##### # ",
        " #  ###  # ",
        " #  ###  # ",
        "##  ###  ##",
    ],
}


# ===========================================================================
#  SDF Template — Khay zone (single link, static, có chữ [A]/[B]/[C]/[T])
# ===========================================================================
def _zone_tray_sdf(name: str, label: str,
                   r: float, g: float, b: float,
                   width: float = 0.07, depth: float = 0.07,
                   height: float = 0.005) -> str:
    """
    Tạo SDF cho khay zone dạng SINGLE LINK:
    - Đáy xám nhạt
    - 4 viền màu trung tính
    - Chữ pixel-art [A]/[B]/[C]/[T] dập nổi trên mặt đáy (hiển thị rõ trong Gazebo)
    """
    border_h = 0.01   # Chiều cao viền
    border_t = 0.003  # Độ dày viền

    # Trích xuất ký tự zone (A, B, C hoặc T)
    char_key = 'A'
    for k in ['A', 'B', 'C', 'T']:
        if k in label.upper() or k in name.upper():
            char_key = k
            break
    if char_key not in ZONE_LETTER_BITMAPS and ("STAGING" in label.upper() or "STAGING" in name.upper() or "TEMP" in label.upper() or "TEMP" in name.upper()):
        char_key = 'T'

    # Tạo visual cho chữ dập nổi
    bitmap = ZONE_LETTER_BITMAPS.get(char_key, ZONE_LETTER_BITMAPS['A'])
    px_size = 0.0050  # 5.0mm mỗi pixel (cân đối tuyệt đẹp với khay 10cm)
    px_th = 0.0008    # Dày 0.8mm
    letter_z = height / 2.0 + px_th / 2.0

    letter_visuals = ""
    v_idx = 0
    num_rows = len(bitmap)
    num_cols = len(bitmap[0])
    for row_idx, row_str in enumerate(bitmap):
        # row 0 ở phía +X (đọc xuôi theo hướng bàn)
        x_pos = ( (num_rows - 1) / 2.0 - row_idx ) * px_size
        for col_idx, ch in enumerate(row_str):
            if ch == '#':
                # col 0 ở phía +Y (bên trái)
                y_pos = ( (num_cols - 1) / 2.0 - col_idx ) * px_size
                letter_visuals += f"""
      <visual name="letter_px_{v_idx}">
        <pose>{x_pos:.5f} {y_pos:.5f} {letter_z:.5f} 0 0 0</pose>
        <geometry><box><size>{px_size*0.95:.5f} {px_size*0.95:.5f} {px_th:.5f}</size></box></geometry>
        <material>
          <ambient>0.08 0.08 0.08 1</ambient>
          <diffuse>0.10 0.10 0.10 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
      </visual>"""
                v_idx += 1

    return f"""<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="{name}">
    <static>true</static>
    <link name="tray_link">
      <!-- Đáy khay -->
      <collision name="collision_base">
        <pose>0 0 0 0 0 0</pose>
        <geometry><box><size>{width} {depth} {height}</size></box></geometry>
      </collision>
      <visual name="visual_base">
        <pose>0 0 0 0 0 0</pose>
        <geometry><box><size>{width} {depth} {height}</size></box></geometry>
        <material>
          <ambient>0.88 0.88 0.88 1</ambient>
          <diffuse>0.92 0.92 0.92 1</diffuse>
          <specular>0.2 0.2 0.2 1</specular>
        </material>
      </visual>

      <!-- Viền trước (+X) -->
      <collision name="col_border_front">
        <pose>{width/2 - border_t/2} 0 {height/2 + border_h/2} 0 0 0</pose>
        <geometry><box><size>{border_t} {depth} {border_h}</size></box></geometry>
      </collision>
      <visual name="vis_border_front">
        <pose>{width/2 - border_t/2} 0 {height/2 + border_h/2} 0 0 0</pose>
        <geometry><box><size>{border_t} {depth} {border_h}</size></box></geometry>
        <material>
          <ambient>{r} {g} {b} 1</ambient>
          <diffuse>{r} {g} {b} 1</diffuse>
        </material>
      </visual>

      <!-- Viền sau (-X) -->
      <collision name="col_border_back">
        <pose>{-width/2 + border_t/2} 0 {height/2 + border_h/2} 0 0 0</pose>
        <geometry><box><size>{border_t} {depth} {border_h}</size></box></geometry>
      </collision>
      <visual name="vis_border_back">
        <pose>{-width/2 + border_t/2} 0 {height/2 + border_h/2} 0 0 0</pose>
        <geometry><box><size>{border_t} {depth} {border_h}</size></box></geometry>
        <material>
          <ambient>{r} {g} {b} 1</ambient>
          <diffuse>{r} {g} {b} 1</diffuse>
        </material>
      </visual>

      <!-- Viền trái (+Y) -->
      <collision name="col_border_left">
        <pose>0 {depth/2 - border_t/2} {height/2 + border_h/2} 0 0 0</pose>
        <geometry><box><size>{width} {border_t} {border_h}</size></box></geometry>
      </collision>
      <visual name="vis_border_left">
        <pose>0 {depth/2 - border_t/2} {height/2 + border_h/2} 0 0 0</pose>
        <geometry><box><size>{width} {border_t} {border_h}</size></box></geometry>
        <material>
          <ambient>{r} {g} {b} 1</ambient>
          <diffuse>{r} {g} {b} 1</diffuse>
        </material>
      </visual>

      <!-- Viền phải (-Y) -->
      <collision name="col_border_right">
        <pose>0 {-depth/2 + border_t/2} {height/2 + border_h/2} 0 0 0</pose>
        <geometry><box><size>{width} {border_t} {border_h}</size></box></geometry>
      </collision>
      <visual name="vis_border_right">
        <pose>0 {-depth/2 + border_t/2} {height/2 + border_h/2} 0 0 0</pose>
        <geometry><box><size>{width} {border_t} {border_h}</size></box></geometry>
        <material>
          <ambient>{r} {g} {b} 1</ambient>
          <diffuse>{r} {g} {b} 1</diffuse>
        </material>
      </visual>

      <!-- Chữ [A]/[B]/[C] trên mặt đáy -->{letter_visuals}
    </link>
  </model>
</sdf>"""


# ===========================================================================
#  SDF Template — Tay kẹp 2 ngón (Two-Finger Gripper)
# ===========================================================================
def _gripper_sdf(name: str = "two_finger_gripper") -> str:
    """Tạo SDF cho mô hình kẹp 2 ngón tay (2-Finger Gripper) gắn đầu công tác."""
    return f"""<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="{name}">
    <static>true</static>
    <link name="gripper_link">
      <!-- Flange adapter cylinder -->
      <visual name="adapter">
        <pose>0 0 0.01 0 0 0</pose>
        <geometry><cylinder><radius>0.035</radius><length>0.02</length></cylinder></geometry>
        <material>
          <ambient>0.2 0.2 0.22 1</ambient>
          <diffuse>0.25 0.25 0.28 1</diffuse>
        </material>
      </visual>
      <!-- Gripper palm / body -->
      <visual name="palm">
        <pose>0 0 0.03 0 0 0</pose>
        <geometry><box><size>0.045 0.08 0.02</size></box></geometry>
        <material>
          <ambient>0.1 0.1 0.12 1</ambient>
          <diffuse>0.15 0.15 0.18 1</diffuse>
        </material>
      </visual>
      <!-- Left Finger Base -->
      <visual name="left_finger_base">
        <pose>0 0.03 0.045 0 0 0</pose>
        <geometry><box><size>0.02 0.015 0.01</size></box></geometry>
        <material>
          <ambient>0.3 0.3 0.35 1</ambient>
          <diffuse>0.4 0.4 0.45 1</diffuse>
        </material>
      </visual>
      <!-- Left Finger Prong -->
      <visual name="left_finger_prong">
        <pose>0 0.025 0.065 0 0 0</pose>
        <geometry><box><size>0.015 0.01 0.035</size></box></geometry>
        <material>
          <ambient>0.15 0.15 0.18 1</ambient>
          <diffuse>0.2 0.2 0.22 1</diffuse>
        </material>
      </visual>
      <!-- Left Finger Rubber Pad -->
      <visual name="left_finger_pad">
        <pose>0 0.021 0.065 0 0 0</pose>
        <geometry><box><size>0.014 0.003 0.03</size></box></geometry>
        <material>
          <ambient>0.9 0.4 0.1 1</ambient>
          <diffuse>0.95 0.45 0.15 1</diffuse>
        </material>
      </visual>
      <!-- Right Finger Base -->
      <visual name="right_finger_base">
        <pose>0 -0.03 0.045 0 0 0</pose>
        <geometry><box><size>0.02 0.015 0.01</size></box></geometry>
        <material>
          <ambient>0.3 0.3 0.35 1</ambient>
          <diffuse>0.4 0.4 0.45 1</diffuse>
        </material>
      </visual>
      <!-- Right Finger Prong -->
      <visual name="right_finger_prong">
        <pose>0 -0.025 0.065 0 0 0</pose>
        <geometry><box><size>0.015 0.01 0.035</size></box></geometry>
        <material>
          <ambient>0.15 0.15 0.18 1</ambient>
          <diffuse>0.2 0.2 0.22 1</diffuse>
        </material>
      </visual>
      <!-- Right Finger Rubber Pad -->
      <visual name="right_finger_pad">
        <pose>0 -0.021 0.065 0 0 0</pose>
        <geometry><box><size>0.014 0.003 0.03</size></box></geometry>
        <material>
          <ambient>0.9 0.4 0.1 1</ambient>
          <diffuse>0.95 0.45 0.15 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""


# ===========================================================================
#  SDF Template — Bàn 4 chân (single link, static — chống crash Gazebo DART)
# ===========================================================================
def _table_sdf(length: float, width: float, depth: float,
               leg_width: float, leg_inset: float,
               table_height_from_floor: float) -> str:
    """
    Tạo SDF cho bàn 4 chân dạng SINGLE LINK (không dùng joint để tránh crash Gazebo DART).
    - Mặt bàn: length x width x depth, tâm tại (0, 0, 0) trong model frame.
    - 4 chân bàn: leg_width x leg_width, kéo dài từ mặt dưới bàn xuống sàn.
    """
    leg_h = table_height_from_floor - depth / 2.0
    if leg_h < 0.05:
        leg_h = 0.5

    lx = length / 2.0 - leg_inset - leg_width / 2.0
    ly = width / 2.0 - leg_inset - leg_width / 2.0
    leg_z = -(depth / 2.0 + leg_h / 2.0)

    legs_elements = ""
    leg_positions = [
        ("leg_fl", +lx, +ly),   # Trước-Trái
        ("leg_fr", +lx, -ly),   # Trước-Phải
        ("leg_bl", -lx, +ly),   # Sau-Trái
        ("leg_br", -lx, -ly),   # Sau-Phải
    ]
    for leg_name, lx_pos, ly_pos in leg_positions:
        legs_elements += f"""
      <!-- {leg_name} -->
      <collision name="col_{leg_name}">
        <pose>{lx_pos} {ly_pos} {leg_z} 0 0 0</pose>
        <geometry><box><size>{leg_width} {leg_width} {leg_h}</size></box></geometry>
      </collision>
      <visual name="vis_{leg_name}">
        <pose>{lx_pos} {ly_pos} {leg_z} 0 0 0</pose>
        <geometry><box><size>{leg_width} {leg_width} {leg_h}</size></box></geometry>
        <material>
          <ambient>0.45 0.30 0.18 1</ambient>
          <diffuse>0.50 0.35 0.22 1</diffuse>
          <specular>0.05 0.05 0.05 1</specular>
        </material>
      </visual>"""

    return f"""<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="work_table">
    <static>true</static>
    <link name="table_link">
      <!-- Mặt bàn -->
      <collision name="collision_top">
        <pose>0 0 0 0 0 0</pose>
        <geometry><box><size>{length} {width} {depth}</size></box></geometry>
        <surface>
          <friction>
            <ode><mu>1.5</mu><mu2>1.5</mu2></ode>
          </friction>
        </surface>
      </collision>
      <visual name="visual_top">
        <pose>0 0 0 0 0 0</pose>
        <geometry><box><size>{length} {width} {depth}</size></box></geometry>
        <material>
          <ambient>0.55 0.37 0.24 1</ambient>
          <diffuse>0.65 0.45 0.30 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
      </visual>

      <!-- 4 chân bàn -->{legs_elements}
    </link>
  </model>
</sdf>"""


# ===========================================================================
#  SDF Template — Đế robot (pedestal) hình trụ (visual only)
# ===========================================================================
def _pedestal_sdf(radius: float, height: float,
                  r: float, g: float, b: float) -> str:
    """Tạo SDF cho đế robot hình trụ (visual-only để tránh xung đột physics với robot base)."""
    return f"""<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="robot_pedestal">
    <static>true</static>
    <link name="pedestal_link">
      <visual name="visual">
        <geometry><cylinder><radius>{radius}</radius><length>{height}</length></cylinder></geometry>
        <material>
          <ambient>{r} {g} {b} 1</ambient>
          <diffuse>{r + 0.05} {g + 0.05} {b + 0.05} 1</diffuse>
          <specular>0.3 0.3 0.3 1</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""


# ===========================================================================
#  SDF Template — Khung giá đỡ camera + Camera sensor (single link, static)
# ===========================================================================
def _camera_rig_sdf(rig_cfg: dict, cam_cfg: dict,
                    base_height: float) -> str:
    """
    Tạo SDF cho khung giá đỡ camera dạng SINGLE LINK gồm:
      - Thanh đứng (vertical post)
      - Thanh ngang (horizontal arm)
      - Đầu gắn camera (camera mount)
      - Camera sensor (RGB) gắn trực tiếp vào link
    """
    vp = rig_cfg["vertical_post"]
    ha = rig_cfg["horizontal_arm"]
    cm = rig_cfg["camera_mount"]
    rc = rig_cfg["color"]

    post_h = vp["height"]
    post_w = vp["width"]
    arm_l = ha["length"]
    arm_w = ha["width"]
    mount_w = cm["width"]
    mount_d = cm["depth"]
    mount_h = cm["height"]

    post_z = post_h / 2.0
    arm_x = -(arm_l / 2.0)
    arm_z = post_h - arm_w / 2.0
    mount_x = -arm_l
    mount_z = post_h - arm_w - mount_h / 2.0
    cam_z = post_h - arm_w - mount_h - 0.01

    hfov = cam_cfg["horizontal_fov"]
    img_w = cam_cfg["image_width"]
    img_h = cam_cfg["image_height"]
    clip_near = cam_cfg["clip_near"]
    clip_far = cam_cfg["clip_far"]
    update_rate = cam_cfg["update_rate"]

    return f"""<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="camera_stand_rig">
    <static>true</static>
    <link name="rig_link">
      <!-- Thanh đứng (vertical post) -->
      <collision name="col_post">
        <pose>0 0 {post_z} 0 0 0</pose>
        <geometry><box><size>{post_w} {post_w} {post_h}</size></box></geometry>
      </collision>
      <visual name="vis_post">
        <pose>0 0 {post_z} 0 0 0</pose>
        <geometry><box><size>{post_w} {post_w} {post_h}</size></box></geometry>
        <material>
          <ambient>{rc['r']} {rc['g']} {rc['b']} 1</ambient>
          <diffuse>{rc['r'] + 0.05} {rc['g'] + 0.05} {rc['b'] + 0.05} 1</diffuse>
          <specular>0.15 0.15 0.15 1</specular>
        </material>
      </visual>

      <!-- Thanh ngang (horizontal arm) -->
      <collision name="col_arm">
        <pose>{arm_x} 0 {arm_z} 0 0 0</pose>
        <geometry><box><size>{arm_l} {arm_w} {arm_w}</size></box></geometry>
      </collision>
      <visual name="vis_arm">
        <pose>{arm_x} 0 {arm_z} 0 0 0</pose>
        <geometry><box><size>{arm_l} {arm_w} {arm_w}</size></box></geometry>
        <material>
          <ambient>{rc['r']} {rc['g']} {rc['b']} 1</ambient>
          <diffuse>{rc['r'] + 0.05} {rc['g'] + 0.05} {rc['b'] + 0.05} 1</diffuse>
          <specular>0.15 0.15 0.15 1</specular>
        </material>
      </visual>

      <!-- Camera mount (đầu gắn camera) -->
      <collision name="col_mount">
        <pose>{mount_x} 0 {mount_z} 0 0 0</pose>
        <geometry><box><size>{mount_w} {mount_d} {mount_h}</size></box></geometry>
      </collision>
      <visual name="vis_mount">
        <pose>{mount_x} 0 {mount_z} 0 0 0</pose>
        <geometry><box><size>{mount_w} {mount_d} {mount_h}</size></box></geometry>
        <material>
          <ambient>0.15 0.15 0.15 1</ambient>
          <diffuse>0.2 0.2 0.2 1</diffuse>
          <specular>0.3 0.3 0.3 1</specular>
        </material>
      </visual>

      <!-- Camera sensor — nhìn thẳng xuống mặt bàn -->
      <sensor name="overhead_camera" type="camera">
        <pose>{mount_x} 0 {cam_z} 0 1.5708 0</pose>
        <always_on>true</always_on>
        <update_rate>{update_rate}</update_rate>
        <topic>/camera/image_raw</topic>
        <camera name="overhead_cam">
          <horizontal_fov>{hfov}</horizontal_fov>
          <image>
            <width>{img_w}</width>
            <height>{img_h}</height>
            <format>R8G8B8</format>
          </image>
          <clip>
            <near>{clip_near}</near>
            <far>{clip_far}</far>
          </clip>
          <camera_info_topic>/camera/camera_info</camera_info_topic>
          <optical_frame_id>camera_optical_frame</optical_frame_id>
        </camera>
      </sensor>
    </link>
  </model>
</sdf>"""


# ===========================================================================
#  Hằng số spawn
# ===========================================================================
MAX_SPAWN_RETRIES = 3
RETRY_DELAY = 2.0
SPAWN_INTERVAL = 1.0


# ===========================================================================
#  Node ROS 2 — SceneSpawnerNode
# ===========================================================================
class SceneSpawnerNode(Node):
    """Spawn toàn bộ môi trường Bài 03 vào Ignition Gazebo."""

    def __init__(self):
        super().__init__("scene_spawner_node")
        self.get_logger().info("═" * 55)
        self.get_logger().info("  SceneSpawnerNode (ur3_vlm — Bài 03) đang khởi tạo...")
        self.get_logger().info("═" * 55)

        # Load scene config
        pkg_dir = get_package_share_directory("ur3_vlm")
        config_path = os.path.join(pkg_dir, "config", "scene.yaml")
        with open(config_path, "r") as f:
            self._scene = yaml.safe_load(f)

        # Chiều cao đế robot = offset world → base_link
        self._base_height = float(self._scene["robot_base_height"])

        # Publishers cho MoveIt và RViz
        qos_transient = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._planning_scene_pub = self.create_publisher(
            PlanningScene, "/planning_scene", 10
        )
        self._marker_pub = self.create_publisher(
            MarkerArray, "/visualization_marker_array", qos_transient
        )
        self._scene_marker_pub = self.create_publisher(
            MarkerArray, "/scene_markers", qos_transient
        )

        # Đợi Gazebo sẵn sàng
        self._wait_for_gazebo()

        # ═══ Spawn theo thứ tự ═══
        self._spawn_pedestal()
        self._spawn_table()
        self._spawn_camera_rig()
        self._spawn_cubes()
        self._spawn_zones()

        # Publish collision objects & visual markers cho MoveIt / RViz
        self._publish_rviz_scene()

        # Timer 1s để duy trì MarkerArray trên RViz
        self._timer = self.create_timer(1.0, self._publish_rviz_scene)

        self.get_logger().info("═" * 55)
        self.get_logger().info("✓ Đã spawn toàn bộ môi trường Bài 03!")
        self.get_logger().info(f"  • Đế robot: z = {self._base_height:.2f} m")
        self.get_logger().info("  • Bàn 4 chân + 5 khối động + 3 zone")
        self.get_logger().info("  • Camera trên giá đỡ nhìn xuống bàn")
        self.get_logger().info("  (Giữ Terminal chạy để duy trì RViz markers)")
        self.get_logger().info("═" * 55)

    # -------------------------------------------------------------------
    #  Chuyển đổi tọa độ base_link → world (Gazebo)
    # -------------------------------------------------------------------
    def _bl_to_world_z(self, z_bl: float) -> float:
        """Chuyển z từ hệ base_link sang hệ world Gazebo."""
        return z_bl + self._base_height

    # -------------------------------------------------------------------
    #  Đợi Gazebo sẵn sàng
    # -------------------------------------------------------------------
    def _wait_for_gazebo(self, timeout_sec: float = 60.0):
        """Đợi Gazebo sẵn sàng bằng cách kiểm tra topic /clock."""
        self.get_logger().info("Đợi Gazebo sẵn sàng (kiểm tra /clock topic)...")
        start = time.time()
        clock_ready = False

        while time.time() - start < timeout_sec:
            topic_list = self.get_topic_names_and_types()
            for topic_name, _ in topic_list:
                if topic_name == "/clock":
                    pubs_info = self.get_publishers_info_by_topic("/clock")
                    if len(pubs_info) > 0:
                        clock_ready = True
                        break
            if clock_ready:
                break
            self.get_logger().info("  ... Gazebo chưa sẵn sàng, đợi 2s...")
            time.sleep(2.0)

        if clock_ready:
            self.get_logger().info("✓ Gazebo đã sẵn sàng! Đợi thêm 5s để ổn định...")
            time.sleep(5.0)
        else:
            self.get_logger().warn(
                f"⚠ Không phát hiện Gazebo sau {timeout_sec}s. Vẫn thử spawn..."
            )
            time.sleep(3.0)

    # -------------------------------------------------------------------
    #  Spawn model vào Gazebo (với retry)
    # -------------------------------------------------------------------
    def _spawn_via_gz(self, sdf_string: str, name: str,
                      x: float, y: float, z: float) -> bool:
        """
        Spawn model vào Gazebo bằng ros_gz_sim create.
        x, y, z: TỌA ĐỘ TRONG HỆ WORLD (đã chuyển đổi).

        Returns True nếu spawn thành công.
        """
        tmp_sdf = f"/tmp/{name}.sdf"
        with open(tmp_sdf, "w") as f:
            f.write(sdf_string)

        for attempt in range(1, MAX_SPAWN_RETRIES + 1):
            self.get_logger().info(
                f"Spawning '{name}' tại world({x:.3f}, {y:.3f}, {z:.3f}) "
                f"[lần {attempt}/{MAX_SPAWN_RETRIES}]..."
            )
            try:
                result = subprocess.run(
                    [
                        "ros2", "run", "ros_gz_sim", "create",
                        "-file", tmp_sdf,
                        "-name", name,
                        "-x", str(x),
                        "-y", str(y),
                        "-z", str(z),
                    ],
                    capture_output=True, text=True, timeout=20.0,
                )

                stdout = result.stdout.strip()
                stderr = result.stderr.strip()

                if result.returncode == 0:
                    self.get_logger().info(f"  ✓ '{name}' spawned thành công!")
                    time.sleep(SPAWN_INTERVAL)
                    return True

                if "already exists" in stderr.lower() or "already exists" in stdout.lower():
                    self.get_logger().info(f"  ✓ '{name}' đã tồn tại trong scene.")
                    time.sleep(SPAWN_INTERVAL)
                    return True

                self.get_logger().warn(
                    f"  ✗ Lần {attempt}: '{name}' spawn lỗi.\n"
                    f"    stdout: {stdout}\n"
                    f"    stderr: {stderr}"
                )

            except subprocess.TimeoutExpired:
                self.get_logger().warn(
                    f"  ✗ Lần {attempt}: '{name}' spawn TIMEOUT (>20s)."
                )
            except Exception as e:
                self.get_logger().error(
                    f"  ✗ Lần {attempt}: Exception khi spawn '{name}': {e}"
                )

            if attempt < MAX_SPAWN_RETRIES:
                self.get_logger().info(f"  Đợi {RETRY_DELAY}s trước khi thử lại...")
                time.sleep(RETRY_DELAY)

        self.get_logger().error(
            f"  ✗✗ '{name}' THẤT BẠI sau {MAX_SPAWN_RETRIES} lần thử!"
        )
        return False

    # -------------------------------------------------------------------
    #  Spawn đế robot (pedestal)
    # -------------------------------------------------------------------
    def _spawn_pedestal(self):
        """Spawn đế robot hình trụ từ sàn lên base_link."""
        ped_cfg = self._scene["pedestal"]
        radius = ped_cfg["radius"]
        height = self._base_height  # Từ sàn (z=0) lên base_link
        color = ped_cfg["color"]

        sdf = _pedestal_sdf(radius, height, color["r"], color["g"], color["b"])
        # Tâm trụ ở giữa chiều cao, tại gốc world (x=0, y=0)
        self._spawn_via_gz(sdf, "robot_pedestal", 0.0, 0.0, height / 2.0)

    # -------------------------------------------------------------------
    #  Spawn bàn 4 chân
    # -------------------------------------------------------------------
    def _spawn_table(self):
        """Spawn bàn làm việc 4 chân."""
        table = self._scene["table"]
        pos = table["position"]
        dims = table["dimensions"]
        leg = table["leg"]

        world_z = self._bl_to_world_z(pos["z"])
        sdf = _table_sdf(
            length=dims["length"],
            width=dims["width"],
            depth=dims["depth"],
            leg_width=leg["width"],
            leg_inset=leg["inset"],
            table_height_from_floor=world_z,
        )
        self._spawn_via_gz(sdf, "work_table", pos["x"], pos["y"], world_z)

    # -------------------------------------------------------------------
    #  Spawn khung giá đỡ camera + camera
    # -------------------------------------------------------------------
    def _spawn_camera_rig(self):
        """Spawn khung giá đỡ camera gồm thanh đứng + thanh ngang + camera."""
        rig_cfg = self._scene["camera_rig"]
        cam_cfg = self._scene["camera"]
        vp = rig_cfg["vertical_post"]

        sdf = _camera_rig_sdf(rig_cfg, cam_cfg, self._base_height)

        # Chân thanh đứng đặt trên mặt bàn, tại vị trí cạnh bàn phía xa robot
        table_surface_world_z = self._bl_to_world_z(
            self._scene["heights"]["table_surface"]
        )
        self._spawn_via_gz(
            sdf, "camera_stand_rig",
            vp["position"]["x"], vp["position"]["y"],
            table_surface_world_z,
        )

    # -------------------------------------------------------------------
    #  Spawn 5 khối lập phương
    # -------------------------------------------------------------------
    def _spawn_cubes(self):
        """Spawn 5 khối lập phương ĐỘNG (static=false) lên mặt bàn."""
        for obj_name, obj_cfg in self._scene["objects"].items():
            pos = obj_cfg["position"]
            color = obj_cfg["color"]
            size = obj_cfg["size"]
            mass = obj_cfg.get("mass", 0.03)
            sdf = _cube_sdf(
                obj_name, size,
                color["r"], color["g"], color["b"], color.get("a", 1.0),
                mass=mass,
            )
            world_z = self._bl_to_world_z(pos["z"])
            self._spawn_via_gz(sdf, obj_name, pos["x"], pos["y"], world_z)

    # -------------------------------------------------------------------
    #  Spawn 3 zone
    # -------------------------------------------------------------------
    def _spawn_zones(self):
        """Spawn 3 khay zone với viền màu trung tính."""
        for zone_name, zone_cfg in self._scene["zones"].items():
            pos = zone_cfg["position"]
            label = zone_cfg["label"]
            bc = zone_cfg["border_color"]
            tray_size = self._scene["zone_detection"]["tray_size"]

            sdf = _zone_tray_sdf(
                zone_name, label,
                bc["r"], bc["g"], bc["b"],
                width=tray_size, depth=tray_size,
            )
            world_z = self._bl_to_world_z(pos["z"])
            self._spawn_via_gz(sdf, zone_name, pos["x"], pos["y"], world_z)

    # -------------------------------------------------------------------
    #  Spawn Tay kẹp 2 ngón (Two-Finger Gripper) vào Gazebo
    # -------------------------------------------------------------------
    def _spawn_gripper(self):
        """Spawn mô hình kẹp 2 ngón tay vào Gazebo."""
        sdf = _gripper_sdf("two_finger_gripper")
        # Khởi tạo tại vị trí Home chuẩn của tool0 trong Gazebo world
        world_z = self._bl_to_world_z(0.30)
        self._spawn_via_gz(sdf, "two_finger_gripper", 0.30, 0.13, world_z)

    # -------------------------------------------------------------------
    #  Publish collision objects + RViz markers
    # -------------------------------------------------------------------
    def _publish_rviz_scene(self):
        """Publish collision objects vào /planning_scene và markers vào RViz."""
        now = self.get_clock().now().to_msg()
        frame_id = "world"

        # ═══ 1. MOVEIT PLANNING SCENE (Table + Pedestal Collision) ═══
        ps = PlanningScene()
        ps.is_diff = True

        tbl = self._scene["table"]
        t_dims = tbl["dimensions"]
        t_pos = tbl["position"]
        tbl_co = CollisionObject()
        tbl_co.id = "work_table"
        tbl_co.header.frame_id = frame_id
        tbl_co.header.stamp = now
        b_box = SolidPrimitive(
            type=SolidPrimitive.BOX,
            dimensions=[t_dims["length"], t_dims["width"], t_dims["depth"]],
        )
        b_pose = Pose()
        b_pose.position.x = float(t_pos["x"])
        b_pose.position.y = float(t_pos["y"])
        b_pose.position.z = float(t_pos["z"])
        b_pose.orientation.w = 1.0
        tbl_co.primitives.append(b_box)
        tbl_co.primitive_poses.append(b_pose)
        tbl_co.operation = CollisionObject.ADD
        ps.world.collision_objects.append(tbl_co)

        tbl_color = ObjectColor()
        tbl_color.id = "work_table"
        tbl_color.color = ColorRGBA(r=0.65, g=0.45, b=0.30, a=0.95)
        ps.object_colors.append(tbl_color)

        self._planning_scene_pub.publish(ps)

        # ═══ 2. RVIZ VISUALIZATION MARKERS ═══
        ma = MarkerArray()

        # --- Marker: Đế robot (Pedestal) ---
        ped_cfg = self._scene["pedestal"]
        ped_r = float(ped_cfg["radius"])
        ped_h = self._base_height
        m_ped = Marker()
        m_ped.header.frame_id = frame_id
        m_ped.header.stamp = now
        m_ped.ns = "scene_pedestal"
        m_ped.id = 0
        m_ped.type = Marker.CYLINDER
        m_ped.action = Marker.ADD
        m_ped.pose.position.x = 0.0
        m_ped.pose.position.y = 0.0
        m_ped.pose.position.z = -ped_h / 2.0
        m_ped.pose.orientation.w = 1.0
        m_ped.scale.x = ped_r * 2.0
        m_ped.scale.y = ped_r * 2.0
        m_ped.scale.z = ped_h
        m_ped.color = ColorRGBA(r=0.35, g=0.35, b=0.38, a=0.95)
        ma.markers.append(m_ped)

        # --- Marker: Mặt bàn ---
        m_tbl = Marker()
        m_tbl.header.frame_id = frame_id
        m_tbl.header.stamp = now
        m_tbl.ns = "scene_table"
        m_tbl.id = 1
        m_tbl.type = Marker.CUBE
        m_tbl.action = Marker.ADD
        m_tbl.pose = b_pose
        m_tbl.scale.x = float(t_dims["length"])
        m_tbl.scale.y = float(t_dims["width"])
        m_tbl.scale.z = float(t_dims["depth"])
        m_tbl.color = ColorRGBA(r=0.65, g=0.45, b=0.30, a=0.95)
        ma.markers.append(m_tbl)

        # --- Markers: 4 Chân bàn ---
        t_leg = tbl["leg"]
        leg_w = float(t_leg["width"])
        leg_inset = float(t_leg["inset"])
        leg_h = self._base_height + float(t_pos["z"]) - float(t_dims["depth"]) / 2.0
        leg_z = float(t_pos["z"]) - float(t_dims["depth"]) / 2.0 - leg_h / 2.0
        lx = float(t_dims["length"]) / 2.0 - leg_inset - leg_w / 2.0
        ly = float(t_dims["width"]) / 2.0 - leg_inset - leg_w / 2.0

        leg_poses = [
            ("leg_fl", float(t_pos["x"]) + lx, float(t_pos["y"]) + ly),
            ("leg_fr", float(t_pos["x"]) + lx, float(t_pos["y"]) - ly),
            ("leg_bl", float(t_pos["x"]) - lx, float(t_pos["y"]) + ly),
            ("leg_br", float(t_pos["x"]) - lx, float(t_pos["y"]) - ly),
        ]
        for l_idx, (l_name, l_x, l_y) in enumerate(leg_poses):
            m_leg = Marker()
            m_leg.header.frame_id = frame_id
            m_leg.header.stamp = now
            m_leg.ns = "scene_table_legs"
            m_leg.id = l_idx
            m_leg.type = Marker.CUBE
            m_leg.action = Marker.ADD
            m_leg.pose.position.x = l_x
            m_leg.pose.position.y = l_y
            m_leg.pose.position.z = leg_z
            m_leg.pose.orientation.w = 1.0
            m_leg.scale.x = leg_w
            m_leg.scale.y = leg_w
            m_leg.scale.z = leg_h
            m_leg.color = ColorRGBA(r=0.50, g=0.35, b=0.22, a=0.95)
            ma.markers.append(m_leg)

        # --- Markers: Khung giá đỡ camera ---
        rig_cfg = self._scene["camera_rig"]
        vp = rig_cfg["vertical_post"]
        ha = rig_cfg["horizontal_arm"]
        cm = rig_cfg["camera_mount"]
        rc = rig_cfg["color"]

        # Cột đứng
        m_vp = Marker()
        m_vp.header.frame_id = frame_id
        m_vp.header.stamp = now
        m_vp.ns = "scene_camera_rig"
        m_vp.id = 10
        m_vp.type = Marker.CUBE
        m_vp.action = Marker.ADD
        m_vp.pose.position.x = float(vp["position"]["x"])
        m_vp.pose.position.y = float(vp["position"]["y"])
        m_vp.pose.position.z = float(vp["height"]) / 2.0
        m_vp.pose.orientation.w = 1.0
        m_vp.scale.x = float(vp["width"])
        m_vp.scale.y = float(vp["width"])
        m_vp.scale.z = float(vp["height"])
        m_vp.color = ColorRGBA(r=float(rc["r"]), g=float(rc["g"]), b=float(rc["b"]), a=0.95)
        ma.markers.append(m_vp)

        # Tay ngang
        m_ha = Marker()
        m_ha.header.frame_id = frame_id
        m_ha.header.stamp = now
        m_ha.ns = "scene_camera_rig"
        m_ha.id = 11
        m_ha.type = Marker.CUBE
        m_ha.action = Marker.ADD
        m_ha.pose.position.x = float(vp["position"]["x"]) - float(ha["length"]) / 2.0
        m_ha.pose.position.y = float(vp["position"]["y"])
        m_ha.pose.position.z = float(vp["height"]) - float(ha["width"]) / 2.0
        m_ha.pose.orientation.w = 1.0
        m_ha.scale.x = float(ha["length"])
        m_ha.scale.y = float(ha["width"])
        m_ha.scale.z = float(ha["width"])
        m_ha.color = ColorRGBA(r=float(rc["r"]), g=float(rc["g"]), b=float(rc["b"]), a=0.95)
        ma.markers.append(m_ha)

        # Đầu gắn Camera (Mount)
        m_cm = Marker()
        m_cm.header.frame_id = frame_id
        m_cm.header.stamp = now
        m_cm.ns = "scene_camera_rig"
        m_cm.id = 12
        m_cm.type = Marker.CUBE
        m_cm.action = Marker.ADD
        m_cm.pose.position.x = float(vp["position"]["x"]) - float(ha["length"])
        m_cm.pose.position.y = float(vp["position"]["y"])
        m_cm.pose.position.z = float(vp["height"]) - float(ha["width"]) - float(cm["height"]) / 2.0
        m_cm.pose.orientation.w = 1.0
        m_cm.scale.x = float(cm["width"])
        m_cm.scale.y = float(cm["depth"])
        m_cm.scale.z = float(cm["height"])
        m_cm.color = ColorRGBA(r=0.15, g=0.15, b=0.15, a=0.95)
        ma.markers.append(m_cm)

        # Thân Camera (Lens & Body) treo trên giá đỡ
        m_cam_body = Marker()
        m_cam_body.header.frame_id = frame_id
        m_cam_body.header.stamp = now
        m_cam_body.ns = "scene_camera_rig"
        m_cam_body.id = 13
        m_cam_body.type = Marker.CYLINDER
        m_cam_body.action = Marker.ADD
        m_cam_body.pose.position.x = float(vp["position"]["x"]) - float(ha["length"])
        m_cam_body.pose.position.y = float(vp["position"]["y"])
        m_cam_body.pose.position.z = float(vp["height"]) - float(ha["width"]) - float(cm["height"]) - 0.01
        m_cam_body.pose.orientation.w = 1.0
        m_cam_body.scale.x = 0.035
        m_cam_body.scale.y = 0.035
        m_cam_body.scale.z = 0.02
        m_cam_body.color = ColorRGBA(r=0.1, g=0.8, b=0.2, a=1.0)  # Xanh lục nổi bật
        ma.markers.append(m_cam_body)

        # --- Markers: 5 Cubes ---
        cube_id = 1
        for obj_name, obj_cfg in self._scene["objects"].items():
            pos = obj_cfg["position"]
            size = obj_cfg["size"]
            col = obj_cfg["color"]

            m_cube = Marker()
            m_cube.header.frame_id = frame_id
            m_cube.header.stamp = now
            m_cube.ns = "scene_cubes"
            m_cube.id = cube_id
            cube_id += 1
            m_cube.type = Marker.CUBE
            m_cube.action = Marker.ADD
            m_cube.pose.position.x = float(pos["x"])
            m_cube.pose.position.y = float(pos["y"])
            m_cube.pose.position.z = float(pos["z"])
            m_cube.pose.orientation.w = 1.0
            m_cube.scale.x = float(size)
            m_cube.scale.y = float(size)
            m_cube.scale.z = float(size)
            m_cube.color = ColorRGBA(
                r=float(col["r"]), g=float(col["g"]),
                b=float(col["b"]), a=1.0,
            )
            ma.markers.append(m_cube)

        # --- Markers: 3 Zones (Trays + Text Labels) ---
        z_idx = 10
        table_top_z = self._scene["heights"]["table_surface"]
        tray_size = self._scene["zone_detection"]["tray_size"]

        for zone_name, zone_cfg in self._scene["zones"].items():
            pos = zone_cfg["position"]
            bc = zone_cfg["border_color"]
            label = zone_cfg["label"]

            # Khay zone
            m_tray = Marker()
            m_tray.header.frame_id = frame_id
            m_tray.header.stamp = now
            m_tray.ns = "scene_zones"
            m_tray.id = z_idx
            m_tray.type = Marker.CUBE
            m_tray.action = Marker.ADD
            m_tray.pose.position.x = float(pos["x"])
            m_tray.pose.position.y = float(pos["y"])
            m_tray.pose.position.z = table_top_z + 0.002
            m_tray.pose.orientation.w = 1.0
            m_tray.scale.x = tray_size
            m_tray.scale.y = tray_size
            m_tray.scale.z = 0.004
            m_tray.color = ColorRGBA(
                r=float(bc["r"]), g=float(bc["g"]),
                b=float(bc["b"]), a=0.7,
            )
            ma.markers.append(m_tray)

            # Trích xuất ký tự zone (A, B, C hoặc T)
            char_key = 'A'
            for k in ['A', 'B', 'C', 'T']:
                if k in label.upper() or k in zone_name.upper():
                    char_key = k
                    break
            if char_key not in ZONE_LETTER_BITMAPS and ("STAGING" in label.upper() or "STAGING" in zone_name.upper() or "TEMP" in label.upper() or "TEMP" in zone_name.upper()):
                char_key = 'T'

            # 3D Symbol [A], [B], [C] vẽ dập nổi trực tiếp trên mặt đáy khay trong RViz
            bitmap = ZONE_LETTER_BITMAPS.get(char_key, ZONE_LETTER_BITMAPS['A'])
            px_size = 0.0035  # 3.5mm
            num_rows = len(bitmap)
            num_cols = len(bitmap[0])

            m_symbol = Marker()
            m_symbol.header.frame_id = frame_id
            m_symbol.header.stamp = now
            m_symbol.ns = "scene_zone_symbols"
            m_symbol.id = z_idx + 200
            m_symbol.type = Marker.CUBE_LIST
            m_symbol.action = Marker.ADD
            m_symbol.scale.x = px_size * 0.95
            m_symbol.scale.y = px_size * 0.95
            m_symbol.scale.z = 0.001
            m_symbol.color = ColorRGBA(r=0.08, g=0.08, b=0.08, a=1.0)  # Đen đậm tương phản

            for row_idx, row_str in enumerate(bitmap):
                x_off = ((num_rows - 1) / 2.0 - row_idx) * px_size
                for col_idx, ch in enumerate(row_str):
                    if ch == '#':
                        y_off = ((num_cols - 1) / 2.0 - col_idx) * px_size
                        p = Point()
                        p.x = float(pos["x"]) + x_off
                        p.y = float(pos["y"]) + y_off
                        p.z = table_top_z + 0.0045  # Nằm nổi trên mặt đáy khay
                        m_symbol.points.append(p)
            ma.markers.append(m_symbol)

            # Text label nổi trên không [A] Zone A
            m_text = Marker()
            m_text.header.frame_id = frame_id
            m_text.header.stamp = now
            m_text.ns = "scene_zone_labels"
            m_text.id = z_idx + 100
            m_text.type = Marker.TEXT_VIEW_FACING
            m_text.action = Marker.ADD
            m_text.pose.position.x = float(pos["x"])
            m_text.pose.position.y = float(pos["y"])
            m_text.pose.position.z = table_top_z + 0.07
            m_text.pose.orientation.w = 1.0
            m_text.scale.z = 0.035
            m_text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            m_text.text = f"[{char_key}] {label}"
            ma.markers.append(m_text)

            z_idx += 1

        # ═══ 3. TAY KẸP 2 NGÓN (Two-Finger Gripper) gắn trực tiếp vào tool0 trong RViz ═══
        # Khung gắn Flange (Adapter)
        m_g_base = Marker()
        m_g_base.header.frame_id = "tool0"
        m_g_base.header.stamp = now
        m_g_base.ns = "robot_gripper"
        m_g_base.id = 1
        m_g_base.type = Marker.CYLINDER
        m_g_base.action = Marker.ADD
        m_g_base.pose.position.z = 0.01
        m_g_base.pose.orientation.w = 1.0
        m_g_base.scale.x = 0.07
        m_g_base.scale.y = 0.07
        m_g_base.scale.z = 0.02
        m_g_base.color = ColorRGBA(r=0.25, g=0.25, b=0.28, a=1.0)
        ma.markers.append(m_g_base)

        # Thân tay kẹp (Palm Body)
        m_g_palm = Marker()
        m_g_palm.header.frame_id = "tool0"
        m_g_palm.header.stamp = now
        m_g_palm.ns = "robot_gripper"
        m_g_palm.id = 2
        m_g_palm.type = Marker.CUBE
        m_g_palm.action = Marker.ADD
        m_g_palm.pose.position.z = 0.03
        m_g_palm.pose.orientation.w = 1.0
        m_g_palm.scale.x = 0.045
        m_g_palm.scale.y = 0.08
        m_g_palm.scale.z = 0.02
        m_g_palm.color = ColorRGBA(r=0.15, g=0.15, b=0.18, a=1.0)
        ma.markers.append(m_g_palm)

        # Ngón tay trái (Left Finger Prong)
        m_g_fl = Marker()
        m_g_fl.header.frame_id = "tool0"
        m_g_fl.header.stamp = now
        m_g_fl.ns = "robot_gripper"
        m_g_fl.id = 3
        m_g_fl.type = Marker.CUBE
        m_g_fl.action = Marker.ADD
        m_g_fl.pose.position.x = 0.0
        m_g_fl.pose.position.y = 0.026
        m_g_fl.pose.position.z = 0.065
        m_g_fl.pose.orientation.w = 1.0
        m_g_fl.scale.x = 0.015
        m_g_fl.scale.y = 0.01
        m_g_fl.scale.z = 0.035
        m_g_fl.color = ColorRGBA(r=0.2, g=0.2, b=0.22, a=1.0)
        ma.markers.append(m_g_fl)

        # Đệm cao su ngón trái (Left Rubber Pad)
        m_g_pad_l = Marker()
        m_g_pad_l.header.frame_id = "tool0"
        m_g_pad_l.header.stamp = now
        m_g_pad_l.ns = "robot_gripper"
        m_g_pad_l.id = 4
        m_g_pad_l.type = Marker.CUBE
        m_g_pad_l.action = Marker.ADD
        m_g_pad_l.pose.position.x = 0.0
        m_g_pad_l.pose.position.y = 0.021
        m_g_pad_l.pose.position.z = 0.065
        m_g_pad_l.pose.orientation.w = 1.0
        m_g_pad_l.scale.x = 0.014
        m_g_pad_l.scale.y = 0.003
        m_g_pad_l.scale.z = 0.03
        m_g_pad_l.color = ColorRGBA(r=0.95, g=0.45, b=0.15, a=1.0)
        ma.markers.append(m_g_pad_l)

        # Ngón tay phải (Right Finger Prong)
        m_g_fr = Marker()
        m_g_fr.header.frame_id = "tool0"
        m_g_fr.header.stamp = now
        m_g_fr.ns = "robot_gripper"
        m_g_fr.id = 5
        m_g_fr.type = Marker.CUBE
        m_g_fr.action = Marker.ADD
        m_g_fr.pose.position.x = 0.0
        m_g_fr.pose.position.y = -0.026
        m_g_fr.pose.position.z = 0.065
        m_g_fr.pose.orientation.w = 1.0
        m_g_fr.scale.x = 0.015
        m_g_fr.scale.y = 0.01
        m_g_fr.scale.z = 0.035
        m_g_fr.color = ColorRGBA(r=0.2, g=0.2, b=0.22, a=1.0)
        ma.markers.append(m_g_fr)

        # Đệm cao su ngón phải (Right Rubber Pad)
        m_g_pad_r = Marker()
        m_g_pad_r.header.frame_id = "tool0"
        m_g_pad_r.header.stamp = now
        m_g_pad_r.ns = "robot_gripper"
        m_g_pad_r.id = 6
        m_g_pad_r.type = Marker.CUBE
        m_g_pad_r.action = Marker.ADD
        m_g_pad_r.pose.position.x = 0.0
        m_g_pad_r.pose.position.y = -0.021
        m_g_pad_r.pose.position.z = 0.065
        m_g_pad_r.pose.orientation.w = 1.0
        m_g_pad_r.scale.x = 0.014
        m_g_pad_r.scale.y = 0.003
        m_g_pad_r.scale.z = 0.03
        m_g_pad_r.color = ColorRGBA(r=0.95, g=0.45, b=0.15, a=1.0)
        ma.markers.append(m_g_pad_r)

        self._marker_pub.publish(ma)
        self._scene_marker_pub.publish(ma)


# ===========================================================================
#  Entrypoint
# ===========================================================================
def main(args=None):
    rclpy.init(args=args)
    node = SceneSpawnerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Dừng SceneSpawnerNode.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
