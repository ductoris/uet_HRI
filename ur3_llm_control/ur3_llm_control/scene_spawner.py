#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scene_spawner.py — Spawn các khối lập phương (red, yellow, blue) và khay zone
vào Ignition Gazebo bằng ros_gz_sim/create service.

Mỗi đối tượng được mô tả bằng SDF inline với visual + collision + màu chính xác.

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
from geometry_msgs.msg import Pose
from std_msgs.msg import ColorRGBA, Header


# ===========================================================================
#  SDF Templates
# ===========================================================================
def _cube_sdf(name: str, size: float, r: float, g: float, b: float, a: float = 1.0, mass: float = 0.05) -> str:
    """Tạo SDF string cho một khối lập phương có màu."""
    half = size / 2.0
    inertia = (1.0/6.0) * mass * size * size
    return f"""<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <inertial>
        <mass>{mass}</mass>
        <inertia>
          <ixx>{inertia:.6f}</ixx><iyy>{inertia:.6f}</iyy><izz>{inertia:.6f}</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry><box><size>{size} {size} {size}</size></box></geometry>
      </collision>
      <visual name="visual">
        <geometry><box><size>{size} {size} {size}</size></box></geometry>
        <material>
          <ambient>{r} {g} {b} {a}</ambient>
          <diffuse>{r} {g} {b} {a}</diffuse>
          <specular>0.3 0.3 0.3 1</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""


def _zone_tray_sdf(name: str, label: str, r: float, g: float, b: float,
                    width: float = 0.06, depth: float = 0.06, height: float = 0.005) -> str:
    """Tạo SDF string cho một khay (zone tray) — hình hộp mỏng có viền màu."""
    mass = 0.01
    ixx = (1.0/12.0) * mass * (depth*depth + height*height)
    iyy = (1.0/12.0) * mass * (width*width + height*height)
    izz = (1.0/12.0) * mass * (width*width + depth*depth)

    # Khay gồm: đáy mỏng (base) + 4 viền nhỏ (borders)
    border_h = 0.01  # Chiều cao viền
    border_t = 0.003  # Độ dày viền

    return f"""<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="{name}">
    <static>true</static>

    <!-- Đáy khay -->
    <link name="base">
      <collision name="collision">
        <geometry><box><size>{width} {depth} {height}</size></box></geometry>
      </collision>
      <visual name="visual">
        <geometry><box><size>{width} {depth} {height}</size></box></geometry>
        <material>
          <ambient>0.85 0.85 0.85 1</ambient>
          <diffuse>0.9 0.9 0.9 1</diffuse>
          <specular>0.2 0.2 0.2 1</specular>
        </material>
      </visual>
    </link>

    <!-- Viền trước -->
    <link name="border_front">
      <pose>{width/2 - border_t/2} 0 {height/2 + border_h/2} 0 0 0</pose>
      <visual name="visual">
        <geometry><box><size>{border_t} {depth} {border_h}</size></box></geometry>
        <material>
          <ambient>{r} {g} {b} 1</ambient>
          <diffuse>{r} {g} {b} 1</diffuse>
        </material>
      </visual>
      <collision name="collision">
        <geometry><box><size>{border_t} {depth} {border_h}</size></box></geometry>
      </collision>
    </link>
    <joint name="j_front" type="fixed"><parent>base</parent><child>border_front</child></joint>

    <!-- Viền sau -->
    <link name="border_back">
      <pose>{-width/2 + border_t/2} 0 {height/2 + border_h/2} 0 0 0</pose>
      <visual name="visual">
        <geometry><box><size>{border_t} {depth} {border_h}</size></box></geometry>
        <material>
          <ambient>{r} {g} {b} 1</ambient>
          <diffuse>{r} {g} {b} 1</diffuse>
        </material>
      </visual>
      <collision name="collision">
        <geometry><box><size>{border_t} {depth} {border_h}</size></box></geometry>
      </collision>
    </link>
    <joint name="j_back" type="fixed"><parent>base</parent><child>border_back</child></joint>

    <!-- Viền trái -->
    <link name="border_left">
      <pose>0 {depth/2 - border_t/2} {height/2 + border_h/2} 0 0 0</pose>
      <visual name="visual">
        <geometry><box><size>{width} {border_t} {border_h}</size></box></geometry>
        <material>
          <ambient>{r} {g} {b} 1</ambient>
          <diffuse>{r} {g} {b} 1</diffuse>
        </material>
      </visual>
      <collision name="collision">
        <geometry><box><size>{width} {border_t} {border_h}</size></box></geometry>
      </collision>
    </link>
    <joint name="j_left" type="fixed"><parent>base</parent><child>border_left</child></joint>

    <!-- Viền phải -->
    <link name="border_right">
      <pose>0 {-depth/2 + border_t/2} {height/2 + border_h/2} 0 0 0</pose>
      <visual name="visual">
        <geometry><box><size>{width} {border_t} {border_h}</size></box></geometry>
        <material>
          <ambient>{r} {g} {b} 1</ambient>
          <diffuse>{r} {g} {b} 1</diffuse>
        </material>
      </visual>
      <collision name="collision">
        <geometry><box><size>{width} {border_t} {border_h}</size></box></geometry>
      </collision>
    </link>
    <joint name="j_right" type="fixed"><parent>base</parent><child>border_right</child></joint>

    <!-- Nhãn label (text trên đáy khay - dùng visual nhỏ để đánh dấu) -->
  </model>
</sdf>"""


def _table_sdf(x: float, y: float, z: float,
               length: float = 0.80, width: float = 0.60, height: float = 0.02) -> str:
    """Tạo SDF cho bàn làm việc."""
    leg_h = abs(z) - height/2 if abs(z) > height else 0.4
    return f"""<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="work_table">
    <static>true</static>
    <!-- Mặt bàn -->
    <link name="tabletop">
      <pose>0 0 0 0 0 0</pose>
      <collision name="collision">
        <geometry><box><size>{length} {width} {height}</size></box></geometry>
      </collision>
      <visual name="visual">
        <geometry><box><size>{length} {width} {height}</size></box></geometry>
        <material>
          <ambient>0.55 0.37 0.24 1</ambient>
          <diffuse>0.65 0.45 0.30 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""


# ===========================================================================
#  Node ROS 2 — SceneSpawner (với retry + đợi Gazebo sẵn sàng)
# ===========================================================================
MAX_SPAWN_RETRIES = 3
RETRY_DELAY = 2.0       # Giây đợi giữa các lần retry
SPAWN_INTERVAL = 1.0    # Giây đợi giữa các object


class SceneSpawnerNode(Node):
    """Spawn các objects và zones vào Ignition Gazebo."""

    def __init__(self):
        super().__init__("scene_spawner_node")
        self.get_logger().info("SceneSpawnerNode đang khởi tạo...")

        # Load scene config
        pkg_dir = get_package_share_directory("ur3_llm_control")
        config_path = os.path.join(pkg_dir, "config", "scene.yaml")
        with open(config_path, "r") as f:
            self._scene = yaml.safe_load(f)

        # Load student config
        student_path = os.path.join(pkg_dir, "config", "student_config.yaml")
        with open(student_path, "r") as f:
            self._student = yaml.safe_load(f)

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

        # ĐỢI GAZEBO SẴN SÀNG — kiểm tra /clock topic
        self._wait_for_gazebo()

        # Spawn tất cả vào Gazebo
        self._spawn_table()
        self._spawn_cubes()
        self._spawn_zones()

        # Publish collision objects & visual markers cho MoveIt / RViz
        self._publish_rviz_scene()

        # Timer chu kỳ 1s để duy trì MarkerArray trên RViz liên tục
        self._timer = self.create_timer(1.0, self._publish_rviz_scene)

        self.get_logger().info("═" * 50)
        self.get_logger().info("✓ Đã spawn vào Gazebo và đồng bộ đầy đủ lên RViz!")
        self.get_logger().info("  (Giữ Terminal này chạy để duy trì visual markers trong RViz)")
        self.get_logger().info("═" * 50)

    def _wait_for_gazebo(self, timeout_sec: float = 60.0):
        """
        Đợi Gazebo sẵn sàng bằng cách kiểm tra topic /clock.
        Nếu /clock có publisher → Gazebo đang chạy và bridge hoạt động.
        """
        self.get_logger().info("Đợi Gazebo sẵn sàng (kiểm tra /clock topic)...")
        start = time.time()
        clock_ready = False

        while time.time() - start < timeout_sec:
            # Kiểm tra xem /clock có publisher nào không
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
                f"⚠ Không phát hiện Gazebo sau {timeout_sec}s. "
                "Vẫn thử spawn..."
            )
            time.sleep(3.0)

    def _spawn_via_gz(self, sdf_string: str, name: str,
                       x: float, y: float, z: float) -> bool:
        """
        Spawn model vào Gazebo với retry logic.

        Returns True nếu spawn thành công.
        """
        tmp_sdf = f"/tmp/{name}.sdf"
        with open(tmp_sdf, "w") as f:
            f.write(sdf_string)

        for attempt in range(1, MAX_SPAWN_RETRIES + 1):
            self.get_logger().info(
                f"Spawning '{name}' at ({x:.3f}, {y:.3f}, {z:.3f}) "
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

                # Kiểm tra thành công (return code 0 HOẶC output chứa
                # dấu hiệu model đã tồn tại)
                if result.returncode == 0:
                    self.get_logger().info(f"  ✓ '{name}' spawned thành công!")
                    time.sleep(SPAWN_INTERVAL)
                    return True

                # Nếu model đã tồn tại thì cũng coi như OK
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

            # Đợi trước khi retry
            if attempt < MAX_SPAWN_RETRIES:
                self.get_logger().info(f"  Đợi {RETRY_DELAY}s trước khi thử lại...")
                time.sleep(RETRY_DELAY)

        self.get_logger().error(
            f"  ✗✗ '{name}' THẤT BẠI sau {MAX_SPAWN_RETRIES} lần thử!"
        )
        return False

    def _spawn_table(self):
        """Spawn bàn làm việc."""
        table = self._scene["table"]
        pos = table["position"]
        dims = table["dimensions"]
        sdf = _table_sdf(
            pos["x"], pos["y"], pos["z"],
            dims["length"], dims["width"], dims["depth"]
        )
        self._spawn_via_gz(sdf, "work_table", pos["x"], pos["y"], pos["z"])

    def _spawn_cubes(self):
        """Spawn 3 khối lập phương với đúng màu."""
        for obj_name, obj_cfg in self._scene["objects"].items():
            pos = obj_cfg["position"]
            color = obj_cfg["color"]
            size = obj_cfg["size"]
            sdf = _cube_sdf(
                obj_name, size,
                color["r"], color["g"], color["b"], color.get("a", 1.0)
            )
            self._spawn_via_gz(sdf, obj_name, pos["x"], pos["y"], pos["z"])

    def _spawn_zones(self):
        """Spawn các khay zone với viền màu phân biệt."""
        zone_colors = {
            "zone_a": (0.2, 0.6, 1.0),    # Xanh dương nhạt
            "zone_b": (1.0, 0.4, 0.2),    # Cam đỏ
            "zone_c": (0.2, 0.8, 0.3),    # Xanh lá
            "temp_zone": (0.7, 0.7, 0.7), # Xám
        }

        for zone_name, zone_cfg in self._scene["zones"].items():
            pos = zone_cfg["position"]
            label = zone_cfg["label"]
            r, g, b = zone_colors.get(zone_name, (0.5, 0.5, 0.5))
            sdf = _zone_tray_sdf(zone_name, label, r, g, b)
            z_tray = self._scene["heights"]["table_surface"] + 0.0025
            self._spawn_via_gz(sdf, zone_name, pos["x"], pos["y"], z_tray)

    def _publish_rviz_scene(self):
        """Publish collision objects vào /planning_scene và markers vào RViz."""
        now = self.get_clock().now().to_msg()
        frame_id = "world"

        # 1. MOVEIT PLANNING SCENE (Collision Objects + Colors)
        ps = PlanningScene()
        ps.is_diff = True

        # --- Table collision object ---
        tbl = self._scene["table"]
        t_dims = tbl["dimensions"]
        t_pos = tbl["position"]
        tbl_co = CollisionObject()
        tbl_co.id = "work_table"
        tbl_co.header.frame_id = frame_id
        tbl_co.header.stamp = now
        b_box = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[t_dims["length"], t_dims["width"], t_dims["depth"]])
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
        tbl_color.color = ColorRGBA(r=0.65, g=0.45, b=0.30, a=0.9)
        ps.object_colors.append(tbl_color)

        self._planning_scene_pub.publish(ps)

        # 2. RVIZ VISUALIZATION MARKER ARRAY
        ma = MarkerArray()

        # Marker: Table
        m_tbl = Marker()
        m_tbl.header.frame_id = frame_id
        m_tbl.header.stamp = now
        m_tbl.ns = "scene_table"
        m_tbl.id = 0
        m_tbl.type = Marker.CUBE
        m_tbl.action = Marker.ADD
        m_tbl.pose = b_pose
        m_tbl.scale.x = float(t_dims["length"])
        m_tbl.scale.y = float(t_dims["width"])
        m_tbl.scale.z = float(t_dims["depth"])
        m_tbl.color = ColorRGBA(r=0.65, g=0.45, b=0.30, a=0.85)
        ma.markers.append(m_tbl)

        # Markers: Cubes
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
                r=float(col["r"]),
                g=float(col["g"]),
                b=float(col["b"]),
                a=1.0
            )
            ma.markers.append(m_cube)

        # Markers: Zones (Trays + Text Labels)
        zone_info_map = {
            "zone_a": ("Zone A\n(Target: Blue Cube)", ColorRGBA(r=0.1, g=0.5, b=0.9, a=0.6)),
            "zone_b": ("Zone B\n(Target: Red Cube)", ColorRGBA(r=0.9, g=0.2, b=0.2, a=0.6)),
            "zone_c": ("Zone C\n(Target: Yellow Cube)", ColorRGBA(r=0.9, g=0.8, b=0.1, a=0.6)),
            "temp_zone": ("Temp Zone", ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.6)),
        }

        z_idx = 10
        table_top_z = self._scene["heights"]["table_surface"]
        for zone_name, zone_cfg in self._scene["zones"].items():
            pos = zone_cfg["position"]
            label_text, tray_color = zone_info_map.get(
                zone_name, (zone_name, ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.6))
            )

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
            m_tray.scale.x = 0.07
            m_tray.scale.y = 0.07
            m_tray.scale.z = 0.004
            m_tray.color = tray_color
            ma.markers.append(m_tray)

            # 3D Floating text label
            m_text = Marker()
            m_text.header.frame_id = frame_id
            m_text.header.stamp = now
            m_text.ns = "scene_zone_labels"
            m_text.id = z_idx + 100
            m_text.type = Marker.TEXT_VIEW_FACING
            m_text.action = Marker.ADD
            m_text.pose.position.x = float(pos["x"])
            m_text.pose.position.y = float(pos["y"])
            m_text.pose.position.z = table_top_z + 0.05
            m_text.pose.orientation.w = 1.0
            m_text.scale.z = 0.025
            m_text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            m_text.text = label_text
            ma.markers.append(m_text)

            z_idx += 1

        # Student info floating banner
        student = self._student["student"]
        m_banner = Marker()
        m_banner.header.frame_id = frame_id
        m_banner.header.stamp = now
        m_banner.ns = "student_banner"
        m_banner.id = 999
        m_banner.type = Marker.TEXT_VIEW_FACING
        m_banner.action = Marker.ADD
        m_banner.pose.position.x = 0.0
        m_banner.pose.position.y = 0.40
        m_banner.pose.position.z = 0.40
        m_banner.pose.orientation.w = 1.0
        m_banner.scale.z = 0.035
        m_banner.color = ColorRGBA(r=0.2, g=0.9, b=1.0, a=1.0)
        m_banner.text = f"{student['name']} | MSSV: {student['mssv']} | P={student['P']}"
        ma.markers.append(m_banner)

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

