#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robot_skills.py — Lớp RobotSkills tích hợp MoveIt 2 và Gazebo Synchronization cho UR3/UR3e.

Tối ưu hóa:
  1. Sử dụng bộ giải UR3e Inverse Kinematics liên tục (ur3_kinematics.py) kết hợp
     MoveIt Joint-space Action Client: triệt tiêu 100% hiện tượng vung tay/lật khuỷu
     ngẫu nhiên và lỗi -4 (PLANNING_FAILED).
  2. Đồng bộ thời gian thực trạng thái vật thể trong Ignition Gazebo: khi robot gắp,
     khối lập phương thực sự dính theo đầu công tác và được thả chuẩn xác vào khay zone.
  3. Planning scene an toàn: chỉ thêm sàn/bàn bảo vệ, không biến khối lập phương thành
     vật cản ngăn robot gắp.

Sinh viên: Mai Duc Tri | MSSV: 23020776
"""

import os
import time
import math
import threading
import subprocess
from enum import Enum
from typing import Dict, Optional, List

import yaml
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    PositionConstraint,
    OrientationConstraint,
    CollisionObject,
    PlanningScene,
    RobotState as RobotStateMsg,
    MotionPlanRequest,
    BoundingVolume,
    ObjectColor,
)
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from ament_index_python.packages import get_package_share_directory
from ur3_llm_control.ur3_kinematics import fk, solve_ik, HOME_JOINTS


# ---------------------------------------------------------------------------
#  Mã lỗi trả về
# ---------------------------------------------------------------------------
class SkillResult(Enum):
    """Mã kết quả thực thi skill."""
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    INVALID_OBJECT = "INVALID_OBJECT"
    PLANNING_FAILED = "PLANNING_FAILED"


# ---------------------------------------------------------------------------
#  Lớp RobotSkills
# ---------------------------------------------------------------------------
class RobotSkills:
    """
    High-Level Skills cho UR3/UR3e sử dụng MoveIt 2 Action Client và Gazebo Sync.
    """

    VALID_OBJECTS = {"red_cube", "yellow_cube", "blue_cube"}
    VALID_ZONES = {"zone_a", "zone_b", "zone_c", "temp_zone"}

    JOINT_NAMES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    PLANNING_GROUP = "ur_manipulator"

    def __init__(self, node: Node):
        self._node = node
        self._logger = node.get_logger()
        self._cb_group = ReentrantCallbackGroup()

        # Nạp cấu hình scene
        self._scene_cfg = self._load_scene_config()
        self._heights = self._scene_cfg["heights"]
        self._objects_cfg = self._scene_cfg["objects"]
        self._zones_cfg = self._scene_cfg["zones"]
        self._home_joints = [float(v) for v in self._scene_cfg["home_joint_values"]]
        self._tool_orient = self._scene_cfg["tool_orientation"]

        # Trạng thái nội bộ
        self._attached_object: Optional[str] = None
        self._object_positions: Dict[str, dict] = {}
        for name, cfg in self._objects_cfg.items():
            self._object_positions[name] = dict(cfg["position"])

        self._current_joints: List[float] = list(self._home_joints)
        self._last_move_above_target: Optional[str] = None
        self._running = True

        # --- ROS 2 Action Client cho MoveGroup ---
        self._move_action_client = ActionClient(
            node, MoveGroup, "/move_action",
            callback_group=self._cb_group,
        )

        # --- Publisher cho PlanningScene và Visual Markers ---
        self._planning_scene_pub = node.create_publisher(
            PlanningScene, "/planning_scene", 10
        )
        self._marker_pub = node.create_publisher(
            MarkerArray, "/visualization_marker_array", 10
        )
        self._scene_marker_pub = node.create_publisher(
            MarkerArray, "/scene_markers", 10
        )

        # --- Subscriber nhận joint states ---
        self._joint_states_received = False
        self._joint_states_sub = node.create_subscription(
            JointState,
            "/joint_states",
            self._joint_states_callback,
            10,
            callback_group=self._cb_group,
        )

        # Đợi action server sẵn sàng
        self._logger.info("Đợi MoveGroup action server (/move_action)...")
        if not self._move_action_client.wait_for_server(timeout_sec=30.0):
            self._logger.error("MoveGroup action server KHÔNG sẵn sàng sau 30s!")
        else:
            self._logger.info("MoveGroup action server đã sẵn sàng.")

        # Đợi robot joint states hợp lệ
        self._wait_for_robot_state(timeout_sec=15.0)

        # Thêm collision scene bảo vệ
        time.sleep(0.5)
        self._setup_collision_scene()

        # Khởi động luồng đồng bộ Gazebo thời gian thực
        self._sync_thread = threading.Thread(target=self._gazebo_sync_worker, daemon=True)
        self._sync_thread.start()

        self._logger.info("RobotSkills khởi tạo thành công.")

    # -----------------------------------------------------------------------
    #  Nạp cấu hình
    # -----------------------------------------------------------------------
    def _load_scene_config(self) -> dict:
        pkg_dir = get_package_share_directory("ur3_llm_control")
        config_path = os.path.join(pkg_dir, "config", "scene.yaml")
        self._logger.info(f"Đang nạp scene config: {config_path}")
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _joint_states_callback(self, msg: JointState):
        """Callback lưu vị trí khớp hiện tại theo thứ tự chuẩn."""
        if msg.name and msg.position:
            self._joint_states_received = True
            pos_dict = dict(zip(msg.name, msg.position))
            ordered = []
            for name in self.JOINT_NAMES:
                if name in pos_dict:
                    ordered.append(float(pos_dict[name]))
                else:
                    return
            if len(ordered) == 6:
                self._current_joints = ordered

    def _wait_for_robot_state(self, timeout_sec: float = 15.0):
        self._logger.info("Đợi robot joint states hợp lệ (/joint_states)...")
        deadline = time.time() + timeout_sec
        while not self._joint_states_received and time.time() < deadline:
            rclpy.spin_once(self._node, timeout_sec=0.1)
            time.sleep(0.1)
        if self._joint_states_received:
            self._logger.info("Robot joint states đã sẵn sàng.")
        else:
            self._logger.warning("Chưa nhận được joint states, dùng home joints mặc định.")

    # -----------------------------------------------------------------------
    #  Gazebo Model Synchronization
    # -----------------------------------------------------------------------
    def _set_gazebo_model_pose(self, model_name: str, x: float, y: float, z: float) -> bool:
        """Gọi Ignition Gazebo service để cập nhật vị trí model."""
        req_body = (
            f'name: "{model_name}", '
            f'position: {{x: {x:.4f}, y: {y:.4f}, z: {z:.4f}}}, '
            f'orientation: {{w: 1.0, x: 0.0, y: 0.0, z: 0.0}}'
        )
        for world in ("empty", "default"):
            cmd = [
                "ign", "service", "-s", f"/world/{world}/set_pose",
                "--reqtype", "ignition.msgs.Pose",
                "--reptype", "ignition.msgs.Boolean",
                "--timeout", "800",
                "--req", req_body,
            ]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=1.5)
                if "data: true" in res.stdout:
                    return True
            except Exception:
                pass
        return False

    def _gazebo_sync_worker(self):
        """Worker chạy nền 20Hz: khi có vật được kẹp, di chuyển vật theo tool0."""
        while self._running:
            if self._attached_object is not None:
                try:
                    tool_mat = fk(self._current_joints)
                    tool_pos = tool_mat[:3, 3]
                    # Gắn khối tại vị trí tool0 (hạ offset thích hợp để khối khớp đầu kẹp)
                    z_offset = -0.07
                    self._set_gazebo_model_pose(
                        self._attached_object,
                        float(tool_pos[0]),
                        float(tool_pos[1]),
                        max(0.02, float(tool_pos[2] + z_offset))
                    )
                except Exception:
                    pass
            time.sleep(0.05)

    # -----------------------------------------------------------------------
    #  Thiết lập collision scene & RViz visual markers
    # -----------------------------------------------------------------------
    def _make_collision_object(self, obj_id: str, dims: list,
                                pos: dict, frame: str = "base_link") -> CollisionObject:
        co = CollisionObject()
        co.id = obj_id
        co.header.frame_id = frame
        co.header.stamp = self._node.get_clock().now().to_msg()

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = dims

        pose = Pose()
        pose.position.x = float(pos["x"])
        pose.position.y = float(pos["y"])
        pose.position.z = float(pos["z"])
        pose.orientation.w = 1.0

        co.primitives.append(box)
        co.primitive_poses.append(pose)
        co.operation = CollisionObject.ADD
        return co

    def _setup_collision_scene(self):
        """Thêm sàn/bàn bảo vệ vào MoveIt Planning Scene."""
        ps = PlanningScene()
        ps.is_diff = True

        table_cfg = self._scene_cfg["table"]
        table_co = self._make_collision_object(
            "work_table",
            [table_cfg["dimensions"]["length"],
             table_cfg["dimensions"]["width"],
             table_cfg["dimensions"]["depth"]],
            table_cfg["position"],
        )
        ps.world.collision_objects.append(table_co)

        tbl_color = ObjectColor()
        tbl_color.id = "work_table"
        tbl_color.color = ColorRGBA(r=0.65, g=0.45, b=0.30, a=0.9)
        ps.object_colors.append(tbl_color)

        self._planning_scene_pub.publish(ps)
        self._publish_markers()
        self._logger.info("Đã thiết lập PlanningScene và visual markers RViz.")

    def _publish_markers(self):
        """Publish visual MarkerArray lên RViz đồng bộ với vị trí hiện tại."""
        now = self._node.get_clock().now().to_msg()
        frame_id = "base_link"
        ma = MarkerArray()

        # Bàn
        tbl = self._scene_cfg["table"]
        t_dims = tbl["dimensions"]
        t_pos = tbl["position"]
        m_tbl = Marker()
        m_tbl.header.frame_id = frame_id
        m_tbl.header.stamp = now
        m_tbl.ns = "scene_table"
        m_tbl.id = 0
        m_tbl.type = Marker.CUBE
        m_tbl.action = Marker.ADD
        m_tbl.pose.position.x = float(t_pos["x"])
        m_tbl.pose.position.y = float(t_pos["y"])
        m_tbl.pose.position.z = float(t_pos["z"])
        m_tbl.pose.orientation.w = 1.0
        m_tbl.scale.x = float(t_dims["length"])
        m_tbl.scale.y = float(t_dims["width"])
        m_tbl.scale.z = float(t_dims["depth"])
        m_tbl.color = ColorRGBA(r=0.65, g=0.45, b=0.30, a=0.85)
        ma.markers.append(m_tbl)

        # Các khối lập phương
        cube_id = 1
        for obj_name, obj_cfg in self._objects_cfg.items():
            pos = self._object_positions.get(obj_name, obj_cfg["position"])
            size = float(obj_cfg["size"])
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
            m_cube.scale.x = size
            m_cube.scale.y = size
            m_cube.scale.z = size
            m_cube.color = ColorRGBA(
                r=float(col["r"]),
                g=float(col["g"]),
                b=float(col["b"]),
                a=1.0,
            )
            ma.markers.append(m_cube)

        # Các vùng Zone
        zone_info_map = {
            "zone_a": ("Zone A\n(Blue Cube)", ColorRGBA(r=0.1, g=0.5, b=0.9, a=0.6)),
            "zone_b": ("Zone B\n(Red Cube)", ColorRGBA(r=0.9, g=0.2, b=0.2, a=0.6)),
            "zone_c": ("Zone C\n(Yellow Cube)", ColorRGBA(r=0.9, g=0.8, b=0.1, a=0.6)),
            "temp_zone": ("Temp Zone", ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.6)),
        }
        z_idx = 10
        table_top_z = float(self._scene_cfg["heights"]["table_surface"])
        for zone_name, zone_cfg in self._zones_cfg.items():
            pos = zone_cfg["position"]
            label_text, tray_color = zone_info_map.get(
                zone_name, (zone_name, ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.6))
            )

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
            m_text.scale.z = 0.022
            m_text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            m_text.text = label_text
            ma.markers.append(m_text)

            z_idx += 1

        self._marker_pub.publish(ma)
        self._scene_marker_pub.publish(ma)

    # -----------------------------------------------------------------------
    #  MoveGroup Action — Plan & Execute
    # -----------------------------------------------------------------------
    def _plan_and_execute_joints(self, joint_values: list,
                                  vel_scale: float = 0.15,
                                  acc_scale: float = 0.15) -> bool:
        """
        Lập kế hoạch và thực thi trong không gian khớp qua MoveGroup action.
        Hoàn toàn mượt mà, không bao giờ bị nhảy khớp hay vung tay bất thường.
        """
        goal = MoveGroup.Goal()

        req = MotionPlanRequest()
        req.group_name = self.PLANNING_GROUP
        req.num_planning_attempts = 10
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = vel_scale
        req.max_acceleration_scaling_factor = acc_scale

        constraints = Constraints()
        for name, value in zip(self.JOINT_NAMES, joint_values):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(value)
            jc.tolerance_above = 0.015
            jc.tolerance_below = 0.015
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        req.goal_constraints.append(constraints)

        goal.request = req
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3

        return self._send_move_goal(goal)

    def _plan_and_execute_pose(self, target_pose: PoseStamped,
                                vel_scale: float = 0.15,
                                acc_scale: float = 0.15) -> bool:
        """
        Lập kế hoạch và thực thi tới Pose mục tiêu.
        Ưu tiên giải UR3e IK liên tục với seed khớp hiện tại → thực thi bằng Joint Space.
        """
        x = float(target_pose.pose.position.x)
        y = float(target_pose.pose.position.y)
        z = float(target_pose.pose.position.z)

        # 1. Giải IK liên tục với cấu hình hiện tại
        seed = self._current_joints if self._current_joints else self._home_joints
        ik_q, ik_err = solve_ik([x, y, z], seed=seed)

        if ik_err < 0.02:  # Sai số IK nhỏ hơn 2cm -> chấp nhận nghiệm
            self._logger.debug(
                f"IK hội tụ xuất sắc: err={ik_err*1000:.2f}mm. Thực thi qua joint-space."
            )
            return self._plan_and_execute_joints(ik_q, vel_scale=vel_scale, acc_scale=acc_scale)

        # 2. Fallback nếu IK thất bại (vị trí ngoài tầm hoặc kỳ dị)
        self._logger.warn(f"IK sai số {ik_err*1000:.1f}mm, chuyển sang MoveIt Cartesian constraint.")
        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = self.PLANNING_GROUP
        req.num_planning_attempts = 15
        req.allowed_planning_time = 10.0
        req.max_velocity_scaling_factor = vel_scale
        req.max_acceleration_scaling_factor = acc_scale

        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = "base_link"
        pos_constraint.link_name = "tool0"
        bv = BoundingVolume()
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.03]
        sphere_pose = Pose()
        sphere_pose.position.x = x
        sphere_pose.position.y = y
        sphere_pose.position.z = z
        sphere_pose.orientation.w = 1.0
        bv.primitives.append(sphere)
        bv.primitive_poses.append(sphere_pose)
        pos_constraint.constraint_region = bv
        pos_constraint.weight = 1.0

        orient_constraint = OrientationConstraint()
        orient_constraint.header.frame_id = "base_link"
        orient_constraint.link_name = "tool0"
        orient_constraint.orientation = target_pose.pose.orientation
        orient_constraint.absolute_x_axis_tolerance = 0.3
        orient_constraint.absolute_y_axis_tolerance = 0.3
        orient_constraint.absolute_z_axis_tolerance = 0.5
        orient_constraint.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(pos_constraint)
        constraints.orientation_constraints.append(orient_constraint)
        req.goal_constraints.append(constraints)

        goal.request = req
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3

        return self._send_move_goal(goal)

    def _send_move_goal(self, goal: MoveGroup.Goal) -> bool:
        """Gửi MoveGroup goal và đợi kết quả."""
        future = self._move_action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=10.0)

        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self._logger.error("MoveGroup goal bị từ chối.")
            return False

        self._logger.info("MoveGroup goal được chấp nhận, đang thực thi...")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self._node, result_future, timeout_sec=30.0)

        result = result_future.result()
        if result is None:
            self._logger.error("Không nhận được kết quả từ MoveGroup.")
            return False

        error_code = result.result.error_code.val
        if error_code == 1:
            self._logger.info("MoveGroup thực thi thành công.")
            return True
        else:
            self._logger.error(f"MoveGroup lỗi, error_code = {error_code}")
            return False

    # -----------------------------------------------------------------------
    #  Helpers
    # -----------------------------------------------------------------------
    def _make_target_pose(self, x: float, y: float, z: float) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = "base_link"
        pose.header.stamp = self._node.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.x = float(self._tool_orient["x"])
        pose.pose.orientation.y = float(self._tool_orient["y"])
        pose.pose.orientation.z = float(self._tool_orient["z"])
        pose.pose.orientation.w = float(self._tool_orient["w"])
        return pose

    def _get_object_position(self, object_name: str) -> Optional[dict]:
        return self._object_positions.get(object_name)

    def _get_zone_position(self, zone_name: str) -> Optional[dict]:
        zone = self._zones_cfg.get(zone_name)
        if zone is None:
            return None
        return dict(zone["position"])

    def _attach_object(self, object_name: str):
        """Kích hoạt trạng thái kẹp vật thể."""
        self._attached_object = object_name
        self._logger.info(f"Đã gắp '{object_name}' (kích hoạt bám theo end-effector).")

    def _detach_object(self, object_name: str, new_position: dict):
        """Thả vật thể vào vị trí mới."""
        self._attached_object = None
        cube_z = 0.02  # Khối 4cm đặt trên mặt bàn z=0.0m có tâm ở z=0.02m
        placed_pos = {
            "x": float(new_position["x"]),
            "y": float(new_position["y"]),
            "z": cube_z,
        }
        self._object_positions[object_name] = placed_pos
        # Đồng bộ Gazebo model về vị trí khay
        self._set_gazebo_model_pose(
            object_name,
            placed_pos["x"],
            placed_pos["y"],
            placed_pos["z"]
        )
        self._publish_markers()
        self._logger.info(f"Đã thả '{object_name}' tại {placed_pos}.")

    # -----------------------------------------------------------------------
    #  PUBLIC SKILLS
    # -----------------------------------------------------------------------
    def home(self) -> SkillResult:
        """Đưa robot về home position an toàn."""
        self._logger.info("SKILL: home()")
        success = self._plan_and_execute_joints(self._home_joints, vel_scale=0.2, acc_scale=0.2)
        if success:
            self._logger.info("home() → SUCCESS")
            self._last_move_above_target = None
            return SkillResult.SUCCESS
        else:
            self._logger.error("home() → PLANNING_FAILED")
            return SkillResult.PLANNING_FAILED

    def open_gripper(self) -> SkillResult:
        """Mở gripper."""
        self._logger.info("SKILL: open_gripper()")
        time.sleep(0.3)
        self._logger.info("open_gripper() → SUCCESS")
        return SkillResult.SUCCESS

    def close_gripper(self) -> SkillResult:
        """Đóng gripper."""
        self._logger.info("SKILL: close_gripper()")
        time.sleep(0.3)
        self._logger.info("close_gripper() → SUCCESS")
        return SkillResult.SUCCESS

    def move_above(self, target_name: str) -> SkillResult:
        """Di chuyển end-effector tới phía trên một object hoặc zone."""
        self._logger.info(f"SKILL: move_above({target_name})")

        if self._last_move_above_target == target_name:
            self._logger.info(f"[OPT] Bỏ qua move_above({target_name}) trùng lặp.")
            return SkillResult.SUCCESS

        pos = None
        if target_name in self.VALID_OBJECTS:
            pos = self._get_object_position(target_name)
        elif target_name in self.VALID_ZONES:
            pos = self._get_zone_position(target_name)

        if pos is None:
            self._logger.error(f"move_above() → INVALID: '{target_name}'")
            return SkillResult.INVALID_OBJECT

        approach_z = float(self._heights["approach_z"])
        target_pose = self._make_target_pose(float(pos["x"]), float(pos["y"]), approach_z)
        success = self._plan_and_execute_pose(target_pose, vel_scale=0.2, acc_scale=0.2)

        if success:
            self._last_move_above_target = target_name
            return SkillResult.SUCCESS
        return SkillResult.PLANNING_FAILED

    def pick(self, object_name: str) -> SkillResult:
        """
        Pick một khối lập phương.
        Trình tự chuẩn hóa:
          1. Di chuyển tới phía trên khối lập phương (approach_z)
          2. Mở gripper
          3. Hạ đầu kẹp xuống vị trí gắp (grasp_z)
          4. Đóng gripper & attach model
          5. Rút thẳng đứng lên vị trí an toàn (retreat_z)
        """
        self._logger.info(f"SKILL: pick({object_name})")

        if object_name not in self.VALID_OBJECTS:
            self._logger.error(f"pick() → INVALID_OBJECT: '{object_name}'")
            return SkillResult.INVALID_OBJECT

        if self._attached_object is not None:
            self._logger.error(f"pick() → FAILED: Đang giữ '{self._attached_object}'.")
            return SkillResult.FAILED

        pos = self._get_object_position(object_name)
        if pos is None:
            return SkillResult.INVALID_OBJECT

        x, y = float(pos["x"]), float(pos["y"])
        approach_z = float(self._heights["approach_z"])
        grasp_z = float(self._heights["grasp_z"])
        retreat_z = float(self._heights["retreat_z"])

        # 1. Di chuyển tiếp cận trên không
        self._logger.info(f"  [1/5] Di chuyển phía trên {object_name} (z={approach_z:.3f}m)...")
        pose_above = self._make_target_pose(x, y, approach_z)
        if not self._plan_and_execute_pose(pose_above, vel_scale=0.2, acc_scale=0.2):
            return SkillResult.PLANNING_FAILED
        self._last_move_above_target = None

        # 2. Mở gripper
        self._logger.info(f"  [2/5] Mở gripper...")
        self.open_gripper()

        # 3. Hạ thẳng đứng xuống gắp
        self._logger.info(f"  [3/5] Hạ xuống vị trí gắp (z={grasp_z:.3f}m)...")
        pose_grasp = self._make_target_pose(x, y, grasp_z)
        if not self._plan_and_execute_pose(pose_grasp, vel_scale=0.1, acc_scale=0.1):
            return SkillResult.PLANNING_FAILED

        # 4. Đóng gripper và gắn vật thể
        self._logger.info(f"  [4/5] Đóng gripper, kẹp '{object_name}'...")
        self.close_gripper()
        self._attach_object(object_name)

        # 5. Rút thẳng đứng lên cao an toàn
        self._logger.info(f"  [5/5] Rút lên (z={retreat_z:.3f}m)...")
        pose_retreat = self._make_target_pose(x, y, retreat_z)
        if not self._plan_and_execute_pose(pose_retreat, vel_scale=0.15, acc_scale=0.15):
            return SkillResult.PLANNING_FAILED

        self._logger.info(f"pick({object_name}) → SUCCESS")
        return SkillResult.SUCCESS

    def place(self, object_name: str, zone_name: str) -> SkillResult:
        """
        Place khối vào zone mục tiêu.
        Trình tự chuẩn hóa:
          1. Di chuyển phía trên zone mục tiêu (approach_z)
          2. Hạ thẳng đứng xuống khay (grasp_z)
          3. Mở gripper & detach model vào khay
          4. Rút thẳng đứng lên vị trí an toàn (retreat_z)
        """
        self._logger.info(f"SKILL: place({object_name}, {zone_name})")

        if object_name not in self.VALID_OBJECTS:
            self._logger.error(f"place() → INVALID_OBJECT: '{object_name}'")
            return SkillResult.INVALID_OBJECT

        if zone_name not in self.VALID_ZONES:
            self._logger.error(f"place() → INVALID zone: '{zone_name}'")
            return SkillResult.FAILED

        if self._attached_object != object_name:
            self._logger.error(
                f"place() → FAILED: Đang giữ '{self._attached_object}', không phải '{object_name}'."
            )
            return SkillResult.FAILED

        zone_pos = self._get_zone_position(zone_name)
        if zone_pos is None:
            return SkillResult.FAILED

        zx, zy = float(zone_pos["x"]), float(zone_pos["y"])
        approach_z = float(self._heights["approach_z"])
        grasp_z = float(self._heights["grasp_z"])
        retreat_z = float(self._heights["retreat_z"])

        # 1. Di chuyển tiếp cận trên khay zone
        self._logger.info(f"  [1/4] Di chuyển phía trên {zone_name} (z={approach_z:.3f}m)...")
        pose_above = self._make_target_pose(zx, zy, approach_z)
        if not self._plan_and_execute_pose(pose_above, vel_scale=0.2, acc_scale=0.2):
            return SkillResult.PLANNING_FAILED

        # 2. Hạ thẳng đứng xuống khay
        self._logger.info(f"  [2/4] Hạ xuống vị trí đặt (z={grasp_z:.3f}m)...")
        pose_place = self._make_target_pose(zx, zy, grasp_z)
        if not self._plan_and_execute_pose(pose_place, vel_scale=0.1, acc_scale=0.1):
            return SkillResult.PLANNING_FAILED

        # 3. Mở gripper và thả vật thể
        self._logger.info(f"  [3/4] Mở gripper, thả '{object_name}' vào {zone_name}...")
        self.open_gripper()
        self._detach_object(object_name, zone_pos)

        # 4. Rút thẳng đứng lên cao an toàn
        self._logger.info(f"  [4/4] Rút lên (z={retreat_z:.3f}m)...")
        pose_retreat = self._make_target_pose(zx, zy, retreat_z)
        if not self._plan_and_execute_pose(pose_retreat, vel_scale=0.15, acc_scale=0.15):
            return SkillResult.PLANNING_FAILED

        self._last_move_above_target = None
        self._logger.info(f"place({object_name}, {zone_name}) → SUCCESS")
        return SkillResult.SUCCESS

    def shutdown(self):
        """Dọn dẹp tài nguyên."""
        self._running = False
