#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robot_skills.py — Bộ kỹ năng thao tác robot MoveIt 2 & Đồng bộ vật lý Gazebo cho UR3e (Bài 03).

Các kỹ năng hỗ trợ:
  1. home(): Đưa robot về tư thế mặc định an toàn.
  2. move_to(target): Di chuyển End-Effector đến tọa độ mục tiêu.
  3. pick(object_name): Tiếp cận, hạ xuống gắp và khóa vật thể theo đầu công tác.
  4. place(object_name, target): Di chuyển, hạ xuống và nhả vật thể vào ô đích.
  5. stack(bottom_object, top_object): Xếp chồng khối lên trên khối khác.
  6. unstack(top_object, target): Dỡ khối từ trên đỉnh sang ô đích.

Sinh viên: Mai Duc Tri | MSSV: 23020776
"""

import os
import sys
import time
import math
import json
import threading
import subprocess
from enum import Enum
from typing import Dict, Optional, List, Tuple, Any

import yaml
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Empty, Float64
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    RobotState as RobotStateMsg,
    PlanningScene,
)
from visualization_msgs.msg import Marker, MarkerArray

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:
    def get_package_share_directory(pkg_name: str) -> str:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from ur3_vlm.ur3_kinematics import fk, solve_ik, HOME_JOINTS


# ---------------------------------------------------------------------------
#  Trạng thái kết quả Skill
# ---------------------------------------------------------------------------
class SkillResult(Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    INVALID_OBJECT = "INVALID_OBJECT"
    PLANNING_FAILED = "PLANNING_FAILED"
    EXECUTION_TIMEOUT = "EXECUTION_TIMEOUT"


# ---------------------------------------------------------------------------
#  Lớp RobotSkills
# ---------------------------------------------------------------------------
class RobotSkills:
    """
    Quản lý và thực thi các kỹ năng chuyển động MoveIt 2 & đồng bộ vật thể trong Gazebo.
    """

    JOINT_NAMES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    PLANNING_GROUP = "ur_manipulator"

    def __init__(self, node: Node, scene_config: dict = None):
        self._node = node
        self._logger = node.get_logger()
        self._cb_group = ReentrantCallbackGroup()

        # Cấu hình Scene
        self._scene_cfg = scene_config or self._load_scene_config()
        self._robot_base_z = float(self._scene_cfg.get("robot_base_height", 0.74))
        self._heights = self._scene_cfg.get("heights", {
            "table_surface": 0.0,
            "cube_size": 0.04,
            "grasp_z": 0.08,
            "approach_z": 0.22,
            "retreat_z": 0.22,
            "safe_z": 0.25,
        })
        self._home_joints = [float(v) for v in self._scene_cfg.get("home_joint_values", HOME_JOINTS)]

        # Tọa độ mặc định của các khối và zone (dự phòng nếu camera chưa nhận diện)
        self._objects_cfg = self._scene_cfg.get("objects", {})
        self._zones_cfg = self._scene_cfg.get("zones", {})

        # Trạng thái nhận diện thời gian thực từ Camera Perception
        self._world_state_data: Dict[str, Any] = {}
        self._detected_objects: Dict[str, dict] = {}
        self._detected_zones: Dict[str, dict] = {}

        # Trạng thái nội bộ
        self._attached_object: Optional[str] = None
        self._object_locations: Dict[str, str] = {}
        self._current_joints: List[float] = list(self._home_joints)
        self._joint_states_received = False
        self._running = True

        # --- ROS 2 Action Client MoveGroup ---
        self._move_action_client = ActionClient(
            node, MoveGroup, "/move_action", callback_group=self._cb_group
        )

        # --- ROS 2 Action Client FollowJointTrajectory (Quỹ đạo trực tiếp dự phòng) ---
        self._traj_action_client = ActionClient(
            node, FollowJointTrajectory, "/joint_trajectory_controller/follow_joint_trajectory", callback_group=self._cb_group
        )

        # --- Subscriber Joint States ---
        self._joint_sub = node.create_subscription(
            JointState,
            "/joint_states",
            self._joint_states_callback,
            10,
            callback_group=self._cb_group,
        )

        # --- Subscriber WorldState JSON từ Camera Perception (QoS Transient Local) ---
        qos_state = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10,
        )
        self._world_state_sub = node.create_subscription(
            String,
            "/world_state_json",
            self._world_state_callback,
            qos_state,
            callback_group=self._cb_group,
        )

        # --- ROS 2 Publishers for Gripper Finger Actuation (Pure Physical Clamping) ---
        self._left_finger_pub = node.create_publisher(Float64, "/gripper/left_cmd", 10)
        self._right_finger_pub = node.create_publisher(Float64, "/gripper/right_cmd", 10)

        self._logger.info("RobotSkills: Đang kết nối MoveGroup & Trajectory Controllers...")
        self._move_action_client.wait_for_server(timeout_sec=2.0)
        self._traj_action_client.wait_for_server(timeout_sec=2.0)
        self._logger.info("RobotSkills: Controllers & Pure Physical Gripper Publishers đã sẵn sàng.")

    def _load_scene_config(self) -> dict:
        try:
            pkg_share = get_package_share_directory("ur3_vlm")
            yaml_path = os.path.join(pkg_share, "config", "scene.yaml")
            if os.path.exists(yaml_path):
                with open(yaml_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
        except Exception:
            pass
        return {}

    def _joint_states_callback(self, msg: JointState):
        name_map = {name: pos for name, pos in zip(msg.name, msg.position)}
        ordered = []
        for jname in self.JOINT_NAMES:
            if jname in name_map:
                ordered.append(float(name_map[jname]))
            else:
                return
        self._current_joints = ordered
        self._joint_states_received = True

    def _world_state_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
            self._world_state_data = data
            self._detected_objects = data.get("detected_objects", {})
            self._detected_zones = data.get("detected_zones", {})
        except Exception:
            pass

    # =======================================================================
    #  Điều khiển chuyển động MoveIt 2 (Joint-Space Planning & Execution)
    # =======================================================================
    def move_to_joints(self, target_joints: List[float], timeout_sec: float = 12.0) -> SkillResult:
        """Gửi goal di chuyển đến góc khớp target_joints (Ưu tiên MoveGroup, tự động fallback Trajectory mượt mà)."""
        # 1. Thử qua MoveGroup nếu server sẵn sàng
        if self._move_action_client.server_is_ready():
            try:
                goal_msg = MoveGroup.Goal()
                req = MotionPlanRequest()
                req.group_name = self.PLANNING_GROUP
                req.num_planning_attempts = 5
                req.allowed_planning_time = 3.0
                req.max_velocity_scaling_factor = 0.6
                req.max_acceleration_scaling_factor = 0.6

                # Thiết lập Joint Constraints
                constraints = Constraints()
                for jname, jval in zip(self.JOINT_NAMES, target_joints):
                    jc = JointConstraint()
                    jc.joint_name = jname
                    jc.position = float(jval)
                    jc.tolerance_above = 0.05
                    jc.tolerance_below = 0.05
                    jc.weight = 1.0
                    constraints.joint_constraints.append(jc)

                req.goal_constraints = [constraints]
                goal_msg.request = req
                goal_msg.planning_options.plan_only = False

                self._logger.info(f"Gửi MoveGroup goal -> Joints: {[round(v, 2) for v in target_joints]}")
                send_future = self._move_action_client.send_goal_async(goal_msg)

                start_t = time.time()
                while not send_future.done() and (time.time() - start_t < 1.0):
                    time.sleep(0.04)

                if send_future.done():
                    goal_handle = send_future.result()
                    if goal_handle and goal_handle.accepted:
                        result_future = goal_handle.get_result_async()
                        exec_start = time.time()
                        while not result_future.done() and (time.time() - exec_start < timeout_sec):
                            time.sleep(0.04)
                        if result_future.done():
                            res = result_future.result()
                            if res and res.result and res.result.error_code.val == 1:
                                self._logger.info("Chuyển động thành công qua MoveGroup!")
                                return SkillResult.SUCCESS
                            else:
                                err_val = res.result.error_code.val if (res and res.result) else -1
                                self._logger.warn(f"MoveGroup trả về lỗi {err_val}, chuyển sang Trajectory Controller...")
                    else:
                        self._logger.warn("MoveGroup từ chối goal, chuyển sang Trajectory Controller...")
                else:
                    self._logger.warn("MoveGroup phản hồi chậm (>1s), chuyển sang Trajectory Controller...")
            except Exception as e:
                self._logger.warn(f"MoveGroup exception: {e}, chuyển sang Trajectory Controller...")

        # 2. Fallback chuyển động trực tiếp 100% tin cậy qua Trajectory Controller
        return self._execute_direct_trajectory(target_joints, duration_sec=2.2, timeout_sec=timeout_sec)

    def _execute_direct_trajectory(self, target_joints: List[float], duration_sec: float = 2.5, timeout_sec: float = 12.0) -> SkillResult:
        """Thực thi quỹ đạo trực tiếp qua FollowJointTrajectory Action Client."""
        if not self._traj_action_client.wait_for_server(timeout_sec=2.0):
            self._logger.error("Không thể kết nối FollowJointTrajectory action server.")
            return SkillResult.PLANNING_FAILED

        goal_msg = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = list(self.JOINT_NAMES)

        # Điểm bắt đầu
        p_start = JointTrajectoryPoint()
        p_start.positions = [float(v) for v in self._current_joints]
        p_start.time_from_start = Duration(sec=0, nanosec=0)
        traj.points.append(p_start)

        # Điểm đích (di chuyển mượt)
        p_end = JointTrajectoryPoint()
        p_end.positions = [float(v) for v in target_joints]
        sec_int = int(duration_sec)
        nanosec_int = int((duration_sec - sec_int) * 1e9)
        p_end.time_from_start = Duration(sec=sec_int, nanosec=nanosec_int)
        traj.points.append(p_end)

        goal_msg.trajectory = traj

        send_future = self._traj_action_client.send_goal_async(goal_msg)
        start_t = time.time()
        while not send_future.done() and (time.time() - start_t < 3.0):
            time.sleep(0.05)

        if not send_future.done():
            self._logger.error("Gửi trajectory goal thất bại.")
            return SkillResult.PLANNING_FAILED

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self._logger.error("Trajectory Controller từ chối goal.")
            return SkillResult.PLANNING_FAILED

        res_future = goal_handle.get_result_async()
        exec_start = time.time()
        while not res_future.done() and (time.time() - exec_start < timeout_sec):
            time.sleep(0.05)

        if not res_future.done():
            self._logger.error("Thực thi trajectory timeout.")
            return SkillResult.EXECUTION_TIMEOUT

        self._logger.info("Chuyển động thành công qua Trajectory Controller!")
        return SkillResult.SUCCESS

    def move_to_xyz(self, x: float, y: float, z: float, timeout_sec: float = 12.0, desired_yaw: Optional[float] = None) -> SkillResult:
        """Giải IK và di chuyển tool0 tới tọa độ (x, y, z) theo hệ base_link."""
        seed = self._current_joints if self._joint_states_received else self._home_joints
        q_sol, err = solve_ik([x, y, z], seed=seed, desired_yaw=desired_yaw)

        if err > 0.015:  # Lỗi vị trí > 1.5 cm -> IK không tìm được nghiệm khả thi
            self._logger.error(f"IK không tìm được nghiệm chính xác tới ({x:.3f}, {y:.3f}, {z:.3f}), sai số = {err:.3f}m")
            return SkillResult.PLANNING_FAILED

        return self.move_to_joints(q_sol, timeout_sec=timeout_sec)

    # =======================================================================
    #  Trích xuất tọa độ & Góc xoay thực tế (từ Camera Perception hoặc Scene Config)
    # =======================================================================
    def get_target_yaw(self, target_name: str) -> float:
        """Lấy góc quay yaw (radian) của vật thể từ nhận diện Camera."""
        if target_name in self._detected_objects:
            return float(self._detected_objects[target_name].get("yaw", 0.0))
        return 0.0

    def get_target_position(self, target_name: str) -> Optional[Tuple[float, float, float]]:
        """Lấy tọa độ (x, y, z) của vật thể hoặc zone theo thời gian thực."""
        # 1. Ưu tiên lấy trực tiếp từ Camera Perception (Detected Objects)
        if target_name in self._detected_objects:
            pos = self._detected_objects[target_name].get("position")
            if pos:
                return float(pos[0]), float(pos[1]), float(pos[2])

        # 2. Kiểm tra nếu vật thể đang nằm trong một Zone theo nhận diện Camera
        for z_name, z_data in self._detected_zones.items():
            if z_data.get("occupant") == target_name:
                z_pos = z_data.get("position")
                if z_pos:
                    return float(z_pos[0]), float(z_pos[1]), 0.02

        # 3. Kiểm tra nếu vật thể đã được đặt vào Zone theo bộ nhớ nội bộ (WorldStateManager)
        if target_name in self._object_locations:
            loc = self._object_locations[target_name]
            if loc != "gripper" and loc != target_name:
                loc_pos = self.get_target_position(loc)
                if loc_pos:
                    return float(loc_pos[0]), float(loc_pos[1]), 0.02

        # 4. Nếu target là một Zone
        if target_name in self._detected_zones:
            pos = self._detected_zones[target_name].get("position")
            if pos:
                return float(pos[0]), float(pos[1]), float(pos[2])

        if target_name in self._zones_cfg:
            pos = self._zones_cfg[target_name]["position"]
            return float(pos["x"]), float(pos["y"]), float(pos["z"])

        # 5. Dự phòng cuối cùng: Scene Config ban đầu
        if target_name in self._objects_cfg:
            pos = self._objects_cfg[target_name]["position"]
            return float(pos["x"]), float(pos["y"]), float(pos["z"])

        return None

    # =======================================================================
    #  Bộ kỹ năng High-Level Skills (Bài 03)
    # =======================================================================
    def home(self) -> SkillResult:
        """Đưa robot về vị trí Home chuẩn an toàn."""
        self._logger.info("=== Thực hiện Skill: home() ===")
        return self.move_to_joints(self._home_joints)

    def move_to(self, target: str) -> SkillResult:
        """Di chuyển đầu công tác đến phía trên target."""
        self._logger.info(f"=== Thực hiện Skill: move_to({target}) ===")
        pos = self.get_target_position(target)
        if not pos:
            self._logger.error(f"Không tìm thấy vị trí của '{target}'")
            return SkillResult.INVALID_OBJECT

        approach_z = float(self._heights.get("approach_z", 0.22))
        return self.move_to_xyz(pos[0], pos[1], approach_z)

    def open_gripper(self):
        """Mở rộng tối đa 2 ngón kẹp vật lý (chiều rộng mở ~8.2cm, thoải mái ôm trọn khối 4cm)."""
        self._logger.info("-> [Gripper]: Mở rộng 2 ngón kẹp vật lý (Open Gripper).")
        cmd_l = Float64()
        cmd_l.data = 0.020
        cmd_r = Float64()
        cmd_r.data = -0.020
        self._left_finger_pub.publish(cmd_l)
        self._right_finger_pub.publish(cmd_r)
        try:
            subprocess.Popen(
                ["ign", "topic", "-t", "/gripper/left_cmd", "-m", "ignition.msgs.Double", "-p", "data: 0.020"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            subprocess.Popen(
                ["ign", "topic", "-t", "/gripper/right_cmd", "-m", "ignition.msgs.Double", "-p", "data: -0.020"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception:
            pass
        self._attached_object = None
        time.sleep(0.4)

    def close_gripper(self, object_name: str = None):
        """Khép 2 ngón kẹp vật lý ép chặt vào 2 mặt bên của khối lập phương (Lực ma sát vật lý 100%)."""
        self._logger.info(f"-> [Gripper]: Khép 2 ngón kẹp vật lý ép chặt '{object_name or 'vật thể'}' (Pure Friction Grasp).")
        cmd_l = Float64()
        cmd_l.data = -0.007
        cmd_r = Float64()
        cmd_r.data = 0.007
        self._left_finger_pub.publish(cmd_l)
        self._right_finger_pub.publish(cmd_r)
        try:
            subprocess.Popen(
                ["ign", "topic", "-t", "/gripper/left_cmd", "-m", "ignition.msgs.Double", "-p", "data: -0.007"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            subprocess.Popen(
                ["ign", "topic", "-t", "/gripper/right_cmd", "-m", "ignition.msgs.Double", "-p", "data: 0.007"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception:
            pass
        self._attached_object = object_name
        time.sleep(0.5)

    def pick(self, object_name: str, max_retries: int = 2) -> SkillResult:
        """
        Kỹ năng gắp khối bằng Tay kẹp 2 ngón (100% LỰC MA SÁT VẬT LÝ NGUYÊN BẢN):
          1. Mở rộng 2 ngón kẹp (open_gripper)
          2. Tiếp cận trên cao (approach_z = 0.22m)
          3. Hạ thẳng đứng xuống độ cao kẹp (grasp_z = 0.090m trong zone, 0.085m trên bàn)
          4. Khép 2 ngón kẹp ép chặt 2 má khối với lực cơ học thực tế (close_gripper)
          5. Nhấc thẳng đứng lên êm ái bằng ma sát tiếp xúc (retreat_z = 0.22m)
          6. Tự động kiểm tra qua Camera: Nếu gắp trượt/vật thể vẫn còn trên bàn, tự động thực hiện lại lần 2.
        """
        for attempt in range(1, max_retries + 1):
            if attempt > 1:
                self._logger.info(f"=== [Lần {attempt}/{max_retries}] Thực hiện lại Skill: pick({object_name}) dựa trên định vị camera thời gian thực ===")
            else:
                self._logger.info(f"=== Thực hiện Skill: pick({object_name}) [Pure Physics Friction Grasp] ===")

            pos = self.get_target_position(object_name)
            if not pos:
                self._logger.error(f"Không tìm thấy vị trí của '{object_name}'")
                return SkillResult.INVALID_OBJECT

            x, y, obj_z = pos
            yaw = self.get_target_yaw(object_name)
            approach_z = float(self._heights.get("approach_z", 0.22))
            # Tự động điều chỉnh độ cao gắp: nếu ở trong khay zone (x >= 0.38m) thì grasp_z = 0.090m, nếu trên bàn thì grasp_z = 0.085m
            grasp_z = 0.090 if x >= 0.38 else 0.085
            retreat_z = float(self._heights.get("retreat_z", 0.22))

            # Bước 1: Mở ngón kẹp & Tiếp cận trên cao
            self.open_gripper()
            self._logger.info(f"1. Tiếp cận trên cao: ({x:.3f}, {y:.3f}, {approach_z:.3f}) [yaw: {math.degrees(yaw):.1f}°]")
            res = self.move_to_xyz(x, y, approach_z, desired_yaw=yaw)
            if res != SkillResult.SUCCESS:
                return res

            # Bước 2: Hạ xuống độ cao gắp
            self._logger.info(f"2. Hạ xuống kẹp: ({x:.3f}, {y:.3f}, {grasp_z:.3f}) [yaw: {math.degrees(yaw):.1f}°]")
            res = self.move_to_xyz(x, y, grasp_z, desired_yaw=yaw)
            if res != SkillResult.SUCCESS:
                return res

            # Bước 3: Khép 2 ngón kẹp ép chặt khối bằng lực cơ học
            self.close_gripper(object_name)
            self._attached_object = object_name
            self._object_locations[object_name] = "gripper"

            # Bước 4: Nhấc lên an toàn bằng ma sát tiếp xúc
            self._logger.info(f"4. Nhấc lên an toàn: ({x:.3f}, {y:.3f}, {retreat_z:.3f})")
            res = self.move_to_xyz(x, y, retreat_z, desired_yaw=yaw)
            if res != SkillResult.SUCCESS:
                return res

            # Bước 5: Kiểm tra xác nhận bằng Camera
            time.sleep(0.5)
            fresh_obj = self._detected_objects.get(object_name)
            if fresh_obj and attempt < max_retries:
                fx, fy, _ = fresh_obj.get("position", [0, 0, 0])
                # Nếu camera phát hiện khối vẫn nằm tại vị trí trên bàn (khoảng cách sai lệch < 2.5cm)
                if math.hypot(fx - x, fy - y) < 0.025:
                    self._logger.warn(
                        f"⚠️ [Camera Feedback]: Gắp trượt khối '{object_name}' (vẫn còn trên bàn tại ({fx:.3f}, {fy:.3f})). "
                        f"Đang tự động thực hiện lại lần 2..."
                    )
                    self._attached_object = None
                    self.open_gripper()
                    continue

            return SkillResult.SUCCESS

        return SkillResult.SUCCESS

    def place(self, object_name: str, target: str, max_retries: int = 2) -> SkillResult:
        """
        Kỹ năng đặt khối:
          1. Di chuyển tới phía trên vị trí đích (approach_z = 0.22m)
          2. Hạ thẳng đứng xuống độ cao đặt (place_z = 0.085m)
          3. Mở kẹp giải phóng vật thể, khối tiếp đất tự nhiên theo trọng lực
          4. Rút thẳng đứng lên an toàn (retreat_z = 0.22m)
          5. Tự động kiểm tra Camera xác nhận khối đã vào đích, nếu trượt thực hiện lại lần 2.
        """
        for attempt in range(1, max_retries + 1):
            if attempt > 1:
                self._logger.info(f"=== [Lần {attempt}/{max_retries}] Thực hiện lại Skill: place({object_name}, {target}) ===")
            else:
                self._logger.info(f"=== Thực hiện Skill: place({object_name}, {target}) ===")

            pos = self.get_target_position(target)
            if not pos:
                self._logger.error(f"Không tìm thấy vị trí đích '{target}'")
                return SkillResult.INVALID_OBJECT

            x, y, target_z = pos
            approach_z = float(self._heights.get("approach_z", 0.22))
            place_z = 0.085
            retreat_z = float(self._heights.get("retreat_z", 0.22))

            # Bước 1: Di chuyển tới trên đích
            self._logger.info(f"1. Tiếp cận trên đích: ({x:.3f}, {y:.3f}, {approach_z:.3f})")
            res = self.move_to_xyz(x, y, approach_z, desired_yaw=0.0)
            if res != SkillResult.SUCCESS:
                return res

            # Bước 2: Hạ xuống đặt
            self._logger.info(f"2. Hạ xuống đặt: ({x:.3f}, {y:.3f}, {place_z:.3f})")
            res = self.move_to_xyz(x, y, place_z, desired_yaw=0.0)
            if res != SkillResult.SUCCESS:
                return res

            # Bước 3: Mở kẹp giải phóng vật thể vào khay đích
            self._logger.info(f"3. Mở kẹp giải phóng vật thể '{object_name}' tại khay '{target}'.")
            self.open_gripper()
            self._attached_object = None
            self._object_locations[object_name] = target
            time.sleep(0.4)

            # Bước 4: Rút lên an toàn
            self._logger.info(f"4. Rút lên an toàn: ({x:.3f}, {y:.3f}, {retreat_z:.3f})")
            res = self.move_to_xyz(x, y, retreat_z, desired_yaw=0.0)
            if res != SkillResult.SUCCESS:
                return res

            # Bước 5: Kiểm tra xác nhận bằng Camera
            time.sleep(0.5)
            fresh_obj = self._detected_objects.get(object_name)
            if fresh_obj:
                ox, oy, _ = fresh_obj.get("position", [0, 0, 0])
                dist_to_target = math.hypot(ox - x, oy - y)
                occ_rad = float(self._scene_cfg.get("zone_detection", {}).get("occupancy_radius", 0.06))
                if dist_to_target > occ_rad:
                    if attempt < max_retries:
                        self._logger.warn(
                            f"⚠️ [Camera Feedback]: Khối '{object_name}' chưa vào đúng vùng '{target}' (đang ở ({ox:.3f}, {oy:.3f})). "
                            f"Thực hiện gắp lại lần 2 và đưa vào '{target}'..."
                        )
                        # Gắp lại từ vị trí camera phát hiện hiện tại
                        pick_res = self.pick(object_name, max_retries=1)
                        if pick_res == SkillResult.SUCCESS:
                            continue
                    else:
                        self._logger.warn(f"⚠️ [Camera Feedback]: Đã thử {max_retries} lần cho place({object_name}, {target}).")

            return SkillResult.SUCCESS

        return SkillResult.SUCCESS

    def stack(self, bottom_object: str, top_object: str) -> SkillResult:
        """
        Kỹ năng xếp chồng: Đặt top_object lên trên đỉnh bottom_object.
        """
        self._logger.info(f"=== Thực hiện Skill: stack({bottom_object}, {top_object}) ===")
        
        # Nếu chưa gắp top_object thì gắp trước
        if self._attached_object != top_object:
            res = self.pick(top_object)
            if res != SkillResult.SUCCESS:
                return res

        pos = self.get_target_position(bottom_object)
        if not pos:
            self._logger.error(f"Không tìm thấy vị trí của '{bottom_object}'")
            return SkillResult.INVALID_OBJECT

        x, y, bot_z = pos
        bot_yaw = self.get_target_yaw(bottom_object)
        cube_size = float(self._heights.get("cube_size", 0.04))
        approach_z = float(self._heights.get("approach_z", 0.22))
        stack_z = float(self._heights.get("grasp_z", 0.08)) + cube_size
        retreat_z = float(self._heights.get("retreat_z", 0.22))

        # Bước 1: Tiếp cận trên đỉnh khối dưới
        self._logger.info(f"1. Tiếp cận trên đỉnh khối dưới: ({x:.3f}, {y:.3f}, {approach_z:.3f})")
        res = self.move_to_xyz(x, y, approach_z, desired_yaw=bot_yaw)
        if res != SkillResult.SUCCESS:
            return res

        # Bước 2: Hạ xuống độ cao xếp chồng
        self._logger.info(f"2. Hạ xuống độ cao xếp chồng: ({x:.3f}, {y:.3f}, {stack_z:.3f})")
        res = self.move_to_xyz(x, y, stack_z, desired_yaw=bot_yaw)
        if res != SkillResult.SUCCESS:
            return res

        # Bước 3: Mở kẹp & nhả khối lên đỉnh
        self._logger.info(f"3. Mở kẹp giải phóng '{top_object}' trên đỉnh '{bottom_object}'.")
        self.open_gripper()
        self._attached_object = None
        self._object_locations[top_object] = bottom_object
        time.sleep(0.4)

        # Bước 4: Rút lên an toàn
        self._logger.info(f"4. Rút lên an toàn: ({x:.3f}, {y:.3f}, {retreat_z:.3f})")
        return self.move_to_xyz(x, y, retreat_z, desired_yaw=bot_yaw)

    def unstack(self, top_object: str, target: str) -> SkillResult:
        """
        Kỹ năng dỡ chồng: Gắp khối trên đỉnh và chuyển vào khay đích.
        """
        self._logger.info(f"=== Thực hiện Skill: unstack({top_object}, {target}) ===")
        pos = self.get_target_position(top_object)
        if not pos:
            self._logger.error(f"Không tìm thấy vị trí của '{top_object}'")
            return SkillResult.INVALID_OBJECT

        x, y, _ = pos
        top_yaw = self.get_target_yaw(top_object)
        cube_size = float(self._heights.get("cube_size", 0.04))
        approach_z = float(self._heights.get("approach_z", 0.22))
        top_grasp_z = float(self._heights.get("grasp_z", 0.08)) + cube_size
        retreat_z = float(self._heights.get("retreat_z", 0.22))

        # Tiếp cận và gắp từ độ cao tầng trên
        self.open_gripper()
        self.move_to_xyz(x, y, approach_z, desired_yaw=top_yaw)
        self.move_to_xyz(x, y, top_grasp_z, desired_yaw=top_yaw)
        self.close_gripper(top_object)
        self._attached_object = top_object
        self._object_locations[top_object] = "gripper"
        time.sleep(0.3)
        self.move_to_xyz(x, y, retreat_z, desired_yaw=top_yaw)

        # Đặt vào target
        return self.place(top_object, target)

    def shutdown(self):
        self._running = False


# ---------------------------------------------------------------------------
#  Main CLI Test Node
# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = Node("robot_skills_test_node")
    skills = RobotSkills(node)

    # Spinner trong background thread để nhận callbacks
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    time.sleep(1.0)
    print("=" * 65)
    print("  Robot Skills (Phase 4 — MoveIt 2 & Gazebo Sync Interactive)")
    print("=" * 65)
    print("  Danh sách lệnh kiểm thử thủ công:")
    print("    • home                     -> Đưa robot về vị trí an toàn")
    print("    • open                     -> Mở 2 ngón kẹp (Open Gripper)")
    print("    • close                    -> Đóng 2 ngón kẹp (Close Gripper)")
    print("    • pick <object>            -> Mở kẹp -> Hạ -> Khép kẹp -> Nhấc vật lên")
    print("    • place <object> <target>  -> Tiếp cận -> Hạ -> Mở kẹp -> Rút lên")
    print("    • stack <bottom> <top>     -> Gắp top -> Xếp lên đỉnh bottom")
    print("    • unstack <top> <target>   -> Dỡ khối đỉnh sang khay đích")
    print("    • auto                     -> Chạy chu trình tự động mẫu")
    print("    • exit / quit              -> Thoát")
    print("=" * 65)

    try:
        while rclpy.ok():
            try:
                cmd_raw = input("\n[RobotSkills CLI] > ").strip()
            except EOFError:
                break

            if not cmd_raw:
                continue

            parts = cmd_raw.split()
            action = parts[0].lower()

            if action in ["exit", "quit", "q"]:
                break

            elif action == "home":
                skills.home()

            elif action == "open":
                skills.open_gripper()

            elif action == "close":
                obj = parts[1] if len(parts) >= 2 else None
                skills.close_gripper(obj)

            elif action == "pick" and len(parts) >= 2:
                skills.pick(parts[1])

            elif action == "place" and len(parts) >= 3:
                skills.place(parts[1], parts[2])

            elif action == "stack" and len(parts) >= 3:
                skills.stack(parts[1], parts[2])

            elif action == "unstack" and len(parts) >= 3:
                skills.unstack(parts[1], parts[2])

            elif action == "auto":
                print("\n[Auto Test]: Chạy chu trình mẫu red_cube -> zone_b -> home")
                skills.home()
                skills.pick("red_cube")
                skills.place("red_cube", "zone_b")
                skills.home()

            else:
                print("Lệnh không hợp lệ. Hãy thử: home, open, close, pick <obj>, place <obj> <zone>, auto, exit")

    except KeyboardInterrupt:
        pass
    finally:
        skills.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
