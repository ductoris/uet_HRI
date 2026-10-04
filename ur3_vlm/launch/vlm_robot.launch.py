#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vlm_robot.launch.py — Launch file cho hệ thống UR3e + VLM (Bài 03).

Chức năng:
  1. Include launch UR3e simulation + MoveIt 2 (ur_sim_moveit.launch.py).
  2. Spawn scene objects vào Gazebo (bàn 4 chân, đế robot, giá camera,
     5 khối động, 3 zone).
  3. Khởi chạy ros_gz_bridge cho camera topics.

Cách chạy:
  Cách 1 — Launch tất cả:
    ros2 launch ur3_vlm vlm_robot.launch.py

  Cách 2 — Bật sim riêng, rồi chạy scene:
    Terminal 1: ros2 launch ur_simulation_gz ur_sim_moveit.launch.py ur_type:=ur3e
    Terminal 2: ros2 run ur3_vlm scene_spawner

Sinh viên: Mai Duc Tri | MSSV: 23020776
"""

import os
import yaml

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def load_yaml(package_name, file_path):
    """Load a YAML file from a ROS 2 package share directory."""
    try:
        pkg_path = get_package_share_directory(package_name)
        with open(os.path.join(pkg_path, file_path), "r") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


def generate_launch_description():
    # --- Launch arguments ---
    ur_type = LaunchConfiguration("ur_type")
    declare_ur_type = DeclareLaunchArgument(
        "ur_type",
        default_value="ur3e",
        description="Loại robot UR (ur3 hoặc ur3e)",
    )

    # --- Include UR simulation + MoveIt 2 ---
    world_file = PathJoinSubstitution([
        FindPackageShare("ur3_vlm"),
        "worlds",
        "vlm_world.sdf",
    ])

    ur_sim_moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("ur_simulation_gz"),
                "launch",
                "ur_sim_moveit.launch.py",
            ])
        ),
        launch_arguments={
            "ur_type": ur_type,
            "robot_base_z": "0.74",
            "world_file": world_file,
        }.items(),
    )

    # --- Load kinematics config ---
    robot_description_kinematics = load_yaml(
        "ur_moveit_config", "config/kinematics.yaml"
    )

    # --- Static TF for Camera (giúp RViz hiển thị Camera Frustum & TF) ---
    camera_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_static_tf_publisher",
        arguments=[
            "--x", "0.40", "--y", "0.0", "--z", "0.84",
            "--roll", "0.0", "--pitch", "1.5708", "--yaw", "0.0",
            "--frame-id", "base_link",
            "--child-frame-id", "camera_link",
        ],
    )

    camera_optical_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_optical_tf_publisher",
        arguments=[
            "--x", "0.0", "--y", "0.0", "--z", "0.0",
            "--roll", "-1.5708", "--pitch", "0.0", "--yaw", "-1.5708",
            "--frame-id", "camera_link",
            "--child-frame-id", "camera_optical_frame",
        ],
    )

    # --- ros_gz_bridge: chuyển camera topics từ Gazebo sang ROS 2 ---
    # Delay 5s để Gazebo sẵn sàng
    camera_bridge_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                name="camera_bridge",
                output="screen",
                arguments=[
                    # RGB image: Ignition → ROS
                    "/camera/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image",
                    # Camera info: Ignition → ROS
                    "/camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
                    # Physical Gripper Finger Actuation (JointPositionController): ROS → Ignition
                    "/gripper/left_cmd@std_msgs/msg/Float64]ignition.msgs.Double",
                    "/gripper/right_cmd@std_msgs/msg/Float64]ignition.msgs.Double",
                ],
                parameters=[{"use_sim_time": True}],
            )
        ],
    )

    # --- Scene Spawner Node (Bàn, Khối màu, Khay Zone, Camera Rig, Gripper) ---
    # Delay 7s để Gazebo /clock sẵn sàng
    scene_spawner_node = TimerAction(
        period=7.0,
        actions=[
            Node(
                package="ur3_vlm",
                executable="scene_spawner",
                name="scene_spawner_node",
                output="screen",
            )
        ],
    )

    # --- Camera Perception Node (CV Pipeline, Pinhole Projection, Live Window) ---
    # Delay 9s để camera bridge và scene đã spawn hoàn chỉnh
    camera_perception_node = TimerAction(
        period=9.0,
        actions=[
            Node(
                package="ur3_vlm",
                executable="camera_perception",
                name="camera_perception_node",
                output="screen",
            )
        ],
    )

    # --- Gripper Joint State Publisher (chạy từ giây 0, loại bỏ hoàn toàn cảnh báo MoveGroup) ---
    gripper_state_node = Node(
        package="ur3_vlm",
        executable="gripper_state_publisher",
        name="gripper_joint_state_publisher",
        output="screen",
    )

    return LaunchDescription([
        declare_ur_type,
        ur_sim_moveit,
        gripper_state_node,
        camera_tf_node,
        camera_optical_tf_node,
        camera_bridge_node,
        scene_spawner_node,
        camera_perception_node,
    ])
