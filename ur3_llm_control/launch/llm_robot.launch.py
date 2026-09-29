#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_robot.launch.py — Launch file cho hệ thống điều khiển UR3e bằng LLM.

Chức năng:
  1. Include launch UR3e simulation + MoveIt 2 (ur_sim_moveit.launch.py).
  2. Spawn scene objects (khối màu + khay zone) vào Gazebo.
  3. Khởi chạy node skill_executor (interactive console).

Cách chạy:
  Cách 1 — Launch tất cả:
    ros2 launch ur3_llm_control llm_robot.launch.py

  Cách 2 — Bật sim riêng, rồi chạy scene + executor:
    Terminal 1: ros2 launch ur_simulation_gz ur_sim_moveit.launch.py ur_type:=ur3e
    Terminal 2: ros2 run ur3_llm_control scene_spawner
    Terminal 3: ros2 run ur3_llm_control skill_executor

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
    ur_sim_moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("ur_simulation_gz"),
                "launch",
                "ur_sim_moveit.launch.py",
            ])
        ),
        launch_arguments={"ur_type": ur_type}.items(),
    )

    # --- Load kinematics config ---
    robot_description_kinematics = load_yaml(
        "ur_moveit_config", "config/kinematics.yaml"
    )

    # --- Node scene_spawner: spawn objects vào Gazebo ---
    # Delay 8s để Gazebo + robot ổn định trước khi spawn
    scene_spawner_node = TimerAction(
        period=8.0,
        actions=[
            Node(
                package="ur3_llm_control",
                executable="scene_spawner",
                name="scene_spawner_node",
                output="screen",
                parameters=[{"use_sim_time": True}],
            )
        ],
    )

    # --- Node skill_executor: interactive console ---
    # Delay 15s để scene spawner xong, MoveIt sẵn sàng
    skill_executor_node = TimerAction(
        period=15.0,
        actions=[
            Node(
                package="ur3_llm_control",
                executable="skill_executor",
                name="skill_executor_node",
                output="screen",
                prefix="xterm -e",  # Terminal riêng cho interactive input
                parameters=[
                    {"use_sim_time": True},
                    {"planning_group": "ur_manipulator"},
                    robot_description_kinematics,
                ],
            )
        ],
    )

    return LaunchDescription([
        declare_ur_type,
        ur_sim_moveit,
        scene_spawner_node,
        skill_executor_node,
    ])
