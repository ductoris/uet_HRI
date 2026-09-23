"""
File 1/2: CHI bat moi truong - UR3e + Gazebo + MoveIt2 (move_group + RViz).
Khong chua node dieu khien ve chu.

Chay:
  ros2 launch ur3_letter_writer bringup_ur3e.launch.py ur_type:=ur3e

Doi cho den khi:
  - Gazebo hien robot dung yen, on dinh
  - RViz hien robot model va MotionPlanning panel san sang (khong con log loi)
roi moi mo terminal khac chay file 2 (write_letter.launch.py).
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ur_type = LaunchConfiguration('ur_type')
    declare_ur_type = DeclareLaunchArgument('ur_type', default_value='ur3e')

    # Sua lai package/ten file neu ban dung launch khac de bat sim+moveit
    ur_sim_moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('ur_simulation_gz'),
                'launch',
                'ur_sim_moveit.launch.py'
            ])
        ),
        launch_arguments={'ur_type': ur_type}.items()
    )

    return LaunchDescription([
        declare_ur_type,
        ur_sim_moveit,
    ])
