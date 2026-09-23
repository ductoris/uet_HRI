import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def load_yaml(package_name, file_path):
    try:
        pkg_path = get_package_share_directory(package_name)
        with open(os.path.join(pkg_path, file_path), 'r') as f:
            return yaml.safe_load(f)
    except Exception:
        return {}

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    declare_use_sim_time = DeclareLaunchArgument('use_sim_time', default_value='true')

    # Load kinematics cấu hình của UR
    robot_description_kinematics = load_yaml('ur_moveit_config', 'config/kinematics.yaml')

    params = {
        'use_sim_time': use_sim_time,
        'planning_group': 'ur_manipulator',
        'base_frame': 'base_link',
        # Tọa độ vùng an toàn: mọi waypoint phải cách gốc robot < 0.50m (tầm với UR3e)
        # Điểm xa nhất = sqrt((0.05+0.12)^2 + (0.28+0.15)^2) ≈ 0.462m < 0.50m ✓
        'origin_x': 0.05,
        'origin_y': 0.28,
        'origin_z': 0.15,
        'letter_width': 0.12,
        'letter_height': 0.15,
        'pen_lift': 0.04,
        'vel_scale': 0.15,
        'acc_scale': 0.15,
        'num_cycles': 2,
        'execute_retries': 3,
    }

    letter_writer = Node(
        package='ur3_letter_writer',
        executable='letter_writer_node',
        name='letter_writer_node',
        output='screen',
        parameters=[params, robot_description_kinematics],
    )

    return LaunchDescription([
        declare_use_sim_time,
        letter_writer,
    ])
