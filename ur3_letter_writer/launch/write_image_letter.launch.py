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
    image_path = LaunchConfiguration('image_path')
    letter_text = LaunchConfiguration('letter_text')
    num_cycles = LaunchConfiguration('num_cycles')

    declare_use_sim_time = DeclareLaunchArgument('use_sim_time', default_value='true')
    declare_image_path = DeclareLaunchArgument('image_path', default_value='')
    declare_letter_text = DeclareLaunchArgument('letter_text', default_value='T')
    declare_num_cycles = DeclareLaunchArgument('num_cycles', default_value='2')
    declare_threshold_method = DeclareLaunchArgument('threshold_method', default_value='otsu',
                                                     description="'otsu' (ảnh đồ họa/scan) hoặc 'adaptive' (ảnh chụp camera/chữ viết tay)")

    # Load kinematics cấu hình của UR
    robot_description_kinematics = load_yaml('ur_moveit_config', 'config/kinematics.yaml')

    params = {
        'use_sim_time': use_sim_time,
        'planning_group': 'ur_manipulator',
        'base_frame': 'base_link',
        'image_path': image_path,
        'letter_text': letter_text,
        'threshold_method': LaunchConfiguration('threshold_method'),
        # Tọa độ vùng an toàn cho UR3e (< 0.50m)
        'origin_x': 0.05,
        'origin_y': 0.28,
        'origin_z': 0.15,
        'letter_width': 0.12,
        'letter_height': 0.15,
        'pen_lift': 0.04,
        'epsilon_approx': 2.0,
        'min_contour_area': 100.0,
        'max_cartesian_step': 0.01,
        'vel_scale': 0.15,
        'acc_scale': 0.15,
        'num_cycles': num_cycles,
    }

    image_letter_writer = Node(
        package='ur3_letter_writer',
        executable='image_letter_writer_node',
        name='image_letter_writer_node',
        output='screen',
        parameters=[params, robot_description_kinematics],
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_image_path,
        declare_letter_text,
        declare_num_cycles,
        declare_threshold_method,
        image_letter_writer,
    ])
