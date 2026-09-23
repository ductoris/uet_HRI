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
    center_x = LaunchConfiguration('center_x')
    center_y = LaunchConfiguration('center_y')
    center_z = LaunchConfiguration('center_z')
    radius = LaunchConfiguration('radius')
    num_cycles = LaunchConfiguration('num_cycles')

    declare_use_sim_time = DeclareLaunchArgument('use_sim_time', default_value='true')
    declare_center_x = DeclareLaunchArgument('center_x', default_value='0.11',
                                             description='Tọa độ tâm X của đường tròn (m) - tránh lệch sâu sang trái')
    declare_center_y = DeclareLaunchArgument('center_y', default_value='0.35',
                                             description='Tọa độ tâm Y của đường tròn (m)')
    declare_center_z = DeclareLaunchArgument('center_z', default_value='0.15',
                                             description='Độ cao mặt phẳng vẽ Z (m)')
    declare_radius = DeclareLaunchArgument('radius', default_value='0.07',
                                           description='Bán kính đường tròn (m) - mặc định 7cm, đường kính 14cm')
    declare_start_angle = DeclareLaunchArgument('start_angle', default_value='1.5707963',
                                               description='Góc bắt đầu (rad) - mặc định PI/2 (đỉnh trên đường tròn, ngay trước mặt robot)')
    declare_num_cycles = DeclareLaunchArgument('num_cycles', default_value='2',
                                               description='Số vòng chu trình vẽ lặp lại')
    declare_orientation_mode = DeclareLaunchArgument('orientation_mode', default_value='down',
                                                description='Tool orientation handling: down (default) or custom')
    # Load kinematics cấu hình của UR
    robot_description_kinematics = load_yaml('ur_moveit_config', 'config/kinematics.yaml')

    params = {
        'use_sim_time': use_sim_time,
        'planning_group': 'ur_manipulator',
        'base_frame': 'base_link',
        'center_x': center_x,
        'center_y': center_y,
        'center_z': center_z,
        'radius': radius,
        'start_angle': LaunchConfiguration('start_angle'),
        'num_points': 72,
        'pen_lift': 0.04,
        'vel_scale': 0.15,
        'acc_scale': 0.15,
        'num_cycles': num_cycles,
        'orientation_mode': LaunchConfiguration('orientation_mode'),
    }

    circle_node = Node(
        package='ur3_letter_writer',
        executable='circle_trajectory_node',
        name='circle_trajectory_node',
        output='screen',
        parameters=[params, robot_description_kinematics],
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_center_x,
        declare_center_y,
        declare_center_z,
        declare_radius,
        declare_start_angle,
        declare_num_cycles,
        declare_orientation_mode,
        circle_node,
    ])
