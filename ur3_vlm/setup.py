from setuptools import setup, find_packages
from glob import glob

package_name = 'ur3_vlm'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # ament index registration
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # package.xml
        ('share/' + package_name, ['package.xml']),
        # config files
        ('share/' + package_name + '/config',
            glob('config/*.yaml')),
        # launch files
        ('share/' + package_name + '/launch',
            glob('launch/*.py')),
        # world files
        ('share/' + package_name + '/worlds',
            glob('worlds/*.sdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mai Duc Tri',
    maintainer_email='23020776@vnu.edu.vn',
    description=(
        'Dieu khien UR3e bang Vision-Language Model: '
        'Camera Perception + LLM Planner + MoveIt 2 + Gripper vat ly (ROS 2 Humble)'
    ),
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'scene_spawner = ur3_vlm.scene_spawner:main',
            'camera_perception = ur3_vlm.camera_perception:main',
            'llm_planner = ur3_vlm.llm_planner:main',
            'robot_skills = ur3_vlm.robot_skills:main',
            'vlm_console = ur3_vlm.vlm_console:main',
            'gripper_state_publisher = ur3_vlm.gripper_joint_state_publisher:main',
        ],
    },
)
