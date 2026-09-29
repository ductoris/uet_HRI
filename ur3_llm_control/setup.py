from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'ur3_llm_control'

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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mai Duc Tri',
    maintainer_email='23020776@vnu.edu.vn',
    description='Dieu khien UR3/UR3e bang LLM va Skill-based Planning (ROS 2 Humble)',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'skill_executor = ur3_llm_control.skill_executor:main',
            'skill_executor_node = ur3_llm_control.skill_executor:main',
            'scene_spawner = ur3_llm_control.scene_spawner:main',
            'scene_spawner_node = ur3_llm_control.scene_spawner:main',
            'reset_world = ur3_llm_control.reset_world:main',
            'reset_world_node = ur3_llm_control.reset_world:main',
        ],
    },

)
