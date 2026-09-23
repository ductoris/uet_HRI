cmake_minimum_required(VERSION 3.8)
project(ur3_letter_writer)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(OpenCV REQUIRED)

# Node 1: Vẽ chữ T theo waypoints thủ công
add_executable(letter_writer_node src/letter_writer_node.cpp)
ament_target_dependencies(letter_writer_node
  rclcpp
  geometry_msgs
  moveit_msgs
  moveit_ros_planning_interface
  visualization_msgs
  tf2
  tf2_geometry_msgs
  shape_msgs
)

# Node 2: Trích xuất quỹ đạo từ ảnh chữ cái bằng OpenCV và điều khiển robot vẽ
add_executable(image_letter_writer_node src/image_letter_writer_node.cpp)
ament_target_dependencies(image_letter_writer_node
  rclcpp
  geometry_msgs
  moveit_msgs
  moveit_ros_planning_interface
  visualization_msgs
  tf2
  tf2_geometry_msgs
  shape_msgs
)
target_include_directories(image_letter_writer_node SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(image_letter_writer_node ${OpenCV_LIBRARIES})

# Node 3: Quỹ đạo hình tròn (Circle Trajectory)
add_executable(circle_trajectory_node src/circle_trajectory_node.cpp)
ament_target_dependencies(circle_trajectory_node
  rclcpp
  geometry_msgs
  moveit_msgs
  moveit_ros_planning_interface
  visualization_msgs
  tf2
  tf2_geometry_msgs
  shape_msgs
)

install(TARGETS
  letter_writer_node
  image_letter_writer_node
  circle_trajectory_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  sample_images
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
