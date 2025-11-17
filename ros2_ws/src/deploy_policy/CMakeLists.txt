cmake_minimum_required(VERSION 3.8)
project(deploy_policy)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

install(DIRECTORY
  scripts
  launch
  policy
  DESTINATION share/${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  scripts/armdog_controller.py
  scripts/go2w_controller.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
