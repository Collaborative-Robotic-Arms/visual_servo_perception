cmake_minimum_required(VERSION 3.8)
project(perception_setup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find only the packages needed to run a Python node
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED) 
find_package(std_msgs REQUIRED)

# Install the Python script as an executable
install(PROGRAMS
  perception_setup/camera_driver.py
  perception_setup/feature_detection.py
  perception_setup/depth_estimation.py
  perception_setup/feature_detection_minimal.py

  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()