cmake_minimum_required(VERSION 3.8)
project(perception_setup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find all required packages
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED) 
find_package(std_msgs REQUIRED)
find_package(VISP REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# --- NEW: MoveIt Dependency ---
find_package(moveit_ros_planning_interface REQUIRED)
# ------------------------------

include_directories(
  ${VISP_INCLUDE_DIRS}
)

# 1. VISP IBVS Controller
add_executable(visp_ibvs_controller src/visp_ibvs_controller.cpp)
ament_target_dependencies(visp_ibvs_controller rclcpp sensor_msgs geometry_msgs std_msgs)
target_link_libraries(visp_ibvs_controller ${VISP_LIBRARIES})

# 2. Z Query Tool
add_executable(z_query_tool src/z_value.cpp)
ament_target_dependencies(z_query_tool rclcpp tf2_ros tf2 tf2_geometry_msgs)
target_link_libraries(z_query_tool tf2_ros::tf2_ros tf2::tf2 tf2_geometry_msgs::tf2_geometry_msgs)

# 3. RPY Pose Commander (NEW)
add_executable(pose_commander src/pose_commander.cpp)
ament_target_dependencies(rpy_pose_commander 
  rclcpp 
  geometry_msgs 
  tf2 
  tf2_geometry_msgs 
  moveit_ros_planning_interface
)

# Install Python scripts
install(PROGRAMS
  perception_setup/camera_driver.py
  perception_setup/feature_detection.py
  perception_setup/depth_estimation.py
  perception_setup/feature_detection_minimal.py
  perception_setup/mock_vel.py
  perception_setup/plot_results.py
  perception_setup/data_logger.py
  perception_setup/pose_validation_logger.py
  perception_setup/plot_pose.py
  perception_setup/joint_plotter.py
  perception_setup/velocity_smoother.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install C++ Executables
install(TARGETS 
  visp_ibvs_controller
  z_query_tool 
  rpy_pose_commander
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()