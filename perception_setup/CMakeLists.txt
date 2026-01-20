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

# --- NEW: Dependencies for Brick Tracking ---
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)
# --------------------------------------------

find_package(moveit_ros_planning_interface REQUIRED)

include_directories(
  ${VISP_INCLUDE_DIRS}
)

# 1. VISP IBVS Controller
add_executable(visp_ibvs_controller src/visp_ibvs_controller.cpp)
ament_target_dependencies(visp_ibvs_controller 
  rclcpp 
  sensor_msgs 
  geometry_msgs 
  std_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)
target_link_libraries(visp_ibvs_controller ${VISP_LIBRARIES})

# 2. Z Query Tool
add_executable(z_query_tool src/z_value.cpp)
ament_target_dependencies(z_query_tool rclcpp tf2_ros tf2 tf2_geometry_msgs)
target_link_libraries(z_query_tool tf2_ros::tf2_ros tf2::tf2 tf2_geometry_msgs::tf2_geometry_msgs)

# 3. RPY Pose Commander
add_executable(pose_commander src/pose_commander.cpp)
ament_target_dependencies(pose_commander 
  rclcpp 
  geometry_msgs 
  tf2 
  tf2_geometry_msgs 
  moveit_ros_planning_interface
)
# Image Saver Tool
add_executable(image_saver src/image_saver.cpp)
ament_target_dependencies(image_saver 
  rclcpp 
  sensor_msgs 
  cv_bridge 
  image_transport 
  OpenCV
)


# 4. Brick Tracking (NEW)
add_executable(brick_tracking src/brick_tracking.cpp)
ament_target_dependencies(brick_tracking 
  rclcpp 
  sensor_msgs 
  std_msgs 
  cv_bridge 
  image_transport
)
# Link ViSP and OpenCV (Required for Moving Edges and Hough Transform)
target_link_libraries(brick_tracking ${VISP_LIBRARIES} ${OpenCV_LIBRARIES})

# 4. Brick Tracking (NEW)
add_executable(trial_controller src/controller_trial.cpp)
ament_target_dependencies(trial_controller 
  rclcpp 
  sensor_msgs 
  std_msgs 
  cv_bridge 
  image_transport
)
# Link ViSP and OpenCV (Required for Moving Edges and Hough Transform)
target_link_libraries(trial_controller ${VISP_LIBRARIES} ${OpenCV_LIBRARIES})

# Install Python scripts
install(PROGRAMS
  perception_setup/camera_driver.py
  perception_setup/feature_detection_bricks.py
  perception_setup/feature_detection_bricks_irl.py
  perception_setup/depth_estimation.py
  perception_setup/feature_detection_aruco.py
  perception_setup/mock_vel.py
  perception_setup/data_logger.py
  perception_setup/pose_validation_logger.py
  perception_setup/plot_pose.py
  perception_setup/joint_plotter.py
  perception_setup/velocity_smoother.py
  perception_setup/plot_workspace.py
  perception_setup/feature_detection_fisheye.py
  perception_setup/feature_detection_irl_fisheye.py
  perception_setup/target_generator.py
  perception_setup/bricks_homography.py
  perception_setup/grasp_shifting_node.py
  perception_setup/ground_truth.py
  perception_setup/visualizer.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install C++ Executables
install(TARGETS 
  visp_ibvs_controller
  z_query_tool 
  pose_commander
  brick_tracking  
  image_saver
  trial_controller
  DESTINATION lib/${PROJECT_NAME}
)
# Install Launch Files
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()