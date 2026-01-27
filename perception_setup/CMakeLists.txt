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

add_executable(trial_controller_irl src/controller_trial_irl.cpp)
ament_target_dependencies(trial_controller_irl
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
  image_transport
)

target_link_libraries(trial_controller_irl ${VISP_LIBRARIES} ${OpenCV_LIBRARIES})

# Install Python scripts
install(PROGRAMS
  perception_setup/camera_driver.py
  perception_setup/depth_estimation.py
  perception_setup/feature_detection_fisheye.py
  perception_setup/feature_detection_fisheye_irl.py
  perception_setup/target_generator.py
  perception_setup/target_generator_irl.py
  perception_setup/bricks_homography.py
  perception_setup/bricks_homography_irl.py
  perception_setup/grasp_shifting_node.py
  perception_setup/grasp_shifting_node_irl.py
  perception_setup/visualizer.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install C++ Executables
install(TARGETS 
  visp_ibvs_controller
  trial_controller
  trial_controller_irl
  DESTINATION lib/${PROJECT_NAME}
)
# Install Launch Files
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()