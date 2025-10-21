cmake_minimum_required(VERSION 3.8)
# It's good practice for the project name to match your package name.
# You might want to change this to "visual_servo_perception" later.
project(perception_setup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --- MODIFICATIONS START HERE ---

# 1. Find necessary dependencies for Python nodes and custom messages.
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
# It's very likely your nodes will use standard ROS messages like Image or CameraInfo.
find_package(sensor_msgs REQUIRED)


# 2. Define the custom message file(s) to be generated.
# This assumes you have a directory named 'msg' in your package
# with a file called 'BrickFeature.msg' inside it.
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BrickFeature.msg"
  # Add dependencies for your custom messages here. For example, if your
  # message uses a header from sensor_msgs, you would add it below.
  DEPENDENCIES sensor_msgs
)

# 3. Install the Python executable scripts.
# This makes them runnable with `ros2 run`.
# This assumes your python files are inside a directory that has the same
# name as your project (e.g., perception_setup/camera_driver.py)
install(PROGRAMS
  perception_setup/camera_driver.py
  perception_setup/brick_detector.py
  DESTINATION lib/${PROJECT_NAME}
)

# --- MODIFICATIONS END HERE ---


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
