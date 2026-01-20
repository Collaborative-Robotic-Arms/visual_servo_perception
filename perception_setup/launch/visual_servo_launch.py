import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_name = 'perception_setup'

    return LaunchDescription([
        # 1. Perception: ROI Tracker
        Node(
            package=pkg_name,
            executable='feature_detection_fisheye.py',
            name='brick_tracker',
            #output='screen'
        ),

        # 2. Perception: Homography Bridge
        Node(
            package=pkg_name,
            executable='bricks_homography.py',
            name='homography_bridge',
            #output='screen'
        ),

        # 3. Perception: Depth Estimator
        Node(
            package=pkg_name,
            executable='depth_estimation.py',
            name='depth_estimator',
            #output='screen'
        ),

        # 4. Logic: Target Generator (Yellow Box)
        Node(
            package=pkg_name,
            executable='target_generator.py',
            name='target_generator',
            #output='screen'
        ),

        # 5. Logic: Grasp Shifting (Cyan Box/Shifted Features)
        Node(
            package=pkg_name,
            executable='grasp_shifting_node.py',
            name='grasp_shifter',
            #output='screen'
        ),

        # --- NEW: RViz Visualization Node ---
        Node(
            package=pkg_name,
            executable='visualizer.py',
            name='vs_visualizer',
            #output='screen'
        ),

        # Optional: Start RViz2 automatically
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            #output='screen'
        )
    ])