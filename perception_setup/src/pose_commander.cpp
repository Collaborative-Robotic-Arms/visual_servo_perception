#include <moveit/move_group_interface/move_group_interface.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <iostream>
#include <memory>
#include <rclcpp/rclcpp.hpp>

// Headers for RPY to Quaternion conversion
#include <tf2/LinearMath/Quaternion.hpp>
#include <tf2/LinearMath/Transform.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// Define your robot's planning group name
static const std::string PLANNING_GROUP = "ar_manipulator";

void go_to_pose(moveit::planning_interface::MoveGroupInterface& move_group_interface,
                const rclcpp::Logger& logger) {
    
    // 1. Clear previous target to ensure fresh planning
    move_group_interface.clearPoseTargets();
    
    // 2. User Input
    double x, y, z, roll_deg, pitch_deg, yaw_deg;

    std::cout << "\n--- Target Pose Input ---\n";
    std::cout << "Enter X position (meters): ";
    std::cin >> x;
    std::cout << "Enter Y position (meters): ";
    std::cin >> y;
    std::cout << "Enter Z position (meters): ";
    std::cin >> z;
    std::cout << "Enter Roll angle (degrees): ";
    std::cin >> roll_deg;
    std::cout << "Enter Pitch angle (degrees): ";
    std::cin >> pitch_deg;
    std::cout << "Enter Yaw angle (degrees): ";
    std::cin >> yaw_deg;

    // 3. RPY to Quaternion Conversion
    // Use M_PI defined in <cmath> (often included via other headers)
    double roll_rad = roll_deg * M_PI / 180.0;
    double pitch_rad = pitch_deg * M_PI / 180.0;
    double yaw_rad = yaw_deg * M_PI / 180.0;

    tf2::Quaternion q_tf2;
    q_tf2.setRPY(roll_rad, pitch_rad, yaw_rad);
    geometry_msgs::msg::Quaternion q_msg = tf2::toMsg(q_tf2);

    // 4. Create Target Pose
    geometry_msgs::msg::Pose target_pose;
    target_pose.position.x = x;
    target_pose.position.y = y;
    target_pose.position.z = z;
    target_pose.orientation = q_msg;

    RCLCPP_INFO(logger, "Attempting to plan to target pose:");
    RCLCPP_INFO(logger, "XYZ: (%.4f, %.4f, %.4f)", x, y, z);
    RCLCPP_INFO(logger, "RPY: (%.1f, %.1f, %.1f) deg", roll_deg, pitch_deg, yaw_deg);

    // 5. Plan and Execute
    move_group_interface.setPoseTarget(target_pose);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = (move_group_interface.plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);

    if (success) {
        RCLCPP_INFO(logger, "Planning successful! Executing motion...");
        // Call stop() before execute() to explicitly halt any residual motion, 
        // mitigating execution errors.
        move_group_interface.stop();
        move_group_interface.execute(plan);
        RCLCPP_INFO(logger, "Motion finished.");
    } else {
        RCLCPP_ERROR(logger, "Planning failed to find a path to the target pose.");
    }
}

void go_to_home(moveit::planning_interface::MoveGroupInterface& move_group_interface,
                const rclcpp::Logger& logger) {
    
    // 1. Clear previous target
    move_group_interface.clearPoseTargets();

    // 2. Set the named target "home"
    RCLCPP_INFO(logger, "Planning to move to 'home' configuration...");
    move_group_interface.setNamedTarget("home");

    // 3. Plan and Execute
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = (move_group_interface.plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);

    if (success) {
        RCLCPP_INFO(logger, "Planning successful! Executing motion...");
        move_group_interface.stop();
        move_group_interface.execute(plan);
        RCLCPP_INFO(logger, "Returned to home configuration.");
    } else {
        RCLCPP_ERROR(logger, "Planning failed to find a path to the 'home' configuration.");
    }
}


int main(int argc, char* argv[]) {
    // 1. Initialize ROS 2
    rclcpp::init(argc, argv);
    auto const node = std::make_shared<rclcpp::Node>(
        "rpy_pose_commander_loop",
        rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

    // Use a separate thread for MoveIt to run its callbacks
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    std::thread([&executor]() { executor.spin(); }).detach();

    auto const logger = rclcpp::get_logger("pose_commander");

    // 2. Setup MoveGroup Interface
    RCLCPP_INFO(logger, "Starting MoveGroupInterface for group: %s", PLANNING_GROUP.c_str());

    using moveit::planning_interface::MoveGroupInterface;
    auto move_group_interface = MoveGroupInterface(node, PLANNING_GROUP);

    // Initial configuration for robustness
    move_group_interface.setPoseReferenceFrame("base_link"); 
    move_group_interface.setPlanningTime(5.0); // Increase planning time for complex moves
    move_group_interface.setMaxVelocityScalingFactor(0.7);
    move_group_interface.setMaxAccelerationScalingFactor(0.5);

    // Goal tolerances
    move_group_interface.setGoalPositionTolerance(0.0005); // Tighter tolerance (0.5 mm)
    move_group_interface.setGoalOrientationTolerance(0.005); // Tighter tolerance (~0.3 degrees)
    move_group_interface.setGoalJointTolerance(0.001);

    // 3. Main Command Loop
    int choice = 0;
    while (rclcpp::ok()) {
        std::cout << "\n====================================\n";
        std::cout << "Robot Command Menu\n";
        std::cout << "1. Move to a new XYZ/RPY Pose\n";
        std::cout << "2. Go to 'home' configuration\n";
        std::cout << "3. Exit\n";
        std::cout << "Enter choice: ";
        
        if (!(std::cin >> choice)) {
            // Handle bad input (e.g., non-numeric)
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            RCLCPP_ERROR(logger, "Invalid input. Please enter a number.");
            continue; 
        }

        if (choice == 1) {
            go_to_pose(move_group_interface, logger);
        } else if (choice == 2) {
            go_to_home(move_group_interface, logger);
        } else if (choice == 3) {
            break; // Exit the loop
        } else {
            RCLCPP_WARN(logger, "Invalid choice. Please enter 1, 2, or 3.");
        }
    }

    // 4. Shutdown
    RCLCPP_INFO(logger, "Shutting down Pose Commander.");
    rclcpp::shutdown();
    return 0;
}