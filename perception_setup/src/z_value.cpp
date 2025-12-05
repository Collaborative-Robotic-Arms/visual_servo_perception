#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <chrono>
#include <memory>
#include <cmath> 
#include <string>

using namespace std::chrono_literals;

class ZQueryTool : public rclcpp::Node
{
public:
    ZQueryTool() : Node("z_query_tool")
    {
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

        // The tool runs once and then shuts down
        timer_ = this->create_wall_timer(
            100ms, // Slightly increased time to allow buffer to fill
            std::bind(&ZQueryTool::query_and_print_z, this));

        RCLCPP_INFO(this->get_logger(), "Z Query Tool Initialized...");
    }

private:
    // --- Helper to Print Coordinates in Readable Format ---
    void print_frame_details(const std::string& label, const geometry_msgs::msg::TransformStamped& t)
    {
        // 1. Position
        double x = t.transform.translation.x;
        double y = t.transform.translation.y;
        double z = t.transform.translation.z;

        // 2. Orientation (Convert Quaternion to Euler RPY)
        tf2::Quaternion q(
            t.transform.rotation.x,
            t.transform.rotation.y,
            t.transform.rotation.z,
            t.transform.rotation.w);
        tf2::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);

        RCLCPP_INFO(this->get_logger(), "  > %s:", label.c_str());
        RCLCPP_INFO(this->get_logger(), "    Position (XYZ): [%.4f, %.4f, %.4f]", x, y, z);
        RCLCPP_INFO(this->get_logger(), "    Rotation (RPY): [%.4f, %.4f, %.4f] rad", roll, pitch, yaw);
    }

    void query_and_print_z()
    {
        const std::string REFERENCE_FRAME = "base_link"; // Change to 'world' if needed
        const std::string CAMERA_FRAME = "ar4_camera_link"; 
        const std::string TARGET_FRAME = "aruco_target_frame"; 

        geometry_msgs::msg::TransformStamped t_cam_to_target;
        geometry_msgs::msg::TransformStamped t_ref_to_cam;
        geometry_msgs::msg::TransformStamped t_ref_to_target;

        try {
            // 1. Get Relative Transform (Camera -> Target)
            t_cam_to_target = tf_buffer_->lookupTransform(
                CAMERA_FRAME, TARGET_FRAME, tf2::TimePointZero);
            
            // 2. Get Global Coordinates (Base -> Camera)
            // We use a separate try-block logic or just do it here if we assume base_link exists
            t_ref_to_cam = tf_buffer_->lookupTransform(
                REFERENCE_FRAME, CAMERA_FRAME, tf2::TimePointZero);

            // 3. Get Global Coordinates (Base -> Target)
            t_ref_to_target = tf_buffer_->lookupTransform(
                REFERENCE_FRAME, TARGET_FRAME, tf2::TimePointZero);

        } catch (const tf2::TransformException & ex) {
            RCLCPP_WARN(this->get_logger(), "Waiting for transforms... (%s)", ex.what());
            return;
        }

        RCLCPP_INFO(this->get_logger(), "==========================================================");
        
        // --- Print Absolute Coordinates ---
        RCLCPP_INFO(this->get_logger(), "1. GLOBAL COORDINATES (Relative to %s)", REFERENCE_FRAME.c_str());
        print_frame_details("Camera Frame", t_ref_to_cam);
        print_frame_details("Target Frame", t_ref_to_target);

        RCLCPP_INFO(this->get_logger(), "----------------------------------------------------------");

        // --- Print Relative Transform ---
        RCLCPP_INFO(this->get_logger(), "2. RELATIVE TRANSFORM (Target inside Camera Frame)");
        print_frame_details("Target relative to Camera", t_cam_to_target);

        // --- Extract Z Depth ---
        double z_depth = t_cam_to_target.transform.translation.x; 

        RCLCPP_INFO(this->get_logger(), "----------------------------------------------------------");
        RCLCPP_INFO(this->get_logger(), "** CALCULATED DEPTH **");
        RCLCPP_INFO(this->get_logger(), "   Using Camera X-axis: %.4f meters", z_depth);
        RCLCPP_INFO(this->get_logger(), "==========================================================");

        // Stop the timer and shut down
        timer_->cancel();
        rclcpp::shutdown();
    }

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ZQueryTool>());
    return 0;
}