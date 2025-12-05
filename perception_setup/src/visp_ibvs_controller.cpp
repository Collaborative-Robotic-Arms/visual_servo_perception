#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp> 
#include <std_msgs/msg/float64.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp> 
#include <sensor_msgs/msg/camera_info.hpp> 

#include <visp3/visual_features/vpFeaturePoint.h>
#include <visp3/vs/vpServo.h>
#include <visp3/core/vpMatrix.h>
#include <visp3/core/vpCameraParameters.h>
#include <visp3/core/vpColVector.h>
#include <visp3/core/vpException.h>

#include <cmath> 
#include <algorithm> 

#define NODE_NAME "visp_ibvs_controller"
#define NUM_FEATURES 3 

class VispIBVSController : public rclcpp::Node
{
public:
    VispIBVSController() : Node(NODE_NAME)
    {
        // Gain - Lowered slightly for simulator stability
        servo_.setLambda(0.2); 
        servo_.setInteractionMatrixType(vpServo::CURRENT);
        servo_.setServo(vpServo::EYEINHAND_CAMERA);

        cam_initialized_ = false;
        
        // FRAME ID: Kept from your simulator code
        cam_frame_id_ = "ar4_camera_link"; 

        // Smoothing variables
        v_prev_ = vpColVector(6, 0.0);
        alpha_ = 0.1; // Increased slightly for smoother Sim behavior

        // --- ROS 2 Communication Setup ---
        cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "cameraAR4/camera_info", 10,
            std::bind(&VispIBVSController::infoCallback, this, std::placeholders::_1));

        feature_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            "/feature_coordinates_6D", 10,
            std::bind(&VispIBVSController::featureCallback, this, std::placeholders::_1));

        depth_sub_ = this->create_subscription<std_msgs::msg::Float64>(
            "/camera_to_marker_depth", 10,
            std::bind(&VispIBVSController::depthCallback, this, std::placeholders::_1));

        velocity_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>(
            "/servo_node/delta_twist_cmds", 10);

        error_pub_ = this->create_publisher<std_msgs::msg::Float64>(
            "/vs/error_norm", 10);

        RCLCPP_INFO(this->get_logger(), "ViSP IBVS Controller (Sim) started.");
    }

private:
    void infoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        if (cam_initialized_) return; 
        double px = msg->k[0]; double py = msg->k[4]; 
        double u0 = msg->k[2]; double v0 = msg->k[5]; 
        cam_.initPersProjWithoutDistortion(px, py, u0, v0);

        // Define Desired Features (Calibrated)
        // Ensure these pixel values match the resolution in your Simulator Camera
        const double Z_desired = 0.2093; 
        double desired_pixels[NUM_FEATURES * 2] = {
            380.0, 291.0,  
            378.0, 432.0,  
            236.0, 432.0    
        };

        for (int i = 0; i < NUM_FEATURES; ++i) {
            double x_d = (desired_pixels[i*2] - cam_.get_u0()) / cam_.get_px();
            double y_d = (desired_pixels[i*2+1] - cam_.get_v0()) / cam_.get_py();
            s_star_[i].set_x(x_d); s_star_[i].set_y(y_d); s_star_[i].set_Z(Z_desired); 
            s_curr_[i].set_x(x_d); s_curr_[i].set_y(y_d); s_curr_[i].set_Z(Z_desired); 
            servo_.addFeature(s_curr_[i], s_star_[i]);
        }
        cam_initialized_ = true;
    }

    void depthCallback(const std_msgs::msg::Float64::SharedPtr msg) {
        if (msg->data > 0.05 && msg->data < 5.0) current_depth_z_ = msg->data;
    }

    void featureCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
        if (!cam_initialized_) return;
        if (msg->data.size() != NUM_FEATURES * 2) return;

        // 1. Update Features
        for (int i = 0; i < NUM_FEATURES; ++i) {
            double x_norm = (msg->data[i*2] - cam_.get_u0()) / cam_.get_px();
            double y_norm = (msg->data[i*2+1] - cam_.get_v0()) / cam_.get_py();
            s_curr_[i].set_x(x_norm); s_curr_[i].set_y(y_norm); s_curr_[i].set_Z(current_depth_z_); 
        }

        // 2. Compute Raw Optical Velocity
        vpColVector v_opt(6, 0.0);
        double error_norm = 0.0;
        try {
            v_opt = servo_.computeControlLaw();
            error_norm = servo_.getError().sumSquare(); 
        } catch (...) { v_opt = 0.0; }
        std_msgs::msg::Float64 err_msg;
        err_msg.data = error_norm;
        error_pub_->publish(err_msg);
        // Sanity Check
        for(int k=0; k<6; k++) if (!std::isfinite(v_opt[k])) v_opt = 0.0;

        // 3. Scaling
        double max_lin = 0.05; // 5 cm/s
        double max_rot = 0.1;  // 0.1 rad/s

        // Linear Scale
        double lin_mag = std::sqrt(v_opt[0]*v_opt[0] + v_opt[1]*v_opt[1] + v_opt[2]*v_opt[2]);
        if (lin_mag > max_lin) {
            double scale = max_lin / lin_mag;
            v_opt[0] *= scale; v_opt[1] *= scale; v_opt[2] *= scale;
        }

        // Angular Scale
        double rot_mag = std::sqrt(v_opt[3]*v_opt[3] + v_opt[4]*v_opt[4] + v_opt[5]*v_opt[5]);
        if (rot_mag > max_rot) {
            double scale = max_rot / rot_mag;
            v_opt[3] *= scale; v_opt[4] *= scale; v_opt[5] *= scale;
        }

        // 4. Low Pass Filter
        v_opt = v_opt * alpha_ + v_prev_ * (1.0 - alpha_);
        v_prev_ = v_opt;

        if (error_norm < 1e-3) {
             v_opt = 0.0;
             v_prev_ = 0.0; 
        }

        auto velocity_msg = std::make_shared<geometry_msgs::msg::TwistStamped>();
        velocity_msg->header.stamp = this->get_clock()->now(); 
        velocity_msg->header.frame_id = cam_frame_id_; 
        
        // ---------------------------------------------------------
        // 5. STANDARD SIMULATOR MAPPING (OPTICAL -> GEOMETRIC)
        // ---------------------------------------------------------
        // ViSP (Optical Frame):
        //   [0] X = Right
        //   [1] Y = Down
        //   [2] Z = Forward (Depth)
        //
        // Robot Camera Link (Standard Geometric Frame):
        //   X = Forward
        //   Y = Left
        //   Z = Up
        // ---------------------------------------------------------
        
        // Mapping Linear Velocities
        velocity_msg->twist.linear.x =  v_opt[2];  // Optical Z (Depth) -> Robot X (Forward)
        velocity_msg->twist.linear.y = -v_opt[0];  // Optical X (Right) -> Robot Y (Left)
        velocity_msg->twist.linear.z = -v_opt[1];  // Optical Y (Down)  -> Robot Z (Up)

        // Mapping Angular Velocities (Rotation follows Right Hand Rule)
        velocity_msg->twist.angular.x =  v_opt[5]; // Roll around Opt Z -> Roll around Rob X
        velocity_msg->twist.angular.y = -v_opt[3]; // Pitch around Opt X -> Pitch around Rob Y
        velocity_msg->twist.angular.z = -v_opt[4]; // Yaw around Opt Y   -> Yaw around Rob Z
        
        velocity_pub_->publish(*velocity_msg);

        // Debug output to check if "Forward" is actually driving X now
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
            "SENT (Robot Frame): Vx=%.3f (Down) Vy=%.3f (Left) Vz=%.3f (Fwd)",
            velocity_msg->twist.linear.x, velocity_msg->twist.linear.y, velocity_msg->twist.linear.z);
    }

    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr feature_sub_; 
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr depth_sub_; 
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_; 
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr velocity_pub_; 
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr error_pub_;
    vpServo servo_;
    vpFeaturePoint s_curr_[NUM_FEATURES], s_star_[NUM_FEATURES]; 
    vpCameraParameters cam_;
    bool cam_initialized_; 
    double current_depth_z_;
    std::string cam_frame_id_; 
    
    // Smoothing members
    vpColVector v_prev_;
    double alpha_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VispIBVSController>());
    rclcpp::shutdown();
    return 0;
}