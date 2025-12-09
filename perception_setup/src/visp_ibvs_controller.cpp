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
// --- CHANGE 1: Revert to 3 features (Triangle) ---
#define NUM_FEATURES 3 

class VispIBVSController : public rclcpp::Node
{
public:
    VispIBVSController() : Node(NODE_NAME)
    {
        // Parameters
        this->declare_parameter("target_depth", 0.20);   
        this->declare_parameter("marker_size", 0.05);    
        
        servo_.setLambda(0.2); 
        servo_.setInteractionMatrixType(vpServo::CURRENT);
        servo_.setServo(vpServo::EYEINHAND_CAMERA);

        cam_initialized_ = false;
        cam_frame_id_ = "ar4_camera_link"; 
        
        // --- CHANGE 4: Initialize Depth safely (Prevent Z=0 Singularity) ---
        current_depth_z_ = 0.4; 

        // Smoothing
        v_prev_ = vpColVector(6, 0.0);
        alpha_ = 0.1; 

        // Subscribers
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

        RCLCPP_INFO(this->get_logger(), "ViSP IBVS Controller (3-Point Mode) started.");
    }

private:
    void infoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        if (cam_initialized_) return; 
        
        double px = msg->k[0]; double py = msg->k[4]; 
        double u0 = msg->k[2]; double v0 = msg->k[5]; 
        cam_.initPersProjWithoutDistortion(px, py, u0, v0);

        double Z_des = this->get_parameter("target_depth").as_double();
        double L = this->get_parameter("marker_size").as_double();
        double h = L / 2.0; 

        // --- CHANGE 2: Define only 3 points (Top-Left, Top-Right, Bottom-Right) ---
        // We skip the 4th point (Bottom-Left)
        double X[3] = {-h,  h,  h};
        double Y[3] = {-h, -h,  h};

        RCLCPP_INFO(this->get_logger(), "Computing Dynamic Goal (3 Points) for Z=%.2f m", Z_des);

        for (int i = 0; i < NUM_FEATURES; ++i) {
            double u_des = u0 + px * (X[i] / Z_des);
            double v_des = v0 + py * (Y[i] / Z_des);

            double x_n = (u_des - u0) / px;
            double y_n = (v_des - v0) / py;

            s_star_[i].set_x(x_n);
            s_star_[i].set_y(y_n);
            s_star_[i].set_Z(Z_des); 
            
            s_curr_[i].set_x(x_n);
            s_curr_[i].set_y(y_n);
            s_curr_[i].set_Z(Z_des); 

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
        
        // --- CHANGE 3: Accept 8 numbers (4 points) but only use 3 ---
        // Check if we have AT LEAST 6 numbers (3 points * 2 coordinates)
        if (msg->data.size() < NUM_FEATURES * 2) return;

        // Loop runs 0, 1, 2. 
        // It reads data[0,1], data[2,3], data[4,5].
        // It completely ignores data[6,7] (the 4th point).
        for (int i = 0; i < NUM_FEATURES; ++i) {
            double x_norm = (msg->data[i*2] - cam_.get_u0()) / cam_.get_px();
            double y_norm = (msg->data[i*2+1] - cam_.get_v0()) / cam_.get_py();
            s_curr_[i].set_x(x_norm); s_curr_[i].set_y(y_norm); s_curr_[i].set_Z(current_depth_z_); 
        }

        vpColVector v_opt(6, 0.0);
        double error_norm = 0.0;
        try {
            v_opt = servo_.computeControlLaw();
            error_norm = servo_.getError().sumSquare(); 
        } catch (...) { v_opt = 0.0; }

        // Publish Error
        std_msgs::msg::Float64 err_msg;
        err_msg.data = error_norm;
        error_pub_->publish(err_msg);
        
        // Safety Checks
        for(int k=0; k<6; k++) if (!std::isfinite(v_opt[k])) v_opt = 0.0;

        // Scaling (Safety Limits)
        double max_lin = 0.1; 
        double max_rot = 0.5; 
        
        double lin_mag = std::sqrt(v_opt[0]*v_opt[0] + v_opt[1]*v_opt[1] + v_opt[2]*v_opt[2]);
        if (lin_mag > max_lin) v_opt *= (max_lin / lin_mag);

        double rot_mag = std::sqrt(v_opt[3]*v_opt[3] + v_opt[4]*v_opt[4] + v_opt[5]*v_opt[5]);
        if (rot_mag > max_rot) {
            double scale = max_rot / rot_mag;
            v_opt[3] *= scale; v_opt[4] *= scale; v_opt[5] *= scale;
        }

        // Smoothing
        v_opt = v_opt * alpha_ + v_prev_ * (1.0 - alpha_);
        v_prev_ = v_opt;

        if (error_norm < 1e-3) { v_opt = 0.0; v_prev_ = 0.0; }

        auto velocity_msg = std::make_shared<geometry_msgs::msg::TwistStamped>();
        velocity_msg->header.stamp = this->get_clock()->now(); 
        velocity_msg->header.frame_id = cam_frame_id_; 

        // Map Optical -> Geometric
        velocity_msg->twist.linear.x =  v_opt[2]; 
        velocity_msg->twist.linear.y = -v_opt[0]; 
        velocity_msg->twist.linear.z = -v_opt[1]; 
        velocity_msg->twist.angular.x =  v_opt[5];
        velocity_msg->twist.angular.y = -v_opt[3];
        velocity_msg->twist.angular.z = -v_opt[4];
        
        velocity_pub_->publish(*velocity_msg);
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
    vpColVector v_prev_;
    double alpha_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VispIBVSController>());
    rclcpp::shutdown();
    return 0;
}