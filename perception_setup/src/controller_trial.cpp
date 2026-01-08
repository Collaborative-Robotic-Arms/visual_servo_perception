#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp> 
#include <std_msgs/msg/float64.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp> 
#include <sensor_msgs/msg/camera_info.hpp> 

#include <visp3/visual_features/vpFeaturePoint.h>
#include <visp3/vs/vpServo.h>
#include <visp3/core/vpCameraParameters.h>

#include <vector>
#include <cmath>
#include <algorithm>

#define NODE_NAME "visp_hybrid_aruco"
#define NUM_FEATURES 4 

// --- SAFETY LIMITS ---
#define MAX_LIN_VEL 0.05  // Limit to 5 cm/s (Very Safe)
#define MAX_ANG_VEL 0.15  // Limit rotational speed

class VispHybridAruco : public rclcpp::Node
{
public:
    VispHybridAruco() : Node(NODE_NAME)
    {
        // --- CONTROL GAINS (LOWERED FOR SAFETY) ---
        // Old: 0.0003 -> New: 0.0001 (Slower reaction to large errors)
        this->declare_parameter("gain_xy", 0.0004); 
        
        // Old: 0.2 -> New: 0.1 (Slower Z movement to prevent table hits)
        this->declare_parameter("gain_z", 0.1);    
        
        this->declare_parameter("gain_i_xy", 0.0001); 
        this->declare_parameter("max_integral_output", 0.01); // Reduced integral cap
        this->declare_parameter("target_depth", 0.2114); 

        // --- VISP SETUP ---
        servo_.setInteractionMatrixType(vpServo::CURRENT);
        servo_.setServo(vpServo::EYEINHAND_CAMERA);

        cam_initialized_ = false;
        goal_initialized_ = false;
        cam_frame_id_ = "ar4_ee_link"; 
        
        sum_err_u_ = 0.0;
        sum_err_v_ = 0.0;
        last_time_ = this->get_clock()->now();

        // --- SUBSCRIBERS & PUBLISHERS ---
        cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/cameraAR4/camera_info", 10, 
            std::bind(&VispHybridAruco::infoCallback, this, std::placeholders::_1));

        feature_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            "/feature_coordinates_6D", 10, 
            std::bind(&VispHybridAruco::featureCallback, this, std::placeholders::_1));

        velocity_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>(
            "/servo_server/delta_twist_cmds", 10);
            
        error_pub_ = this->create_publisher<std_msgs::msg::Float64>(
            "/vs/error_norm", 10);

        RCLCPP_INFO(this->get_logger(), "Hybrid Controller Started (SAFETY LIMITS: 0.05 m/s)");
    }

private:
    void infoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        if (cam_initialized_) return; 
        cam_.initPersProjWithoutDistortion(msg->k[0], msg->k[4], msg->k[2], msg->k[5]);
        cam_initialized_ = true;
        if (!goal_initialized_) setHardcodedGoal();
    }

    void setHardcodedGoal()
    {
        std::vector<std::pair<double, double>> desired_pixels = {
            {312.8, 309.4}, {384.5, 310.2}, 
            {383.5, 408.6}, {311.8, 407.8}  
        };

        double Z_des = this->get_parameter("target_depth").as_double();
        double u_sum = 0, v_sum = 0;
        servo_.kill(); 

        for (int i = 0; i < NUM_FEATURES; ++i) {
            double u_d = desired_pixels[i].first;
            double v_d = desired_pixels[i].second;
            u_sum += u_d; v_sum += v_d;

            double x_n = (u_d - cam_.get_u0()) / cam_.get_px();
            double y_n = (v_d - cam_.get_v0()) / cam_.get_py();

            s_star_[i].set_x(x_n); s_star_[i].set_y(y_n); s_star_[i].set_Z(Z_des);
            servo_.addFeature(s_curr_[i], s_star_[i]);
        }
        u_des_center_ = u_sum / NUM_FEATURES;
        v_des_center_ = v_sum / NUM_FEATURES;
        goal_initialized_ = true;
    }

    void featureCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
        if (!cam_initialized_ || !goal_initialized_ || msg->data.size() < NUM_FEATURES * 2) return;

        rclcpp::Time current_time = this->get_clock()->now();
        double dt = (current_time - last_time_).seconds();
        last_time_ = current_time;
        if (dt > 0.1) dt = 0.1; 

        double u_sum = 0, v_sum = 0;
        double Z_curr_est = this->get_parameter("target_depth").as_double(); 

        for (int i = 0; i < NUM_FEATURES; ++i) {
            double u_raw = msg->data[i*2];
            double v_raw = msg->data[i*2+1];
            u_sum += u_raw; v_sum += v_raw;
            
            double x_n = (u_raw - cam_.get_u0()) / cam_.get_px();
            double y_n = (v_raw - cam_.get_v0()) / cam_.get_py();
            s_curr_[i].set_x(x_n); s_curr_[i].set_y(y_n); s_curr_[i].set_Z(Z_curr_est); 
        }
        
        // --- 1. XY CONTROL (PI) ---
        double u_curr = u_sum / NUM_FEATURES;
        double v_curr = v_sum / NUM_FEATURES;
        
        double k_p = this->get_parameter("gain_xy").as_double();
        double k_i = this->get_parameter("gain_i_xy").as_double();
        double max_i_out = this->get_parameter("max_integral_output").as_double();

        double err_u = u_des_center_ - u_curr;
        double err_v = v_des_center_ - v_curr;

        sum_err_u_ += err_u * dt;
        sum_err_v_ += err_v * dt;

        auto clamp = [](double v, double limit) { return std::max(std::min(v, limit), -limit); };

        // Calculate Terms
        double p_term_x = -1.0 * err_u * k_p;
        double p_term_y = -1.0 * err_v * k_p;
        double i_term_x = clamp(-1.0 * sum_err_u_ * k_i, max_i_out);
        double i_term_y = clamp(-1.0 * sum_err_v_ * k_i, max_i_out);

        double v_x = p_term_x + i_term_x;
        double v_y = p_term_y + i_term_y;

        // --- 2. Z/ROT CONTROL (VISP) ---
        vpColVector v_visp(6, 0.0);
        try { v_visp = servo_.computeControlLaw(); } catch(...) { v_visp = 0.0; }
        
        double v_z = v_visp[2] * this->get_parameter("gain_z").as_double();
        double w_z = v_visp[5]*0.15; 

        // --- 3. SAFETY CLAMPING (CRITICAL) ---
        // Ensure that NO command exceeds the safety limit
        v_x = clamp(v_x, MAX_LIN_VEL);
        v_y = clamp(v_y, MAX_LIN_VEL);
        v_z = clamp(v_z, MAX_LIN_VEL);
        w_z = clamp(w_z, MAX_ANG_VEL);

        // Publish
        auto velocity_msg = std::make_unique<geometry_msgs::msg::TwistStamped>();
        velocity_msg->header.stamp = current_time;
        velocity_msg->header.frame_id = cam_frame_id_; 

        velocity_msg->twist.linear.x = v_x;
        velocity_msg->twist.linear.y = v_y;
        velocity_msg->twist.linear.z = v_z;
        velocity_msg->twist.angular.z = w_z; 

        velocity_pub_->publish(std::move(velocity_msg));
        
        // Reset integral if error is small
        double total_err_norm = sqrt(err_u*err_u + err_v*err_v);
        if (total_err_norm < 5.0) { sum_err_u_ = 0.0; sum_err_v_ = 0.0; }

        RCLCPP_INFO(this->get_logger(), 
            "Err: %.0f px | Cmd: Vx=%.3f Vy=%.3f Vz=%.3f Wz=%.3f", 
            total_err_norm, v_x, v_y, v_z, w_z);

        std_msgs::msg::Float64 err_msg;
        err_msg.data = total_err_norm;
        error_pub_->publish(err_msg);
    }

    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_; 
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr feature_sub_; 
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr velocity_pub_; 
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr error_pub_;

    vpServo servo_;
    vpFeaturePoint s_curr_[NUM_FEATURES], s_star_[NUM_FEATURES]; 
    vpCameraParameters cam_;
    bool cam_initialized_, goal_initialized_; 
    std::string cam_frame_id_;
    double u_des_center_, v_des_center_;
    double sum_err_u_, sum_err_v_;
    rclcpp::Time last_time_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VispHybridAruco>());
    rclcpp::shutdown();
    return 0;
}