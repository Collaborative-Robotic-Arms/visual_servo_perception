#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp> 
#include <std_msgs/msg/float64.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp> 

#include <visp3/visual_features/vpFeaturePoint.h>
#include <visp3/vs/vpServo.h>
#include <visp3/core/vpCameraParameters.h>

#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>

#define NODE_NAME "visp_hybrid_aruco"
#define NUM_FEATURES 4 

class VispHybridAruco : public rclcpp::Node
{
public:
    VispHybridAruco() : Node(NODE_NAME)
    {
        // --- PHYSICAL LIMITS ---
        this->declare_parameter("max_linear_vel", 0.15); 
        this->declare_parameter("max_angular_vel", 0.5);
        this->declare_parameter("max_linear_accel", 0.25);
        this->declare_parameter("max_angular_accel", 1.5);

        // --- GAINS ---
        this->declare_parameter("gain_xy", 0.0006);  
        this->declare_parameter("gain_i_xy", 0.0000); 
        this->declare_parameter("gain_d_xy", 0.00015); 
        this->declare_parameter("gain_yaw", 0.6); 

        // Z-Axis Params
        this->declare_parameter("target_depth", 0.2);
        this->declare_parameter("lambda_z_0", 0.15);   
        this->declare_parameter("lambda_z_inf", 0.02); 
        this->declare_parameter("beta_z", 5.0); 
        
        // Logging
        this->declare_parameter("enable_logging", true);
        this->declare_parameter("log_file_path", "/home/omar-magdy/vs_log.csv");

        // --- VISP INIT (HARDCODED) ---
        double px = 190.68;
        double py = 190.68;
        double u0 = 320.0;
        double v0 = 240.0;
        
        cam_.initPersProjWithoutDistortion(px, py, u0, v0);
        cam_initialized_ = true; 
        
        servo_.setInteractionMatrixType(vpServo::CURRENT);
        servo_.setServo(vpServo::EYEINHAND_CAMERA);
        goal_initialized_ = false;
        cam_frame_id_ = "ar4_ee_link"; 
        
        // State Init
        sum_err_u_ = 0.0; sum_err_v_ = 0.0; prev_err_u_ = 0.0; prev_err_v_ = 0.0; 
        v_prev_x_ = 0.0; v_prev_y_ = 0.0; v_prev_z_ = 0.0; w_prev_z_ = 0.0;
        
        last_time_ = this->get_clock()->now();
        start_time_ = this->get_clock()->now(); 
        current_depth_z_ = this->get_parameter("target_depth").as_double();

        if (this->get_parameter("enable_logging").as_bool()) {
            std::string path = this->get_parameter("log_file_path").as_string();
            log_file_.open(path);
            if (log_file_.is_open()) log_file_ << "Time,ErrorNorm,Vx,Vy,Vz,Wz\n";
        }

        // --- SUBSCRIBERS ---
        feature_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            "/visual_servo/shifted_features", 10, std::bind(&VispHybridAruco::featureCallback, this, std::placeholders::_1));

        desired_feature_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            "/visual_servo/desired_features", 10, std::bind(&VispHybridAruco::desiredFeatureCallback, this, std::placeholders::_1));

        depth_sub_ = this->create_subscription<std_msgs::msg::Float64>(
            "/camera_to_brick_depth", 10, std::bind(&VispHybridAruco::depthCallback, this, std::placeholders::_1));

        velocity_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>("/servo_node/delta_twist_cmds", 10);
        error_pub_ = this->create_publisher<std_msgs::msg::Float64>("/vs/error_norm", 10);

        RCLCPP_INFO(this->get_logger(), "Controller Online (Debug Enabled). Using HARDCODED Intrinsics.");
    }

    ~VispHybridAruco() { if (log_file_.is_open()) log_file_.close(); }

private:
    double calculateArea() {
        double area = 0.0;
        for (int i = 0; i < NUM_FEATURES; i++) {
            int next = (i + 1) % NUM_FEATURES;
            area += (s_curr_[i].get_x() * s_curr_[next].get_y()) - (s_curr_[next].get_x() * s_curr_[i].get_y());
        }
        return 0.5 * std::abs(area);
    }

    double calculateDesiredArea() {
        double area = 0.0;
        for (int i = 0; i < NUM_FEATURES; i++) {
            int next = (i + 1) % NUM_FEATURES;
            area += (s_star_[i].get_x() * s_star_[next].get_y()) - (s_star_[next].get_x() * s_star_[i].get_y());
        }
        return 0.5 * std::abs(area);
    }

    void depthCallback(const std_msgs::msg::Float64::SharedPtr msg) {
        if (msg->data > 0.05 && msg->data < 5.0) current_depth_z_ = msg->data;
    }

    void desiredFeatureCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
        if (msg->data.size() < 8) return;
        double Z_des = this->get_parameter("target_depth").as_double();
        double u_sum = 0, v_sum = 0;
        servo_.kill(); 
        for (int i = 0; i < NUM_FEATURES; ++i) {
            double u_d = msg->data[i*2]; double v_d = msg->data[i*2+1];
            u_sum += u_d; v_sum += v_d;
            s_star_[i].set_x((u_d - cam_.get_u0()) / cam_.get_px());
            s_star_[i].set_y((v_d - cam_.get_v0()) / cam_.get_py());
            s_star_[i].set_Z(Z_des);
            servo_.addFeature(s_curr_[i], s_star_[i]);
        }
        u_des_center_ = u_sum / NUM_FEATURES;
        v_des_center_ = v_sum / NUM_FEATURES;
        goal_initialized_ = true;
        v_prev_x_ = 0; v_prev_y_ = 0; v_prev_z_ = 0; w_prev_z_ = 0;
    }

    void featureCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
        if (!goal_initialized_ || msg->data.size() < 8) return;
        
        rclcpp::Time now = this->get_clock()->now();
        double dt = std::min(0.1, (now - last_time_).seconds());
        last_time_ = now;
        if (dt <= 0.0001) return; 

        double u_shift_sum = 0, v_shift_sum = 0;
        
        for (int i = 0; i < NUM_FEATURES; ++i) {
            double u = msg->data[i*2]; double v = msg->data[i*2+1];
            u_shift_sum += u; v_shift_sum += v;
            
            s_curr_[i].set_x((u - cam_.get_u0()) / cam_.get_px());
            s_curr_[i].set_y((v - cam_.get_v0()) / cam_.get_py());
            s_curr_[i].set_Z(current_depth_z_); 
        }
        double shift_centroid_u = u_shift_sum / NUM_FEATURES;
        double shift_centroid_v = v_shift_sum / NUM_FEATURES;

        vpColVector v_visp(6, 0.0);
        try { v_visp = servo_.computeControlLaw(); } catch(...) {}

        double err_u = u_des_center_ - shift_centroid_u;
        double err_v = v_des_center_ - shift_centroid_v;
        double err_norm = std::sqrt(err_u*err_u + err_v*err_v);
        
        sum_err_u_ += err_u * dt; sum_err_v_ += err_v * dt;
        double d_err_u = (err_u - prev_err_u_) / dt;
        double d_err_v = (err_v - prev_err_v_) / dt;
        prev_err_u_ = err_u; prev_err_v_ = err_v;

        double kp = this->get_parameter("gain_xy").as_double();
        double ki = this->get_parameter("gain_i_xy").as_double();
        double kd = this->get_parameter("gain_d_xy").as_double();
        double v_target_x = (-err_u * kp) + (-sum_err_u_ * ki) + (-d_err_u * kd);
        double v_target_y = (-err_v * kp) + (-sum_err_v_ * ki) + (-d_err_v * kd);

        double area_curr = calculateArea();
        double area_des = calculateDesiredArea();
        double area_err = (area_des - area_curr) / area_des;
        
        double l_z_0 = this->get_parameter("lambda_z_0").as_double();
        double l_z_inf = this->get_parameter("lambda_z_inf").as_double();
        double b_z = this->get_parameter("beta_z").as_double();
        double lambda_z = (l_z_0 - l_z_inf) * std::abs(area_err) * std::exp(-b_z * std::abs(area_err)) + l_z_inf;
        double v_target_z = area_err * lambda_z;

        double w_target_z = v_visp[5] * this->get_parameter("gain_yaw").as_double();

        auto ramp = [](double target, double prev, double acc, double dt) {
            double step = acc * dt;
            return prev + std::max(std::min(target - prev, step), -step);
        };
        double v_out_x = ramp(v_target_x, v_prev_x_, this->get_parameter("max_linear_accel").as_double(), dt);
        double v_out_y = ramp(v_target_y, v_prev_y_, this->get_parameter("max_linear_accel").as_double(), dt);
        double v_out_z = ramp(v_target_z, v_prev_z_, this->get_parameter("max_linear_accel").as_double(), dt);
        double w_out_z = ramp(w_target_z, w_prev_z_, this->get_parameter("max_angular_accel").as_double(), dt);

        double max_lin = this->get_parameter("max_linear_vel").as_double();
        double max_ang = this->get_parameter("max_angular_vel").as_double();
        auto clamp = [](double v, double lim) { return std::max(std::min(v, lim), -lim); };
        
        v_out_x = clamp(v_out_x, max_lin);
        v_out_y = clamp(v_out_y, max_lin);
        v_out_z = clamp(v_out_z, max_lin);
        w_out_z = clamp(w_out_z, max_ang);

        v_prev_x_ = v_out_x; v_prev_y_ = v_out_y; v_prev_z_ = v_out_z; w_prev_z_ = w_out_z;

        auto tw = std::make_unique<geometry_msgs::msg::TwistStamped>();
        tw->header.stamp = now;
        tw->header.frame_id = cam_frame_id_; 
        tw->twist.linear.x = v_out_x; tw->twist.linear.y = v_out_y; tw->twist.linear.z = v_out_z; 
        tw->twist.angular.z = w_out_z;

        velocity_pub_->publish(std::move(tw));
        
        std_msgs::msg::Float64 err_msg; err_msg.data = err_norm;
        error_pub_->publish(err_msg);

        // --- DEBUG PRINTS (RESTORED) ---
        // Prints status every 0.5 seconds to the terminal
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
            "ERR: %.2f | AreaErr: %.3f | Vz_cmd: %.3f |  Vx_cmd: %.3f |  Vy_cmd: %.3f | Depth: %.3f" , 
            err_norm, area_err, v_out_z, v_out_x, v_out_y, current_depth_z_);

        if (log_file_.is_open()) {
            log_file_ << (now - start_time_).seconds() << "," << err_norm << "," 
                      << v_out_x << "," << v_out_y << "," << v_out_z << "," << w_out_z << "\n";
        }
    }

    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr feature_sub_, desired_feature_sub_; 
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr depth_sub_; 
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr velocity_pub_; 
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr error_pub_;

    vpServo servo_;
    vpFeaturePoint s_curr_[NUM_FEATURES], s_star_[NUM_FEATURES]; 
    vpCameraParameters cam_;
    bool cam_initialized_, goal_initialized_; 
    std::string cam_frame_id_;
    double u_des_center_, v_des_center_, sum_err_u_, sum_err_v_, prev_err_u_, prev_err_v_, current_depth_z_;
    
    rclcpp::Time last_time_, start_time_;
    std::ofstream log_file_;
    double v_prev_x_, v_prev_y_, v_prev_z_, w_prev_z_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VispHybridAruco>());
    rclcpp::shutdown();
    return 0;
}