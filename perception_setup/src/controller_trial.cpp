#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp> 
#include <std_msgs/msg/float64.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp> 
#include <sensor_msgs/msg/camera_info.hpp> 

#include <visp3/visual_features/vpFeaturePoint.h>
#include <visp3/vs/vpServo.h>
#include <visp3/core/vpCameraParameters.h>

#define NODE_NAME "visp_ibvs_fixed_z"
#define NUM_FEATURES 4 

class VispIBVSFixedZ : public rclcpp::Node
{
public:
    VispIBVSFixedZ() : Node(NODE_NAME)
    {
        // --- TUNING GAINS ---
        // How fast to correct X/Y pixel error? (Meters/sec per pixel)
        // 0.0005 means 100px error -> 5cm/s speed
        this->declare_parameter("gain_xy", 0.001);
        
        // How fast to correct Z depth error?
        // 1.0 means 10cm error -> 10cm/s speed
        this->declare_parameter("gain_z", 0.2); 
        
        // Target Z Depth (Meters)
        this->declare_parameter("target_z", 0.2587);

        // ViSP Setup (Used only for Yaw calculation)
        servo_.setInteractionMatrixType(vpServo::MEAN);
        servo_.setServo(vpServo::EYEINHAND_CAMERA);

        // Parameters
        cam_initialized_ = false;
        cam_frame_id_ = "ar4_ee_link"; 
        
        // We assume 320, 240 is the center of the image (Perfect alignment)
        u_des_center_ = 320.0;
        v_des_center_ = 240.0;
        
        // --- DESIRED PIXEL COORDINATES ---
        // Coordinates for Yaw calculation (Desired Square)
        // UPDATED: Using exact coordinates from Python script to fix Z-height
        double des_px[8] = {
            265.0, 185.0, // Top-Left
            375.0, 185.0, // Top-Right
            375.0, 295.0, // Bottom-Right
            265.0, 295.0  // Bottom-Left
        };
        for(int i=0; i<8; i++) desired_pixels_[i] = des_px[i];

        // Subs/Pubs
        cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "cameraAR4/camera_info", 10, std::bind(&VispIBVSFixedZ::infoCallback, this, std::placeholders::_1));

        feature_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            "/feature_coordinates_6D", 10, std::bind(&VispIBVSFixedZ::featureCallback, this, std::placeholders::_1));

        velocity_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>(
            "/servo_server/delta_twist_cmds", 10);
            
        error_pub_ = this->create_publisher<std_msgs::msg::Float64>("/vs/error_norm", 10);

        RCLCPP_INFO(this->get_logger(), "Fixed-Z P-Controller Started.");
    }

private:
    void infoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        if (cam_initialized_) return; 
        cam_.initPersProjWithoutDistortion(msg->k[0], msg->k[4], msg->k[2], msg->k[5]);
        
        // Initialize ViSP features for Yaw calculation
        double Z_target = this->get_parameter("target_z").as_double();
        for (int i = 0; i < NUM_FEATURES; ++i) {
            double x_d = (desired_pixels_[i*2] - cam_.get_u0()) / cam_.get_px();
            double y_d = (desired_pixels_[i*2+1] - cam_.get_v0()) / cam_.get_py();
            s_star_[i].set_x(x_d); s_star_[i].set_y(y_d); s_star_[i].set_Z(Z_target); 
            s_curr_[i].set_x(x_d); s_curr_[i].set_y(y_d); s_curr_[i].set_Z(Z_target); 
            servo_.addFeature(s_curr_[i], s_star_[i]);
        }
        cam_initialized_ = true;
    }

    void featureCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
        if (!cam_initialized_ || msg->data.size() < NUM_FEATURES * 2) return;

        // --- 1. COMPUTE CURRENT CENTER (For X/Y) ---
        double u_sum = 0, v_sum = 0;
        
        // Also update ViSP for Yaw/Z calculation
        for (int i = 0; i < NUM_FEATURES; ++i) {
            double u_raw = msg->data[i*2];
            double v_raw = msg->data[i*2+1];
            u_sum += u_raw;
            v_sum += v_raw;
            
            // For ViSP (Yaw only)
            double x_n = (u_raw - cam_.get_u0()) / cam_.get_px();
            double y_n = (v_raw - cam_.get_v0()) / cam_.get_py();
            s_curr_[i].set_x(x_n); s_curr_[i].set_y(y_n); s_curr_[i].set_Z(0.3); // Z doesn't matter much for yaw
        }
        
        double u_curr = u_sum / 4.0;
        double v_curr = v_sum / 4.0;

        // --- 2. CONTROL LAW: X and Y (Pure P-Control) ---
        // We define: Right Pixel Error -> Move Right (+X)
        //            Down Pixel Error  -> Move Down  (+Y)
        // CHECK YOUR FRAME: If +X is Right and +Y is Down in your EE frame.
        double k_xy = this->get_parameter("gain_xy").as_double();
        
        double err_u = u_des_center_ - u_curr; // Positive if target is to the Right
        double err_v = v_des_center_ - v_curr; // Positive if target is Down
        
        // Note: Sign depends on mounting. If moving wrong way, flip signs here.
        // Assuming Camera X = Robot EE X, Camera Y = Robot EE Y
        double v_x = -1.0 * err_u * k_xy; 
        double v_y = -1.0 * err_v * k_xy; 

        // --- 3. CONTROL LAW: Z (Depth Hold) ---
        // Instead of ViSP, we use Area. Larger Area = Closer.
        // We want to maintain a specific "Reference Area".
        // BUT simpler: Use ViSP's "Z" output from interaction matrix, or just lock it if you have external depth.
        // Since you rely on monocular, let's trust ViSP for Z and Yaw only.
        
        vpColVector v_visp(6, 0.0);
        try { v_visp = servo_.computeControlLaw(); } catch(...) { v_visp = 0.0; }
        
        double gain_z = this->get_parameter("gain_z").as_double();
        double v_z = v_visp[2] * gain_z; // Use ViSP's depth convergence logic

        // --- 4. CONTROL LAW: Yaw (Rotation around Z) ---
        double w_x = 0.0; // Use ViSP to align orientation
        double w_y = 0.0; // Use ViSP to align orientation
        double w_z = v_visp[5]; // Use ViSP to align orientation

        // --- 6. LIMITS & PUBLISH ---
        auto velocity_msg = std::make_unique<geometry_msgs::msg::TwistStamped>();
        velocity_msg->header.stamp = this->get_clock()->now(); 
        velocity_msg->header.frame_id = cam_frame_id_; 

        auto clamp = [](double v, double max_v) { return std::max(std::min(v, max_v), -max_v); };

        velocity_msg->twist.linear.x = clamp(v_x, 0.1);
        velocity_msg->twist.linear.y = clamp(v_y, 0.1);
        velocity_msg->twist.linear.z = clamp(v_z, 0.1);
        velocity_msg->twist.angular.x = clamp(w_x, 0.2);
        velocity_msg->twist.angular.y = clamp(w_y, 0.2);
        velocity_msg->twist.angular.z = clamp(w_z, 0.2);

        velocity_pub_->publish(std::move(velocity_msg));
        
        // Debug
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500, 
            "Err U: %.1f V: %.1f | Cmd Vx: %.3f Vy: %.3f Vz: %.3f", 
            err_u, err_v, v_x, v_y, v_z);
            
        std_msgs::msg::Float64 err_msg;
        err_msg.data = sqrt(err_u*err_u + err_v*err_v);
        error_pub_->publish(err_msg);
    }

    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_; 
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr feature_sub_; 
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr velocity_pub_; 
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr error_pub_;
    
    vpServo servo_;
    vpFeaturePoint s_curr_[NUM_FEATURES], s_star_[NUM_FEATURES]; 
    vpCameraParameters cam_;
    bool cam_initialized_; 
    std::string cam_frame_id_;
    double u_des_center_, v_des_center_;
    double desired_pixels_[8];
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VispIBVSFixedZ>());
    rclcpp::shutdown();
    return 0;
}