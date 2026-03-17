#include <rclcpp/rclcpp.hpp>
#include "mapper_pkg/mapper_node.hpp"

int main(int argc, char **argv) {
    // 1. Initialize the ROS 2 network interfaces
    rclcpp::init(argc, argv);

    // 2. Create the MapperNode instance as a shared pointer
    auto node = std::make_shared<MapperNode>();

    // 3. Spin the node so it continuously listens to the depth and YOLO topics
    // This function blocks and will not return until the node is shut down (e.g., Ctrl+C)
    rclcpp::spin(node);

    // 4. Clean up resources and safely shut down the ROS 2 network
    rclcpp::shutdown();
    
    return 0;
}