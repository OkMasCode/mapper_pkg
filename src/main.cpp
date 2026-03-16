#include <rclcpp/rclcpp.hpp>

// We will include your ROS node header here once we write it
// #include "semantic_mapper_cpp/mapper_node.hpp" 

int main(int argc, char **argv) {
    // 1. Initialize the ROS 2 network
    rclcpp::init(argc, argv);

    // 2. Create the executor and the node (Commented out until we write mapper_node.hpp)
    // auto node = std::make_shared<semantic_mapper::MapperNode>();
    // rclcpp::spin(node);

    // 3. Clean shutdown
    rclcpp::shutdown();
    return 0;
}