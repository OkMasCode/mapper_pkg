cmake_minimum_required(VERSION 3.8)
project(mapper_pkg)
cmake_policy(SET CMP0074 NEW)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(yolo11_seg_interfaces REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)
# Eigen3 for fast matrix math (Bipartite Matching / Cosine Distance)
find_package(Eigen3 REQUIRED)
# PCL for 3D Geometry (Filtering, OBB, PointClouds)
find_package(PCL REQUIRED COMPONENTS common io filters features)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(mapper_node
  src/mapper_node.cpp
  src/semantic_object_map.cpp
  src/Hungarian.cpp
)

target_include_directories(mapper_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

ament_target_dependencies(mapper_node
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  message_filters
  tf2
  tf2_ros
  tf2_geometry_msgs
  cv_bridge
  pcl_conversions
  pcl_ros
  yolo11_seg_interfaces
)

target_link_libraries(mapper_node 
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
)

target_compile_definitions(mapper_node PUBLIC ${PCL_DEFINITIONS})

install(TARGETS mapper_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)


ament_package()
