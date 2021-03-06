cmake_minimum_required(VERSION 2.8.3)
project(target_detection)

add_compile_options(-std=c++14)
#set(CMAKE_PREFIX_PATH "$CMAKE_PREFIX_PATH:/opt/libtorch/share/cmake/Torch")

set(EIGEN3_INCLUDE_DIR /usr/include/eigen3)
set(USE_CUDA 0)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package (Torch REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  message_generation std_msgs geometry_msgs nav_msgs
)

catkin_package(
  INCLUDE_DIRS ${TORCH_INCLUDE_DIRS}
  LIBRARIES sparcsnode systems_lib
)

include_directories (
	../../sparcslib/include
	/usr/include/eigen3
	eigen3
	${TORCH_INCLUDE_DIRS}
	${TORCH_INCLUDE_DIRS}/include
	${PROJECT_SOURCE_DIR}
	${Boost_INCLUDE_DIRS}
	${catkin_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIR}
)


add_executable (fake_target src/fake_target_node.cpp)
target_link_libraries (fake_target ${catkin_LIBRARIES} ${TORCH_LIBRARIES} sparcsnode systems_lib)
add_dependencies(fake_target ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
