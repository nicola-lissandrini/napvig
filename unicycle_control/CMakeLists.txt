cmake_minimum_required(VERSION 3.0.2)
project(unicycle_control)

add_compile_options(-std=c++14)

set(EIGEN3_INCLUDE_DIR /usr/include/eigen3)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  message_generation std_msgs geometry_msgs nav_msgs
)

catkin_package(
  INCLUDE_DIRS
  LIBRARIES sparcsnode systems_lib
)

include_directories (
	../../sparcslib/include
	/usr/include/eigen3
	eigen3
	${PROJECT_SOURCE_DIR}
	${Boost_INCLUDE_DIRS}
	${catkin_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIR}
)

add_executable (unicycle_control src/unicycle_control.cpp)
target_link_libraries (unicycle_control ${catkin_LIBRARIES} ${TORCH_LIBRARIES} sparcsnode systems_lib)
add_dependencies(unicycle_control ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
