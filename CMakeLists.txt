cmake_minimum_required(VERSION 2.8.3)
project(napvig)

add_compile_options(-std=c++14)
list(APPEND CMAKE_PREFIX_PATH ":/opt/libtorch")

set(EIGEN3_INCLUDE_DIR /usr/include/eigen3)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Torch REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  message_generation std_msgs geometry_msgs nav_msgs
)

add_message_files (
	FILES
	SearchHistory.msg
)

generate_messages (DEPENDENCIES std_msgs geometry_msgs nav_msgs)

catkin_package(
  INCLUDE_DIRS include ${TORCH_INCLUDE_DIRS}
  LIBRARIES sparcsnode systems_lib
)

include_directories (
	include
	../sparcslib/include
	/usr/include/eigen3
	eigen3
	${TORCH_INCLUDE_DIRS}
	${TORCH_INCLUDE_DIRS}/include
	${PROJECT_SOURCE_DIR}
	${Boost_INCLUDE_DIRS}
	${catkin_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIR}
)

add_executable (napvig src/napvig_node.cpp src/napvig.cpp src/napvig_map.cpp)
target_link_libraries (napvig ${catkin_LIBRARIES} ${TORCH_LIBRARIES} sparcsnode systems_lib)
add_dependencies(napvig ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable (unicycle_control src/unicycle_control.cpp)
target_link_libraries (unicycle_control ${catkin_LIBRARIES} ${TORCH_LIBRARIES} sparcsnode systems_lib)
add_dependencies(unicycle_control ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})


