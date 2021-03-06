cmake_minimum_required(VERSION 2.8.3)
project(napvig)

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

add_executable (napvig
	src/napvig_node_debugger.cpp
	src/napvig_handler.cpp
	src/napvig_node.cpp
	src/napvig.cpp
	src/landscape.cpp
	src/rotation.cpp
	src/implementations/napvig_legacy.cpp)
target_link_libraries (napvig ${catkin_LIBRARIES} ${TORCH_LIBRARIES} sparcsnode systems_lib)
add_dependencies(napvig ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

#add_executable (fake_target src/fake_target_node.cpp)
#target_link_libraries (fake_target ${catkin_LIBRARIES} ${TORCH_LIBRARIES} sparcsnode systems_lib)
#add_dependencies(fake_target ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

catkin_install_python(PROGRAMS scripts/display.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
