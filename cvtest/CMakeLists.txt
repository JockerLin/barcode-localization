cmake_minimum_required(VERSION 3.5)
add_compile_options(-std=gnu++11)

project(cvtest)
find_package(OpenCV REQUIRED)
find_package(Open3D)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(run main.cpp )
target_link_libraries(run ${OpenCV_LIBS})