cmake_minimum_required(VERSION 3.15)
add_compile_options(-std=gnu++11)

project(BC)

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
set(CMAKE_CXX_STANDARD 11)

add_executable(BC main.cpp )
target_link_libraries(BC ${OpenCV_LIBS})