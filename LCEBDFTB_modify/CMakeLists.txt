cmake_minimum_required(VERSION 3.14)
add_compile_options(-std=gnu++11)
#SET(CMAKE_BUILD_TYPE Release)
SET(CMAKE_BUILD_TYPE Debug)
project(LCEBDFTB)

#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")

## Compiler flags
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-O3")        ## Optimize
    set(CMAKE_EXE_LINKER_FLAGS "-s")  ## Strip binary
endif()

find_package(OpenCV REQUIRED)
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

#find_package(MathGL REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

FILE(GLOB_RECURSE LibFiles "*.h")
add_custom_target(headers SOURCES ${LibFiles})

add_library(LCEBDFTB SHARED barcode_localization.cpp barcode_localization.h main.cpp)
#target_link_libraries(barcodelocalizer ${OpenCV_LIBS} ${MathGL_LIBRARIES} zbar)
target_link_libraries(LCEBDFTB ${OpenCV_LIBS})

# Install library
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_SOURCE_DIR}/lib/${PROJECT_NAME})

# Install library headers
file(GLOB HEADERS *.h)
install(FILES ${HEADERS} DESTINATION ${CMAKE_SOURCE_DIR}/include/${PROJECT_NAME})

add_executable(run main.cpp barcode_localization.cpp barcode_localization.h)
target_link_libraries(run ${OpenCV_LIBS} "/usr/lib/x86_64-linux-gnu/libzbar.so"
        "/home/pilcq/personal/PROJECT_CODE!/c_project/LCEBDFTB/lib/LCEBDFTB/libLCEBDFTB.so")