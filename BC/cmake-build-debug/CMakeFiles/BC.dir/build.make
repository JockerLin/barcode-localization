# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/clion/88/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /snap/clion/88/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/pilcq/personal/PROJECT_CODE!/c_project/BC

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pilcq/personal/PROJECT_CODE!/c_project/BC/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/BC.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/BC.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/BC.dir/flags.make

CMakeFiles/BC.dir/main.cpp.o: CMakeFiles/BC.dir/flags.make
CMakeFiles/BC.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pilcq/personal/PROJECT_CODE!/c_project/BC/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/BC.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/BC.dir/main.cpp.o -c /home/pilcq/personal/PROJECT_CODE!/c_project/BC/main.cpp

CMakeFiles/BC.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/BC.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pilcq/personal/PROJECT_CODE!/c_project/BC/main.cpp > CMakeFiles/BC.dir/main.cpp.i

CMakeFiles/BC.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/BC.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pilcq/personal/PROJECT_CODE!/c_project/BC/main.cpp -o CMakeFiles/BC.dir/main.cpp.s

# Object files for target BC
BC_OBJECTS = \
"CMakeFiles/BC.dir/main.cpp.o"

# External object files for target BC
BC_EXTERNAL_OBJECTS =

BC: CMakeFiles/BC.dir/main.cpp.o
BC: CMakeFiles/BC.dir/build.make
BC: /usr/local/lib/libopencv_calib3d.a
BC: /usr/local/lib/libopencv_core.a
BC: /usr/local/lib/libopencv_dnn.a
BC: /usr/local/lib/libopencv_features2d.a
BC: /usr/local/lib/libopencv_flann.a
BC: /usr/local/lib/libopencv_highgui.a
BC: /usr/local/lib/libopencv_imgcodecs.a
BC: /usr/local/lib/libopencv_imgproc.a
BC: /usr/local/lib/libopencv_ml.a
BC: /usr/local/lib/libopencv_objdetect.a
BC: /usr/local/lib/libopencv_photo.a
BC: /usr/local/lib/libopencv_shape.a
BC: /usr/local/lib/libopencv_stitching.a
BC: /usr/local/lib/libopencv_superres.a
BC: /usr/local/lib/libopencv_video.a
BC: /usr/local/lib/libopencv_videoio.a
BC: /usr/local/lib/libopencv_videostab.a
BC: /usr/local/lib/libopencv_line_descriptor.a
BC: /usr/local/lib/libopencv_rgbd.a
BC: /usr/local/lib/libopencv_text.a
BC: /usr/local/lib/libopencv_photo.a
BC: /usr/local/lib/libopencv_video.a
BC: /usr/local/lib/libopencv_calib3d.a
BC: /usr/local/lib/libopencv_dnn.a
BC: /usr/local/share/OpenCV/3rdparty/lib/liblibprotobuf.a
BC: /usr/local/lib/libopencv_features2d.a
BC: /usr/local/lib/libopencv_flann.a
BC: /usr/local/lib/libopencv_highgui.a
BC: /usr/local/lib/libopencv_ml.a
BC: /usr/local/lib/libopencv_videoio.a
BC: /usr/local/lib/libopencv_imgcodecs.a
BC: /usr/local/share/OpenCV/3rdparty/lib/liblibjpeg.a
BC: /usr/lib/x86_64-linux-gnu/libwebp.so
BC: /usr/local/share/OpenCV/3rdparty/lib/liblibpng.a
BC: /usr/lib/x86_64-linux-gnu/libjasper.so
BC: /usr/lib/x86_64-linux-gnu/libImath.so
BC: /usr/lib/x86_64-linux-gnu/libIlmImf.so
BC: /usr/lib/x86_64-linux-gnu/libIex.so
BC: /usr/lib/x86_64-linux-gnu/libHalf.so
BC: /usr/lib/x86_64-linux-gnu/libIlmThread.so
BC: /usr/local/lib/libopencv_imgproc.a
BC: /usr/local/lib/libopencv_core.a
BC: /usr/lib/x86_64-linux-gnu/libz.so
BC: /usr/lib/liblapack.so
BC: /usr/lib/libcblas.so
BC: /usr/lib/libatlas.so
BC: /usr/local/share/OpenCV/3rdparty/lib/libittnotify.a
BC: /usr/local/share/OpenCV/3rdparty/lib/libippiw.a
BC: /usr/local/share/OpenCV/3rdparty/lib/libippicv.a
BC: CMakeFiles/BC.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/pilcq/personal/PROJECT_CODE!/c_project/BC/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable BC"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/BC.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/BC.dir/build: BC

.PHONY : CMakeFiles/BC.dir/build

CMakeFiles/BC.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/BC.dir/cmake_clean.cmake
.PHONY : CMakeFiles/BC.dir/clean

CMakeFiles/BC.dir/depend:
	cd /home/pilcq/personal/PROJECT_CODE!/c_project/BC/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pilcq/personal/PROJECT_CODE!/c_project/BC /home/pilcq/personal/PROJECT_CODE!/c_project/BC /home/pilcq/personal/PROJECT_CODE!/c_project/BC/cmake-build-debug /home/pilcq/personal/PROJECT_CODE!/c_project/BC/cmake-build-debug /home/pilcq/personal/PROJECT_CODE!/c_project/BC/cmake-build-debug/CMakeFiles/BC.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/BC.dir/depend

