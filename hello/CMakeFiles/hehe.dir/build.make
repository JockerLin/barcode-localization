# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

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
CMAKE_COMMAND = /snap/clion/81/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /snap/clion/81/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/pilcq/personal/PROJECT_CODE!/c_project/hello

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pilcq/personal/PROJECT_CODE!/c_project/hello

# Include any dependencies generated for this target.
include CMakeFiles/hehe.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/hehe.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/hehe.dir/flags.make

CMakeFiles/hehe.dir/main.cpp.o: CMakeFiles/hehe.dir/flags.make
CMakeFiles/hehe.dir/main.cpp.o: main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pilcq/personal/PROJECT_CODE!/c_project/hello/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/hehe.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hehe.dir/main.cpp.o -c /home/pilcq/personal/PROJECT_CODE!/c_project/hello/main.cpp

CMakeFiles/hehe.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hehe.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pilcq/personal/PROJECT_CODE!/c_project/hello/main.cpp > CMakeFiles/hehe.dir/main.cpp.i

CMakeFiles/hehe.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hehe.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pilcq/personal/PROJECT_CODE!/c_project/hello/main.cpp -o CMakeFiles/hehe.dir/main.cpp.s

# Object files for target hehe
hehe_OBJECTS = \
"CMakeFiles/hehe.dir/main.cpp.o"

# External object files for target hehe
hehe_EXTERNAL_OBJECTS =

hehe: CMakeFiles/hehe.dir/main.cpp.o
hehe: CMakeFiles/hehe.dir/build.make
hehe: /usr/local/lib/libopencv_ml.so.3.4.0
hehe: /usr/local/lib/libopencv_objdetect.so.3.4.0
hehe: /usr/local/lib/libopencv_shape.so.3.4.0
hehe: /usr/local/lib/libopencv_stitching.so.3.4.0
hehe: /usr/local/lib/libopencv_superres.so.3.4.0
hehe: /usr/local/lib/libopencv_videostab.so.3.4.0
hehe: /usr/local/lib/libopencv_calib3d.so.3.4.0
hehe: /usr/local/lib/libopencv_features2d.so.3.4.0
hehe: /usr/local/lib/libopencv_flann.so.3.4.0
hehe: /usr/local/lib/libopencv_highgui.so.3.4.0
hehe: /usr/local/lib/libopencv_photo.so.3.4.0
hehe: /usr/local/lib/libopencv_video.so.3.4.0
hehe: /usr/local/lib/libopencv_videoio.so.3.4.0
hehe: /usr/local/lib/libopencv_imgcodecs.so.3.4.0
hehe: /usr/local/lib/libopencv_imgproc.so.3.4.0
hehe: /usr/local/lib/libopencv_core.so.3.4.0
hehe: CMakeFiles/hehe.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/pilcq/personal/PROJECT_CODE!/c_project/hello/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable hehe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/hehe.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/hehe.dir/build: hehe

.PHONY : CMakeFiles/hehe.dir/build

CMakeFiles/hehe.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/hehe.dir/cmake_clean.cmake
.PHONY : CMakeFiles/hehe.dir/clean

CMakeFiles/hehe.dir/depend:
	cd /home/pilcq/personal/PROJECT_CODE!/c_project/hello && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pilcq/personal/PROJECT_CODE!/c_project/hello /home/pilcq/personal/PROJECT_CODE!/c_project/hello /home/pilcq/personal/PROJECT_CODE!/c_project/hello /home/pilcq/personal/PROJECT_CODE!/c_project/hello /home/pilcq/personal/PROJECT_CODE!/c_project/hello/CMakeFiles/hehe.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/hehe.dir/depend

