# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/nvidia/.local/lib/python3.8/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/nvidia/.local/lib/python3.8/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/nvidia/3437-3762/yolov7-tiny-pose-trt-main/YoloLayer_TRT_v7.0

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/nvidia/3437-3762/yolov7-tiny-pose-trt-main/YoloLayer_TRT_v7.0/build

# Include any dependencies generated for this target.
include CMakeFiles/yolo.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/yolo.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/yolo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/yolo.dir/flags.make

CMakeFiles/yolo.dir/yolo_generated_yololayer.cu.o: /media/nvidia/3437-3762/yolov7-tiny-pose-trt-main/YoloLayer_TRT_v7.0/yololayer.cu
CMakeFiles/yolo.dir/yolo_generated_yololayer.cu.o: CMakeFiles/yolo.dir/yolo_generated_yololayer.cu.o.depend
CMakeFiles/yolo.dir/yolo_generated_yololayer.cu.o: CMakeFiles/yolo.dir/yolo_generated_yololayer.cu.o.Release.cmake
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/media/nvidia/3437-3762/yolov7-tiny-pose-trt-main/YoloLayer_TRT_v7.0/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/yolo.dir/yolo_generated_yololayer.cu.o"
	cd /media/nvidia/3437-3762/yolov7-tiny-pose-trt-main/YoloLayer_TRT_v7.0/build/CMakeFiles/yolo.dir && /home/nvidia/.local/lib/python3.8/site-packages/cmake/data/bin/cmake -E make_directory /media/nvidia/3437-3762/yolov7-tiny-pose-trt-main/YoloLayer_TRT_v7.0/build/CMakeFiles/yolo.dir//.
	cd /media/nvidia/3437-3762/yolov7-tiny-pose-trt-main/YoloLayer_TRT_v7.0/build/CMakeFiles/yolo.dir && /home/nvidia/.local/lib/python3.8/site-packages/cmake/data/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING=Release -D generated_file:STRING=/media/nvidia/3437-3762/yolov7-tiny-pose-trt-main/YoloLayer_TRT_v7.0/build/CMakeFiles/yolo.dir//./yolo_generated_yololayer.cu.o -D generated_cubin_file:STRING=/media/nvidia/3437-3762/yolov7-tiny-pose-trt-main/YoloLayer_TRT_v7.0/build/CMakeFiles/yolo.dir//./yolo_generated_yololayer.cu.o.cubin.txt -P /media/nvidia/3437-3762/yolov7-tiny-pose-trt-main/YoloLayer_TRT_v7.0/build/CMakeFiles/yolo.dir//yolo_generated_yololayer.cu.o.Release.cmake

# Object files for target yolo
yolo_OBJECTS =

# External object files for target yolo
yolo_EXTERNAL_OBJECTS = \
"/media/nvidia/3437-3762/yolov7-tiny-pose-trt-main/YoloLayer_TRT_v7.0/build/CMakeFiles/yolo.dir/yolo_generated_yololayer.cu.o"

libyolo.so: CMakeFiles/yolo.dir/yolo_generated_yololayer.cu.o
libyolo.so: CMakeFiles/yolo.dir/build.make
libyolo.so: /usr/local/cuda-11.4/lib64/libcudart.so
libyolo.so: /usr/local/cuda-11.4/lib64/libcudart.so
libyolo.so: CMakeFiles/yolo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/media/nvidia/3437-3762/yolov7-tiny-pose-trt-main/YoloLayer_TRT_v7.0/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libyolo.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/yolo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/yolo.dir/build: libyolo.so
.PHONY : CMakeFiles/yolo.dir/build

CMakeFiles/yolo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/yolo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/yolo.dir/clean

CMakeFiles/yolo.dir/depend: CMakeFiles/yolo.dir/yolo_generated_yololayer.cu.o
	cd /media/nvidia/3437-3762/yolov7-tiny-pose-trt-main/YoloLayer_TRT_v7.0/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/nvidia/3437-3762/yolov7-tiny-pose-trt-main/YoloLayer_TRT_v7.0 /media/nvidia/3437-3762/yolov7-tiny-pose-trt-main/YoloLayer_TRT_v7.0 /media/nvidia/3437-3762/yolov7-tiny-pose-trt-main/YoloLayer_TRT_v7.0/build /media/nvidia/3437-3762/yolov7-tiny-pose-trt-main/YoloLayer_TRT_v7.0/build /media/nvidia/3437-3762/yolov7-tiny-pose-trt-main/YoloLayer_TRT_v7.0/build/CMakeFiles/yolo.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/yolo.dir/depend

