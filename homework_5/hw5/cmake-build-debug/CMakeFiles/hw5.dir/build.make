# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.20

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

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\JetBrains\CLion 2021.2.1\bin\cmake\win\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\JetBrains\CLion 2021.2.1\bin\cmake\win\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\kdex9\OneDrive\Documents\School\Fall_2021\advanced_computing\homework_5\hw5

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\kdex9\OneDrive\Documents\School\Fall_2021\advanced_computing\homework_5\hw5\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/hw5.dir/depend.make
# Include the progress variables for this target.
include CMakeFiles/hw5.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/hw5.dir/flags.make

CMakeFiles/hw5.dir/main.cpp.obj: CMakeFiles/hw5.dir/flags.make
CMakeFiles/hw5.dir/main.cpp.obj: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\kdex9\OneDrive\Documents\School\Fall_2021\advanced_computing\homework_5\hw5\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/hw5.dir/main.cpp.obj"
	C:\PROGRA~1\MINGW-~1\X86_64~1.0-P\mingw64\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\hw5.dir\main.cpp.obj -c C:\Users\kdex9\OneDrive\Documents\School\Fall_2021\advanced_computing\homework_5\hw5\main.cpp

CMakeFiles/hw5.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hw5.dir/main.cpp.i"
	C:\PROGRA~1\MINGW-~1\X86_64~1.0-P\mingw64\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\kdex9\OneDrive\Documents\School\Fall_2021\advanced_computing\homework_5\hw5\main.cpp > CMakeFiles\hw5.dir\main.cpp.i

CMakeFiles/hw5.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hw5.dir/main.cpp.s"
	C:\PROGRA~1\MINGW-~1\X86_64~1.0-P\mingw64\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\kdex9\OneDrive\Documents\School\Fall_2021\advanced_computing\homework_5\hw5\main.cpp -o CMakeFiles\hw5.dir\main.cpp.s

# Object files for target hw5
hw5_OBJECTS = \
"CMakeFiles/hw5.dir/main.cpp.obj"

# External object files for target hw5
hw5_EXTERNAL_OBJECTS =

hw5.exe: CMakeFiles/hw5.dir/main.cpp.obj
hw5.exe: CMakeFiles/hw5.dir/build.make
hw5.exe: CMakeFiles/hw5.dir/linklibs.rsp
hw5.exe: CMakeFiles/hw5.dir/objects1.rsp
hw5.exe: CMakeFiles/hw5.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\kdex9\OneDrive\Documents\School\Fall_2021\advanced_computing\homework_5\hw5\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable hw5.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\hw5.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/hw5.dir/build: hw5.exe
.PHONY : CMakeFiles/hw5.dir/build

CMakeFiles/hw5.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\hw5.dir\cmake_clean.cmake
.PHONY : CMakeFiles/hw5.dir/clean

CMakeFiles/hw5.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\kdex9\OneDrive\Documents\School\Fall_2021\advanced_computing\homework_5\hw5 C:\Users\kdex9\OneDrive\Documents\School\Fall_2021\advanced_computing\homework_5\hw5 C:\Users\kdex9\OneDrive\Documents\School\Fall_2021\advanced_computing\homework_5\hw5\cmake-build-debug C:\Users\kdex9\OneDrive\Documents\School\Fall_2021\advanced_computing\homework_5\hw5\cmake-build-debug C:\Users\kdex9\OneDrive\Documents\School\Fall_2021\advanced_computing\homework_5\hw5\cmake-build-debug\CMakeFiles\hw5.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/hw5.dir/depend

