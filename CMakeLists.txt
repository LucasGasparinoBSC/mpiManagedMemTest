# Set minimum CMake version
cmake_minimum_required(VERSION 3.15)

# Set project name and languages
project(testACC LANGUAGES CXX)

# Set OpenACC option
option(USE_ACC "Use OpenACC" ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set compiler flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
	message("Compiling with NVIDIA HPC compiler")
	set(CMAKE_CXX_FLAGS "-cuda -Minfo=all -lnvToolsExt")
	set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
	set(CMAKE_CXX_FLAGS_RELEASE "-O3")
	if(USE_ACC)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gpu=cc86,managed,lineinfo -acc")

	endif()
endif()

# Set MPI
find_package(MPI REQUIRED)
if(NOT MPI_FOUND)
	message(FATAL_ERROR "MPI not found")
endif()

# Add executable
if(USE_ACC)
	add_executable(test_ACC mpiTest.cxx)
	add_executable(rma_ACC rmaTest.cxx)
else()
	add_executable(test_noACC mpiTest.cxx)
	add_executable(rma_noACC rmaTest.cxx)
endif()