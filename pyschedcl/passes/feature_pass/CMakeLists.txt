cmake_minimum_required(VERSION 2.8.8)

project("coarsening_pass")
set(CMAKE_C_COMPILER "gcc-4.8")
set(CMAKE_CXX_COMPILER "g++-4.8")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
add_definitions("-std=c++11")

# Directories.
set(INSTALL_LIB_DIR "lib")

set(THRUD_DIR "thrud")



# LLVM.
# Set this to your LLVM cmake directory.
set(LLVM_DIR "/home/anirban/Tools/Compilers/llvm-analysis-tools/object/share/llvm/cmake/")


add_subdirectory(${THRUD_DIR})
