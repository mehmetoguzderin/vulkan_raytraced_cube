#	#
#	to do
#	#
# Test compilers other than MSVC
# Test operating systems other than Windows 10
#	#
#	license
#	#
# The MIT License (MIT)
#
# Copyright (c) 2017 Mehmet Oguz Derin <dev@mehmetoguzderin.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#	#
#	cmake versioning
#	#
cmake_minimum_required(VERSION 3.7 FATAL_ERROR)
cmake_policy(VERSION 3.7)
#	#
#	specify project
#	#
project(vulkan_raytraced_cube)
#	#
#	include vulkan
#	#
find_path(VULKAN_INCLUDE_DIR NAMES vulkan/vulkan.h HINTS
    "$ENV{VULKAN_SDK}/Include"
    "$ENV{VK_SDK_PATH}/Include")
if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    find_library(VULKAN_LIBRARY NAMES vulkan-1 HINTS
        "$ENV{VULKAN_SDK}/Lib"
        "$ENV{VULKAN_SDK}/Bin"
        "$ENV{VK_SDK_PATH}/Bin")
else()
    find_library(VULKAN_LIBRARY NAMES vulkan-1 HINTS
        "$ENV{VULKAN_SDK}/Lib32"
        "$ENV{VULKAN_SDK}/Bin32"
        "$ENV{VK_SDK_PATH}/Bin32")
endif()
include_directories(${VULKAN_INCLUDE_DIR})
#	#
#	include source and modules
#	#
include_directories(modules/glm)
include_directories(modules/stb)
include_directories(${PROJECT_SOURCE_DIR}/source)
#	#
#	include glfw
#	#
option(GLFW_BUILD_EXAMPLES "Build the GLFW example programs" OFF)
option(GLFW_BUILD_TESTS "Build the GLFW test programs" OFF)
option(GLFW_BUILD_DOCS "Build the GLFW documentation" OFF)
option(GLFW_INSTALL "Generate installation target" OFF)
add_subdirectory(modules/glfw)
#	#
#	preprocessor definitions
#	#
add_definitions(-std=c++14)
#	#
#	add and link executable
#	#
add_executable(
vulkan_raytraced_cube
${PROJECT_SOURCE_DIR}/source/cube.cpp)
set_target_properties(
vulkan_raytraced_cube
PROPERTIES
ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/lib
LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/lib
RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/bin
)
target_link_libraries(
vulkan_raytraced_cube
${VULKAN_LIBRARY}
glfw)
#	#
#	compile shaders
#	#
add_custom_command(
TARGET vulkan_raytraced_cube
PRE_BUILD
COMMAND glslangValidator ${PROJECT_SOURCE_DIR}/source/cube.vert -V -o ${PROJECT_SOURCE_DIR}/source/cube-vert.spv
WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
add_custom_command(
TARGET vulkan_raytraced_cube
PRE_BUILD
COMMAND glslangValidator ${PROJECT_SOURCE_DIR}/source/cube.frag -V -o ${PROJECT_SOURCE_DIR}/source/cube-frag.spv
WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
#	#
#	copy to build/bin
#	#
add_custom_command(
TARGET vulkan_raytraced_cube
PRE_BUILD
COMMAND ${CMAKE_COMMAND} -E copy
        ${PROJECT_SOURCE_DIR}/source/cube-vert.spv
        ${PROJECT_SOURCE_DIR}/build/bin/cube-vert.spv)
add_custom_command(
TARGET vulkan_raytraced_cube
PRE_BUILD
COMMAND ${CMAKE_COMMAND} -E copy
        ${PROJECT_SOURCE_DIR}/source/cube-frag.spv
        ${PROJECT_SOURCE_DIR}/build/bin/cube-frag.spv)
add_custom_command(
TARGET vulkan_raytraced_cube
PRE_BUILD
COMMAND ${CMAKE_COMMAND} -E copy
        ${PROJECT_SOURCE_DIR}/source/vulkan_raytraced_cube_texture.png
        ${PROJECT_SOURCE_DIR}/build/bin/vulkan_raytraced_cube_texture.png)
#	#
#	copy to build/bin/Debug
#	#
add_custom_command(
TARGET vulkan_raytraced_cube
PRE_BUILD
COMMAND ${CMAKE_COMMAND} -E copy
        ${PROJECT_SOURCE_DIR}/source/cube-vert.spv
        ${PROJECT_SOURCE_DIR}/build/bin/Debug/cube-vert.spv)
add_custom_command(
TARGET vulkan_raytraced_cube
PRE_BUILD
COMMAND ${CMAKE_COMMAND} -E copy
        ${PROJECT_SOURCE_DIR}/source/cube-frag.spv
        ${PROJECT_SOURCE_DIR}/build/bin/Debug/cube-frag.spv)
add_custom_command(
TARGET vulkan_raytraced_cube
PRE_BUILD
COMMAND ${CMAKE_COMMAND} -E copy
        ${PROJECT_SOURCE_DIR}/source/vulkan_raytraced_cube_texture.png
        ${PROJECT_SOURCE_DIR}/build/bin/Debug/vulkan_raytraced_cube_texture.png)
#	#
#	copy to build/bin/Release
#	#
add_custom_command(
TARGET vulkan_raytraced_cube
PRE_BUILD
COMMAND ${CMAKE_COMMAND} -E copy
        ${PROJECT_SOURCE_DIR}/source/cube-vert.spv
        ${PROJECT_SOURCE_DIR}/build/bin/Release/cube-vert.spv)
add_custom_command(
TARGET vulkan_raytraced_cube
PRE_BUILD
COMMAND ${CMAKE_COMMAND} -E copy
        ${PROJECT_SOURCE_DIR}/source/cube-frag.spv
        ${PROJECT_SOURCE_DIR}/build/bin/Release/cube-frag.spv)
add_custom_command(
TARGET vulkan_raytraced_cube
PRE_BUILD
COMMAND ${CMAKE_COMMAND} -E copy
        ${PROJECT_SOURCE_DIR}/source/vulkan_raytraced_cube_texture.png
        ${PROJECT_SOURCE_DIR}/build/bin/Release/vulkan_raytraced_cube_texture.png)