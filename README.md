# Vulkan Raytraced Cube
| Raytraced | Original |
| --- | --- |
| ![Raytraced Cube](https://github.com/OguzDerin/vulkan_raytraced_cube/blob/master/docs/img/raytraced_cube.png) | ![Original Cube](https://github.com/OguzDerin/vulkan_raytraced_cube/blob/master/docs/img/lunarg_cube.png) |

## Before Reading
I'd like to get your opinions and help on this project, it would be super cool if we can turn this into a referenceable code for Vulkan learners, users and a solid base for quick Vulkan projects and experimentations. So every contribution would be more than welcome! And as a side note, if there is a code source I forgot to mention, please tell me to add it.
## https://github.com/LunarG/VulkanSamples/blob/master/demos/cube.cpp
This project is based on this awesome Cube demo built by LunarG
## What's Different?
* Ray tracing
  * This project draws the cube using ray-tracing instead of rasterizing the mesh
* Camera transform
  * This project transforms camera instead of cube
* More modern C++ usage
  * This project tries to utilize modern C++ features more often
* Usage of well-established C++ libraries for cleaner code and cross-platform support
  * This project utilizes GLFW, glm, stb_image
* Automated cross-platform project generation and SPIR-V compilation
  * This project uses CMake to automate cross-platform project generation and SPIR-V compilation
* Different cube texture 😆 
  * A cool Vulkan labeled construction site texture

## How to Install
1. Clone repository recursively
  > `git clone --recursive https://github.com/OguzDerin/vulkan_raytraced_cube.git`
2. Use CMake to generate project files
  > If you are using Visual Studio 2017 and above, just right click the root folder and choose "Open in Visual Studio", it will handle the rest
3. Run!

## To Do List
* Reorder member functions
* Improve code consistency
* Improve naming conventions
* Remove redundant code
* Convert the class to an extendable class which can serve as a Vulkan base
* Move repeated parts of code into functions
* Improve modern C++ usage including exceptions
* Improve math calculations
* Improve data layouts
* Load cube and quad data from a glTF file using a solid (preferably) header-only and up-to-date glTF library
* Move GLSL shaders to HLSL (with a cross-platform automated pipeline attached to CMake)
* Reduce memory usage
* Improve shader intersection function
* Improve overall performance
