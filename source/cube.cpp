// To Do
/*
 * Improve code consistency
 * Improve naming convention
 * Remove redundant code
 * Move repeated parts of code to functions
 * Improve exception support
 * Fix unnecessary memory usage
 * Improve data layouts
 * Load cube and plane data from glTF file using a solid (preferably) header-only and up-to-date glTF library
 */
/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2017 Mehmet Oguz Derin <dev@mehmetoguzderin.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
/*
 * Copyright (c) 2015-2016 The Khronos Group Inc.
 * Copyright (c) 2015-2016 Valve Corporation
 * Copyright (c) 2015-2016 LunarG, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Jeremy Hayes <jeremy@lunarg.com>
 */

#include <fstream>
#include <iostream>
#include <iterator>

#include <vulkan/vulkan.hpp>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace vulkan_raytraced_cube_demo {
class vulkan_raytraced_cube {
public:
private:
#define ARRAY_SIZE(a) (sizeof(a) / sizeof(a[0]))
#define FRAME_LAG 2

  struct texture_object {
    vk::Sampler sampler;

    vk::Image image;
    vk::ImageLayout imageLayout{vk::ImageLayout::eUndefined};

    vk::MemoryAllocateInfo mem_alloc;
    vk::DeviceMemory mem;
    vk::ImageView view;

    int32_t tex_width{0};
    int32_t tex_height{0};
  };

  struct vktexcube_vs_uniform {
    /*
     * Data Layout
     * eye[0].xyz = eye left
     * eye[1].xyz = eye right
     * eye[2].xyz = eye down
     * eye[3].xyz = eye up
     * eyeOrigin = eye[0][3], eye[1][3], eye[2][3]
     */
    glm::mat4 eye;
    glm::vec4 pos_uv[6];
    glm::vec4 screen_size;
  };

  struct vktexcube_triangle_storage {
    /*
     * Data Layout
     * pos_uv[0] = vertex position, u[0]
     * pos_uv[1] = vertex position, v[0]
     * pos_uv[2] = vertex position, -
     * pos_uv[3] = u[1], v[1], u[2], v[2]
     */
    glm::vec4 pos_uv[4];
  };

  typedef struct {
    vk::Image image;
    vk::CommandBuffer cmd;
    vk::CommandBuffer graphics_to_present_cmd;
    vk::ImageView view;
    vk::Buffer uniform_buffer;
    vk::DeviceMemory uniform_memory;
    vk::Buffer storage_buffer;
    vk::DeviceMemory storage_memory;
    vk::Framebuffer framebuffer;
    vk::DescriptorSet descriptor_set;
  } SwapchainImageResources;

private:
  std::string app_name;

  uint32_t width, height;

  const std::vector<std::string> tex_files = {
      "vulkan_raytraced_cube_texture.png"};

  const std::vector<glm::vec4> g_vertex_buffer_data = {
      glm::vec4(-1.0f, 1.0f, 0.0f, 1.0f),  glm::vec4(1.0f, -1.0f, 1.0f, 0.0f),
      glm::vec4(-1.0f, -1.0f, 0.0f, 0.0f), glm::vec4(-1.0f, 1.0f, 0.0f, 1.0f),
      glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),   glm::vec4(1.0f, -1.0f, 1.0f, 0.0f)};

  const std::vector<vktexcube_triangle_storage> g_cube_buffer_data = {
      {glm::vec4(-1.0f, -1.0f, -1.0f, 1.0f), // 0
       glm::vec4(-1.0f, -1.0f, 1.0f, 1.0f), glm::vec4(-1.0f, 1.0f, 1.0f, 0.0f),
       glm::vec4(0.0f, 1.0f, 0.0f, 0.0f)},

      {glm::vec4(-1.0f, 1.0f, 1.0f, 0.0f), // 1
       glm::vec4(-1.0f, 1.0f, -1.0f, 0.0f),
       glm::vec4(-1.0f, -1.0f, -1.0f, 0.0f), glm::vec4(1.0f, 0.0f, 1.0f, 1.0f)},

      {glm::vec4(-1.0f, -1.0f, -1.0f, 0.0f), // 2
       glm::vec4(1.0f, 1.0f, -1.0f, 1.0f), glm::vec4(1.0f, -1.0f, -1.0f, 0.0f),
       glm::vec4(1.0f, 0.0f, 1.0f, 1.0f)},

      {glm::vec4(-1.0f, -1.0f, -1.0f, 0.0f), // 3
       glm::vec4(-1.0f, 1.0f, -1.0f, 1.0f), glm::vec4(1.0f, 1.0f, -1.0f, 0.0f),
       glm::vec4(0.0f, 0.0f, 1.0f, 0.0f)},

      {glm::vec4(-1.0f, -1.0f, -1.0f, 0.0f), // 4
       glm::vec4(1.0f, -1.0f, -1.0f, 0.0f), glm::vec4(1.0f, -1.0f, 1.0f, 0.0f),
       glm::vec4(0.0f, 1.0f, 1.0f, 1.0f)},

      {glm::vec4(-1.0f, -1.0f, -1.0f, 0.0f), // 5
       glm::vec4(1.0f, -1.0f, 1.0f, 0.0f), glm::vec4(-1.0f, -1.0f, 1.0f, 0.0f),
       glm::vec4(1.0f, 1.0f, 1.0f, 0.0f)},

      {glm::vec4(-1.0f, 1.0f, -1.0f, 0.0f), // 6
       glm::vec4(-1.0f, 1.0f, 1.0f, 0.0f), glm::vec4(1.0f, 1.0f, 1.0f, 0.0f),
       glm::vec4(1.0f, 0.0f, 1.0f, 1.0f)},

      {glm::vec4(-1.0f, 1.0f, -1.0f, 0.0f), // 7
       glm::vec4(1.0f, 1.0f, 1.0f, 0.0f), glm::vec4(1.0f, 1.0f, -1.0f, 0.0f),
       glm::vec4(1.0f, 1.0f, 0.0f, 1.0f)},

      {glm::vec4(1.0f, 1.0f, -1.0f, 0.0f), // 8
       glm::vec4(1.0f, 1.0f, 1.0f, 0.0f), glm::vec4(1.0f, -1.0f, 1.0f, 0.0f),
       glm::vec4(1.0f, 0.0f, 1.0f, 1.0f)},

      {glm::vec4(1.0f, -1.0f, 1.0f, 1.0f), // 9
       glm::vec4(1.0f, -1.0f, -1.0f, 1.0f), glm::vec4(1.0f, 1.0f, -1.0f, 0.0f),
       glm::vec4(0.0f, 1.0f, 0.0f, 0.0f)},

      {glm::vec4(-1.0f, 1.0f, 1.0f, 1.0f), // 10
       glm::vec4(-1.0f, -1.0f, 1.0f, 0.0f), glm::vec4(1.0f, 1.0f, 1.0f, 0.0f),
       glm::vec4(1.0f, 1.0f, 0.0f, 0.0f)},

      {glm::vec4(-1.0f, -1.0f, 1.0f, 1.0f), // 11
       glm::vec4(1.0f, -1.0f, 1.0f, 1.0f), glm::vec4(1.0f, 1.0f, 1.0f, 0.0f),
       glm::vec4(0.0f, 1.0f, 0.0f, 0.0f)}};

  std::unique_ptr<GLFWwindow, void (*)(GLFWwindow *)> window;
  vk::SurfaceKHR surface;
  bool prepared;
  bool use_staging_buffer;
  bool separate_present_queue;

  vk::Instance inst;
  vk::PhysicalDevice gpu;
  vk::Device device;
  vk::Queue graphics_queue;
  vk::Queue present_queue;
  uint32_t graphics_queue_family_index;
  uint32_t present_queue_family_index;
  vk::Semaphore image_acquired_semaphores[FRAME_LAG];
  vk::Semaphore draw_complete_semaphores[FRAME_LAG];
  vk::Semaphore image_ownership_semaphores[FRAME_LAG];
  vk::PhysicalDeviceProperties gpu_props;
  std::vector<vk::QueueFamilyProperties> queue_props;
  vk::PhysicalDeviceMemoryProperties memory_properties;

  uint32_t enabled_extension_count;
  uint32_t enabled_layer_count;
  char const *extension_names[64];
  char const *enabled_layers[64];

  vk::Format format;
  vk::ColorSpaceKHR color_space;

  uint32_t swapchainImageCount;
  vk::SwapchainKHR swapchain;
  std::vector<SwapchainImageResources> swapchain_image_resources;
  vk::PresentModeKHR presentMode;
  vk::Fence fences[FRAME_LAG];
  uint32_t frame_index;

  vk::CommandPool cmd_pool;
  vk::CommandPool present_cmd_pool;

  struct {
    vk::Format format;
    vk::Image image;
    vk::MemoryAllocateInfo mem_alloc;
    vk::DeviceMemory mem;
    vk::ImageView view;
  } depth;

  static int32_t const texture_count = 1;
  texture_object textures[texture_count];
  texture_object staging_texture;

  vk::CommandBuffer cmd; // Buffer for initialization commands
  vk::PipelineLayout pipeline_layout;
  vk::DescriptorSetLayout desc_layout;
  vk::PipelineCache pipelineCache;
  vk::RenderPass render_pass;
  vk::Pipeline pipeline;

  glm::mat4 model_matrix;
  glm::mat4 eye_matrix;
  glm::vec3 eye;
  glm::vec3 origin;
  glm::vec3 up;
  glm::vec3 f;
  glm::vec3 s;
  glm::vec3 u;

  float spin_angle;
  float spin_increment;
  bool pause;

  vk::ShaderModule vert_shader_module;
  vk::ShaderModule frag_shader_module;

  vk::DescriptorPool desc_pool;

  bool quit;
  bool callback_exception;
  std::exception_ptr callback_exception_ptr;
  uint32_t curFrame;
  uint32_t frameCount;
  bool validate;
  bool use_break;
  bool suppress_popups;

  uint32_t current_buffer;

public:
  vulkan_raytraced_cube(std::string _app_name, uint32_t _width,
                        uint32_t _height) try
      : window(nullptr, glfwDestroyWindow),
        prepared(false),
        use_staging_buffer(false),
        graphics_queue_family_index(0),
        present_queue_family_index(0),
        enabled_extension_count(0),
        enabled_layer_count(0),
        width(0),
        height(0),
        eye(0.0f, 3.0f, 5.0f),
        origin(0.0f, 0.0f, 0.0f),
        up(0.0f, -1.0f, 0.0f),
        f(glm::normalize(origin - eye)),
        s(glm::normalize(cross(f, up))),
        u(glm::cross(s, f)),
        swapchainImageCount(0),
        frame_index(0),
        spin_angle(0.0f),
        spin_increment(0.0f),
        pause(false),
        quit(false),
        callback_exception(false),
        callback_exception_ptr(nullptr),
        curFrame(0),
        frameCount(0),
        validate(false),
        use_break(false),
        suppress_popups(false),
        current_buffer(0) {
    app_name = _app_name;
    width = _width;
    height = _height;

    init();
    create_window();
    init_vk_swapchain();
    prepare();

    run();

    cleanup();
  } catch (...) {
    throw;
  };

  ~vulkan_raytraced_cube(){};

private:
  static void refresh_callback(GLFWwindow *window) try {
    vulkan_raytraced_cube *cube =
        (vulkan_raytraced_cube *)glfwGetWindowUserPointer(window);
    cube->draw();
  } catch (...) {
    vulkan_raytraced_cube *cube =
        (vulkan_raytraced_cube *)glfwGetWindowUserPointer(window);
    cube->callback_exception_ptr = std::current_exception();
    cube->callback_exception = true;
  };

  static void resize_callback(GLFWwindow *window, int width, int height) try {
    vulkan_raytraced_cube *cube =
        (vulkan_raytraced_cube *)glfwGetWindowUserPointer(window);
    cube->width = (uint32_t)width;
    cube->height = (uint32_t)height;
    cube->resize();
  } catch (...) {
    vulkan_raytraced_cube *cube =
        (vulkan_raytraced_cube *)glfwGetWindowUserPointer(window);
    cube->callback_exception_ptr = std::current_exception();
    cube->callback_exception = true;
  };

  static void key_callback(GLFWwindow *window, int key, int scancode,
                           int action, int mods) try {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE)
      glfwSetWindowShouldClose(window, GLFW_TRUE);
  } catch (...) {
    vulkan_raytraced_cube *cube =
        (vulkan_raytraced_cube *)glfwGetWindowUserPointer(window);
    cube->callback_exception_ptr = std::current_exception();
    cube->callback_exception = true;
  };

  void init() try {
    presentMode = vk::PresentModeKHR::eFifo;
    frameCount = UINT32_MAX;

    init_connection();

    init_vk();

    spin_angle = 4.0f;
    spin_increment = 0.2f;
    pause = false;

    model_matrix = glm::mat4(1.0f);
    calculate_eye_matrix();
  } catch (...) {
    throw;
  };

  void calculate_eye_matrix() try {
    float tana = tanf(glm::radians(45.0f));
    float aspectRatio = float(width) / float(height);

    glm::vec4 rotated_eye = model_matrix * glm::vec4(eye, 1.0f);
    glm::vec3 rotated_f = glm::vec3(model_matrix * glm::vec4(f, 1.0f));
    glm::vec3 rotated_s =
        glm::vec3(model_matrix * glm::vec4(s, 1.0f)) * tana * aspectRatio;
    glm::vec3 rotated_u = glm::vec3(model_matrix * glm::vec4(u, 1.0f)) * tana;

    glm::vec3 eye_left = rotated_f + rotated_s * -1.0f;
    glm::vec3 eye_right = eye_left + rotated_s * 2.0f;
    glm::vec3 eye_down = rotated_f + rotated_u * -1.0f;
    glm::vec3 eye_up = eye_down + rotated_u * 2.0f;

    eye_matrix = glm::mat4(1.0f);
    eye_matrix[0] =
        glm::vec4(eye_left[0], eye_left[1], eye_left[2], rotated_eye[0]);
    eye_matrix[1] =
        glm::vec4(eye_right[0], eye_right[1], eye_right[2], rotated_eye[1]);
    eye_matrix[2] =
        glm::vec4(eye_down[0], eye_down[1], eye_down[2], rotated_eye[2]);
    eye_matrix[3] = glm::vec4(eye_up[0], eye_up[1], eye_up[2], rotated_eye[3]);
  } catch (...) {
    throw;
  };

  void create_window() try {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    window.reset(glfwCreateWindow((int)width, (int)height, app_name.c_str(),
                                  NULL, NULL));

    glfwSetWindowUserPointer(window.get(), this);
    glfwSetWindowRefreshCallback(window.get(),
                                 &vulkan_raytraced_cube::refresh_callback);
    glfwSetFramebufferSizeCallback(window.get(),
                                   &vulkan_raytraced_cube::resize_callback);
    glfwSetKeyCallback(window.get(), &vulkan_raytraced_cube::key_callback);
  } catch (...) {
    throw;
  };

  void init_vk_swapchain() try {
    // Create a WSI surface for the window:
    VkSurfaceKHR glfwSurface;
    glfwCreateWindowSurface(static_cast<VkInstance>(inst), window.get(), NULL,
                            &glfwSurface);
    surface = vk::SurfaceKHR(glfwSurface);

    // Iterate over each queue to learn whether it supports presenting:
    std::vector<vk::Bool32> supportsPresent;
    for (uint32_t i = 0; i < queue_props.size(); i++) {
      supportsPresent.push_back(gpu.getSurfaceSupportKHR(i, surface));
    }

    uint32_t graphicsQueueFamilyIndex = UINT32_MAX;
    uint32_t presentQueueFamilyIndex = UINT32_MAX;
    for (uint32_t i = 0; i < queue_props.size(); i++) {
      if (queue_props[i].queueFlags & vk::QueueFlagBits::eGraphics) {
        if (graphicsQueueFamilyIndex == UINT32_MAX) {
          graphicsQueueFamilyIndex = i;
        }

        if (supportsPresent[i] == VK_TRUE) {
          graphicsQueueFamilyIndex = i;
          presentQueueFamilyIndex = i;
          break;
        }
      }
    }

    if (presentQueueFamilyIndex == UINT32_MAX) {
      // If didn't find a queue that supports both graphics and present,
      // then
      // find a separate present queue.
      for (uint32_t i = 0; i < queue_props.size(); ++i) {
        if (supportsPresent[i] == VK_TRUE) {
          presentQueueFamilyIndex = i;
          break;
        }
      }
    }

    // Generate error if could not find both a graphics and a present queue
    if (graphicsQueueFamilyIndex == UINT32_MAX ||
        presentQueueFamilyIndex == UINT32_MAX) {
      throw;
    }

    graphics_queue_family_index = graphicsQueueFamilyIndex;
    present_queue_family_index = presentQueueFamilyIndex;
    separate_present_queue =
        (graphics_queue_family_index != present_queue_family_index);

    create_device();

    device.getQueue(graphics_queue_family_index, 0, &graphics_queue);
    if (!separate_present_queue) {
      present_queue = graphics_queue;
    } else {
      device.getQueue(present_queue_family_index, 0, &present_queue);
    }

    // Get the list of VkFormat's that are supported:
    std::vector<vk::SurfaceFormatKHR> surfFormats =
        gpu.getSurfaceFormatsKHR(surface);

    // If the format list includes just one entry of VK_FORMAT_UNDEFINED,
    // the surface has no preferred format.  Otherwise, at least one
    // supported format will be returned.
    if (surfFormats.size() == 1 &&
        surfFormats[0].format == vk::Format::eUndefined) {
      format = vk::Format::eB8G8R8A8Unorm;
    } else {
      if (surfFormats.size() < 1) {
        throw;
      }
      format = surfFormats[0].format;
    }
    color_space = surfFormats[0].colorSpace;

    quit = false;
    curFrame = 0;

    // Create semaphores to synchronize acquiring presentable buffers before
    // rendering and waiting for drawing to be complete before presenting
    auto const semaphoreCreateInfo = vk::SemaphoreCreateInfo();

    // Create fences that we can use to throttle if we get too far
    // ahead of the image presents
    auto const fence_ci =
        vk::FenceCreateInfo().setFlags(vk::FenceCreateFlagBits::eSignaled);
    for (uint32_t i = 0; i < FRAME_LAG; i++) {
      fences[i] = device.createFence(fence_ci);

      image_acquired_semaphores[i] =
          device.createSemaphore(semaphoreCreateInfo);

      draw_complete_semaphores[i] = device.createSemaphore(semaphoreCreateInfo);

      if (separate_present_queue) {
        image_ownership_semaphores[i] =
            device.createSemaphore(semaphoreCreateInfo);
      }
    }
    frame_index = 0;

    // Get Memory information and properties
    memory_properties = gpu.getMemoryProperties();
  } catch (...) {
    throw;
  };

  void prepare() try {
    calculate_eye_matrix();

    auto const cmd_pool_info = vk::CommandPoolCreateInfo().setQueueFamilyIndex(
        graphics_queue_family_index);
    cmd_pool = device.createCommandPool(cmd_pool_info);

    auto const cmd_allocate_info =
        vk::CommandBufferAllocateInfo()
            .setCommandPool(cmd_pool)
            .setLevel(vk::CommandBufferLevel::ePrimary)
            .setCommandBufferCount(1);

    cmd = device.allocateCommandBuffers(cmd_allocate_info)[0];

    auto const cmd_buf_info =
        vk::CommandBufferBeginInfo().setPInheritanceInfo(nullptr);

    cmd.begin(cmd_buf_info);

    prepare_buffers();
    prepare_depth();
    prepare_textures();
    prepare_cube_data_buffers();

    prepare_descriptor_layout();
    prepare_render_pass();
    prepare_pipeline();

    for (uint32_t i = 0; i < swapchainImageCount; ++i) {
      swapchain_image_resources[i].cmd =
          device.allocateCommandBuffers(cmd_allocate_info)[0];
    }

    if (separate_present_queue) {
      auto const present_cmd_pool_info =
          vk::CommandPoolCreateInfo().setQueueFamilyIndex(
              present_queue_family_index);

      present_cmd_pool = device.createCommandPool(present_cmd_pool_info);

      auto const present_cmd = vk::CommandBufferAllocateInfo()
                                   .setCommandPool(present_cmd_pool)
                                   .setLevel(vk::CommandBufferLevel::ePrimary)
                                   .setCommandBufferCount(1);

      for (uint32_t i = 0; i < swapchainImageCount; i++) {
        swapchain_image_resources[i].graphics_to_present_cmd =
            device.allocateCommandBuffers(present_cmd)[0];

        build_image_ownership_cmd(i);
      }
    }

    prepare_descriptor_pool();
    prepare_descriptor_set();

    prepare_framebuffers();

    for (uint32_t i = 0; i < swapchainImageCount; ++i) {
      current_buffer = i;
      draw_build_cmd(swapchain_image_resources[i].cmd);
    }

    /*
     * Prepare functions above may generate pipeline commands
     * that need to be flushed before beginning the render loop.
     */
    flush_init_cmd();
    if (staging_texture.image) {
      destroy_texture_image(&staging_texture);
    }

    current_buffer = 0;
    prepared = true;
  } catch (...) {
    throw;
  };

  void run() try {
    while (!glfwWindowShouldClose(window.get())) {
      glfwPollEvents();

      if (callback_exception) {
        if (callback_exception_ptr) {
          std::rethrow_exception(callback_exception_ptr);
        }
      }

      draw();

      // Wait for work to finish before updating MVP.
      device.waitIdle();

      curFrame++;
      if (frameCount != INT32_MAX && curFrame == frameCount)
        glfwSetWindowShouldClose(window.get(), GLFW_TRUE);
    }
  } catch (...) {
    throw;
  };

  void cleanup() try {
    prepared = false;
    device.waitIdle();

    // Wait for fences from present operations
    for (uint32_t i = 0; i < FRAME_LAG; i++) {
      device.waitForFences(1, &fences[i], VK_TRUE, UINT64_MAX);
      device.destroyFence(fences[i]);
      device.destroySemaphore(image_acquired_semaphores[i]);
      device.destroySemaphore(draw_complete_semaphores[i]);
      if (separate_present_queue) {
        device.destroySemaphore(image_ownership_semaphores[i]);
      }
    }

    for (uint32_t i = 0; i < swapchainImageCount; i++) {
      device.destroyFramebuffer(swapchain_image_resources[i].framebuffer);
    }
    device.destroyDescriptorPool(desc_pool);

    device.destroyPipeline(pipeline);
    device.destroyPipelineCache(pipelineCache);
    device.destroyRenderPass(render_pass);
    device.destroyPipelineLayout(pipeline_layout);
    device.destroyDescriptorSetLayout(desc_layout);

    for (uint32_t i = 0; i < texture_count; i++) {
      device.destroyImageView(textures[i].view);
      device.destroyImage(textures[i].image);
      device.freeMemory(textures[i].mem);
      device.destroySampler(textures[i].sampler);
    }
    device.destroySwapchainKHR(swapchain);

    device.destroyImageView(depth.view);
    device.destroyImage(depth.image);
    device.freeMemory(depth.mem);

    for (uint32_t i = 0; i < swapchainImageCount; i++) {
      device.destroyImageView(swapchain_image_resources[i].view);
      device.freeCommandBuffers(cmd_pool, 1, &swapchain_image_resources[i].cmd);
      device.destroyBuffer(swapchain_image_resources[i].uniform_buffer);
      device.freeMemory(swapchain_image_resources[i].uniform_memory);
      device.destroyBuffer(swapchain_image_resources[i].storage_buffer);
      device.freeMemory(swapchain_image_resources[i].storage_memory);
    }

    device.destroyCommandPool(cmd_pool);

    if (separate_present_queue) {
      device.destroyCommandPool(present_cmd_pool);
    }
    device.waitIdle();
    device.destroy();
    inst.destroySurfaceKHR(surface);

    glfwDestroyWindow(window.get());
    glfwTerminate();

    inst.destroy();
  } catch (...) {
    throw;
  };

  void resize() try {
    uint32_t i;

    // Don't react to resize until after first initialization.
    if (!prepared) {
      return;
    }

    // In order to properly resize the window, we must re-create the
    // swapchain
    // AND redo the command buffers, etc.
    //
    // First, perform part of the cleanup() function:
    prepared = false;
    device.waitIdle();

    for (i = 0; i < swapchainImageCount; i++) {
      device.destroyFramebuffer(swapchain_image_resources[i].framebuffer);
    }

    device.destroyDescriptorPool(desc_pool);

    device.destroyPipeline(pipeline);
    device.destroyPipelineCache(pipelineCache);
    device.destroyRenderPass(render_pass);
    device.destroyPipelineLayout(pipeline_layout);
    device.destroyDescriptorSetLayout(desc_layout);

    for (i = 0; i < texture_count; i++) {
      device.destroyImageView(textures[i].view);
      device.destroyImage(textures[i].image);
      device.freeMemory(textures[i].mem);
      device.destroySampler(textures[i].sampler);
    }

    device.destroyImageView(depth.view);
    device.destroyImage(depth.image);
    device.freeMemory(depth.mem);

    for (i = 0; i < swapchainImageCount; i++) {
      device.destroyImageView(swapchain_image_resources[i].view);
      device.freeCommandBuffers(cmd_pool, 1, &swapchain_image_resources[i].cmd);
      device.destroyBuffer(swapchain_image_resources[i].uniform_buffer);
      device.freeMemory(swapchain_image_resources[i].uniform_memory);
      device.destroyBuffer(swapchain_image_resources[i].storage_buffer);
      device.freeMemory(swapchain_image_resources[i].storage_memory);
    }

    device.destroyCommandPool(cmd_pool);
    if (separate_present_queue) {
      device.destroyCommandPool(present_cmd_pool);
    }

    // Second, re-perform the prepare() function, which will re-create the
    // swapchain.
    prepare();
  } catch (...) {
    throw;
  };

  void draw() try {
    // Ensure no more than FRAME_LAG renderings are outstanding
    device.waitForFences(1, &fences[frame_index], VK_TRUE, UINT64_MAX);
    device.resetFences(1, &fences[frame_index]);

    vk::Result result;
    do {
      result = device.acquireNextImageKHR(
          swapchain, UINT64_MAX, image_acquired_semaphores[frame_index],
          vk::Fence(), &current_buffer);
      if (result == vk::Result::eErrorOutOfDateKHR) {
        // demo->swapchain is out of date (e.g. the window was resized) and
        // must be recreated:
        resize();
      } else if (result == vk::Result::eSuboptimalKHR) {
        // swapchain is not as optimal as it could be, but the platform's
        // presentation engine will still present the image correctly.
        break;
      } else {
        if (result != vk::Result::eSuccess) {
          throw;
        }
      }
    } while (result != vk::Result::eSuccess);

    update_data_buffer();

    // Wait for the image acquired semaphore to be signaled to ensure
    // that the image won't be rendered to until the presentation
    // engine has fully released ownership to the application, and it is
    // okay to render to the image.
    vk::PipelineStageFlags const pipe_stage_flags =
        vk::PipelineStageFlagBits::eColorAttachmentOutput;
    auto const submit_info =
        vk::SubmitInfo()
            .setPWaitDstStageMask(&pipe_stage_flags)
            .setWaitSemaphoreCount(1)
            .setPWaitSemaphores(&image_acquired_semaphores[frame_index])
            .setCommandBufferCount(1)
            .setPCommandBuffers(&swapchain_image_resources[current_buffer].cmd)
            .setSignalSemaphoreCount(1)
            .setPSignalSemaphores(&draw_complete_semaphores[frame_index]);

    graphics_queue.submit(submit_info, fences[frame_index]);

    if (separate_present_queue) {
      // If we are using separate queues, change image ownership to the
      // present queue before presenting, waiting for the draw complete
      // semaphore and signalling the ownership released semaphore when
      // finished
      auto const present_submit_info =
          vk::SubmitInfo()
              .setPWaitDstStageMask(&pipe_stage_flags)
              .setWaitSemaphoreCount(1)
              .setPWaitSemaphores(&draw_complete_semaphores[frame_index])
              .setCommandBufferCount(1)
              .setPCommandBuffers(&swapchain_image_resources[current_buffer]
                                       .graphics_to_present_cmd)
              .setSignalSemaphoreCount(1)
              .setPSignalSemaphores(&image_ownership_semaphores[frame_index]);

      present_queue.submit(present_submit_info, vk::Fence());
    }

    // If we are using separate queues we have to wait for image ownership,
    // otherwise wait for draw complete
    auto const presentInfo =
        vk::PresentInfoKHR()
            .setWaitSemaphoreCount(1)
            .setPWaitSemaphores(separate_present_queue
                                    ? &image_ownership_semaphores[frame_index]
                                    : &draw_complete_semaphores[frame_index])
            .setSwapchainCount(1)
            .setPSwapchains(&swapchain)
            .setPImageIndices(&current_buffer);

    result = present_queue.presentKHR(&presentInfo);
    frame_index += 1;
    frame_index %= FRAME_LAG;
    if (result == vk::Result::eErrorOutOfDateKHR) {
      // swapchain is out of date (e.g. the window was resized) and
      // must be recreated:
      resize();
    } else if (result == vk::Result::eSuboptimalKHR) {
      // swapchain is not as optimal as it could be, but the platform's
      // presentation engine will still present the image correctly.
    } else {
      if (result != vk::Result::eSuccess) {
        throw;
      }
    }
  } catch (...) {
    throw;
  };

  void init_connection() try {
    if (!glfwInit()) {
      throw;
    }

    if (!glfwVulkanSupported()) {
      throw;
    }
  } catch (...) {
    throw;
  };

  void init_vk() try {
    uint32_t required_extension_count = 0;
    uint32_t validation_layer_count = 0;
    char const *const *required_extensions = nullptr;
    char const *const *instance_validation_layers = nullptr;
    enabled_extension_count = 0;
    enabled_layer_count = 0;

    char const *const instance_validation_layers_alt1[] = {
        "VK_LAYER_LUNARG_standard_validation"};

    char const *const instance_validation_layers_alt2[] = {
        "VK_LAYER_GOOGLE_threading", "VK_LAYER_LUNARG_parameter_validation",
        "VK_LAYER_LUNARG_object_tracker", "VK_LAYER_LUNARG_core_validation",
        "VK_LAYER_GOOGLE_unique_objects"};

    // Look for validation layers
    vk::Bool32 validation_found = VK_FALSE;
    if (validate) {
      std::vector<vk::LayerProperties> instance_layers =
          vk::enumerateInstanceLayerProperties();

      instance_validation_layers = instance_validation_layers_alt1;
      if (instance_layers.size() > 0) {
        validation_found =
            check_layers(ARRAY_SIZE(instance_validation_layers_alt1),
                         instance_validation_layers, instance_layers);
        if (validation_found) {
          enabled_layer_count = ARRAY_SIZE(instance_validation_layers_alt1);
          enabled_layers[0] = "VK_LAYER_LUNARG_standard_validation";
          validation_layer_count = 1;
        } else {
          // use alternative set of validation layers
          instance_validation_layers = instance_validation_layers_alt2;
          enabled_layer_count = ARRAY_SIZE(instance_validation_layers_alt2);
          validation_found =
              check_layers(ARRAY_SIZE(instance_validation_layers_alt2),
                           instance_validation_layers, instance_layers);
          validation_layer_count = ARRAY_SIZE(instance_validation_layers_alt2);
          for (uint32_t i = 0; i < validation_layer_count; i++) {
            enabled_layers[i] = instance_validation_layers[i];
          }
        }
      }
      if (!validation_found) {
        throw;
      }
    }

    /* Look for instance extensions */
    memset(extension_names, 0, sizeof(extension_names));

    required_extensions =
        glfwGetRequiredInstanceExtensions(&required_extension_count);
    if (!required_extensions) {
      throw;
    }

    for (uint32_t i = 0; i < required_extension_count; i++) {
      extension_names[enabled_extension_count++] = required_extensions[i];

      if (enabled_extension_count >= 64) {
        throw;
      }
    }

    std::vector<vk::ExtensionProperties> instance_extensions =
        vk::enumerateInstanceExtensionProperties();
    if (instance_extensions.size() > 0) {
      for (uint32_t i = 0; i < instance_extensions.size(); i++) {
        if (!strcmp(VK_EXT_DEBUG_REPORT_EXTENSION_NAME,
                    instance_extensions[i].extensionName)) {
          if (validate) {
            extension_names[enabled_extension_count++] =
                VK_EXT_DEBUG_REPORT_EXTENSION_NAME;
          }
        }

        if (enabled_extension_count >= 64) {
          throw;
        }
      }
    }

    auto const app = vk::ApplicationInfo()
                         .setPApplicationName(app_name.c_str())
                         .setApplicationVersion(0)
                         .setPEngineName(app_name.c_str())
                         .setEngineVersion(0)
                         .setApiVersion(VK_API_VERSION_1_0);
    auto const inst_info =
        vk::InstanceCreateInfo()
            .setPApplicationInfo(&app)
            .setEnabledLayerCount(enabled_layer_count)
            .setPpEnabledLayerNames(instance_validation_layers)
            .setEnabledExtensionCount(enabled_extension_count)
            .setPpEnabledExtensionNames(extension_names);

    inst = vk::createInstance(inst_info);

    /* Make initial call to query gpu_count, then second call for gpu info*/
    std::vector<vk::PhysicalDevice> physical_devices =
        inst.enumeratePhysicalDevices();

    if (physical_devices.size() > 0) {
      gpu = physical_devices[0];
    } else {
      throw;
    }

    /* Look for device extensions */
    vk::Bool32 swapchainExtFound = VK_FALSE;
    enabled_extension_count = 0;
    memset(extension_names, 0, sizeof(extension_names));

    std::vector<vk::ExtensionProperties> device_extensions =
        gpu.enumerateDeviceExtensionProperties();
    if (device_extensions.size() > 0) {
      for (uint32_t i = 0; i < device_extensions.size(); i++) {
        if (!strcmp(VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                    device_extensions[i].extensionName)) {
          swapchainExtFound = 1;
          extension_names[enabled_extension_count++] =
              VK_KHR_SWAPCHAIN_EXTENSION_NAME;
        }

        if (enabled_extension_count >= 64) {
          throw;
        }
      }
    }

    if (!swapchainExtFound) {
      throw;
    }

    gpu_props = gpu.getProperties();

    /* Call with nullptr data to get count */
    queue_props = gpu.getQueueFamilyProperties();
    if (queue_props.size() < 1) {
      throw;
    }

    // Query fine-grained feature support for this device.
    //  If app has specific feature requirements it should check supported
    //  features based on this query
    vk::PhysicalDeviceFeatures physDevFeatures;
    physDevFeatures = gpu.getFeatures();
  } catch (...) {
    throw;
  };

  vk::Bool32 check_layers(uint32_t check_count,
                          char const *const *const check_names,
                          std::vector<vk::LayerProperties> layers) try {
    for (uint32_t i = 0; i < check_count; i++) {
      vk::Bool32 found = VK_FALSE;
      for (uint32_t j = 0; j < layers.size(); j++) {
        if (!strcmp(check_names[i], layers[j].layerName)) {
          found = VK_TRUE;
          break;
        }
      }
      if (!found) {
        throw;
      }
    }
    return VK_TRUE;
  } catch (...) {
    throw;
  };

  void create_device() try {
    float const priorities[1] = {0.0};

    vk::DeviceQueueCreateInfo queues[2];
    queues[0].setQueueFamilyIndex(graphics_queue_family_index);
    queues[0].setQueueCount(1);
    queues[0].setPQueuePriorities(priorities);

    auto deviceInfo =
        vk::DeviceCreateInfo()
            .setQueueCreateInfoCount(1)
            .setPQueueCreateInfos(queues)
            .setEnabledLayerCount(0)
            .setPpEnabledLayerNames(nullptr)
            .setEnabledExtensionCount(enabled_extension_count)
            .setPpEnabledExtensionNames((const char *const *)extension_names)
            .setPEnabledFeatures(nullptr);

    if (separate_present_queue) {
      queues[1].setQueueFamilyIndex(present_queue_family_index);
      queues[1].setQueueCount(1);
      queues[1].setPQueuePriorities(priorities);
      deviceInfo.setQueueCreateInfoCount(2);
    }

    device = gpu.createDevice(deviceInfo);
  } catch (...) {
    throw;
  };

  void prepare_buffers() try {
    vk::SwapchainKHR oldSwapchain = swapchain;

    // Check the surface capabilities and formats
    vk::SurfaceCapabilitiesKHR surfCapabilities;
    surfCapabilities = gpu.getSurfaceCapabilitiesKHR(surface);

    std::vector<vk::PresentModeKHR> presentModes =
        gpu.getSurfacePresentModesKHR(surface);

    vk::Extent2D swapchainExtent;
    // width and height are either both -1, or both not -1.
    if (surfCapabilities.currentExtent.width == (uint32_t)-1) {
      // If the surface size is undefined, the size is set to
      // the size of the images requested.
      swapchainExtent.width = width;
      swapchainExtent.height = height;
    } else {
      // If the surface size is defined, the swap chain size must match
      swapchainExtent = surfCapabilities.currentExtent;
      width = surfCapabilities.currentExtent.width;
      height = surfCapabilities.currentExtent.height;
    }

    // The FIFO present mode is guaranteed by the spec to be supported
    // and to have no tearing.  It's a great default present mode to use.
    vk::PresentModeKHR swapchainPresentMode = vk::PresentModeKHR::eFifo;

    //  There are times when you may wish to use another present mode.  The
    //  following code shows how to select them, and the comments provide some
    //  reasons you may wish to use them.
    //
    // It should be noted that Vulkan 1.0 doesn't provide a method for
    // synchronizing rendering with the presentation engine's display.  There
    // is a method provided for throttling rendering with the display, but
    // there are some presentation engines for which this method will not work.
    // If an application doesn't throttle its rendering, and if it renders much
    // faster than the refresh rate of the display, this can waste power on
    // mobile devices.  That is because power is being spent rendering images
    // that may never be seen.

    // VK_PRESENT_MODE_IMMEDIATE_KHR is for applications that don't care
    // about
    // tearing, or have some way of synchronizing their rendering with the
    // display.
    // VK_PRESENT_MODE_MAILBOX_KHR may be useful for applications that
    // generally render a new presentable image every refresh cycle, but are
    // occasionally early.  In this case, the application wants the new
    // image
    // to be displayed instead of the previously-queued-for-presentation
    // image
    // that has not yet been displayed.
    // VK_PRESENT_MODE_FIFO_RELAXED_KHR is for applications that generally
    // render a new presentable image every refresh cycle, but are
    // occasionally
    // late.  In this case (perhaps because of stuttering/latency concerns),
    // the application wants the late image to be immediately displayed,
    // even
    // though that may mean some tearing.

    if (presentMode != swapchainPresentMode) {
      for (size_t i = 0; i < presentModes.size(); ++i) {
        if (presentModes[i] == presentMode) {
          swapchainPresentMode = presentMode;
          break;
        }
      }
    }

    if (swapchainPresentMode != presentMode) {
      throw;
    }

    // Determine the number of VkImages to use in the swap chain.
    // Application desires to acquire 3 images at a time for triple
    // buffering
    uint32_t desiredNumOfSwapchainImages = 3;
    if (desiredNumOfSwapchainImages < surfCapabilities.minImageCount) {
      desiredNumOfSwapchainImages = surfCapabilities.minImageCount;
    }

    // If maxImageCount is 0, we can ask for as many images as we want,
    // otherwise
    // we're limited to maxImageCount
    if ((surfCapabilities.maxImageCount > 0) &&
        (desiredNumOfSwapchainImages > surfCapabilities.maxImageCount)) {
      // Application must settle for fewer images than desired:
      desiredNumOfSwapchainImages = surfCapabilities.maxImageCount;
    }

    vk::SurfaceTransformFlagBitsKHR preTransform;
    if (surfCapabilities.supportedTransforms &
        vk::SurfaceTransformFlagBitsKHR::eIdentity) {
      preTransform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
    } else {
      preTransform = surfCapabilities.currentTransform;
    }

    // Find a supported composite alpha mode - one of these is guaranteed to be
    // set
    vk::CompositeAlphaFlagBitsKHR compositeAlpha =
        vk::CompositeAlphaFlagBitsKHR::eOpaque;
    vk::CompositeAlphaFlagBitsKHR compositeAlphaFlags[4] = {
        vk::CompositeAlphaFlagBitsKHR::eOpaque,
        vk::CompositeAlphaFlagBitsKHR::ePreMultiplied,
        vk::CompositeAlphaFlagBitsKHR::ePostMultiplied,
        vk::CompositeAlphaFlagBitsKHR::eInherit,
    };
    for (uint32_t i = 0; i < sizeof(compositeAlphaFlags); i++) {
      if (surfCapabilities.supportedCompositeAlpha & compositeAlphaFlags[i]) {
        compositeAlpha = compositeAlphaFlags[i];
        break;
      }
    }

    auto const swapchain_ci =
        vk::SwapchainCreateInfoKHR()
            .setSurface(surface)
            .setMinImageCount(desiredNumOfSwapchainImages)
            .setImageFormat(format)
            .setImageColorSpace(color_space)
            .setImageExtent({swapchainExtent.width, swapchainExtent.height})
            .setImageArrayLayers(1)
            .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
            .setImageSharingMode(vk::SharingMode::eExclusive)
            .setQueueFamilyIndexCount(0)
            .setPQueueFamilyIndices(nullptr)
            .setPreTransform(preTransform)
            .setCompositeAlpha(compositeAlpha)
            .setPresentMode(swapchainPresentMode)
            .setClipped(true)
            .setOldSwapchain(oldSwapchain);

    swapchain = device.createSwapchainKHR(swapchain_ci);

    // If we just re-created an existing swapchain, we should destroy the
    // old
    // swapchain at this point.
    // Note: destroying the swapchain also cleans up all its associated
    // presentable images once the platform is done with them.
    if (oldSwapchain) {
      device.destroySwapchainKHR(oldSwapchain, nullptr);
    }

    std::vector<vk::Image> swapchainImages =
        device.getSwapchainImagesKHR(swapchain);
    swapchainImageCount = (uint32_t)swapchainImages.size();

    swapchain_image_resources.clear();
    swapchain_image_resources.shrink_to_fit();
    swapchain_image_resources.resize(swapchainImageCount);

    for (uint32_t i = 0; i < swapchainImages.size(); ++i) {
      auto color_image_view =
          vk::ImageViewCreateInfo()
              .setViewType(vk::ImageViewType::e2D)
              .setFormat(format)
              .setSubresourceRange(vk::ImageSubresourceRange(
                  vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

      swapchain_image_resources[i].image = swapchainImages[i];

      color_image_view.image = swapchain_image_resources[i].image;

      swapchain_image_resources[i].view =
          device.createImageView(color_image_view);
    }
  } catch (...) {
    throw;
  };

  void prepare_depth() try {
    depth.format = vk::Format::eD16Unorm;

    auto const image =
        vk::ImageCreateInfo()
            .setImageType(vk::ImageType::e2D)
            .setFormat(depth.format)
            .setExtent({(uint32_t)width, (uint32_t)height, 1})
            .setMipLevels(1)
            .setArrayLayers(1)
            .setSamples(vk::SampleCountFlagBits::e1)
            .setTiling(vk::ImageTiling::eOptimal)
            .setUsage(vk::ImageUsageFlagBits::eDepthStencilAttachment)
            .setSharingMode(vk::SharingMode::eExclusive)
            .setQueueFamilyIndexCount(0)
            .setPQueueFamilyIndices(nullptr)
            .setInitialLayout(vk::ImageLayout::eUndefined);

    depth.image = device.createImage(image);

    vk::MemoryRequirements mem_reqs;
    device.getImageMemoryRequirements(depth.image, &mem_reqs);

    depth.mem_alloc.setAllocationSize(mem_reqs.size);
    depth.mem_alloc.setMemoryTypeIndex(0);

    auto const pass = memory_type_from_properties(
        mem_reqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal,
        &depth.mem_alloc.memoryTypeIndex);
    if (!pass) {
      throw;
    }

    depth.mem = device.allocateMemory(depth.mem_alloc);

    device.bindImageMemory(depth.image, depth.mem, 0);

    auto const view = vk::ImageViewCreateInfo()
                          .setImage(depth.image)
                          .setViewType(vk::ImageViewType::e2D)
                          .setFormat(depth.format)
                          .setSubresourceRange(vk::ImageSubresourceRange(
                              vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1));
    depth.view = device.createImageView(view);
  } catch (...) {
    throw;
  };

  void prepare_textures() try {
    vk::Format const tex_format = vk::Format::eR8G8B8A8Unorm;
    vk::FormatProperties props;
    gpu.getFormatProperties(tex_format, &props);

    for (uint32_t i = 0; i < texture_count; i++) {
      if ((props.linearTilingFeatures &
           vk::FormatFeatureFlagBits::eSampledImage) &&
          !use_staging_buffer) {
        /* Device can texture using linear textures */
        prepare_texture_image(tex_files[i].c_str(), &textures[i],
                              vk::ImageTiling::eLinear,
                              vk::ImageUsageFlagBits::eSampled,
                              vk::MemoryPropertyFlagBits::eHostVisible |
                                  vk::MemoryPropertyFlagBits::eHostCoherent);
        // Nothing in the pipeline needs to be complete to start, and don't
        // allow fragment
        // shader to run until layout transition completes
        set_image_layout(textures[i].image, vk::ImageAspectFlagBits::eColor,
                         vk::ImageLayout::ePreinitialized,
                         textures[i].imageLayout,
                         vk::AccessFlagBits::eHostWrite,
                         vk::PipelineStageFlagBits::eTopOfPipe,
                         vk::PipelineStageFlagBits::eAllGraphics);
        staging_texture.image = vk::Image();
      } else if (props.optimalTilingFeatures &
                 vk::FormatFeatureFlagBits::eSampledImage) {
        /* Must use staging buffer to copy linear texture to optimized */

        prepare_texture_image(tex_files[i].c_str(), &staging_texture,
                              vk::ImageTiling::eLinear,
                              vk::ImageUsageFlagBits::eTransferSrc,
                              vk::MemoryPropertyFlagBits::eHostVisible |
                                  vk::MemoryPropertyFlagBits::eHostCoherent);

        prepare_texture_image(tex_files[i].c_str(), &textures[i],
                              vk::ImageTiling::eOptimal,
                              vk::ImageUsageFlagBits::eTransferDst |
                                  vk::ImageUsageFlagBits::eSampled,
                              vk::MemoryPropertyFlagBits::eDeviceLocal);

        set_image_layout(staging_texture.image, vk::ImageAspectFlagBits::eColor,
                         vk::ImageLayout::ePreinitialized,
                         vk::ImageLayout::eTransferSrcOptimal,
                         vk::AccessFlagBits::eHostWrite,
                         vk::PipelineStageFlagBits::eTopOfPipe,
                         vk::PipelineStageFlagBits::eTransfer);

        set_image_layout(textures[i].image, vk::ImageAspectFlagBits::eColor,
                         vk::ImageLayout::ePreinitialized,
                         vk::ImageLayout::eTransferDstOptimal,
                         vk::AccessFlagBits::eHostWrite,
                         vk::PipelineStageFlagBits::eTopOfPipe,
                         vk::PipelineStageFlagBits::eTransfer);

        auto const subresource =
            vk::ImageSubresourceLayers()
                .setAspectMask(vk::ImageAspectFlagBits::eColor)
                .setMipLevel(0)
                .setBaseArrayLayer(0)
                .setLayerCount(1);

        auto const copy_region =
            vk::ImageCopy()
                .setSrcSubresource(subresource)
                .setSrcOffset({0, 0, 0})
                .setDstSubresource(subresource)
                .setDstOffset({0, 0, 0})
                .setExtent({(uint32_t)staging_texture.tex_width,
                            (uint32_t)staging_texture.tex_height, 1});

        cmd.copyImage(staging_texture.image,
                      vk::ImageLayout::eTransferSrcOptimal, textures[i].image,
                      vk::ImageLayout::eTransferDstOptimal, 1, &copy_region);

        set_image_layout(textures[i].image, vk::ImageAspectFlagBits::eColor,
                         vk::ImageLayout::eTransferDstOptimal,
                         textures[i].imageLayout,
                         vk::AccessFlagBits::eTransferWrite,
                         vk::PipelineStageFlagBits::eTransfer,
                         vk::PipelineStageFlagBits::eAllGraphics);
      } else {
        throw;
      }

      auto const samplerInfo =
          vk::SamplerCreateInfo()
              .setMagFilter(vk::Filter::eNearest)
              .setMinFilter(vk::Filter::eNearest)
              .setMipmapMode(vk::SamplerMipmapMode::eNearest)
              .setAddressModeU(vk::SamplerAddressMode::eClampToEdge)
              .setAddressModeV(vk::SamplerAddressMode::eClampToEdge)
              .setAddressModeW(vk::SamplerAddressMode::eClampToEdge)
              .setMipLodBias(0.0f)
              .setAnisotropyEnable(VK_FALSE)
              .setMaxAnisotropy(1)
              .setCompareEnable(VK_FALSE)
              .setCompareOp(vk::CompareOp::eNever)
              .setMinLod(0.0f)
              .setMaxLod(0.0f)
              .setBorderColor(vk::BorderColor::eFloatOpaqueWhite)
              .setUnnormalizedCoordinates(VK_FALSE);

      textures[i].sampler = device.createSampler(samplerInfo);

      auto const viewInfo =
          vk::ImageViewCreateInfo()
              .setImage(textures[i].image)
              .setViewType(vk::ImageViewType::e2D)
              .setFormat(tex_format)
              .setSubresourceRange(vk::ImageSubresourceRange(
                  vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

      textures[i].view = device.createImageView(viewInfo);
    }
  } catch (...) {
    throw;
  };

  void prepare_cube_data_buffers() try {
    calculate_eye_matrix();

    vktexcube_vs_uniform data;
    data.screen_size = glm::vec4(width, height, 0.0f, 1.0f);
    data.eye = eye_matrix;
    //    dumpMatrix("MVP", MVP)

    for (uint32_t i = 0; i < g_vertex_buffer_data.size(); i++) {
      data.pos_uv[i] = g_vertex_buffer_data[i];
    }
    {
      auto const buf_info =
          vk::BufferCreateInfo()
              .setSize(sizeof(data))
              .setUsage(vk::BufferUsageFlagBits::eUniformBuffer);

      for (unsigned int i = 0; i < swapchainImageCount; i++) {
        swapchain_image_resources[i].uniform_buffer =
            device.createBuffer(buf_info);

        vk::MemoryRequirements mem_reqs = device.getBufferMemoryRequirements(
            swapchain_image_resources[i].uniform_buffer);

        auto mem_alloc = vk::MemoryAllocateInfo()
                             .setAllocationSize(mem_reqs.size)
                             .setMemoryTypeIndex(0);

        bool const pass = memory_type_from_properties(
            mem_reqs.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent,
            &mem_alloc.memoryTypeIndex);
        if (!pass) {
          throw;
        }
        swapchain_image_resources[i].uniform_memory =
            device.allocateMemory(mem_alloc);

        auto pData =
            device.mapMemory(swapchain_image_resources[i].uniform_memory, 0,
                             VK_WHOLE_SIZE, vk::MemoryMapFlags());

        memcpy(pData, &data, sizeof(data));

        device.unmapMemory(swapchain_image_resources[i].uniform_memory);

        device.bindBufferMemory(swapchain_image_resources[i].uniform_buffer,
                                swapchain_image_resources[i].uniform_memory, 0);
      }
    }
    {
      auto const buf_info =
          vk::BufferCreateInfo()
              .setSize(sizeof(vktexcube_triangle_storage) *
                       g_cube_buffer_data.size())
              .setUsage(vk::BufferUsageFlagBits::eStorageBuffer);

      for (unsigned int i = 0; i < swapchainImageCount; i++) {
        swapchain_image_resources[i].storage_buffer =
            device.createBuffer(buf_info);

        vk::MemoryRequirements mem_reqs = device.getBufferMemoryRequirements(
            swapchain_image_resources[i].storage_buffer);

        auto mem_alloc = vk::MemoryAllocateInfo()
                             .setAllocationSize(mem_reqs.size)
                             .setMemoryTypeIndex(0);

        bool const pass = memory_type_from_properties(
            mem_reqs.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent,
            &mem_alloc.memoryTypeIndex);
        if (!pass) {
          throw;
        }
        swapchain_image_resources[i].storage_memory =
            device.allocateMemory(mem_alloc);

        auto pData =
            device.mapMemory(swapchain_image_resources[i].storage_memory, 0,
                             VK_WHOLE_SIZE, vk::MemoryMapFlags());

        memcpy(pData, g_cube_buffer_data.data(),
               sizeof(vktexcube_triangle_storage) * g_cube_buffer_data.size());

        device.unmapMemory(swapchain_image_resources[i].storage_memory);

        device.bindBufferMemory(swapchain_image_resources[i].storage_buffer,
                                swapchain_image_resources[i].storage_memory, 0);
      }
    }
  } catch (...) {
    throw;
  };

  void prepare_descriptor_layout() try {
    std::vector<vk::DescriptorSetLayoutBinding> layout_bindings = {
        vk::DescriptorSetLayoutBinding()
            .setBinding(0)
            .setDescriptorType(vk::DescriptorType::eUniformBuffer)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eAllGraphics)
            .setPImmutableSamplers(nullptr),
        vk::DescriptorSetLayoutBinding()
            .setBinding(1)
            .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
            .setDescriptorCount(texture_count)
            .setStageFlags(vk::ShaderStageFlagBits::eAllGraphics)
            .setPImmutableSamplers(nullptr),
        vk::DescriptorSetLayoutBinding()
            .setBinding(2)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eAllGraphics)
            .setPImmutableSamplers(nullptr)};

    auto const descriptor_layout = vk::DescriptorSetLayoutCreateInfo()
                                       .setBindingCount(layout_bindings.size())
                                       .setPBindings(layout_bindings.data());

    desc_layout = device.createDescriptorSetLayout(descriptor_layout);

    auto const pPipelineLayoutCreateInfo =
        vk::PipelineLayoutCreateInfo().setSetLayoutCount(1).setPSetLayouts(
            &desc_layout);

    pipeline_layout = device.createPipelineLayout(pPipelineLayoutCreateInfo);
  } catch (...) {
    throw;
  };

  void prepare_render_pass() try {
    // The initial layout for the color and depth attachments will be
    // LAYOUT_UNDEFINED
    // because at the start of the renderpass, we don't care about their
    // contents.
    // At the start of the subpass, the color attachment's layout will be
    // transitioned
    // to LAYOUT_COLOR_ATTACHMENT_OPTIMAL and the depth stencil attachment's
    // layout
    // will be transitioned to LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL.  At the
    // end of
    // the renderpass, the color attachment's layout will be transitioned to
    // LAYOUT_PRESENT_SRC_KHR to be ready to present.  This is all done as part
    // of
    // the renderpass, no barriers are necessary.
    const vk::AttachmentDescription attachments[2] = {
        vk::AttachmentDescription()
            .setFormat(format)
            .setSamples(vk::SampleCountFlagBits::e1)
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eStore)
            .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
            .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
            .setInitialLayout(vk::ImageLayout::eUndefined)
            .setFinalLayout(vk::ImageLayout::ePresentSrcKHR),
        vk::AttachmentDescription()
            .setFormat(depth.format)
            .setSamples(vk::SampleCountFlagBits::e1)
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eDontCare)
            .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
            .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
            .setInitialLayout(vk::ImageLayout::eUndefined)
            .setFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal)};

    auto const color_reference =
        vk::AttachmentReference().setAttachment(0).setLayout(
            vk::ImageLayout::eColorAttachmentOptimal);

    auto const depth_reference =
        vk::AttachmentReference().setAttachment(1).setLayout(
            vk::ImageLayout::eDepthStencilAttachmentOptimal);

    auto const subpass =
        vk::SubpassDescription()
            .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
            .setInputAttachmentCount(0)
            .setPInputAttachments(nullptr)
            .setColorAttachmentCount(1)
            .setPColorAttachments(&color_reference)
            .setPResolveAttachments(nullptr)
            .setPDepthStencilAttachment(&depth_reference)
            .setPreserveAttachmentCount(0)
            .setPPreserveAttachments(nullptr);

    auto const rp_info = vk::RenderPassCreateInfo()
                             .setAttachmentCount(2)
                             .setPAttachments(attachments)
                             .setSubpassCount(1)
                             .setPSubpasses(&subpass)
                             .setDependencyCount(0)
                             .setPDependencies(nullptr);

    render_pass = device.createRenderPass(rp_info);
  } catch (...) {
    throw;
  };

  void prepare_pipeline() try {
    vk::PipelineCacheCreateInfo const pipelineCacheInfo;
    pipelineCache = device.createPipelineCache(pipelineCacheInfo);

    vk::PipelineShaderStageCreateInfo const shaderStageInfo[2] = {
        vk::PipelineShaderStageCreateInfo()
            .setStage(vk::ShaderStageFlagBits::eVertex)
            .setModule(prepare_vs())
            .setPName("main"),
        vk::PipelineShaderStageCreateInfo()
            .setStage(vk::ShaderStageFlagBits::eFragment)
            .setModule(prepare_fs())
            .setPName("main")};

    vk::PipelineVertexInputStateCreateInfo const vertexInputInfo;

    auto const inputAssemblyInfo =
        vk::PipelineInputAssemblyStateCreateInfo().setTopology(
            vk::PrimitiveTopology::eTriangleList);

    // TODO: Where are pViewports and pScissors set?
    auto const viewportInfo = vk::PipelineViewportStateCreateInfo()
                                  .setViewportCount(1)
                                  .setScissorCount(1);

    auto const rasterizationInfo =
        vk::PipelineRasterizationStateCreateInfo()
            .setDepthClampEnable(VK_FALSE)
            .setRasterizerDiscardEnable(VK_FALSE)
            .setPolygonMode(vk::PolygonMode::eFill)
            .setCullMode(vk::CullModeFlagBits::eBack)
            .setFrontFace(vk::FrontFace::eCounterClockwise)
            .setDepthBiasEnable(VK_FALSE)
            .setLineWidth(1.0f);

    auto const multisampleInfo = vk::PipelineMultisampleStateCreateInfo();

    auto const stencilOp = vk::StencilOpState()
                               .setFailOp(vk::StencilOp::eKeep)
                               .setPassOp(vk::StencilOp::eKeep)
                               .setCompareOp(vk::CompareOp::eAlways);

    auto const depthStencilInfo =
        vk::PipelineDepthStencilStateCreateInfo()
            .setDepthTestEnable(VK_TRUE)
            .setDepthWriteEnable(VK_TRUE)
            .setDepthCompareOp(vk::CompareOp::eLessOrEqual)
            .setDepthBoundsTestEnable(VK_FALSE)
            .setStencilTestEnable(VK_FALSE)
            .setFront(stencilOp)
            .setBack(stencilOp);

    vk::PipelineColorBlendAttachmentState const colorBlendAttachments[1] = {
        vk::PipelineColorBlendAttachmentState().setColorWriteMask(
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA)};

    auto const colorBlendInfo = vk::PipelineColorBlendStateCreateInfo()
                                    .setAttachmentCount(1)
                                    .setPAttachments(colorBlendAttachments);

    vk::DynamicState const dynamicStates[2] = {vk::DynamicState::eViewport,
                                               vk::DynamicState::eScissor};

    auto const dynamicStateInfo = vk::PipelineDynamicStateCreateInfo()
                                      .setPDynamicStates(dynamicStates)
                                      .setDynamicStateCount(2);

    auto const pipeline_ci = vk::GraphicsPipelineCreateInfo()
                                 .setStageCount(2)
                                 .setPStages(shaderStageInfo)
                                 .setPVertexInputState(&vertexInputInfo)
                                 .setPInputAssemblyState(&inputAssemblyInfo)
                                 .setPViewportState(&viewportInfo)
                                 .setPRasterizationState(&rasterizationInfo)
                                 .setPMultisampleState(&multisampleInfo)
                                 .setPDepthStencilState(&depthStencilInfo)
                                 .setPColorBlendState(&colorBlendInfo)
                                 .setPDynamicState(&dynamicStateInfo)
                                 .setLayout(pipeline_layout)
                                 .setRenderPass(render_pass);

    pipeline = device.createGraphicsPipelines(pipelineCache, pipeline_ci)[0];

    device.destroyShaderModule(frag_shader_module, nullptr);
    device.destroyShaderModule(vert_shader_module, nullptr);
  } catch (...) {
    throw;
  };

  void build_image_ownership_cmd(uint32_t const &i) try {
    auto const cmd_buf_info = vk::CommandBufferBeginInfo().setFlags(
        vk::CommandBufferUsageFlagBits::eSimultaneousUse);
    swapchain_image_resources[i].graphics_to_present_cmd.begin(cmd_buf_info);

    auto const image_ownership_barrier =
        vk::ImageMemoryBarrier()
            .setSrcAccessMask(vk::AccessFlags())
            .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite)
            .setOldLayout(vk::ImageLayout::ePresentSrcKHR)
            .setNewLayout(vk::ImageLayout::ePresentSrcKHR)
            .setSrcQueueFamilyIndex(graphics_queue_family_index)
            .setDstQueueFamilyIndex(present_queue_family_index)
            .setImage(swapchain_image_resources[i].image)
            .setSubresourceRange(vk::ImageSubresourceRange(
                vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

    swapchain_image_resources[i].graphics_to_present_cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::DependencyFlagBits(), 0, nullptr, 0, nullptr, 1,
        &image_ownership_barrier);

    swapchain_image_resources[i].graphics_to_present_cmd.end();
  } catch (...) {
    throw;
  };

  void prepare_descriptor_pool() try {
    std::vector<vk::DescriptorPoolSize> poolSizes = {
        vk::DescriptorPoolSize()
            .setType(vk::DescriptorType::eUniformBuffer)
            .setDescriptorCount(swapchainImageCount),
        vk::DescriptorPoolSize()
            .setType(vk::DescriptorType::eCombinedImageSampler)
            .setDescriptorCount(swapchainImageCount * texture_count),
        vk::DescriptorPoolSize()
            .setType(vk::DescriptorType::eStorageBuffer)
            .setDescriptorCount(swapchainImageCount)};

    auto const descriptor_pool = vk::DescriptorPoolCreateInfo()
                                     .setMaxSets(swapchainImageCount)
                                     .setPoolSizeCount(poolSizes.size())
                                     .setPPoolSizes(poolSizes.data());

    desc_pool = device.createDescriptorPool(descriptor_pool);
  } catch (...) {
    throw;
  };

  void prepare_descriptor_set() try {
    auto const alloc_info = vk::DescriptorSetAllocateInfo()
                                .setDescriptorPool(desc_pool)
                                .setDescriptorSetCount(1)
                                .setPSetLayouts(&desc_layout);

    auto buffer_info = vk::DescriptorBufferInfo().setOffset(0).setRange(
        sizeof(vktexcube_vs_uniform));
    auto storage_buffer_info = vk::DescriptorBufferInfo().setOffset(0).setRange(
        sizeof(vktexcube_triangle_storage) * g_cube_buffer_data.size());

    vk::DescriptorImageInfo tex_descs[texture_count];
    for (uint32_t i = 0; i < texture_count; i++) {
      tex_descs[i].setSampler(textures[i].sampler);
      tex_descs[i].setImageView(textures[i].view);
      tex_descs[i].setImageLayout(vk::ImageLayout::eGeneral);
    }

    std::vector<vk::WriteDescriptorSet> writes = {
        vk::WriteDescriptorSet()
            .setDstBinding(0)
            .setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eUniformBuffer)
            .setPBufferInfo(&buffer_info),
        vk::WriteDescriptorSet()
            .setDstBinding(1)
            .setDescriptorCount(texture_count)
            .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
            .setPImageInfo(tex_descs),
        vk::WriteDescriptorSet()
            .setDstBinding(2)
            .setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setPBufferInfo(&storage_buffer_info)};

    for (unsigned int i = 0; i < swapchainImageCount; i++) {
      swapchain_image_resources[i].descriptor_set =
          device.allocateDescriptorSets(alloc_info)[0];

      buffer_info.setBuffer(swapchain_image_resources[i].uniform_buffer);
      storage_buffer_info.setBuffer(
          swapchain_image_resources[i].storage_buffer);
      for (auto &write : writes)
        write.setDstSet(swapchain_image_resources[i].descriptor_set);
      device.updateDescriptorSets(writes.size(), writes.data(), 0, nullptr);
    }
  } catch (...) {
    throw;
  };

  void prepare_framebuffers() try {
    vk::ImageView attachments[2];
    attachments[1] = depth.view;

    auto const fb_info = vk::FramebufferCreateInfo()
                             .setRenderPass(render_pass)
                             .setAttachmentCount(2)
                             .setPAttachments(attachments)
                             .setWidth((uint32_t)width)
                             .setHeight((uint32_t)height)
                             .setLayers(1);

    for (uint32_t i = 0; i < swapchainImageCount; i++) {
      attachments[0] = swapchain_image_resources[i].view;
      swapchain_image_resources[i].framebuffer =
          device.createFramebuffer(fb_info);
    }
  } catch (...) {
    throw;
  };

  void draw_build_cmd(vk::CommandBuffer commandBuffer) try {
    auto const commandInfo = vk::CommandBufferBeginInfo().setFlags(
        vk::CommandBufferUsageFlagBits::eSimultaneousUse);

    vk::ClearValue const clearValues[2] = {
        vk::ClearColorValue(std::array<float, 4>({{0.2f, 0.2f, 0.2f, 0.2f}})),
        vk::ClearDepthStencilValue(1.0f, 0u)};

    auto const passInfo =
        vk::RenderPassBeginInfo()
            .setRenderPass(render_pass)
            .setFramebuffer(
                swapchain_image_resources[current_buffer].framebuffer)
            .setRenderArea(
                vk::Rect2D(vk::Offset2D(0, 0),
                           vk::Extent2D((uint32_t)width, (uint32_t)height)))
            .setClearValueCount(2)
            .setPClearValues(clearValues);

    commandBuffer.begin(commandInfo);

    commandBuffer.beginRenderPass(&passInfo, vk::SubpassContents::eInline);
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics, pipeline_layout, 0, 1,
        &swapchain_image_resources[current_buffer].descriptor_set, 0, nullptr);

    auto const viewport = vk::Viewport()
                              .setWidth((float)width)
                              .setHeight((float)height)
                              .setMinDepth((float)0.0f)
                              .setMaxDepth((float)1.0f);
    commandBuffer.setViewport(0, 1, &viewport);

    vk::Rect2D const scissor(vk::Offset2D(0, 0), vk::Extent2D(width, height));
    commandBuffer.setScissor(0, 1, &scissor);
    commandBuffer.draw((uint32_t)g_vertex_buffer_data.size(), 1, 0, 0);
    // Note that ending the renderpass changes the image's layout from
    // COLOR_ATTACHMENT_OPTIMAL to PRESENT_SRC_KHR
    commandBuffer.endRenderPass();

    if (separate_present_queue) {
      // We have to transfer ownership from the graphics queue family to
      // the
      // present queue family to be able to present.  Note that we don't
      // have
      // to transfer from present queue family back to graphics queue
      // family at
      // the start of the next frame because we don't care about the
      // image's
      // contents at that point.
      auto const image_ownership_barrier =
          vk::ImageMemoryBarrier()
              .setSrcAccessMask(vk::AccessFlags())
              .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite)
              .setOldLayout(vk::ImageLayout::ePresentSrcKHR)
              .setNewLayout(vk::ImageLayout::ePresentSrcKHR)
              .setSrcQueueFamilyIndex(graphics_queue_family_index)
              .setDstQueueFamilyIndex(present_queue_family_index)
              .setImage(swapchain_image_resources[current_buffer].image)
              .setSubresourceRange(vk::ImageSubresourceRange(
                  vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

      commandBuffer.pipelineBarrier(
          vk::PipelineStageFlagBits::eColorAttachmentOutput,
          vk::PipelineStageFlagBits::eBottomOfPipe, vk::DependencyFlagBits(), 0,
          nullptr, 0, nullptr, 1, &image_ownership_barrier);
    }

    commandBuffer.end();
  } catch (...) {
    throw;
  };

  void flush_init_cmd() try {
    // TODO: hmm.
    // This function could get called twice if the texture uses a staging
    // buffer
    // In that case the second call should be ignored
    if (!cmd) {
      return;
    }

    cmd.end();

    auto const fenceInfo = vk::FenceCreateInfo();
    vk::Fence fence = device.createFence(fenceInfo);

    vk::CommandBuffer const commandBuffers[] = {cmd};
    auto const submitInfo =
        vk::SubmitInfo().setCommandBufferCount(1).setPCommandBuffers(
            commandBuffers);

    graphics_queue.submit(submitInfo, fence);

    device.waitForFences(fence, VK_TRUE, UINT64_MAX);

    device.freeCommandBuffers(cmd_pool, 1, commandBuffers);
    device.destroyFence(fence, nullptr);

    cmd = vk::CommandBuffer();
  } catch (...) {
    throw;
  };

  void destroy_texture_image(texture_object *tex_objs) try {
    // clean up staging resources
    device.freeMemory(tex_objs->mem);
    device.destroyImage(tex_objs->image);
  } catch (...) {
    throw;
  };

  void update_data_buffer() try {
    // Rotate around the Y axis
    model_matrix = glm::rotate(model_matrix, glm::radians(spin_angle),
                               glm::vec3(0.0f, 1.0f, 0.0f));

    calculate_eye_matrix();

    vktexcube_vs_uniform data;
    data.screen_size = glm::vec4(width, height, 0.0f, 1.0f);
    data.eye = eye_matrix;
    for (uint32_t i = 0; i < g_vertex_buffer_data.size(); i++) {
      data.pos_uv[i] = g_vertex_buffer_data[i];
    }

    auto pData = device.mapMemory(
        swapchain_image_resources[current_buffer].uniform_memory, 0,
        VK_WHOLE_SIZE, vk::MemoryMapFlags());

    memcpy(pData, &data, sizeof(vktexcube_vs_uniform));

    device.unmapMemory(
        swapchain_image_resources[current_buffer].uniform_memory);
  } catch (...) {
    throw;
  };

  bool memory_type_from_properties(uint32_t typeBits,
                                   vk::MemoryPropertyFlags requirements_mask,
                                   uint32_t *typeIndex) try {
    // Search memtypes to find first index with those properties
    for (uint32_t i = 0; i < VK_MAX_MEMORY_TYPES; i++) {
      if ((typeBits & 1) == 1) {
        // Type is available, does it match user properties?
        if ((memory_properties.memoryTypes[i].propertyFlags &
             requirements_mask) == requirements_mask) {
          *typeIndex = i;
          return true;
        }
      }
      typeBits >>= 1;
    }

    // No memory types matched, return failure
    return false;
  } catch (...) {
    throw;
  };

  void prepare_texture_image(const char *filename, texture_object *tex_obj,
                             vk::ImageTiling tiling, vk::ImageUsageFlags usage,
                             vk::MemoryPropertyFlags required_props) try {
    int32_t tex_width;
    int32_t tex_height;
    if (!load_texture(filename, nullptr, nullptr, &tex_width, &tex_height)) {
      throw;
    }

    tex_obj->tex_width = tex_width;
    tex_obj->tex_height = tex_height;

    auto const image_create_info =
        vk::ImageCreateInfo()
            .setImageType(vk::ImageType::e2D)
            .setFormat(vk::Format::eR8G8B8A8Unorm)
            .setExtent({(uint32_t)tex_width, (uint32_t)tex_height, 1})
            .setMipLevels(1)
            .setArrayLayers(1)
            .setSamples(vk::SampleCountFlagBits::e1)
            .setTiling(tiling)
            .setUsage(usage)
            .setSharingMode(vk::SharingMode::eExclusive)
            .setQueueFamilyIndexCount(0)
            .setPQueueFamilyIndices(nullptr)
            .setInitialLayout(vk::ImageLayout::ePreinitialized);

    tex_obj->image = device.createImage(image_create_info);

    vk::MemoryRequirements mem_reqs;
    device.getImageMemoryRequirements(tex_obj->image, &mem_reqs);

    tex_obj->mem_alloc.setAllocationSize(mem_reqs.size);
    tex_obj->mem_alloc.setMemoryTypeIndex(0);

    auto pass =
        memory_type_from_properties(mem_reqs.memoryTypeBits, required_props,
                                    &tex_obj->mem_alloc.memoryTypeIndex);
    if (pass != true) {
      throw;
    }

    tex_obj->mem = device.allocateMemory(tex_obj->mem_alloc);

    device.bindImageMemory(tex_obj->image, tex_obj->mem, 0);

    if (required_props & vk::MemoryPropertyFlagBits::eHostVisible) {
      auto const subres = vk::ImageSubresource()
                              .setAspectMask(vk::ImageAspectFlagBits::eColor)
                              .setMipLevel(0)
                              .setArrayLayer(0);
      vk::SubresourceLayout layout;
      device.getImageSubresourceLayout(tex_obj->image, &subres, &layout);

      auto data =
          device.mapMemory(tex_obj->mem, 0, tex_obj->mem_alloc.allocationSize);

      if (!load_texture(filename, (uint8_t *)data, &layout, &tex_width,
                        &tex_height)) {
        throw;
      }

      device.unmapMemory(tex_obj->mem);
    }

    tex_obj->imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
  } catch (...) {
    throw;
  };

  bool load_texture(const char *filename, uint8_t *rgba_data,
                    vk::SubresourceLayout *layout, int32_t *img_width,
                    int32_t *img_height) try {
    int channels_in_file = 0;
    stbi_uc *texture_data =
        stbi_load(filename, img_width, img_height, &channels_in_file, 0);
    if (rgba_data == nullptr) {
    } else {
      for (uint32_t y = 0; y < (uint32_t)*img_height; y++) {
        for (uint32_t x = 0; x < (uint32_t)*img_width; x++) {
          uint32_t index = y * uint32_t(*img_width) * 4 + x * 4 + 0;
          uint32_t tex_index = y * uint32_t(*img_width) * channels_in_file +
                               x * channels_in_file + 0;
          rgba_data[index + 0] = (channels_in_file >= 1)
                                     ? texture_data[tex_index + 0]
                                     : (uint8_t)0;
          rgba_data[index + 1] = (channels_in_file >= 2)
                                     ? texture_data[tex_index + 1]
                                     : (uint8_t)0;
          rgba_data[index + 2] = (channels_in_file >= 3)
                                     ? texture_data[tex_index + 2]
                                     : (uint8_t)0;
          rgba_data[index + 3] = (channels_in_file >= 4)
                                     ? texture_data[tex_index + 3]
                                     : (uint8_t)0;
        }
      }
    }
    stbi_image_free(texture_data);
    return true;
  } catch (...) {
    throw;
  };

  void set_image_layout(vk::Image image, vk::ImageAspectFlags aspectMask,
                        vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
                        vk::AccessFlags srcAccessMask,
                        vk::PipelineStageFlags src_stages,
                        vk::PipelineStageFlags dest_stages) try {
    auto DstAccessMask = [](vk::ImageLayout const &layout) {
      vk::AccessFlags flags;

      switch (layout) {
      case vk::ImageLayout::eTransferDstOptimal:
        // Make sure anything that was copying from this image has
        // completed
        flags = vk::AccessFlagBits::eTransferWrite;
        break;
      case vk::ImageLayout::eColorAttachmentOptimal:
        flags = vk::AccessFlagBits::eColorAttachmentWrite;
        break;
      case vk::ImageLayout::eDepthStencilAttachmentOptimal:
        flags = vk::AccessFlagBits::eDepthStencilAttachmentWrite;
        break;
      case vk::ImageLayout::eShaderReadOnlyOptimal:
        // Make sure any Copy or CPU writes to image are flushed
        flags = vk::AccessFlagBits::eShaderRead |
                vk::AccessFlagBits::eInputAttachmentRead;
        break;
      case vk::ImageLayout::eTransferSrcOptimal:
        flags = vk::AccessFlagBits::eTransferRead;
        break;
      case vk::ImageLayout::ePresentSrcKHR:
        flags = vk::AccessFlagBits::eMemoryRead;
        break;
      default:
        break;
      }

      return flags;
    };

    auto const barrier = vk::ImageMemoryBarrier()
                             .setSrcAccessMask(srcAccessMask)
                             .setDstAccessMask(DstAccessMask(newLayout))
                             .setOldLayout(oldLayout)
                             .setNewLayout(newLayout)
                             .setSrcQueueFamilyIndex(0)
                             .setDstQueueFamilyIndex(0)
                             .setImage(image)
                             .setSubresourceRange(vk::ImageSubresourceRange(
                                 aspectMask, 0, 1, 0, 1));

    cmd.pipelineBarrier(src_stages, dest_stages, vk::DependencyFlagBits(), 0,
                        nullptr, 0, nullptr, 1, &barrier);
  } catch (...) {
    throw;
  };

  vk::ShaderModule prepare_vs() try {
    vert_shader_module = prepare_shader_module(read_spv("cube-vert.spv"));

    return vert_shader_module;
  } catch (...) {
    throw;
  };

  vk::ShaderModule prepare_fs() try {
    frag_shader_module = prepare_shader_module(read_spv("cube-frag.spv"));

    return frag_shader_module;
  } catch (...) {
    throw;
  };

  vk::ShaderModule prepare_shader_module(std::vector<char> code) try {
    auto const moduleCreateInfo = vk::ShaderModuleCreateInfo()
                                      .setCodeSize(code.size())
                                      .setPCode((uint32_t const *)code.data());

    vk::ShaderModule module = device.createShaderModule(moduleCreateInfo);

    code.clear();
    code.shrink_to_fit();

    return module;
  } catch (...) {
    throw;
  };

  std::vector<char> read_spv(std::string filename) try {
    std::ifstream fp(filename, std::ios::binary);

    std::vector<char> shader_code((std::istreambuf_iterator<char>(fp)),
                                  (std::istreambuf_iterator<char>()));

    fp.close();

    return shader_code;
  } catch (...) {
    throw;
  };
};
} // namespace vulkan_raytraced_cube_demo

int main(int argc, char **argv) {
  try {
    vulkan_raytraced_cube_demo::vulkan_raytraced_cube(
        "Vulkan Raytraced Cube Demo", 500, 500);
  } catch (std::exception e) {
    std::cout << "Exception thrown!" << std::endl
              << e.what() << std::endl
              << "Press enter to exit." << std::endl;
    std::cin.get();
  }
}