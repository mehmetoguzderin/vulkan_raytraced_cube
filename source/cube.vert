// To Do
/*
 * Improve code consistency
 * Improve naming convention
 * Improve performance
 * Improve data layouts
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
 */

#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

// Structures
struct vktexcube_triangle_storage {
  /*
   * Data Layout
   * pos_uv[0] = vertex position, u[0]
   * pos_uv[1] = vertex position, v[0]
   * pos_uv[2] = vertex position, -
   * pos_uv[3] = u[1], v[1], u[2], v[2]
   */
  vec4 pos_uv[4];
};

// Bindings
layout(std140, binding = 0) uniform plane_buffer {
  /*
   * Data Layout
   * eye[0].xyz = eye left
   * eye[1].xyz = eye right
   * eye[2].xyz = eye down
   * eye[3].xyz = eye up
   * eyeOrigin = eye[0][3], eye[1][3], eye[2][3]
   */
  mat4 eye;
  vec4 pos_uv[6];
  vec4 screen_size;
}
plane;
layout(binding = 1) uniform sampler2D tex;
layout(std140, binding = 2) buffer Cube {
  vktexcube_triangle_storage triangles[];
};

// Outputs
layout(location = 0) out vec2 plane_uv;
layout(location = 1) out vec3 ray_origin;
layout(location = 2) out vec3 ray_direction;

// Export
void main() {
  // Fetching current vertex
  vec4 current_vertex = plane.pos_uv[gl_VertexIndex];

  // Ray direction calculation
  vec3 ray_direction_x =
      mix(plane.eye[0].xyz, plane.eye[1].xyz, current_vertex[2]);
  vec3 ray_direction_y =
      mix(plane.eye[2].xyz, plane.eye[3].xyz, current_vertex[3]);

  // gl_Position
  gl_Position = vec4(current_vertex[0], current_vertex[1], 0.0, 1.0);

  // Outputs
  plane_uv = vec2(current_vertex[2], current_vertex[3]);
  ray_origin = vec3(plane.eye[0][3], plane.eye[1][3], plane.eye[2][3]);
  ray_direction = normalize(ray_direction_x + ray_direction_y);
}
