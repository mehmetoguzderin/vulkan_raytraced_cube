// To Do
/*
 * Improve code consistency
 * Improve naming convention
 * Improve intersection function
 * Improve
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

// Inputs
layout(location = 0) in vec2 plane_uv;
layout(location = 1) in vec3 ray_origin;
layout(location = 2) in vec3 ray_direction;

// Outputs
layout(location = 0) out vec4 uFragColor;

// Functions
// Moller-Trumbore Ray-Triangle Intersection
float intersection(vec3 origin, vec3 direction, vec3 v0, vec3 v1, vec3 v2,
                   vec2 uv0, vec2 uv1, vec2 uv2, out vec3 position,
                   out vec2 texcoord) {
  vec3 e1 = v1 - v0;
  vec3 e2 = v2 - v0;
  vec3 normal = normalize(cross(e1, e2));
  float b = dot(normal, direction);
  vec3 w0 = origin - v0;
  float a = -dot(normal, w0);
  float t = a / b;
  vec3 p = origin + t * direction;
  float uu, uv, vv, wu, wv, invd;
  uu = dot(e1, e1);
  uv = dot(e1, e2);
  vv = dot(e2, e2);
  vec3 w = p - v0;
  wu = dot(w, e1);
  wv = dot(w, e2);
  invd = uv * uv - uu * vv;
  invd = 1.0f / invd;
  float u = (uv * wv - vv * wu) * invd;
  if (u < 0.0f || u > 1.0f)
    return -1.0f;
  float v = (uv * wu - uu * wv) * invd;
  if (v < 0.0f || (u + v) > 1.0f)
    return -1.0f;
  vec3 bc = vec3(1 - u - v, u, v);
  position = bc.x * v0 + bc.y * v1 + bc.z * v2;
  texcoord = bc.x * uv0 + bc.y * uv1 + bc.z * uv2;
  return t;
}

// Export
void main() {
  // Input normalization
  vec3 normalized_ray_direction = normalize(ray_direction);

  // Ray tracing
  bool result = false;
  float result_time = 1000000.0;
  vec3 result_position = vec3(0.0);
  vec2 result_texcoord = vec2(0.0);
  for (int i = 0; i < triangles.length(); i++) {
    vec3 a = triangles[i].pos_uv[0].xyz;
    vec3 b = triangles[i].pos_uv[1].xyz;
    vec3 c = triangles[i].pos_uv[2].xyz;
    vec2 x = vec2(triangles[i].pos_uv[0][3], triangles[i].pos_uv[1][3]);
    vec2 y = vec2(triangles[i].pos_uv[3][0], triangles[i].pos_uv[3][1]);
    vec2 z = vec2(triangles[i].pos_uv[3][2], triangles[i].pos_uv[3][3]);

    vec3 position;
    vec2 texcoord;
    float time = intersection(ray_origin, normalized_ray_direction, a, b, c, x,
                              y, z, position, texcoord);

    if (time > 0.0 && time < result_time) {
      result = true;
      result_time = time;
      result_position = position;
      result_texcoord = texcoord;
    }
  }

  // Outputs
  if (result) {
    uFragColor = texture(tex, result_texcoord);
  } else {
    uFragColor = vec4(0.2f, 0.2f, 0.2f, 0.2f);
  }
}