/*!
 * This program does k-means classification on data points on ARM
 * based CPUs. Where possible, hardware acceleration is used.
 * Copyright (C) 2018  Dejice Jacob
 *
 *
 * This file is part of kmeans-rpi3.
 *
 * hetero-examples is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * kmeans-rpi3 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with kmeans-rpi3.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <vector>
#include <memory>
#include <limits>
#include <exception>
#include <chrono>
#include <arm_neon.h>

#include <g_types.h>
#include <utils.h>
#include <kmeans.h>
#include <hw/interface.h>
#include <hw/simd.h>

namespace algo
{

Kmeans_HW<float, g_type::hw_simd>::Kmeans_HW(std::vector<float>& buff, uint32_t cols,
                                             uint32_t num_k, uint32_t max_iter)
    : Kmeans_CPU<float>(buff, cols, num_k, g_type::hw_simd, max_iter)
{
}

Kmeans_HW<float, g_type::hw_simd>::Kmeans_HW(std::vector<float>& buff, uint32_t cols,
                                             std::vector<float> c_list, uint32_t max_iter)
    : Kmeans_CPU<float>(buff, cols, c_list, g_type::hw_simd, max_iter)
{
}
Kmeans_HW<float, g_type::hw_simd>::Kmeans_HW(std::vector<float>& buff, uint32_t cols,
                                             std::vector<float, util::Align_Mem<float, 128>> c_list,
                                             uint32_t max_iter)
    : Kmeans_CPU<float>(buff, cols, c_list, g_type::hw_simd, max_iter)
{
}

void Kmeans_HW<float, g_type::hw_simd>::alloc_centroid()
{
  float acc;
  uint32_t data_rows = this->data_plane().size(), cdata_rows = this->cdata_plane().size();
  uint32_t d_idx, c_idx, inew = 0;
  for (d_idx = 0; d_idx < data_rows; d_idx++) {
    float best = std::numeric_limits<float>::infinity();
    for (c_idx = 0; c_idx < cdata_rows; c_idx++) {
      acc = this->distance(d_idx, c_idx);
      if (acc < best) {
        best = acc;
        inew = c_idx;
      }
    }
    if (this->clist()[d_idx] != inew)
      this->clist()[d_idx] = inew;
  }
}

void Kmeans_HW<float, g_type::hw_simd>::zero_centroids()
{
  uint32_t idx, _size = this->cdata().size();
  uint32_t _strides = _size / 4;

  float* _buff = &(this->cdata()[0]);

  float32x4_t _vzero = vmovq_n_f32(0.0f);
  for (idx = 0; idx < _strides; idx++)
    vst1q_f32(&_buff[idx * 4], _vzero);

  for (idx = 4 * idx; idx < _size; idx++)
    _buff[idx] = 0;
}

void Kmeans_HW<float, g_type::hw_simd>::zero_num_points()
{
  uint32_t idx, _size = this->num_pt().size();
  uint32_t _strides = _size / 4;

  uint32_t* _buff = &(this->num_pt()[0]);

  uint32x4_t _vzero = vmovq_n_u32(0u);
  for (idx = 0; idx < _strides; idx++)
    vst1q_u32(&_buff[idx * 4], _vzero);

  for (idx = 4 * idx; idx < _size; idx++)
    _buff[idx] = 0;
}

void Kmeans_HW<float, g_type::hw_simd>::calc()
{
  bool short_circuit = false;
  std::chrono::high_resolution_clock::time_point cstart, cend;

  this->profile(true);

  this->alloc_centroid();
  this->zero_centroids();
  this->zero_num_points();
  this->reinit_centroids();
  short_circuit = this->compute_centroids();

  this->profile(false);
}

/*!
 *  \note This function will work only on assumption that the number of columns
 *        is a mulitple of 4
 */
float Kmeans_HW<float, g_type::hw_simd>::distance(uint32_t data_row, uint32_t centroid_row)
{
  uint32_t idx = 0, cols = this->cols();
  float* d_row = this->data_plane()[data_row];
  float* c_row = this->cdata_plane()[centroid_row];
  float tot = 0.0f;

  uint32_t _blks = cols / 4;
  float32x4_t _vtot = vmovq_n_f32(0.0f);
  for (idx = 0; idx < _blks; idx++) {
    float32x4_t _vdiff = vsubq_f32(vld1q_f32(&d_row[idx * 4]), vld1q_f32(&c_row[idx * 4]));
    _vtot = vmlaq_f32(_vtot, _vdiff, _vdiff);
  }
  float32x2_t _vpart = vadd_f32(vget_high_f32(_vtot), vget_low_f32(_vtot));
  _vpart = vpadd_f32(_vpart, _vpart);
  return vget_lane_f32(_vpart, 0);
}

/*!
 *  \note This function will work only on assumption that the number of columns
 *        is a mulitple of 4
 */
void Kmeans_HW<float, g_type::hw_simd>::reinit_centroids()
{
  uint32_t row, col, it;
  uint32_t num_rows = this->data_plane().size(), num_cols = this->cols();
  uint32_t _dstrides, _cstrides = num_cols / 4;

  /*!
   * for each row accummulate num-of-pts and
   *  column-wise totals for each centroid
   */
  for (uint32_t row = 0; row < num_rows; row++) {
    /* accumulate number of points in each centroid */
    it = this->clist()[row];
    this->num_pt()[it]++;
    for (uint32_t col = 0; col < _cstrides; col++) {
      float32x4_t _vdata = vld1q_f32(&(this->data_plane()[row][col * 4]));
      float32x4_t _vcdata = vld1q_f32(&(this->cdata_plane()[it][col * 4]));
      _vcdata = vaddq_f32(_vcdata, _vdata);
      vst1q_f32(&(this->cdata_plane()[it][col * 4]), _vcdata);

    } /* for each 4 columns - accumulate */
  }

  /* get the average of all the axes in each centroid */
  num_rows = this->cdata_plane().size();
  _dstrides = num_rows / 4;
  for (row = 0; row < _dstrides; row++) {
    float32x4_t _vf_numpt0 = vcvtq_f32_u32(vld1q_u32(&(this->num_pt()[row * 4])));

    /*!
     * take reciprocal of num-points. To improve accuracy
     * do a single newton-raphson iteration first approximation
     */
    float32x4_t _vdiv_x0 = vrecpeq_f32(_vf_numpt0);
    _vf_numpt0 = vmulq_f32(_vdiv_x0, vrecpsq_f32(_vf_numpt0, _vdiv_x0));

    /* broadcast to 4 vectors for simultaneous division */
    float32x4_t _vf_numpt1 = vdupq_n_f32(vgetq_lane_f32(_vf_numpt0, 1));
    float32x4_t _vf_numpt2 = vdupq_n_f32(vgetq_lane_f32(_vf_numpt0, 2));
    float32x4_t _vf_numpt3 = vdupq_n_f32(vgetq_lane_f32(_vf_numpt0, 3));
    _vf_numpt0 = vdupq_n_f32(vgetq_lane_f32(_vf_numpt0, 0));

    /* average out each accumulated dimension */
    for (col = 0; col < _cstrides; col++) {
      float32x4_t _vcdata0 = vld1q_f32(&(this->cdata_plane()[row * 4][col * 4]));
      float32x4_t _vcdata1 = vld1q_f32(&(this->cdata_plane()[row * 4 + 1][col * 4]));
      float32x4_t _vcdata2 = vld1q_f32(&(this->cdata_plane()[row * 4 + 2][col * 4]));
      float32x4_t _vcdata3 = vld1q_f32(&(this->cdata_plane()[row * 4 + 3][col * 4]));

      /*!
       * Vector division to obtain average
       * _vf_numpt contains inverse of distance
       */
      _vcdata0 = vmulq_f32(_vcdata0, _vf_numpt0);
      _vcdata1 = vmulq_f32(_vcdata1, _vf_numpt1);
      _vcdata2 = vmulq_f32(_vcdata2, _vf_numpt2);
      _vcdata3 = vmulq_f32(_vcdata3, _vf_numpt3);

      /* Store back into centroid data */
      vst1q_f32(&(this->cdata_plane()[row * 4][col * 4]), _vcdata0);
      vst1q_f32(&(this->cdata_plane()[row * 4 + 1][col * 4]), _vcdata1);
      vst1q_f32(&(this->cdata_plane()[row * 4 + 2][col * 4]), _vcdata2);
      vst1q_f32(&(this->cdata_plane()[row * 4 + 3][col * 4]), _vcdata3);
    }
  }

  /* Do the remainder of the rows over here. dont unroll the loop */
  for (row = row * 4; row < num_rows; row++) {
    float32x4_t _vf_numpt = vcvtq_f32_u32(vmovq_n_u32(this->num_pt()[row]));
    /*!
     * take reciprocal of num-points. To improve accuracy
     * do a single newton-raphson iteration first approximation
     */
    float32x4_t _vdiv_x0 = vrecpeq_f32(_vf_numpt);
    _vf_numpt = vmulq_f32(_vdiv_x0, vrecpsq_f32(_vf_numpt, _vdiv_x0));

    for (col = 0; col < _cstrides; col++) {
      float32x4_t _vcdata = vld1q_f32(&(this->cdata_plane()[row][col * 4]));
      _vcdata = vmulq_f32(_vcdata, _vf_numpt);
      vst1q_f32(&(this->cdata_plane()[row][col * 4]), _vcdata);
    }
  }
}

void Kmeans_HW<float, g_type::hw_simd>::move_data_pt(uint32_t dest_row, uint32_t src_row,
                                                     uint32_t data_row)
{
  uint32_t num_cols = this->cols(), _stride = num_cols / 4;
  float* dest = this->cdata_plane()[dest_row];
  float* src = this->cdata_plane()[src_row];
  float* data = this->data_plane()[data_row];
  uint32_t* dest_numpt = &(this->num_pt()[dest_row]);
  uint32_t* src_numpt = &(this->num_pt()[src_row]);

  for (uint32_t col = 0; col < _stride; col++) {
    float32x4_t _vsrc = vld1q_f32(&src[col * 4]);
    float32x4_t _vdata = vld1q_f32(&data[col * 4]);
    float32x4_t _vdest = vld1q_f32(&dest[col * 4]);
    float32x4_t _vf_dest_numpt = vcvtq_f32_u32(vdupq_n_u32(this->num_pt()[dest_row]));
    float32x4_t _vf_src_numpt = vcvtq_f32_u32(vdupq_n_u32(this->num_pt()[src_row]));

    /*!
     * take reciprocal of num-points for both destination and source rows.
     * To improve accuracy do * a single newton-raphson iteration first
     * approximation
     */
    float32x4_t _vdest_x0 = vrecpeq_f32(_vf_dest_numpt);
    _vf_dest_numpt = vmulq_f32(_vdest_x0, vrecpsq_f32(_vf_dest_numpt, _vdest_x0));

    float32x4_t _vsrc_x0 = vrecpeq_f32(_vf_src_numpt);
    _vf_src_numpt = vmulq_f32(_vsrc_x0, vrecpsq_f32(_vf_src_numpt, _vsrc_x0));

    _vsrc = vaddq_f32(_vsrc, vmulq_f32(vsubq_f32(_vsrc, _vdata), _vf_src_numpt));
    _vdest = vaddq_f32(_vdest, vmulq_f32(vsubq_f32(_vdata, _vdest), _vf_dest_numpt));

    vst1q_f32(&src[col * 4], _vsrc);
    vst1q_f32(&dest[col * 4], _vdest);
  }
}
}
