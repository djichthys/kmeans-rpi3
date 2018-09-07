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

namespace algo
{

template <>
class Kmeans_HW<float, g_type::hw_simd> : public Kmeans_CPU<float>
{
public:
  Kmeans_HW() = delete;
#if 0 
  Kmeans_HW(std::vector<float>& buff, uint32_t cols, uint32_t num_k, uint32_t max_iter)
      : Kmeans_CPU<float>(buff, cols, num_k, g_type::hw_simd, max_iter)
  {
  }
  Kmeans_HW(std::vector<float>& buff, uint32_t cols, std::vector<float> c_list, uint32_t max_iter)
      : Kmeans_CPU<float>(buff, cols, c_list, g_type::hw_simd, max_iter)
  {
  }
  Kmeans_HW(std::vector<float>& buff, uint32_t cols,
     std::vector<float, util::Align_Mem<float, 128>> c_list, uint32_t max_iter)
      : Kmeans_CPU<float>(buff, cols, c_list, g_type::hw_simd, max_iter)
  {
  }
#endif
  Kmeans_HW(std::vector<float>&, uint32_t, uint32_t, uint32_t);
  Kmeans_HW(std::vector<float>&, uint32_t, std::vector<float>, uint32_t);
  Kmeans_HW(std::vector<float>&, uint32_t, std::vector<float, util::Align_Mem<float, 128>>,
            uint32_t);

  virtual void calc();

protected:
  virtual float distance(uint32_t, uint32_t);
  virtual void alloc_centroid();
  virtual void zero_centroids();
  virtual void zero_num_points();
  virtual void reinit_centroids();
  virtual void move_data_pt(uint32_t, uint32_t, uint32_t);
};
}
