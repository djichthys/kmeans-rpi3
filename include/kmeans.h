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

template <typename T>
class Kmeans_CPU
{
private:
  g_type::Hardware_Type hw_type;
  std::vector<T, util::Align_Mem<T, 128>> _data;
  std::vector<T*> _data_plane;
  std::vector<T, util::Align_Mem<T, 128>> _cdata;
  std::vector<T*> _cdata_plane;

  /* items will be equal num of centroids */
  std::vector<float, util::Align_Mem<T, 128>> _avg_list;
  /* point->centroid map using a vector for speed */
  std::vector<uint32_t, util::Align_Mem<T, 128>> _clist;
  /* centroid-> num_of_points map */
  std::vector<uint32_t, util::Align_Mem<T, 128>> _num_pt;
  uint32_t _cols;
  uint32_t _num_k;
  uint32_t _max_iter;

  void create_centroids(uint32_t);
  std::chrono::high_resolution_clock::time_point clk_start, clk_end;

public:
  Kmeans_CPU() = delete;
  Kmeans_CPU(std::vector<T>& buff, uint32_t cols, uint32_t num_k, uint32_t max_iter)
      : Kmeans_CPU(buff, cols, num_k, g_type::hw_cpu, max_iter)
  {
  }
  Kmeans_CPU(std::vector<T>& buff, uint32_t cols, std::vector<T> c_list, uint32_t max_iter)
      : Kmeans_CPU(buff, cols, c_list, g_type::hw_cpu, max_iter)
  {
  }
  Kmeans_CPU(std::vector<T>& buff, uint32_t cols, std::vector<T, util::Align_Mem<T, 128>> c_list,
             uint32_t max_iter)
      : Kmeans_CPU(buff, cols, c_list, g_type::hw_cpu, max_iter)
  {
  }
  Kmeans_CPU(std::vector<T>&, uint32_t, uint32_t, g_type::Hardware_Type, uint32_t);
  Kmeans_CPU(std::vector<T>&, uint32_t, std::vector<T>&, g_type::Hardware_Type, uint32_t);
  Kmeans_CPU(std::vector<T>&, uint32_t, std::vector<T, util::Align_Mem<T, 128>>&,
             g_type::Hardware_Type, uint32_t);

  std::vector<T, util::Align_Mem<T, 128>>& data() { return this->_data; }
  std::vector<T*>& data_plane() { return this->_data_plane; }

  std::vector<T, util::Align_Mem<T, 128>>& cdata() { return this->_cdata; }
  std::vector<T*>& cdata_plane() { return this->_cdata_plane; }

  std::vector<float, util::Align_Mem<T, 128>>& avg_list() { return this->_avg_list; }
  std::vector<uint32_t, util::Align_Mem<T, 128>>& clist() { return this->_clist; }
  std::vector<uint32_t, util::Align_Mem<T, 128>>& num_pt() { return this->_num_pt; }

  uint32_t cols() { return this->_cols; }
  g_type::Hardware_Type accelerator() { return this->hw_type; }
  uint32_t& max_iter() { return this->_max_iter; }

  virtual void calc();

  template <typename Alloc = std::allocator<T>>
  std::unique_ptr<std::vector<T, Alloc>> copy_data(Alloc&& = std::allocator<T>());

  template <typename Alloc = std::allocator<T>>
  std::unique_ptr<std::vector<T, Alloc>> copy_centroids(Alloc&& = std::allocator<T>());

  uint64_t duration();

protected:
  void profile(bool);
  virtual T distance(uint32_t, uint32_t);
  virtual void alloc_centroid();
  virtual void zero_centroids();
  virtual void zero_num_points();
  virtual void reinit_centroids();
  virtual bool compute_centroids();
  virtual void move_data_pt(uint32_t, uint32_t, uint32_t);
};

template <typename T>
void Kmeans_CPU<T>::create_centroids(uint32_t num_k)
{
  /* Create centroid points */
  T max = std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity()
                                               : std::numeric_limits<T>::max();
  uint32_t cols = this->cols(), rows = this->data().size() / cols;
  uint32_t _seg_size = rows / num_k;
  uint32_t idx_i, idx_j;

  this->_cdata.reserve(num_k * cols);
  this->_avg_list.reserve(num_k);

  for (idx_i = 0; idx_i < num_k; idx_i++) {
    uint32_t c_row = util::random_pt(_seg_size, 1024) + (idx_i * _seg_size);

    for (idx_j = 0; idx_j < cols; idx_j++) {
      this->_cdata.push_back(this->_data_plane[c_row][idx_j]);
    }
    this->_cdata_plane.push_back(&(this->_cdata[idx_i * cols]));
    this->_avg_list.push_back(max);
  }
}

template <typename T>
Kmeans_CPU<T>::Kmeans_CPU(std::vector<T>& buff, uint32_t cols, uint32_t num_k,
                          g_type::Hardware_Type hw_type, uint32_t max_iter)
    : hw_type(hw_type),
      _cols(cols),
      _num_k(num_k),
      _clist(std::vector<uint32_t, util::Align_Mem<T, 128>>(buff.size() / cols, 0)),
      _num_pt(std::vector<uint32_t, util::Align_Mem<T, 128>>(num_k, 0)),
      _max_iter(max_iter)
{
  T max = std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity()
                                               : std::numeric_limits<T>::max();
  this->_data.reserve(buff.size());
  for (auto& it : buff)
    this->_data.push_back(it);

  uint32_t rows = buff.size() / this->_cols;
  this->_data_plane.reserve(rows);
  for (uint32_t idx = 0; idx < rows; idx++)
    this->_data_plane.push_back(&(this->_data[0]) + idx * this->cols());

  this->create_centroids(this->_num_k);
}

template <typename T>
Kmeans_CPU<T>::Kmeans_CPU(std::vector<T>& buff, uint32_t cols, std::vector<T>& c_list,
                          g_type::Hardware_Type hw_type, uint32_t max_iter)
    : hw_type(hw_type),
      _cols(cols),
      _num_k(c_list.size() / cols),
      _clist(std::vector<uint32_t, util::Align_Mem<T, 128>>(buff.size() / cols, 0)),
      _num_pt(std::vector<uint32_t, util::Align_Mem<T, 128>>(c_list.size() / cols, 0)),
      _max_iter(max_iter)
{
  T max = std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity()
                                               : std::numeric_limits<T>::max();
  this->_data.reserve(buff.size());
  for (auto& it : buff)
    this->_data.push_back(it);

  uint32_t rows = buff.size() / this->_cols;
  this->_data_plane.reserve(rows);
  for (uint32_t idx = 0; idx < rows; idx++)
    this->_data_plane.push_back(&(this->_data[0]) + idx * this->cols());

  /* Copy over the centroids */
  this->_cdata.reserve(this->_num_k);
  for (auto& it : c_list)
    this->_cdata.push_back(it);

  rows = this->_num_k;
  this->_cdata_plane.reserve(this->_num_k);
  for (uint32_t idx = 0; idx < rows; idx++)
    this->_cdata_plane.push_back(&(this->_cdata[0]) + idx * this->cols());

  this->_avg_list.reserve(this->_num_k);
  for (uint32_t idx = 0; idx < this->_num_k; idx++)
    this->_avg_list.push_back(max);
}

template <typename T>
Kmeans_CPU<T>::Kmeans_CPU(std::vector<T>& buff, uint32_t cols,
                          std::vector<T, util::Align_Mem<T, 128>>& c_list,
                          g_type::Hardware_Type type, uint32_t max_iter)
    : hw_type(hw_type),
      _cols(cols),
      _cdata(c_list),
      _num_k(c_list.size() / cols),
      _clist(std::vector<uint32_t, util::Align_Mem<T, 128>>(buff.size() / cols, 0)),
      _num_pt(std::vector<uint32_t, util::Align_Mem<T, 128>>(c_list.size() / cols, 0)),
      _max_iter(max_iter)
{
  T max = std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity()
                                               : std::numeric_limits<T>::max();

  this->_data.reserve(buff.size());
  for (auto& it : buff)
    this->_data.push_back(it);

  uint32_t rows = buff.size() / this->_cols;
  this->_data_plane.reserve(rows);
  for (uint32_t idx = 0; idx < rows; idx++)
    this->_data_plane.push_back(&(this->_data[0]) + idx * this->cols());

  rows = this->_num_k;
  this->_cdata_plane.reserve(this->_num_k);
  for (uint32_t idx = 0; idx < rows; idx++)
    this->_cdata_plane.push_back(&(this->_cdata[0]) + idx * this->cols());

  this->_avg_list.reserve(this->_num_k);
  for (uint32_t idx = 0; idx < this->_num_k; idx++)
    this->_avg_list.push_back(max);
}
template <typename T>
void Kmeans_CPU<T>::alloc_centroid()
{
  T acc;
  uint32_t num_data = this->data_plane().size(), num_cdata = this->cdata_plane().size();
  uint32_t d_idx, c_idx, inew = 0;
  for (d_idx = 0; d_idx < num_data; d_idx++) {
    T best = std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity()
                                                  : std::numeric_limits<T>::max();
    for (c_idx = 0; c_idx < num_cdata; c_idx++) {
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

template <typename T>
void Kmeans_CPU<T>::zero_centroids()
{
  for (auto& it : this->cdata())
    it = 0;
}

template <typename T>
void Kmeans_CPU<T>::zero_num_points()
{
  for (auto& it : this->num_pt())
    it = 0;
}

template <typename T>
void Kmeans_CPU<T>::calc()
{
  std::chrono::high_resolution_clock::time_point cstart, cend;

  this->profile(true);

  this->alloc_centroid();
  this->zero_centroids();
  this->zero_num_points();
  this->reinit_centroids();
  this->compute_centroids();

  this->profile(false);
}

template <typename T>
T Kmeans_CPU<T>::distance(uint32_t data_row, uint32_t centroid_row)
{
  uint32_t idx_i, idx_j;
  T* d_row = this->_data_plane[data_row];
  T* c_row = this->_cdata_plane[centroid_row];
  T tot = 0.0f;
  for (idx_i = 0; idx_i < this->_cols; idx_i++) {
    T _tmp = d_row[idx_i] - c_row[idx_i];
    tot += _tmp * _tmp;
  }
  return tot;
}

template <typename T>
void Kmeans_CPU<T>::reinit_centroids()
{
  uint32_t num_rows = this->data_plane().size();
  uint32_t it, row, col;
  for (row = 0; row < num_rows; row++) {
    /* accumulate number of points in each centroid */
    it = this->clist()[row];
    this->num_pt()[it]++;
    for (col = 0; col < this->_cols; col++)
      this->cdata_plane()[it][col] += this->data_plane()[row][col];
  }

  /* get the average of all the axes in each centroid */
  num_rows = this->cdata_plane().size();
  for (row = 0; row < num_rows; row++) {
    for (col = 0; col < this->_cols; col++)
      this->cdata_plane()[row][col] /= this->num_pt()[row];
  }
}

template <typename T>
bool Kmeans_CPU<T>::compute_centroids()
{
  bool updated = true;
  T acc;
  uint32_t num_data = this->data_plane().size(), num_cdata = this->cdata_plane().size();
  uint32_t pt_old, pt_new;

  /*!
   * Continue algorithm until no inter-centroid migration of data points occur
   * or until we reach the maximum number of iterations
   */
  for (uint32_t iter = 0; updated && (iter < this->max_iter()); iter++) {
    updated = false;
    /* for each data point ascertain and recalculate centroids */
    for (uint32_t d_idx = 0; d_idx < num_data; d_idx++) {
      T best = std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity()
                                                    : std::numeric_limits<T>::max();

      /* obtain least distance between data point and each centroid */
      for (uint32_t c_idx = 0; c_idx < num_cdata; c_idx++) {
        acc = this->distance(d_idx, c_idx);
        if (acc < best) {
          best = acc;
          pt_new = c_idx;
        }
      } /* obtain least distance between data point and each centroid */

      /* check if any point has moved from one centroid to another */
      if ((pt_old = this->clist()[d_idx]) != pt_new) {
        updated = true;
        this->clist()[d_idx] = pt_new;
        this->num_pt()[pt_new]++;
        this->num_pt()[pt_old]--;
        this->move_data_pt(pt_new, pt_old, d_idx);
      }
    } /* for each data point ascertain and recalculate centroids */
  }   /* do until no inter-data migration -or- maximum iterations */

  return updated; /* if true - we have reached max iterations */
}

template <typename T>
void Kmeans_CPU<T>::move_data_pt(uint32_t dest_row, uint32_t src_row, uint32_t data_row)
{
  uint32_t num_cols = this->cols();
  T* dest = this->cdata_plane()[dest_row];
  T* src = this->cdata_plane()[src_row];
  T* data = this->data_plane()[data_row];

  for (uint32_t col = 0; col < num_cols; col++) {
    src[col] += (src[col] - data[col]) / this->num_pt()[src_row];
    dest[col] += (data[col] - dest[col]) / this->num_pt()[dest_row];
  }
}

/*!
 * \return  difference between profile(true) and profile(false)
 */
template <typename T>
uint64_t Kmeans_CPU<T>::duration()
{
  return std::chrono::duration_cast<std::chrono::microseconds>(this->clk_end - this->clk_start)
      .count();
}

template <typename T>
void Kmeans_CPU<T>::profile(bool restart)
{
  if (restart)
    this->clk_start = std::chrono::high_resolution_clock::now();
  else
    this->clk_end = std::chrono::high_resolution_clock::now();
}

template <typename T>
template <typename A>
std::unique_ptr<std::vector<T, A>> Kmeans_CPU<T>::copy_data(A&& allocator)
{
  std::unique_ptr<std::vector<T, A>> _ptr = nullptr;
  _ptr = std::make_unique<std::vector<T, A>>(this->data().begin(), this->data().end(), allocator);
  return std::move(_ptr);
}

template <typename T>
template <typename A>
std::unique_ptr<std::vector<T, A>> Kmeans_CPU<T>::copy_centroids(A&& allocator)
{
  std::unique_ptr<std::vector<T, A>> _ptr = nullptr;
  _ptr = std::make_unique<std::vector<T, A>>(this->cdata().begin(), this->cdata().end(), allocator);
  return std::move(_ptr);
}
}
