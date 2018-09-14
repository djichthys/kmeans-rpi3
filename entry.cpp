/*!
 * This program does k-means classification on data points on ARM
 * based CPUs. Where possible, hardware acceleration is used.
 * Copyright (C) 2018  Dejice Jacob
 *
 *
 * This file is part of kmeans-rpi3
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
#include <exception>
#include <sstream>
#include <fstream>
#include <cstring>
#include <limits>
#include <chrono>

#include <api_error.h>
#include <g_types.h>
#include <cmdline.h>
#include <parser.h>
#include <utils.h>
#include <data_container.h>

#include <kmeans.h>
#include <hw/interface.h>
#include <hw/simd.h>

/* Local Function Declarations */
static err::api_Err_Status read_file(g_type::Data_Type ty, std::unique_ptr<std::string>,
                                     std::string&, parser::DC_Wrapper*&);

template <typename T1>
static err::api_Err_Status _read_file_t(std::unique_ptr<std::string>, std::string&,
                                        parser::DC_Wrapper*&);
template <typename T1>
static std::unique_ptr<algo::Kmeans_CPU<T1>>
    get_exec_ctx(parser::Data_Container<T1, 2>*,
                 util::Expected<parser::Data_Container<T1, 2>*, uint32_t>&, g_type::Hardware_Type,
                 uint32_t = DefaultMaxIterations);

/* program options should be globally accessible */
std::weak_ptr<parser::Program_Options> g_opt;

int main(int argc, char* argv[])
{
  err::api_Err_Status _err = err::api_Success;

  std::shared_ptr<parser::Program_Options> opt =
      std::make_shared<parser::Program_Options>(argc, argv);
  g_opt = opt;
  std::unique_ptr<algo::Kmeans_CPU<float>> kmeans = nullptr, kmeans_simd = nullptr;

  /*!
   * Parse the raw options and store user options
   */
  try {
    opt->parse(argc, argv);
    opt->display_options();
  } catch (std::exception& parse_x) {
    std::cout << "=================================" << std::endl;
    std::cout << "exception reading commandline options . Exception >> " << parse_x.what()
              << std::endl;
    std::exit(-256);
  }

  // Set-up data and pick same centroids for both CPU and SIMD versions
  try {
    parser::DC_Wrapper *d_wrap = nullptr, *c_wrap = nullptr;
    parser::Data_Container<float, 2>* data_2d = nullptr;
    parser::Data_Container<float, 2>* centroid_2d = nullptr;

    // Read data file for input data
    parser::File_Parser<std::string, char> data_pt(opt->filename());
    data_pt.read_file(); /* Read raw text file and populate memory */

    _err = read_file(opt->data_type(), data_pt.mv_raw_buff(), opt->separators(), d_wrap);
    if (_err != err::api_Success) {
      std::cerr << "Error Reading / Creating Data Container for Data points" << std::endl;
      throw std::runtime_error("Error Reading / Creating Data Container for Data points");
    }

    data_2d = dynamic_cast<parser::Data_Container<float, 2>*>(d_wrap);
    if (data_2d == nullptr) {
      std::cerr << "Data :: K-means is only done for 2-Dimensional float values" << std::endl;
      throw std::runtime_error("Data :: K-means is only done for 2-Dimensional float values");
    }

    // Create initial centroids by either
    // 1) reading from a file that user provides -or-
    // 2) random points (num_k) within data set, num of centroids are to be
    // provided by the user
    if (opt->k_val()) {
      parser::File_Parser<std::string, char> _cbuff_txt(opt->k_val().expected());
      _cbuff_txt.read_file(); /* Read raw text file and populate memory */

      // Read and format data from file and create a data-container
      _err = read_file(opt->data_type(), _cbuff_txt.mv_raw_buff(), opt->separators(), c_wrap);
      if (_err != err::api_Success) {
        std::cerr << "Error Reading / Creating Data Container for Centroids" << std::endl;
        throw std::runtime_error("Error Reading / Creating Data Container for Centroids");
      }

      centroid_2d = dynamic_cast<parser::Data_Container<float, 2>*>(c_wrap);
      if (centroid_2d == nullptr) {
        std::cerr << "Centroids :: K-means is only done for 2-Dimensional "
                     "float values"
                  << std::endl;
        throw std::runtime_error("Centroids ::K-means is only done for 2-Dimensional float values");
      }

      util::Expected<parser::Data_Container<float, 2>*, uint32_t> centroid(centroid_2d);

      // Get execution context for standard CPU version of code
      kmeans = get_exec_ctx<float>(data_2d, centroid, g_type::hw_cpu);

    } else {
      util::Expected<parser::Data_Container<float, 2>*, uint32_t> centroid(
          opt->k_val().unexpected());
      // Get execution context for standard CPU version of code
      kmeans = get_exec_ctx<float>(data_2d, centroid, g_type::hw_cpu);

      // Extract same centroids from the CPU context and copy over to SIMD
      // context */
      std::unique_ptr<std::vector<float>> _initial_centroids = kmeans->copy_centroids();
      c_wrap = new parser::Data_Container<float, 2>(*_initial_centroids,
                                                    _initial_centroids->size() /
                                                        data_2d->dimension()->cols(),
                                                    data_2d->dimension()->cols());

      centroid_2d = dynamic_cast<parser::Data_Container<float, 2>*>(c_wrap);
      if (centroid_2d == nullptr) {
        std::cerr << "Centroids :: K-means is only done for 2-Dimensional "
                     "float values"
                  << std::endl;
        throw std::runtime_error("Centroids ::K-means is only done for 2-Dimensional float values");
      }
    }

    // Get execution context for SIMD execution. However, if
    // the number of columns is not a multiple of 4, function
    // will default back to normal CPU execution
    util::Expected<parser::Data_Container<float, 2>*, uint32_t> centroid(centroid_2d);
    kmeans_simd = get_exec_ctx<float>(data_2d, centroid, g_type::hw_simd);

    // Clean-up initial data and centroid points
    if (d_wrap) {
      delete d_wrap;
      d_wrap = nullptr;
    }
    if (c_wrap) {
      delete c_wrap;
      c_wrap = nullptr;
    }
  } catch (std::exception& parse_x) {
    std::cout << "=================================" << std::endl;
    std::cout << "exception during parsing. Exception >> " << parse_x.what() << std::endl;
    std::exit(-256);
  }

  // Do kmeans
  try {
    kmeans->calc();
    /* Display Calculated centroids */
    std::cout << "=================================" << std::endl;
    std::cout << "CPU k-means ::: time = " << kmeans->duration() << " (micro-secs)" << std::endl
              << "calculated centroids : " << std::endl;
    for (auto& it : kmeans->cdata_plane()) {
      for (uint32_t col = 0; col < kmeans->cols(); col++)
        std::cout << it[col] << ", ";
      std::cout << std::endl;
    }

    kmeans_simd->calc();
    /* Display Calculated centroids */
    std::cout << "=================================" << std::endl;
    std::cout << "SIMD k-means ::: time = " << kmeans_simd->duration() << " (micro-secs)"
              << std::endl
              << "calculated centroids : " << std::endl;
    for (auto& it : kmeans_simd->cdata_plane()) {
      for (uint32_t col = 0; col < kmeans_simd->cols(); col++)
        std::cout << it[col] << ", ";
      std::cout << std::endl;
    }
  } catch (std::exception& parse_x) {
    std::cout << "=================================" << std::endl;
    std::cout << "exception during K-means calculation. Exception >> " << parse_x.what()
              << std::endl;
    std::exit(-256);
  }

  return 0;
}

/*!
 * \param[in]  *sep - List of separators (upto 3) in 'ascending' order.
 *
 * \note       *sep The data will be parsed and spatially co-located in order of
 *             separators and array subscripts are in order of separators
 *             i.e if sep = "\n|,", then data is assumed to be in the format
 *                    a0,b0,c0,....|a1,b1,c1,.....|............\n
 *                    d0,e0,f0,....|d1,e1,f1,.....|............\n
 *                            .........................        \m
 *                            .........................        \m
 *                            .........................        \m
 *                            .........................        \m
 *                    ax,bx,cx  are in contiguous memory.
 *                    buff[i][j][k] subscripting map is i->\n, j->| , k->,
 */
template <typename T1>
static err::api_Err_Status _read_file_t(std::unique_ptr<std::string> buff, std::string& sep,
                                        parser::DC_Wrapper*& data_container)
{
  uint32_t no_of_dims = 0, max_dims = sep.length();
  err::api_Err_Status _err = err::api_Success;
  err::Debug_Level _lvl = err::debug_MaxLevel;

  {
    std::shared_ptr<parser::Program_Options> s_opt = g_opt.lock();
    _lvl = (s_opt) ? (err::Debug_Level)s_opt->verbosity() : err::debug_Critical;
  }

  /*!
   * First obtain the number of dimensions.  We will only search
   * up to the number of delimiters in the 'separator' string
   * Assumption : each of the delimiters in 'sep' are unique
   */
  for (uint32_t idx_i = 0; idx_i < (*buff).size() && no_of_dims < max_dims; idx_i++) {
    for (uint32_t idx_j = no_of_dims; idx_j < max_dims; idx_j++) {
      if ((*buff)[idx_i] == sep[idx_j]) {
        no_of_dims++;
        break;
      }
    }
  }

  if (_lvl >= err::debug_Trace) {
    /* for debugging purpose only */
    std::cout << "max-dimensionss = " << max_dims << ", num dimensions detected = " << no_of_dims
              << std::endl;
  }

  /* verify results */
  switch (no_of_dims) {
    case 1:
      data_container = new parser::Data_Container<T1, 1>();
      _err = data_container->populate_data(buff, sep);
      data_container->display(_lvl);
      break;
    case 2:
      data_container = new parser::Data_Container<T1, 2>();
      _err = data_container->populate_data(buff, sep);
      data_container->display(_lvl);
      break;
    case 3:
      data_container = new parser::Data_Container<T1, 3>();
      _err = data_container->populate_data(buff, sep);
      data_container->display(_lvl);
      break;
    default:
      std::cerr << "Can only parse 1/2/3 Dimensional data" << std::endl;
      _err = err::api_Err_Param;
      throw std::runtime_error("Can only parse 1/2/3 Dimensional data");
  }

  return _err;
}

static err::api_Err_Status read_file(g_type::Data_Type ty, std::unique_ptr<std::string> buff,
                                     std::string& sep, parser::DC_Wrapper*& data_container)
{
  err::api_Err_Status _err = err::api_Success;
  switch (ty) {
    case g_type::DataType_uint8:
      _err = _read_file_t<uint8_t>(std::move(buff), sep, data_container);
      break;
    case g_type::DataType_uint16:
      _err = _read_file_t<uint16_t>(std::move(buff), sep, data_container);
      break;
    case g_type::DataType_uint32:
      _err = _read_file_t<uint32_t>(std::move(buff), sep, data_container);
      break;
    case g_type::DataType_uint64:
      _err = _read_file_t<uint64_t>(std::move(buff), sep, data_container);
      break;
    case g_type::DataType_int8:
      _err = _read_file_t<int8_t>(std::move(buff), sep, data_container);
      break;
    case g_type::DataType_int16:
      _err = _read_file_t<int16_t>(std::move(buff), sep, data_container);
      break;
    case g_type::DataType_int32:
      _err = _read_file_t<int32_t>(std::move(buff), sep, data_container);
      break;
    case g_type::DataType_int64:
      _err = _read_file_t<int64_t>(std::move(buff), sep, data_container);
      break;
    case g_type::DataType_float:
      _err = _read_file_t<float>(std::move(buff), sep, data_container);
      break;
    case g_type::DataType_double:
      _err = _read_file_t<double>(std::move(buff), sep, data_container);
      break;
    case g_type::DataType_long_double:
      _err = _read_file_t<long double>(std::move(buff), sep, data_container);
      break;
    default:
      std::cerr << "Unknown Data type enum g_type::Data_Type(" << (uint32_t)ty << ")" << std::endl;
      throw std::runtime_error("Can only parse 1/2/3 Dimensional data");
      _err = err::api_Err_Param;
  }

  if (_err != err::api_Success)
    std::cerr << "Reading File caused error (" << _err << ")" << std::endl;

  return _err;
}

/*!
 * param[in]  num_k  - number of centroids to be generated. Overriden by
 * centroid_2d
 */
template <typename T1>
static std::unique_ptr<algo::Kmeans_CPU<T1>>
    get_exec_ctx(parser::Data_Container<T1, 2>* data_2d,
                 util::Expected<parser::Data_Container<T1, 2>*, uint32_t>& centroid,
                 g_type::Hardware_Type hw_type, uint32_t max_iter)
{
  std::unique_ptr<algo::Kmeans_CPU<T1>> ctx = nullptr;
  if (data_2d == nullptr) {
    std::cerr << "Cannot init kmeans with no data" << std::endl;
    throw std::runtime_error("Cannot init kmeans with no data");
  }

  switch (hw_type) {
    case g_type::hw_best: // fall through option
    case g_type::hw_simd:
      if (centroid) { // use given centroid list to start
        if (((data_2d->dimension()->cols() * sizeof(T1)) % 16) == 0) {
          ctx = std::make_unique<algo::Kmeans_HW<T1, g_type::hw_simd, Align128>>(
              data_2d->raw_buffer(),
              data_2d->dimension()->cols(),
              centroid.expected()->raw_buffer(),
              max_iter);
        } else if (((data_2d->dimension()->cols() * sizeof(T1)) % 8) == 0) {
          ctx = std::make_unique<algo::Kmeans_HW<T1, g_type::hw_simd, Align64>>(
              data_2d->raw_buffer(),
              data_2d->dimension()->cols(),
              centroid.expected()->raw_buffer(),
              max_iter);
        } else {
          ctx = std::make_unique<algo::Kmeans_CPU<T1>>(data_2d->raw_buffer(),
                                                       data_2d->dimension()->cols(),
                                                       centroid.expected()->raw_buffer(),
                                                       max_iter);
        }
      } else { // randomly assign data points as centroids
        if (((data_2d->dimension()->cols() * sizeof(T1)) % 16) == 0) {
          ctx = std::make_unique<algo::Kmeans_HW<T1, g_type::hw_simd, Align128>>(
              data_2d->raw_buffer(), data_2d->dimension()->cols(), centroid.unexpected(), max_iter);
        } else if (((data_2d->dimension()->cols() * sizeof(T1)) % 8) == 0) {
          ctx = std::make_unique<algo::Kmeans_HW<T1, g_type::hw_simd, Align64>>(
              data_2d->raw_buffer(), data_2d->dimension()->cols(), centroid.unexpected(), max_iter);
        } else {
          ctx = std::make_unique<algo::Kmeans_CPU<T1>>(
              data_2d->raw_buffer(), data_2d->dimension()->cols(), centroid.unexpected(), max_iter);
        }
      }
      break;

    case g_type::hw_cpu: // Fall through option  - same as default
    default:
      if (centroid) { // use given centroid list to start
        ctx = std::make_unique<algo::Kmeans_CPU<T1>>(data_2d->raw_buffer(),
                                                     data_2d->dimension()->cols(),
                                                     centroid.expected()->raw_buffer(),
                                                     max_iter);
      } else {
        ctx = std::make_unique<algo::Kmeans_CPU<T1>>(
            data_2d->raw_buffer(), data_2d->dimension()->cols(), centroid.unexpected(), max_iter);
      }
  }

  return std::move(ctx);
}
