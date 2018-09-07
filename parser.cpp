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

#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <sstream>

#include <cstring>
#include <stdint.h>
#include <getopt.h>
#include <stdio.h>

#include <api_error.h>
#include <g_types.h>
#include <cmdline.h>

namespace parser
{

Option_Help help_strings[] = {
    {.option = 'f',
     .option_text = "-f,--file..........: input data file. Each "
                    "data point is line-separated"},
    {.option = 'k',
     .option_text =
         "-k,--initial.......: initial centroid file. Each data point is line-separated.\n\
                                    Optionally provide an integer as the number of clusters (k) and \n\
                                    the intial centroids will be automatically calculated"},
    {.option = 's',
     .option_text = "-s,--separator.....: separator field. default space. "
                    "Combination of separators can be used"},
    {.option = 'd',
     .option_text = "-d,--dtype.........: data-types. Use "
                    "(u)int8,(u)int16,(u)int32,(u)int64,float,double,longdouble"},
    {.option = 'h', .option_text = "-h,--help..........: this help menu"},
    {.option = 'i',
     .option_text = "-i,--iter..........: maximum iterations after which processing "
                    "aborts without converging"},
    {.option = 'a', .option_text = "-a, --accelerator..: best/cpu/simd/gpu optimisation"},
    {.option = 'v', .option_text = "-v, --verbose......: verbose mode"},
    {.option = 0, .option_text = nullptr}};

struct option g_option_list[] = {
    {.name = "file", .has_arg = required_argument, .flag = nullptr, .val = 'f'},
    {.name = "initial", .has_arg = required_argument, .flag = nullptr, .val = 'k'},
    {.name = "separator", .has_arg = required_argument, .flag = nullptr, .val = 's'},
    {.name = "dtype", .has_arg = required_argument, .flag = nullptr, .val = 'd'},
    {.name = "help", .has_arg = no_argument, .flag = nullptr, .val = 'h'},
    {.name = "iter", .has_arg = required_argument, .flag = nullptr, .val = 'i'},
    {.name = "accel", .has_arg = required_argument, .flag = nullptr, .val = 'a'},
    {.name = "verbose", .has_arg = optional_argument, .flag = nullptr, .val = 'v'},
    {.name = nullptr, .has_arg = 0, .flag = nullptr, .val = 0}};

// Base_Options::Base_Options( const int argc , const char **argv )
Base_Options::Base_Options(const int argc, char* const* argv)
{
  if ((argv != nullptr) && (argv[0] != nullptr))
    this->program = argv[0];

  for (int idx = 1; idx < argc; idx++)
    this->raw_args.push_back(argv[idx]);
}

std::unique_ptr<std::vector<std::string>> Base_Options::raw_options()
{
  return std::make_unique<std::vector<std::string>>(this->raw_args);
}

std::unique_ptr<std::string> Base_Options::gen_opt_string(struct option* lopt)
{
  std::unique_ptr<std::string> opt_str = std::make_unique<std::string>("");

  /* sanity check inputs */
  if (lopt == nullptr) {
    debug("options list parameter = NULL");
    return nullptr;
  }

  /* generate short option string */
  for (uint32_t idx_i = 0; lopt[idx_i].name != NULL; idx_i++) {
    (*opt_str) += lopt[idx_i].val;
    switch (lopt[idx_i].has_arg) {
      case optional_argument:
        (*opt_str) += ':'; /* intentional fall-through */
      case required_argument:
        (*opt_str) += ':'; /* intentional fall-through */
      case no_argument: break;
      default: break;
    }
  }

  return opt_str;
}

void Base_Options::usage(const char* prog, const struct option* opt_list,
                         const Option_Help* help_strings)
{
  /* sanity check inputs */
  if (opt_list == nullptr) {
    std::cerr << "List of long options = nullptr" << std::endl;
    return;
  }
  if (help_strings == NULL) {
    std::cerr << "Help documentation for options list is empty" << std::endl;
    return;
  }

  std::cout << "==============================================================="
               "================="
            << std::endl;
  std::cout << prog << " Options description." << std::endl;
  std::cout << "==============================================================="
               "================="
            << std::endl;

  /*!
   * for each option in the long-options list, search help strings
   * for corresponding entry and display
   */
  for (uint32_t idx_i = 0; opt_list[idx_i].name != NULL; idx_i++) {
    for (uint32_t idx_j = 0; help_strings[idx_j].option_text != NULL; idx_j++) {
      if (opt_list[idx_i].val == help_strings[idx_j].option) {
        std::cout << "[" << idx_i + 1 << "]. " << help_strings[idx_j].option_text << std::endl;
        break;
      }
    }
  }
}

Program_Options::Program_Options(const int argc, char* const* argv)
    : Base_Options(argc, argv),
      _init(false),
      _verbose(err::debug_Critical),
      _k_val(""),
      _dtype(g_type::DataType_uint8),
      _max_iter(DefaultMaxIterations),
      _hw_type(g_type::hw_cpu)
{
}

err::api_Err_Status Program_Options::parse(const int argc, char* const* argv)
{
  err::api_Err_Status _err = err::api_Success;
  uint32_t idx, len;

  /*!
   * Check whether user has provided the required options
   * We require atleast the data sets to proceed
   */

  std::unique_ptr<std::string> opt_str = this->gen_opt_string(g_option_list);
  if (*opt_str == "") {
    std::cerr << "No options to parse" << std::endl;
    return err::api_Err_Init;
  }

  /* Parse and match list of options */
  const char* short_opt = (*opt_str).c_str();
  optind = 1;
  for (int opt = getopt_long(argc, argv, short_opt, g_option_list, nullptr); opt != -1;
       opt = getopt_long(argc, argv, short_opt, g_option_list, nullptr)) {
    switch (opt) {
      case '?': /* intentional fallthrough */
      case ':':
        this->usage(this->program.c_str(), g_option_list, help_strings);
        throw std::runtime_error("Unknown Option / Missing argument");
      case 'h':
        this->usage(this->program.c_str(), g_option_list, help_strings);
        throw std::runtime_error("-h option encountered");

      case 'f': this->filename() = optarg; break;

      case 'k':
        uint32_t idx, len;
        len = std::strlen(optarg);
        for (idx = 0; idx < len; idx++) {
          if (!std::isdigit(optarg[idx]))
            break;
        }
        // if there was a non-digit within the string, treat it as a filename
        if (idx < len) {
          this->k_val() = util::Expected<std::string, uint32_t>(std::string(optarg));
        } else {
          uint32_t num_k;
          std::stringstream ss(optarg);
          ss >> num_k;
          this->k_val() = util::Expected<std::string, uint32_t>(num_k);
        }
        break;

      case 'i': this->max_iter() = std::stoul(optarg, 0, 0); break;

      case 's': this->separators() = optarg; break;

      case 'd':
        _err = this->map_data_type(optarg);
        if (_err != err::api_Success) {
          std::cerr << "Data type [" << optarg << "] not recognised" << std::endl;
          throw std::runtime_error("Unknown data type");
        }
        break;

      case 'a':
        _err = this->map_accelerator(optarg);
        if (_err != err::api_Success) {
          std::cerr << "Accelerator type [" << optarg << "] not recognised" << std::endl;
          throw std::runtime_error("Unknown accelerator type");
        }
        break;

      case 'v':
        if (optarg == nullptr) {
          this->_verbose = err::debug_Error; /* option -v */
        } else {
          uint32_t len = std::strlen(optarg) + 1;
          if ((err::debug_Error < len) && (len < err::debug_MaxLevel)) {
            this->_verbose = len; /* option -vv (debug_Warning) , -vvv (debug_Trave ) */
          } else {
            std::cout << "Only options -v. -vv and -vvv allowed" << std::endl;
            throw std::runtime_error("Unknown verbosity option");
          }
        }
        break;

      default:
        std::cerr << "Unknown Option [-" << (char)opt << "] encountered" << std::endl;
        throw std::runtime_error("Unknown option specified");
    }
  }
  this->_init = true;
  return _err;
}

err::api_Err_Status Program_Options::map_data_type(std::string arg)
{
  err::api_Err_Status _err = err::api_Success;
  if (arg == "uint8") {
    this->_dtype = g_type::DataType_uint8;
  } else if (arg == "uint16") {
    this->_dtype = g_type::DataType_uint16;
  } else if (arg == "uint32") {
    this->_dtype = g_type::DataType_uint32;
  } else if (arg == "uint64") {
    this->_dtype = g_type::DataType_uint64;
  } else if (arg == "int8") {
    this->_dtype = g_type::DataType_int8;
  } else if (arg == "int16") {
    this->_dtype = g_type::DataType_int16;
  } else if (arg == "int32") {
    this->_dtype = g_type::DataType_int32;
  } else if (arg == "int64") {
    this->_dtype = g_type::DataType_int64;
  } else if (arg == "float") {
    this->_dtype = g_type::DataType_float;
  } else if (arg == "double") {
    this->_dtype = g_type::DataType_double;
  } else if (arg == "longdouble") {
    this->_dtype = g_type::DataType_long_double;
  } else {
    this->_dtype = g_type::DataType_MaxTypes;
    _err = err::api_Err_Param;
  }

  return _err;
}

err::api_Err_Status Program_Options::map_accelerator(std::string arg)
{
  err::api_Err_Status _err = err::api_Success;
  if (arg == "best") {
    this->hw_type() = g_type::hw_best;
  } else if (arg == "cpu") {
    this->hw_type() = g_type::hw_cpu;
  } else if (arg == "simd") {
    this->hw_type() = g_type::hw_simd;
  } else if (arg == "gpu") {
    this->hw_type() = g_type::hw_gpu;
  } else {
    this->hw_type() = g_type::hw_MaxTypes;
  }

  return _err;
}

void Program_Options::display_options()
{
  if (this->verbosity() < err::debug_Trace)
    return;

  std::cout << "=====================================================================" << std::endl;
  std::cout << "Parsed options for : " << this->program << std::endl;
  std::cout << "=====================================================================" << std::endl;
  std::cout << "-f,--file.........: " << this->filename() << std::endl;
  if (this->k_val())
    std::cout << "-k,--initial......: " << this->k_val().expected() << std::endl;
  else
    std::cout << "-k,--initial......: " << this->k_val().unexpected() << std::endl;
  std::cout << "-s,--separator....: " << this->separators() << std::endl;
  std::cout << "-d,--dtype........: " << this->data_type() << std::endl;
  std::cout << "-i,--iter.........: " << this->max_iter() << std::endl;
  std::cout << "-a,--accelerator..: " << this->hw_type() << std::endl;
  std::cout << "-v,--verbose......: " << (uint32_t) this->verbosity() << std::endl;
  std::cout << "=====================================================================" << std::endl;
}
}
