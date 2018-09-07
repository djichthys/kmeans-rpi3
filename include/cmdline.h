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

namespace parser
{

typedef struct __Option_Help_Description__
{
  char option;
  const char* option_text;
} Option_Help;

class Base_Options
{
protected:
  std::string program;
  std::vector<std::string> raw_args;

  std::unique_ptr<std::string> gen_opt_string(struct option*);
  void usage(const char*, const struct option*, const Option_Help*);

public:
  Base_Options() = delete;
  Base_Options(const int, char* const*);

  /* Accessor methods */
  std::unique_ptr<std::vector<std::string>> raw_options();
};

class Program_Options : public Base_Options
{
private:
  std::string _filename;
  util::Expected<std::string, uint32_t> _k_val;
  std::string _separators;
  g_type::Data_Type _dtype;
  uint32_t _max_iter;
  g_type::Hardware_Type _hw_type;
  uint8_t _verbose;

  bool _init;

  /* private util functions */
  err::api_Err_Status map_data_type(std::string);
  err::api_Err_Status map_accelerator(std::string);

public:
  Program_Options() = delete;
  Program_Options(const int, char* const*);
  err::api_Err_Status parse(const int, char* const*);
  void display_options();
  bool is_initialised() { return this->_init; }
  std::string& filename() { return this->_filename; }
  util::Expected<std::string, uint32_t>& k_val() { return this->_k_val; }
  std::string& separators() { return this->_separators; }
  g_type::Data_Type data_type() { return this->_dtype; }
  uint32_t& max_iter() { return this->_max_iter; }
  g_type::Hardware_Type& hw_type() { return this->_hw_type; }
  uint8_t verbosity() { return this->_verbose; }
};
}
