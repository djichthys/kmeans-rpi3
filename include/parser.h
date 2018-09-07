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
/*!
 * Base class to hold raw text data being read from a file
 */
template <typename T1>
class File_Parser_Base
{
private:
  std::unique_ptr<std::string> _raw_buffer;

public:
  std::unique_ptr<std::string> mv_raw_buff();
  std::unique_ptr<std::string>& raw_buff();
};

template <typename T1>
std::unique_ptr<std::string> File_Parser_Base<T1>::mv_raw_buff()
{
  return std::move(this->_raw_buffer);
}

template <typename T1>
std::unique_ptr<std::string>& File_Parser_Base<T1>::raw_buff()
{
  return this->_raw_buffer;
}

template <typename T0, typename T1>
class File_Parser : public File_Parser_Base<T1>
{
};

/* Specialisation for filename held as string */
template <typename T1>
class File_Parser<std::string, T1> : public File_Parser_Base<T1>
{
private:
  std::string handle;

public:
  File_Parser() = delete;
  File_Parser(std::string);
  void read_file();
};

/* Specialisation for file instantiated with an fstream */
template <typename T1>
class File_Parser<std::ifstream, T1> : public File_Parser_Base<T1>
{
private:
  std::ifstream& handle;

public:
  File_Parser() = delete;
  File_Parser(std::ifstream&);
  void read_file();
};

template <typename T>
File_Parser<std::string, T>::File_Parser(std::string filename) : handle(filename)
{
}

template <typename T>
File_Parser<std::ifstream, T>::File_Parser(std::ifstream& f_str) : handle(f_str)
{
}

template <typename T>
void File_Parser<std::ifstream, T>::read_file()
{
  if (this->handle.is_open())
    this->raw_buff() = std::make_unique<std::string>(
        static_cast<std::stringstream const&>(std::stringstream() << this->handle.rdbuf()).str());
}

template <typename T>
void File_Parser<std::string, T>::read_file()
{
  std::ifstream f_str(this->handle);
  if (!f_str.is_open()) {
    std::cerr << "File " << this->handle << " cannot be opened" << std::endl;
    throw std::runtime_error("File Cannot be opened");
  }

  File_Parser<std::ifstream, T> _fparser(f_str);
  f_str.seekg(0);
  _fparser.read_file(); /* populate raw buffer */
  this->raw_buff() = _fparser.mv_raw_buff();
  f_str.close();
}
}
