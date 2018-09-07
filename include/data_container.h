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

template <typename Type>
class Base_Vector_Metadata
{
private:
  uint32_t _axes;

public:
  Base_Vector_Metadata() : _axes(0) {}
  Base_Vector_Metadata(uint32_t axes) : _axes(axes) {}
  uint32_t num_of_axes() { return this->_axes; }
  virtual uint32_t size() = 0;
  virtual uint32_t rows() = 0;
  virtual uint32_t cols() = 0;
  virtual uint32_t x() = 0;
  virtual uint32_t y() = 0;
  virtual uint32_t z() = 0;
  virtual err::api_Err_Status probe_buffer(const std::unique_ptr<std::string>&, const std::string&,
                                           std::vector<Type>&) = 0;
};

template <typename Type, int NDim>
class Vector_Metadata : public Base_Vector_Metadata<Type>
{
};

template <typename Type>
class Vector_Metadata<Type, 1> : public Base_Vector_Metadata<Type>
{
private:
  uint32_t _items;

public:
  Vector_Metadata() : Base_Vector_Metadata<Type>(1), _items(0) {}
  Vector_Metadata(uint32_t _items) : Base_Vector_Metadata<Type>(1), _items(_items) {}
  virtual err::api_Err_Status probe_buffer(const std::unique_ptr<std::string>&, const std::string&,
                                           std::vector<Type>&) final;
  uint32_t size() final { return this->_items; }
  uint32_t rows() final { return 1; }
  uint32_t cols() final { return this->_items; }
  uint32_t x() final { return this->_items; }
  uint32_t y() final { return 1; }
  uint32_t z() final { return 1; }
};

template <typename Type>
err::api_Err_Status
    Vector_Metadata<Type, 1>::probe_buffer(const std::unique_ptr<std::string>& raw_buff,
                                           const std::string& delim, std::vector<Type>& buff)
{
  /*!
   * the class constructor created from a pre-existing buffer
   * Hence do nothing
   */
  if (this->size() != 0)
    return err::api_Err_Init;

  /*!
   * First create a copy of the raw buffer so that we can use std::strtok
   * this is because std::strtok clobbers the delimiters with \0
   */
  char* l_ptr = nullptr;
  std::stringstream _str;
  std::unique_ptr<std::string> dup = std::make_unique<std::string>(*raw_buff);
  const char* sep = delim.c_str();
  Type _tmp;

  /* We used the default constructor.Hence find out size as we parse */
  for (char* linebuff = strtok_r(&(*dup)[0], sep, &l_ptr); linebuff != nullptr;
       linebuff = strtok_r(nullptr, sep, &l_ptr)) {
    this->_items++;
    _str.flush();
    _str.clear();
    _str << linebuff;
    _str >> _tmp;
    buff.push_back(_tmp);
  }

  return err::api_Success;
}

template <typename Type>
class Vector_Metadata<Type, 2> : public Base_Vector_Metadata<Type>
{
private:
  uint32_t _rows, _cols;

public:
  Vector_Metadata() : Base_Vector_Metadata<Type>(2), _rows(0), _cols(0) {}
  Vector_Metadata(uint32_t rows, uint32_t cols)
      : Base_Vector_Metadata<Type>(2), _rows(rows), _cols(cols)
  {
  }
  /* in 2D form returns total number of bytes */
  uint32_t size() final { return this->rows() * this->cols(); }

  uint32_t rows() final { return this->_rows; }
  uint32_t cols() final { return this->_cols; }

  uint32_t x() final { return this->rows(); }
  uint32_t y() final { return this->cols(); }
  uint32_t z() final { return 1; }
  virtual err::api_Err_Status probe_buffer(const std::unique_ptr<std::string>&, const std::string&,
                                           std::vector<Type>&) final;
};

template <typename Type>
err::api_Err_Status
    Vector_Metadata<Type, 2>::probe_buffer(const std::unique_ptr<std::string>& raw_buff,
                                           const std::string& delim, std::vector<Type>& buff)
{
  /*!
   * the class constructor created from a pre-existing buffer
   * Hence do nothing
   */
  if (this->size() != 0)
    return err::api_Err_Init;

  char sep_x[2] = {0}, sep_y[2] = {0};
  sep_y[0] = delim[1];
  sep_x[0] = delim[0];

  /*!
   * First create a copy of the raw buffer so that we can use std::strtok
   * this is because std::strtok clobbers the delimiters with \0
   */
  char *w_ptr = nullptr, *l_ptr = nullptr;
  std::stringstream _str;
  std::unique_ptr<std::string> dup = std::make_unique<std::string>(*raw_buff);
  Type _tmp;

  this->_rows = this->_cols = 0;
  for (char* row_ptr = strtok_r(&(*dup)[0], sep_y, &w_ptr); row_ptr != nullptr;
       row_ptr = strtok_r(nullptr, sep_y, &w_ptr)) {
    this->_rows++;
    this->_cols = 0;
    for (char* col_ptr = strtok_r(row_ptr, sep_x, &l_ptr); col_ptr != nullptr;
         col_ptr = strtok_r(nullptr, sep_x, &l_ptr)) {
      this->_cols++;
      _str.flush();
      _str.clear();
      _str << col_ptr;
      _str >> _tmp;
      buff.push_back(_tmp);
    }
  }
  return err::api_Success;
}

template <typename Type>
class Vector_Metadata<Type, 3> : public Base_Vector_Metadata<Type>
{
private:
  uint32_t _x, _y, _z;

public:
  Vector_Metadata() : Base_Vector_Metadata<Type>(3), _x(0), _y(0), _z(0) {}
  Vector_Metadata(uint32_t z, uint32_t y, uint32_t x)
      : Base_Vector_Metadata<Type>(3), _z(z), _y(y), _x(x)
  {
  }
  /* in 3D form returns total number of bytes */
  uint32_t size() final { return this->z() * this->y() * this->x(); }

  uint32_t rows() final { return this->y(); }
  uint32_t cols() final { return this->x(); }

  uint32_t x() final { return this->_x; }
  uint32_t y() final { return this->_y; }
  uint32_t z() final { return this->_z; }
  virtual err::api_Err_Status probe_buffer(const std::unique_ptr<std::string>&, const std::string&,
                                           std::vector<Type>&) final;
};

template <typename Type>
err::api_Err_Status
    Vector_Metadata<Type, 3>::probe_buffer(const std::unique_ptr<std::string>& raw_buff,
                                           const std::string& delim, std::vector<Type>& buff)
{
  /*!
   * the class constructor created from a pre-existing buffer
   * Hence do nothing
   */
  if (this->size() != 0)
    return err::api_Err_Init;

  char sep_z[2] = {0}, sep_y[2] = {0}, sep_x[2] = {0};
  sep_z[0] = delim[2];
  sep_y[0] = delim[1];
  sep_x[0] = delim[0];

  /*!
   * First create a copy of the raw buffer so that we can use std::strtok
   * this is because std::strtok clobbers the delimiters with \0
   */
  char *w_ptr = nullptr, *l_ptr = nullptr, *d_ptr = nullptr;
  std::stringstream _str;
  std::unique_ptr<std::string> dup = std::make_unique<std::string>(*raw_buff);
  Type _tmp;

  for (char* z_ptr = strtok_r(&(*dup)[0], sep_z, &d_ptr); z_ptr != nullptr;
       z_ptr = strtok_r(nullptr, sep_z, &d_ptr)) {
    this->_z++;
    this->_y = 0;
    for (char* y_ptr = strtok_r(z_ptr, sep_y, &l_ptr); y_ptr != nullptr;
         y_ptr = strtok_r(nullptr, sep_y, &l_ptr)) {
      this->_y++;
      this->_x = 0;
      for (char* x_ptr = strtok_r(y_ptr, sep_x, &w_ptr); x_ptr != nullptr;
           x_ptr = strtok_r(nullptr, sep_x, &w_ptr)) {
        this->_x++;
        _str.flush();
        _str.clear();
        _str << x_ptr;
        _str >> _tmp;
        buff.push_back(_tmp);
      } /* For each item within a line */
    }   /* For each 1-d y line */
  }     /* For each 2-d (y,x) plane */

  return err::api_Success;
}

class DC_Wrapper
{
private:
  g_type::Data_Type _dtype;

public:
  DC_Wrapper();
  DC_Wrapper(g_type::Data_Type);
  g_type::Data_Type& type() { return this->_dtype; }
  virtual ~DC_Wrapper(){};
  virtual err::api_Err_Status populate_data(const std::unique_ptr<std::string>&, std::string&) = 0;
  virtual void display(err::Debug_Level = err::debug_Critical) = 0;
};

DC_Wrapper::DC_Wrapper() : _dtype(g_type::DataType_float) {}
DC_Wrapper::DC_Wrapper(g_type::Data_Type ty) : _dtype(ty) {}

template <typename T1>
class Data_Container_Base : public DC_Wrapper
{
private:
  Base_Vector_Metadata<T1>* _meta;

  /* internal function to generate run-time type info */
  g_type::Data_Type _data_container_rtti();

public:
  Data_Container_Base();
  Data_Container_Base(Base_Vector_Metadata<T1>* meta);
  virtual ~Data_Container_Base() { delete _meta; }
  Base_Vector_Metadata<T1>*& dimension() { return this->_meta; }
  virtual err::api_Err_Status populate_data(const std::unique_ptr<std::string>&, std::string&) = 0;
  virtual void display(err::Debug_Level = err::debug_Critical) = 0;
};

template <typename T1>
Data_Container_Base<T1>::Data_Container_Base() : _meta(nullptr)
{
  this->type() = this->_data_container_rtti();
}

template <typename T1>
Data_Container_Base<T1>::Data_Container_Base(Base_Vector_Metadata<T1>* meta) : _meta(meta)
{
  this->type() = this->_data_container_rtti();
}

template <typename T1>
g_type::Data_Type Data_Container_Base<T1>::_data_container_rtti()
{
  g_type::Data_Type _type;

  if (std::is_same<T1, uint8_t>::value) {
    _type = g_type::DataType_uint8;
  } else if (std::is_same<T1, uint16_t>::value) {
    _type = g_type::DataType_uint16;
  } else if (std::is_same<T1, uint32_t>::value) {
    _type = g_type::DataType_uint32;
  } else if (std::is_same<T1, uint64_t>::value) {
    _type = g_type::DataType_uint64;
  } else if (std::is_same<T1, int8_t>::value) {
    _type = g_type::DataType_int8;
  } else if (std::is_same<T1, int16_t>::value) {
    _type = g_type::DataType_int16;
  } else if (std::is_same<T1, int32_t>::value) {
    _type = g_type::DataType_int32;
  } else if (std::is_same<T1, int64_t>::value) {
    _type = g_type::DataType_int64;
  } else if (std::is_same<T1, float>::value) {
    _type = g_type::DataType_float;
  } else if (std::is_same<T1, double>::value) {
    _type = g_type::DataType_double;
  } else if (std::is_same<T1, long double>::value) {
    _type = g_type::DataType_long_double;
  } else {
    _type = g_type::DataType_MaxTypes;
  }

  return _type;
}

template <typename T1, int Axes>
class Data_Container : public Data_Container_Base<T1>
{
};

/* Declare Container for Data */
template <typename T1>
class Data_Container<T1, 1> : public Data_Container_Base<T1>
{
private:
  std::vector<T1> _data;

public:
  Data_Container();
  Data_Container(const std::vector<T1>&);
  virtual ~Data_Container() final {}
  std::vector<T1>& buffer() { return this->_data; }
  std::vector<T1>& raw_buffer() { return this->_data; }
  err::api_Err_Status populate_data(const std::unique_ptr<std::string>&, std::string&);
  virtual void display(err::Debug_Level = err::debug_Critical) final;
};

template <typename T1>
Data_Container<T1, 1>::Data_Container()
    : Data_Container_Base<T1>(new Vector_Metadata<T1, 1>(0)), _data(std::vector<T1>(0))
{
}

template <typename T1>
Data_Container<T1, 1>::Data_Container(const std::vector<T1>& data)
    : Data_Container_Base<T1>(new Vector_Metadata<T1, 1>(data.size()))
{
  /* First copy over data into primary contigous aligned memory */
  this->_data.reserve(data.size());
  for (auto& it : data)
    this->_data.push_back(it);
}

template <typename T1>
err::api_Err_Status
    Data_Container<T1, 1>::populate_data(const std::unique_ptr<std::string>& raw_buff,
                                         std::string& delim)
{
  err::api_Err_Status _err = err::api_Success;

  if (raw_buff == nullptr) {
    std::cerr << "Empty buffer cannot be parsed" << std::endl;
    _err = err::api_Err_Param;
    throw std::runtime_error("Null buffer cannot be parsed");
  }

  /* Probe and Populate array 1D dimensions within Metadata */
  Base_Vector_Metadata<T1>*& meta = this->dimension();
  if (meta == nullptr) {
    std::cerr << "Dimension Class not populated" << std::endl;
    throw std::runtime_error("Dimension Class not populated");
  }

  _err = meta->probe_buffer(raw_buff, delim, this->_data);
  if (_err != err::api_Success) {
    std::cerr << "Probing Dimensions of 1D space returned err = " << _err << std::endl;
    throw std::runtime_error("Error Probing Dimensions of 1D space");
  }

  return _err;
}

template <typename T1>
void Data_Container<T1, 1>::display(err::Debug_Level lvl)
{
  Base_Vector_Metadata<T1>*& meta = this->dimension();
  if (meta == nullptr) {
    std::cerr << "Dimension Class not populated" << std::endl;
    throw std::runtime_error("Dimension Class not populated");
  }

  if (lvl < err::debug_Trace)
    return;

  std::cout << "===========================================+" << std::endl;

  std::cout << "Detected sizes ::" << std::endl
            << "Num of axes = " << meta->num_of_axes() << std::endl
            << "num of items = " << meta->size() << std::endl
            << "Capacity = " << this->_data.capacity() << ", Size = " << this->_data.size()
            << std::endl;
  std::cout << ">>>>>>>>>>" << std::endl;
  for (T1& it : this->_data)
    std::cout << "Value = " << it << std::endl;
  std::cout << "===========================================+" << std::endl;
}

/* Declare Container for Data - 2D */
template <typename T1>
class Data_Container<T1, 2> : public Data_Container_Base<T1>
{
private:
  std::vector<T1*> _data_plane;
  std::vector<T1> _buff;

public:
  Data_Container();
  Data_Container(const std::vector<T1>&, uint32_t rows, uint32_t cols);
  virtual ~Data_Container() final{};
  std::vector<T1*>& buffer() { return this->_data_plane; }
  std::vector<T1>& raw_buffer() { return this->_buff; }
  err::api_Err_Status populate_data(const std::unique_ptr<std::string>&, std::string&);
  virtual void display(err::Debug_Level = err::debug_Critical) final;
};

template <typename T1>
Data_Container<T1, 2>::Data_Container()
    : Data_Container_Base<T1>(new Vector_Metadata<T1, 2>(0, 0)),
      _data_plane(std::vector<T1*>(0)),
      _buff(std::vector<T1>(0))
{
}

template <typename T1>
Data_Container<T1, 2>::Data_Container(const std::vector<T1>& data, uint32_t rows, uint32_t cols)
    : Data_Container_Base<T1>(new Vector_Metadata<T1, 2>(rows, cols)), _data_plane(rows, nullptr)
{
  /* First copy over vector data into a vector that is aligned for SIMD */
  this->_buff.reserve(data.size());
  for (auto& it : data)
    this->_buff.push_back(it);

  for (uint32_t idx = 0; idx < rows; idx++)
    this->_data_plane[idx] = &(this->_buff[0]) + idx * sizeof(T1);
}

template <typename T1>
err::api_Err_Status
    Data_Container<T1, 2>::populate_data(const std::unique_ptr<std::string>& raw_buff,
                                         std::string& delim)
{
  err::api_Err_Status _err = err::api_Success;

  if (raw_buff == nullptr) {
    std::cerr << "Empty buffer cannot be parsed" << std::endl;
    _err = err::api_Err_Param;
    throw std::runtime_error("Null buffer cannot be parsed");
  }

  /* Probe and Populate array 1D dimensions within Metadata */
  Base_Vector_Metadata<T1>*& meta = this->dimension();
  if (meta == nullptr) {
    std::cerr << "Dimension Class not populated" << std::endl;
    throw std::runtime_error("Dimension Class not populated");
  }

  _err = meta->probe_buffer(raw_buff, delim, this->_buff);
  if (_err != err::api_Success) {
    std::cerr << "Probing Dimensions of 2D space returned err = " << _err << std::endl;
    throw std::runtime_error("Error Probing Dimensions of 1D space");
  }

  /* Populate pointer indirection to use [][] notation for 2D array */
  for (uint32_t idx = 0; idx < meta->rows(); idx++)
    this->_data_plane.push_back(&(this->_buff[0]) + idx * meta->cols());

  return _err;
}

template <typename T1>
void Data_Container<T1, 2>::display(err::Debug_Level lvl)
{
  Base_Vector_Metadata<T1>*& meta = this->dimension();
  if (meta == nullptr) {
    std::cerr << "Dimension Class not populated" << std::endl;
    throw std::runtime_error("Dimension Class not populated");
  }

  if (lvl < err::debug_Trace)
    return;

  std::cout << "===========================================+" << std::endl;

  std::cout << "Detected sizes ::" << std::endl
            << "Num of axes = " << meta->num_of_axes() << std::endl
            << "Num of rows = " << meta->rows() << " , cols = " << meta->cols() << std::endl
            << "Capacity = " << this->_buff.capacity() << ", Size = " << this->_buff.size()
            << std::endl;

  std::cout << ">>>>>>>>>>" << std::endl;

  uint32_t idx_x = 0, idx_y = 0;
  const T1** _ptrC = (const T1**)(&(this->buffer()[0]));

  for (idx_x = 0; idx_x < meta->rows(); idx_x++) {
    for (idx_y = 0; idx_y < meta->cols(); idx_y++)
      std::cout << "data[" << idx_x << "][" << idx_y << "] = " << _ptrC[idx_x][idx_y] << std::endl;
  }
  std::cout << "===========================================+" << std::endl;
}

/* Declare Container for Data - 3D */
template <typename T1>
class Data_Container<T1, 3> : public Data_Container_Base<T1>
{
private:
  std::vector<T1**> _data_plane;
  std::vector<T1> _buff;

public:
  Data_Container();
  Data_Container(const std::vector<T1>&, uint32_t z, uint32_t y, uint32_t x);
  virtual ~Data_Container() final;
  std::vector<T1**>& buffer() { return this->_data_plane; }
  std::vector<T1>& raw_buffer() { return this->_buff; }
  err::api_Err_Status populate_data(const std::unique_ptr<std::string>&, std::string&);
  virtual void display(err::Debug_Level = err::debug_Critical) final;
};

template <typename T1>
Data_Container<T1, 3>::Data_Container()
    : Data_Container_Base<T1>(new Vector_Metadata<T1, 3>(0, 0, 0)),
      _data_plane(std::vector<T1**>(0)),
      _buff(std::vector<T1>(0))
{
}

template <typename T1>
Data_Container<T1, 3>::Data_Container(const std::vector<T1>& data, uint32_t z, uint32_t y,
                                      uint32_t x)
    : Data_Container_Base<T1>(new Vector_Metadata<T1, 2>(z, y, x)), _data_plane(z, nullptr)
{
  /* First copy over data into primary contigous aligned memory */
  this->_buff.reserve(data.size());
  for (auto& it : data)
    this->_buff.push_back(it);

  /* First create an array to hold pointers to every y-axis ( = z * y lines) */
  T1** _tmp = new T1*[z * y]();

  for (uint32_t idx_z = 0; idx_z < z; idx_z++) {
    this->_data_plane[idx_z] = &_tmp[0] + (idx_z * y);
    for (uint32_t idx_y = 0; idx_y < y; idx_y++)
      this->_data_plane[idx_z][idx_y] = &this->_buff[0] + (idx_z * y * x) + (idx_y * x);
  }
}

template <typename T1>
Data_Container<T1, 3>::~Data_Container()
{
  if ((this->_data_plane.size() > 0) && (this->_data_plane[0] != nullptr))
    delete[] this->_data_plane[0];
}

template <typename T1>
err::api_Err_Status
    Data_Container<T1, 3>::populate_data(const std::unique_ptr<std::string>& raw_buff,
                                         std::string& delim)
{
  err::api_Err_Status _err = err::api_Success;

  if (raw_buff == nullptr) {
    std::cerr << "Empty buffer cannot be parsed" << std::endl;
    _err = err::api_Err_Param;
    throw std::runtime_error("Null buffer cannot be parsed");
  }

  /* Probe and Populate array 1D dimensions within Metadata */
  Base_Vector_Metadata<T1>*& meta = this->dimension();
  if (meta == nullptr) {
    std::cerr << "Dimension Class not populated" << std::endl;
    throw std::runtime_error("Dimension Class not populated");
  }

  _err = meta->probe_buffer(raw_buff, delim, this->_buff);
  if (_err != err::api_Success) {
    std::cerr << "Probing Dimensions of 2D space returned err = " << _err << std::endl;
    throw std::runtime_error("Error Probing Dimensions of 1D space");
  }

  /* First create an array to hold pointers to every y-axis ( = z * y lines) */
  T1** _tmp = new T1*[meta->z() * meta->y()]();
  this->_data_plane.reserve(meta->z());

  /* Populate pointer indirection to use [][] notation for 2D array */
  for (uint32_t idx_z = 0; idx_z < meta->z(); idx_z++) {
    this->_data_plane.push_back(&_tmp[0] + (idx_z * meta->y()));
    for (uint32_t idx_y = 0; idx_y < meta->y(); idx_y++)
      this->_data_plane[idx_z][idx_y] =
          &(this->_buff[0]) + (idx_z * meta->y() * meta->x()) + (idx_y * meta->x());
  }

  return _err;
}

template <typename T1>
void Data_Container<T1, 3>::display(err::Debug_Level lvl)
{
  Base_Vector_Metadata<T1>*& meta = this->dimension();
  if (meta == nullptr) {
    std::cerr << "Dimension Class not populated" << std::endl;
    throw std::runtime_error("Dimension Class not populated");
  }

  if (lvl < err::debug_Trace)
    return;

  std::cout << "===========================================+" << std::endl;

  std::cout << "Detected sizes ::" << std::endl
            << "Num of axes = " << meta->num_of_axes() << std::endl
            << "z = " << meta->z() << " , y = " << meta->y() << ", x = " << meta->x() << std::endl
            << "Capacity = " << this->_buff.capacity() << ", Size = " << this->_buff.size()
            << std::endl;

  std::cout << ">>>>>>>>>>" << std::endl;
  uint32_t idx_x = 0, idx_y = 0, idx_z = 0;
  const T1*** _ptrC = (const T1***)(&(this->buffer()[0]));

  std::cout << "_ptrC  = " << _ptrC << ", *_ptrC = " << _ptrC[0] << std::endl;
  for (idx_z = 0; idx_z < meta->z(); idx_z++) {
    std::cout << "_ptrC  = " << _ptrC << ", _ptrC[" << idx_z << "] = " << _ptrC[idx_z] << std::endl;
    for (idx_y = 0; idx_y < meta->y(); idx_y++) {
      for (idx_x = 0; idx_x < meta->x(); idx_x++)
        std::cout << "data[" << idx_z << "][" << idx_y << "][" << idx_x
                  << "] = " << _ptrC[idx_z][idx_y][idx_x] << std::endl;
    }
  }

  std::cout << "===========================================+" << std::endl;
}
}
