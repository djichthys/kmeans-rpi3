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

/*!
 * Enumeration types for all APIs to talk to
 * each other at the application layer
 */
namespace err
{
typedef enum __api_Err_Status_Types__ {
  api_Err_Param = -256,
  api_Err_Init,
  api_Err_Hardware,
  api_Err_Failure,
  api_Err_Memory,
  api_Err_File,
  api_Success = 0,
  api_Stat_InProgress,
  api_Stat_Complete,
  api_Err_Max_Type
} api_Err_Status;

typedef enum __Debug_Level__ {
  debug_Critical = 0,
  debug_Error,
  debug_Warning,
  debug_Trace,
  debug_MaxLevel
} Debug_Level;
}

namespace util
{
/* Class to return an either-or type class */
template <typename Ex, typename UnEx>
class Expected
{
public:
  Expected() = delete;
  Expected(Ex&);
  Expected(Ex&&);
  Expected(UnEx&);
  Expected(UnEx&&);
  operator bool() const;
  Ex& expected();
  UnEx& unexpected();

private:
  bool _is_expected;
  Ex _expected;
  UnEx _unexpected;
};

template <typename Ex, typename UnEx>
Expected<Ex, UnEx>::Expected(UnEx& unex) : _unexpected(unex), _is_expected(false)
{
}

template <typename Ex, typename UnEx>
Expected<Ex, UnEx>::Expected(UnEx&& unex) : _unexpected(unex), _is_expected(false)
{
}

template <typename Ex, typename UnEx>
Expected<Ex, UnEx>::Expected(Ex& val) : _expected(val), _is_expected(true)
{
}

template <typename Ex, typename UnEx>
Expected<Ex, UnEx>::Expected(Ex&& val) : _expected(val), _is_expected(true)
{
}

template <typename Ex, typename UnEx>
Expected<Ex, UnEx>::operator bool() const
{
  return this->_is_expected;
}

template <typename Ex, typename UnEx>
Ex& Expected<Ex, UnEx>::expected()
{
  return this->_expected;
}

template <typename Ex, typename UnEx>
UnEx& Expected<Ex, UnEx>::unexpected()
{
  return this->_unexpected;
}
}

#define debug(_fmt, args...) printf("[%s:%u] | " _fmt "\r\n", __FUNCTION__, __LINE__, ##args)
