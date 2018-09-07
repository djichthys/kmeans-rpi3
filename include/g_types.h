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

#define DefaultMaxIterations 256

namespace g_type
{
typedef enum __Supported_Data_Type__ {
  DataType_uint8 = 0,
  DataType_uint16,
  DataType_uint32,
  DataType_uint64,
  DataType_int8,
  DataType_int16,
  DataType_int32,
  DataType_int64,
  DataType_float,
  DataType_double,
  DataType_long_double,
  DataType_MaxTypes /* Sentinel value for error checking */
} Data_Type;

typedef enum __Accelerator_Hardware_Type__ {
  hw_best = 0,
  hw_cpu,
  hw_simd,
  hw_gpu,
  hw_MaxTypes /* Sentinel value for error checking */
} Hardware_Type;
}
