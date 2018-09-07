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

#include <cstdlib>
#include <cstdint>
#include <vector>
#include <memory>
#include <exception>

#include <utils.h>

namespace util
{

uint32_t random_pt(uint32_t max, uint32_t seed)
{
  static uint32_t stateful_seed = 0;
  uint32_t rval;
  if (stateful_seed != seed) {
    stateful_seed = seed;
    srand(seed);
  }

  if (max == RAND_MAX) {
    rval = rand();
  } else {
    long limit = (RAND_MAX / max) * max;
    while ((rval = rand()) >= limit)
      ;
  }
  return rval % max;
}
}
