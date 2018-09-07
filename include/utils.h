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

namespace util
{

/* Definitions of some values */
#define InitSeed 32
uint32_t random_pt(uint32_t max, uint32_t seed);

#define Align64 8
#define Align128 16
#define Align256 32
#define Align512 54
/**
 * Allocator for aligned data.
 *
 * Modified from the Mallocator from Stephan T. Lavavej.
 * <http://blogs.msdn.com/b/vcblog/archive/2008/08/28/the-mallocator.aspx>
 */
template <typename T, std::size_t Alignment>
class Align_Mem
{
public:
  // The following will be the same for virtually all allocators.
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef T value_type;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;

  T* address(T& r) const { return &r; }

  const T* address(const T& s) const { return &s; }

  std::size_t max_size() const
  {
    // The following has been carefully written to be independent of
    // the definition of size_t and to avoid signed/unsigned warnings.
    return (static_cast<std::size_t>(0) - static_cast<std::size_t>(1)) / sizeof(T);
  }

  // The following must be the same for all allocators.
  template <typename U>
  struct rebind
  {
    typedef Align_Mem<U, Alignment> other;
  };

  bool operator!=(const Align_Mem& other) const { return !(*this == other); }

  void construct(T* const p, const T& t) const
  {
    void* const pv = static_cast<void*>(p);
    new (pv) T(t);
  }

  void destroy(T* const p) const { p->~T(); }

  // Returns true if and only if storage allocated from *this
  // can be deallocated from other, and vice versa.
  // Always returns true for stateless allocators.
  bool operator==(const Align_Mem& other) const { return true; }

  // Default constructor, copy constructor, rebinding constructor, and
  // destructor.
  // Empty for stateless allocators.
  Align_Mem() {}

  Align_Mem(const Align_Mem&) {}

  template <typename U>
  Align_Mem(const Align_Mem<U, Alignment>&)
  {
  }

  ~Align_Mem() {}

  // The following will be different for each allocator.
  T* allocate(const std::size_t n) const
  {
    // The return value of allocate(0) is unspecified.
    // Mallocator returns NULL in order to avoid depending
    // on malloc(0)'s implementation-defined behavior
    // (the implementation can define malloc(0) to return NULL,
    // in which case the bad_alloc check below would fire).
    // All allocators can return NULL in this case.
    if (n == 0) {
      return NULL;
    }

    // All allocators should contain an integer overflow check.
    // The Standardization Committee recommends that std::length_error
    // be thrown in the case of integer overflow.
    if (n > max_size()) {
      throw std::length_error("Align_Mem<T>::allocate() - Integer overflow.");
    }

    // Mallocator wraps malloc().
    void* pv;
    if (int _err = posix_memalign(&pv, Alignment, n * sizeof(T)))
      throw std::bad_alloc();

    // Allocators should throw std::bad_alloc in the case of memory allocation
    // failure.
    if (pv == NULL) {
      throw std::bad_alloc();
    }

    return static_cast<T*>(pv);
  }

  void deallocate(T* const p, const std::size_t n) const { free(p); }

  // The following will be the same for all allocators that ignore hints.
  template <typename U>
  T* allocate(const std::size_t n, const U* /* const hint */) const
  {
    return allocate(n);
  }

  // Allocators are not required to be assignable, so
  // all allocators should have a private unimplemented
  // assignment operator. Note that this will trigger the
  // off-by-default (enabled under /Wall) warning C4626
  // "assignment operator could not be generated because a
  // base class assignment operator is inaccessible" within
  // the STL headers, but that warning is useless.
private:
  Align_Mem& operator=(const Align_Mem&);
};
}
