// HIP memfill kernels for AMD GPUs.
// Functionally identical to memfill.cu — HIP's <<<>>> syntax is source-compatible.

#include <hip/hip_runtime.h>
#include "types.hpp"

template<typename T>
__global__ void ulib_fill_memory_fixed_hip_kernel(T *a, usize len, const T val) {
  usize i = blockIdx.x * (usize)blockDim.x + threadIdx.x;
  if(i >= len) return;
  a[i] = val;
}

extern "C"
void ulib_fill_memory_1byte_hip(u8 *a, usize len, u8 val) {
  ulib_fill_memory_fixed_hip_kernel<<<(len + 256 - 1) / 256, 256>>>(
    a, len, val
    );
}

extern "C"
void ulib_fill_memory_2byte_hip(u16 *a, usize len, u16 val) {
  ulib_fill_memory_fixed_hip_kernel<<<(len + 256 - 1) / 256, 256>>>(
    a, len, val
    );
}

extern "C"
void ulib_fill_memory_4byte_hip(u32 *a, usize len, u32 val) {
  ulib_fill_memory_fixed_hip_kernel<<<(len + 256 - 1) / 256, 256>>>(
    a, len, val
    );
}

extern "C"
void ulib_fill_memory_8byte_hip(u64 *a, usize len, u64 val) {
  ulib_fill_memory_fixed_hip_kernel<<<(len + 256 - 1) / 256, 256>>>(
    a, len, val
    );
}

__global__ void ulib_fill_memory_anybyte_hip_kernel(u8 *a, usize len, const u8 *val, usize size) {
  usize i = blockIdx.x * (usize)blockDim.x + threadIdx.x;
  if(i >= len) return;
  for(u8 j = 0; j < size; ++j) {
    a[i * size + j] = val[j];
  }
}

extern "C"
void ulib_fill_memory_anybyte_hip(u8 *a, usize len, const u8 *val, usize size) {
  ulib_fill_memory_anybyte_hip_kernel<<<(len + 256 - 1) / 256, 256>>>(
    a, len, val, size
    );
}
