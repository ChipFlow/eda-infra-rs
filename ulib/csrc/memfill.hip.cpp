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

// ---- HIP runtime FFI wrappers ----
// On NVIDIA backend (HIP_PLATFORM=nvidia), HIP runtime functions are
// header-only wrappers around CUDA; no libamdhip64.so exists.  Provide
// compiled wrapper symbols so Rust FFI can link without that library.

extern "C"
hipError_t hip_ffi_malloc(void **ptr, size_t size) {
  return hipMalloc(ptr, size);
}

extern "C"
hipError_t hip_ffi_free(void *ptr) {
  return hipFree(ptr);
}

extern "C"
hipError_t hip_ffi_memcpy(void *dst, const void *src, size_t size, int kind) {
  return hipMemcpy(dst, src, size, (hipMemcpyKind)kind);
}

extern "C"
hipError_t hip_ffi_memset(void *ptr, int value, size_t size) {
  return hipMemset(ptr, value, size);
}

extern "C"
hipError_t hip_ffi_device_synchronize() {
  return hipDeviceSynchronize();
}

extern "C"
hipError_t hip_ffi_get_device_count(int *count) {
  return hipGetDeviceCount(count);
}

extern "C"
hipError_t hip_ffi_set_device(int device_id) {
  return hipSetDevice(device_id);
}

extern "C"
const char *hip_ffi_get_error_string(hipError_t error) {
  return hipGetErrorString(error);
}

extern "C"
hipError_t hip_ffi_device_get_attribute(int *value, int attr, int device) {
  return hipDeviceGetAttribute(value, (hipDeviceAttribute_t)attr, device);
}
