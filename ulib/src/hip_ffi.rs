//! Minimal raw FFI bindings for the HIP runtime API (AMD GPUs).
//!
//! Calls go through thin C wrapper functions (hip_ffi_*) compiled by hipcc
//! in memfill.hip.cpp.  This avoids linking against `libamdhip64` directly,
//! which does not exist on the NVIDIA backend (HIP_PLATFORM=nvidia) where
//! HIP runtime functions are header-only wrappers around CUDA runtime.

use std::os::raw::{c_char, c_int, c_void};

/// HIP error codes (subset — full enum has ~100 variants).
pub type hipError_t = c_int;
pub const HIP_SUCCESS: hipError_t = 0;

/// Memory copy direction.
pub type hipMemcpyKind = c_int;
pub const hipMemcpyHostToDevice: hipMemcpyKind = 1;
pub const hipMemcpyDeviceToHost: hipMemcpyKind = 2;
pub const hipMemcpyDeviceToDevice: hipMemcpyKind = 3;

extern "C" {
    #[link_name = "hip_ffi_malloc"]
    pub fn hipMalloc(dev_ptr: *mut *mut c_void, size: usize) -> hipError_t;
    #[link_name = "hip_ffi_free"]
    pub fn hipFree(dev_ptr: *mut c_void) -> hipError_t;
    #[link_name = "hip_ffi_memcpy"]
    pub fn hipMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        size_bytes: usize,
        kind: hipMemcpyKind,
    ) -> hipError_t;
    #[link_name = "hip_ffi_memset"]
    pub fn hipMemset(dev_ptr: *mut c_void, value: c_int, size_bytes: usize) -> hipError_t;
    #[link_name = "hip_ffi_device_synchronize"]
    pub fn hipDeviceSynchronize() -> hipError_t;
    #[link_name = "hip_ffi_get_device_count"]
    pub fn hipGetDeviceCount(count: *mut c_int) -> hipError_t;
    #[link_name = "hip_ffi_set_device"]
    pub fn hipSetDevice(device_id: c_int) -> hipError_t;
    #[link_name = "hip_ffi_get_error_string"]
    pub fn hipGetErrorString(error: hipError_t) -> *const c_char;
    #[link_name = "hip_ffi_device_get_attribute"]
    pub fn hipDeviceGetAttribute(
        value: *mut c_int,
        attr: c_int,
        device: c_int,
    ) -> hipError_t;
}

/// Device attribute for warp/wave size.
pub const HIP_DEVICE_ATTRIBUTE_WARP_SIZE: c_int = 10;

/// Panic with a descriptive message if `err` is not `HIP_SUCCESS`.
#[inline]
pub fn check_hip(err: hipError_t, context: &str) {
    if err != HIP_SUCCESS {
        let msg = unsafe {
            let ptr = hipGetErrorString(err);
            if ptr.is_null() {
                "unknown HIP error".to_string()
            } else {
                std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        panic!("HIP error in {}: {} (code {})", context, msg, err);
    }
}
