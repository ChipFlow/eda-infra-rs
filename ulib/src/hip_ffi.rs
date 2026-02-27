//! Minimal raw FFI bindings for the HIP runtime API (AMD GPUs).
//!
//! Only the subset needed by ulib is exposed here. No external Rust crate
//! is required; we link against the `amdhip64` shared library at build time.

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
    pub fn hipMalloc(dev_ptr: *mut *mut c_void, size: usize) -> hipError_t;
    pub fn hipFree(dev_ptr: *mut c_void) -> hipError_t;
    pub fn hipMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        size_bytes: usize,
        kind: hipMemcpyKind,
    ) -> hipError_t;
    pub fn hipMemset(dev_ptr: *mut c_void, value: c_int, size_bytes: usize) -> hipError_t;
    pub fn hipDeviceSynchronize() -> hipError_t;
    pub fn hipGetDeviceCount(count: *mut c_int) -> hipError_t;
    pub fn hipSetDevice(device_id: c_int) -> hipError_t;
    pub fn hipGetErrorString(error: hipError_t) -> *const c_char;
}

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
