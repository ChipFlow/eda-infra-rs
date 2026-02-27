//! Universal vector-like array storage [`UVec`].

use super::*;
use bytemuck::Zeroable;
use std::cell::UnsafeCell;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::sync::Mutex;

#[cfg(feature = "cuda")]
use cust::context::{Context, CurrentContext};
#[cfg(feature = "cuda")]
use cust::memory::{CopyDestination, DeviceBuffer};

#[cfg(feature = "metal")]
use metal::MTLResourceOptions;

/// A HIP device buffer wrapping a raw device pointer.
/// Automatically frees the allocation on drop.
#[cfg(feature = "hip")]
struct HipBuffer<T: UniversalCopy> {
    ptr: *mut T,
    len: usize,
}

#[cfg(feature = "hip")]
impl<T: UniversalCopy> HipBuffer<T> {
    /// Allocate `len` elements on the HIP device (uninitialized).
    unsafe fn alloc_uninit(len: usize) -> Self {
        let byte_size = len * std::mem::size_of::<T>();
        let mut ptr: *mut std::os::raw::c_void = std::ptr::null_mut();
        let err = super::hip_ffi::hipMalloc(&mut ptr, byte_size);
        super::hip_ffi::check_hip(err, "hipMalloc");
        HipBuffer { ptr: ptr as *mut T, len }
    }

    /// Allocate `len` elements on the HIP device, zeroed.
    unsafe fn alloc_zeroed(len: usize) -> Self {
        let buf = Self::alloc_uninit(len);
        let byte_size = len * std::mem::size_of::<T>();
        let err = super::hip_ffi::hipMemset(buf.ptr as *mut std::os::raw::c_void, 0, byte_size);
        super::hip_ffi::check_hip(err, "hipMemset");
        buf
    }

    fn as_ptr(&self) -> *const T { self.ptr as *const T }
    fn as_mut_ptr(&self) -> *mut T { self.ptr }
}

#[cfg(feature = "hip")]
impl<T: UniversalCopy> Drop for HipBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let err = unsafe { super::hip_ffi::hipFree(self.ptr as *mut std::os::raw::c_void) };
            // Don't panic in drop — just log
            if err != super::hip_ffi::HIP_SUCCESS {
                eprintln!("Warning: hipFree failed with code {}", err);
            }
        }
    }
}

// Safety: HipBuffer holds a device pointer that can be sent across threads.
#[cfg(feature = "hip")]
unsafe impl<T: UniversalCopy> Send for HipBuffer<T> {}
#[cfg(feature = "hip")]
unsafe impl<T: UniversalCopy> Sync for HipBuffer<T> {}

/// Universal vector-like array storage.
///
/// `UVec` is thread-safe (`Send` + `Sync`). Specifically, its
/// read-only reference can be shared across different threads.
/// This is nontrivial because a read in `UVec` might schedule
/// a copy across device.
pub struct UVec<T: UniversalCopy>(UnsafeCell<UVecInternal<T>>);

unsafe impl<T: UniversalCopy> Sync for UVec<T> {}

impl<T: UniversalCopy> UVec<T> {
    fn get_intl_mut(&mut self) -> &mut UVecInternal<T> {
        self.0.get_mut()
    }

    fn get_intl(&self) -> &UVecInternal<T> {
        unsafe { &*self.0.get() }
    }

    unsafe fn get_intl_mut_unsafe(&self) -> &mut UVecInternal<T> {
        unsafe { &mut *self.0.get() }
    }
}

/// defines the reallocation heuristic. current we allocate 50\% more.
#[inline]
fn realloc_heuristic(new_len: usize) -> usize {
    (new_len as f64 * 1.5).round() as usize
}

/// The unsafe cell-wrapped internal.
struct UVecInternal<T: UniversalCopy> {
    data_cpu: Option<Box<[T]>>,
    #[cfg(feature = "cuda")]
    data_cuda: [Option<DeviceBuffer<T>>; MAX_NUM_CUDA_DEVICES],
    #[cfg(feature = "hip")]
    data_hip: [Option<HipBuffer<T>>; MAX_NUM_HIP_DEVICES],
    /// Metal buffer using shared memory (Apple Silicon UMA).
    /// With MTLResourceStorageModeShared, CPU and GPU share the same memory.
    #[cfg(feature = "metal")]
    data_metal: [Option<metal::Buffer>; MAX_NUM_METAL_DEVICES],
    /// A flag array recording the data presence and dirty status.
    /// A true entry means the data is valid on that device.
    valid_flag: [bool; MAX_DEVICES],
    /// Read locks for all devices
    ///
    /// This will not be locked for any operation originating
    /// from a write access -- no need to do so because Rust
    /// guarantees exclusive mutable reference.
    ///
    /// This will not be locked for readonly reference as long as
    /// our interested device is already ready for read (valid)
    /// -- no need to do so because Rust guarantees no mutation
    /// operation ever possible when a read-only reference
    /// is alive.
    ///
    /// This will ONLY be locked when a copy across device
    /// need to be launched with a read-only reference.
    /// The lock, in this case, is also per receiver device.
    read_locks: [Mutex<()>; MAX_DEVICES],
    /// the length of content
    len: usize,
    /// the length of buffer
    capacity: usize,
}

impl<T: UniversalCopy + fmt::Debug> fmt::Debug for UVec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.len() == 0 {
            return write!(f, "empty uvec");
        }
        let slice = self.as_ref();
        write!(f, "uvec[{}] = [", slice.len())?;
        for (i, e) in slice.iter().enumerate() {
            if i != 0 {
                write!(f, ", ")?;
            }
            if f.alternate() {
                write!(f, "{:#?}", e)?;
            } else {
                write!(f, "{:?}", e)?;
            }
        }
        write!(f, "]")
    }
}

impl<T: UniversalCopy> Default for UVec<T> {
    #[inline]
    fn default() -> Self {
        Self(UnsafeCell::new(UVecInternal {
            data_cpu: None,
            #[cfg(feature = "cuda")]
            data_cuda: Default::default(),
            #[cfg(feature = "hip")]
            data_hip: Default::default(),
            #[cfg(feature = "metal")]
            data_metal: Default::default(),
            valid_flag: [false; MAX_DEVICES],
            read_locks: Default::default(),
            len: 0,
            capacity: 0,
        }))
    }
}

impl<T: UniversalCopy> From<Box<[T]>> for UVec<T> {
    #[inline]
    fn from(b: Box<[T]>) -> UVec<T> {
        let len = b.len();
        let mut valid_flag = [false; MAX_DEVICES];
        valid_flag[Device::CPU.to_id()] = true;
        Self(UnsafeCell::new(UVecInternal {
            data_cpu: Some(b),
            #[cfg(feature = "cuda")]
            data_cuda: Default::default(),
            #[cfg(feature = "hip")]
            data_hip: Default::default(),
            #[cfg(feature = "metal")]
            data_metal: Default::default(),
            valid_flag,
            read_locks: Default::default(),
            len,
            capacity: len,
        }))
    }
}

impl<T: UniversalCopy> UVec<T> {
    /// Create a UVec by cloning from a universal pointer.
    ///
    /// Safety: the given pointer must be valid for `len` elements,
    /// and can be queried from the specific device.
    #[inline]
    pub unsafe fn from_uptr_cloned(ptr: impl AsUPtr<T>, len: usize, device: Device) -> UVec<T> {
        let mut uvec = UVec::new_uninitialized(len, device);
        uvec.copy_from(device, ptr, device, len);
        uvec
    }
}

impl<T: UniversalCopy> From<Vec<T>> for UVec<T> {
    #[inline]
    fn from(v: Vec<T>) -> UVec<T> {
        v.into_boxed_slice().into()
    }
}

impl<T: UniversalCopy> FromIterator<T> for UVec<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Vec::from_iter(iter).into()
    }
}

impl<T: UniversalCopy> From<UVec<T>> for Box<[T]> {
    #[inline]
    fn from(mut v: UVec<T>) -> Box<[T]> {
        v.schedule_device_read(Device::CPU);
        v.get_intl_mut().data_cpu.take().unwrap()
    }
}

impl<T: UniversalCopy> From<UVec<T>> for Vec<T> {
    #[inline]
    fn from(v: UVec<T>) -> Vec<T> {
        Box::<[T]>::from(v).into()
    }
}

#[cfg(feature = "rayon")]
mod uvec_rayon {
    use super::*;
    use rayon::prelude::*;

    impl<'i, T: UniversalCopy + Sync + 'i> IntoParallelIterator for &'i UVec<T> {
        type Iter = <&'i [T] as IntoParallelIterator>::Iter;
        type Item = &'i T;

        #[inline]
        fn into_par_iter(self) -> Self::Iter {
            self.as_ref().into_par_iter()
        }
    }

    impl<'i, T: UniversalCopy + Send + 'i> IntoParallelIterator for &'i mut UVec<T> {
        type Iter = <&'i mut [T] as IntoParallelIterator>::Iter;
        type Item = &'i mut T;

        #[inline]
        fn into_par_iter(self) -> Self::Iter {
            self.as_mut().into_par_iter()
        }
    }

    impl<T: UniversalCopy + Send> IntoParallelIterator for UVec<T> {
        type Iter = <Vec<T> as IntoParallelIterator>::Iter;
        type Item = T;

        #[inline]
        fn into_par_iter(self) -> Self::Iter {
            Vec::<T>::from(self).into_par_iter()
        }
    }

    impl<T: UniversalCopy + Send> FromParallelIterator<T> for UVec<T> {
        #[inline]
        fn from_par_iter<I: IntoParallelIterator<Item = T>>(par_iter: I) -> Self {
            Vec::from_par_iter(par_iter).into()
        }
    }
}

impl<T: UniversalCopy + Zeroable> UVecInternal<T> {
    /// private function to allocate space for one device.
    ///
    /// Guaranteed to only modify the buffer and the validity
    /// bit of the specified device.
    /// (which is useful in the safety of read-schedule
    /// interior mutability.)
    #[inline]
    fn alloc_zeroed(&mut self, device: Device) {
        use Device::*;
        match device {
            CPU => {
                use std::alloc;
                self.data_cpu = Some(unsafe {
                    let ptr = alloc::alloc_zeroed(alloc::Layout::array::<T>(self.capacity).unwrap())
                        as *mut T;
                    Box::from_raw(core::ptr::slice_from_raw_parts_mut(ptr, self.len))
                    // Box::new_zeroed_slice(sz).assume_init()
                });
            }
            #[cfg(feature = "cuda")]
            CUDA(c) => {
                let _context = Context::new(CUDA_DEVICES[c as usize].0).unwrap();
                self.data_cuda[c as usize] = Some(DeviceBuffer::zeroed(self.capacity).unwrap());
            }
            #[cfg(feature = "hip")]
            HIP(h) => {
                let _ctx = HIP(h).get_context();
                self.data_hip[h as usize] = Some(unsafe { HipBuffer::alloc_zeroed(self.capacity) });
            }
            #[cfg(feature = "metal")]
            Metal(m) => {
                let device = &METAL_DEVICES[m as usize];
                let byte_size = self.capacity * std::mem::size_of::<T>();
                // Use shared storage mode for Apple Silicon UMA
                let buffer =
                    device.new_buffer(byte_size as u64, MTLResourceOptions::StorageModeShared);
                // Zero the buffer
                unsafe {
                    std::ptr::write_bytes(buffer.contents() as *mut u8, 0, byte_size);
                }
                self.data_metal[m as usize] = Some(buffer);
            }
        }
    }
}

#[inline]
unsafe fn alloc_cpu_uninit<T: UniversalCopy>(sz: usize) -> Box<[T]> {
    use std::alloc;
    let ptr = alloc::alloc(alloc::Layout::array::<T>(sz).unwrap()) as *mut T;
    Box::from_raw(core::ptr::slice_from_raw_parts_mut(ptr, sz))
}

#[cfg(feature = "cuda")]
#[inline]
unsafe fn alloc_cuda_uninit<T: UniversalCopy>(sz: usize, dev: u8) -> DeviceBuffer<T> {
    let _context = Context::new(CUDA_DEVICES[dev as usize].0).unwrap();
    DeviceBuffer::uninitialized(sz).unwrap()
}

#[cfg(feature = "hip")]
#[inline]
unsafe fn alloc_hip_uninit<T: UniversalCopy>(sz: usize, dev: u8) -> HipBuffer<T> {
    let _ctx = Device::HIP(dev).get_context();
    HipBuffer::alloc_uninit(sz)
}

#[cfg(feature = "metal")]
#[inline]
fn alloc_metal_uninit<T: UniversalCopy>(sz: usize, dev: u8) -> metal::Buffer {
    let device = &METAL_DEVICES[dev as usize];
    let byte_size = sz * std::mem::size_of::<T>();
    // Use shared storage mode for Apple Silicon UMA
    device.new_buffer(byte_size as u64, MTLResourceOptions::StorageModeShared)
}

impl<T: UniversalCopy> UVecInternal<T> {
    /// private function to allocate space for one device.
    ///
    /// Guaranteed to only modify the buffer and the validity
    /// bit of the specified device.
    /// (which is useful in the safety of read-schedule
    /// interior mutability.)
    #[inline]
    unsafe fn alloc_uninitialized(&mut self, device: Device) {
        use Device::*;
        match device {
            CPU => {
                self.data_cpu = Some(alloc_cpu_uninit(self.capacity));
            }
            #[cfg(feature = "cuda")]
            CUDA(c) => {
                self.data_cuda[c as usize] = Some(alloc_cuda_uninit(self.capacity, c));
            }
            #[cfg(feature = "hip")]
            HIP(h) => {
                self.data_hip[h as usize] = Some(alloc_hip_uninit(self.capacity, h));
            }
            #[cfg(feature = "metal")]
            Metal(m) => {
                self.data_metal[m as usize] = Some(alloc_metal_uninit::<T>(self.capacity, m));
            }
        }
    }

    /// private function to get one device with valid data
    #[inline]
    fn device_valid(&self) -> Option<Device> {
        self.valid_flag
            .iter()
            .enumerate()
            .find(|(_i, v)| **v)
            .map(|(i, _v)| Device::from_id(i))
    }

    #[inline]
    fn drop_all_buf(&mut self) {
        self.data_cpu = None;
        #[cfg(feature = "cuda")]
        for d in &mut self.data_cuda {
            *d = None;
        }
        #[cfg(feature = "hip")]
        for d in &mut self.data_hip {
            *d = None;
        }
        #[cfg(feature = "metal")]
        for d in &mut self.data_metal {
            *d = None;
        }
    }

    #[inline]
    unsafe fn realloc_uninit_nopreserve(&mut self, device: Device) {
        self.drop_all_buf();
        if self.capacity > 10000000 {
            clilog::debug!("large realloc: capacity {}", self.capacity);
        }
        self.alloc_uninitialized(device);
        self.valid_flag.fill(false);
        self.valid_flag[device.to_id()] = true;
    }

    #[inline]
    unsafe fn realloc_uninit_preserve(&mut self, device: Device) {
        use Device::*;
        match device {
            CPU => {
                let old = self.data_cpu.take().unwrap();
                self.drop_all_buf();
                self.alloc_uninitialized(device);
                self.data_cpu.as_mut().unwrap()[..self.len].copy_from_slice(&old[..self.len]);
            }
            #[cfg(feature = "cuda")]
            CUDA(c) => {
                let _context = CUDA(c).get_context();
                let c = c as usize;
                let old = self.data_cuda[c].take().unwrap();
                self.drop_all_buf();
                self.alloc_uninitialized(device);
                self.data_cuda[c]
                    .as_mut()
                    .unwrap()
                    .index(..self.len)
                    .copy_from(&old.index(..self.len))
                    .unwrap();
            }
            #[cfg(feature = "hip")]
            HIP(h) => {
                let _ctx = HIP(h).get_context();
                let h = h as usize;
                let old = self.data_hip[h].take().unwrap();
                let byte_len = self.len * std::mem::size_of::<T>();
                self.drop_all_buf();
                self.alloc_uninitialized(device);
                let err = super::hip_ffi::hipMemcpy(
                    self.data_hip[h].as_ref().unwrap().as_mut_ptr() as *mut std::os::raw::c_void,
                    old.as_ptr() as *const std::os::raw::c_void,
                    byte_len,
                    super::hip_ffi::hipMemcpyDeviceToDevice,
                );
                super::hip_ffi::check_hip(err, "hipMemcpy D2D realloc");
            }
            #[cfg(feature = "metal")]
            Metal(m) => {
                let m = m as usize;
                let old = self.data_metal[m].take().unwrap();
                let byte_len = self.len * std::mem::size_of::<T>();
                self.drop_all_buf();
                self.alloc_uninitialized(device);
                // Copy data from old buffer to new buffer (both in shared memory)
                std::ptr::copy_nonoverlapping(
                    old.contents() as *const u8,
                    self.data_metal[m].as_ref().unwrap().contents() as *mut u8,
                    byte_len,
                );
            }
        }
        self.valid_flag.fill(false);
        self.valid_flag[device.to_id()] = true;
    }

    /// schedule a device to make its data available.
    ///
    /// Guaranteed to only modify the buffer and the validity
    /// bit of the specified device.
    /// (which is useful in the safety of read-schedule
    /// interior mutability.)
    #[inline]
    fn schedule_device_read(&mut self, device: Device) {
        if self.valid_flag[device.to_id()] {
            return;
        }
        use Device::*;
        let is_none = match device {
            CPU => self.data_cpu.is_none(),
            #[cfg(feature = "cuda")]
            CUDA(c) => self.data_cuda[c as usize].is_none(),
            #[cfg(feature = "hip")]
            HIP(h) => self.data_hip[h as usize].is_none(),
            #[cfg(feature = "metal")]
            Metal(m) => self.data_metal[m as usize].is_none(),
        };
        if is_none {
            unsafe {
                self.alloc_uninitialized(device);
            }
        }
        if self.capacity == 0 {
            return;
        }
        let device_valid = self.device_valid().expect("no valid dev");
        let byte_len = self.len * std::mem::size_of::<T>();
        match (device_valid, device) {
            (CPU, CPU) => {}
            #[cfg(feature = "cuda")]
            (CPU, CUDA(c)) => {
                let _context = CUDA(c).get_context();
                let c = c as usize;
                self.data_cuda[c]
                    .as_mut()
                    .unwrap()
                    .index(..self.len)
                    .copy_from(&self.data_cpu.as_ref().unwrap()[..self.len])
                    .unwrap();
            }
            #[cfg(feature = "cuda")]
            (CUDA(c), CPU) => {
                let _context = CUDA(c).get_context();
                let c = c as usize;
                self.data_cuda[c]
                    .as_ref()
                    .unwrap()
                    .index(..self.len)
                    .copy_to(&mut self.data_cpu.as_mut().unwrap()[..self.len])
                    .unwrap();
                CurrentContext::synchronize().unwrap();
            }
            #[cfg(feature = "cuda")]
            (CUDA(c1), CUDA(c2)) => {
                let _context = CUDA(c2).get_context();
                let (c1, c2) = (c1 as usize, c2 as usize);
                assert_ne!(c1, c2);
                // unsafe is used to access one mutable element.
                // safety guaranteed by the above `assert_ne!`.
                let c2_mut = unsafe {
                    &mut *(self.data_cuda[c2].as_mut().unwrap() as *const DeviceBuffer<T>
                        as *mut DeviceBuffer<T>)
                };
                self.data_cuda[c1]
                    .as_ref()
                    .unwrap()
                    .index(..self.len)
                    .copy_to(&mut c2_mut.index(..self.len))
                    .unwrap();
            }
            // HIP device copies
            #[cfg(feature = "hip")]
            (CPU, HIP(h)) => {
                let _ctx = HIP(h).get_context();
                let h = h as usize;
                unsafe {
                    let err = super::hip_ffi::hipMemcpy(
                        self.data_hip[h].as_ref().unwrap().as_mut_ptr() as *mut std::os::raw::c_void,
                        self.data_cpu.as_ref().unwrap().as_ptr() as *const std::os::raw::c_void,
                        byte_len,
                        super::hip_ffi::hipMemcpyHostToDevice,
                    );
                    super::hip_ffi::check_hip(err, "hipMemcpy H2D schedule");
                }
            }
            #[cfg(feature = "hip")]
            (HIP(h), CPU) => {
                let _ctx = HIP(h).get_context();
                let h = h as usize;
                unsafe {
                    let err = super::hip_ffi::hipMemcpy(
                        self.data_cpu.as_mut().unwrap().as_mut_ptr() as *mut std::os::raw::c_void,
                        self.data_hip[h].as_ref().unwrap().as_ptr() as *const std::os::raw::c_void,
                        byte_len,
                        super::hip_ffi::hipMemcpyDeviceToHost,
                    );
                    super::hip_ffi::check_hip(err, "hipMemcpy D2H schedule");
                    let err = super::hip_ffi::hipDeviceSynchronize();
                    super::hip_ffi::check_hip(err, "hipDeviceSynchronize after D2H");
                }
            }
            #[cfg(feature = "hip")]
            (HIP(h1), HIP(h2)) => {
                let _ctx = HIP(h2).get_context();
                let (h1, h2) = (h1 as usize, h2 as usize);
                assert_ne!(h1, h2);
                unsafe {
                    let err = super::hip_ffi::hipMemcpy(
                        self.data_hip[h2].as_ref().unwrap().as_mut_ptr() as *mut std::os::raw::c_void,
                        self.data_hip[h1].as_ref().unwrap().as_ptr() as *const std::os::raw::c_void,
                        byte_len,
                        super::hip_ffi::hipMemcpyDeviceToDevice,
                    );
                    super::hip_ffi::check_hip(err, "hipMemcpy D2D schedule");
                }
            }
            // Metal with shared memory - CPU and Metal can access the same memory
            #[cfg(feature = "metal")]
            (CPU, Metal(m)) => {
                let m = m as usize;
                // Copy from CPU to Metal buffer
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.data_cpu.as_ref().unwrap().as_ptr() as *const u8,
                        self.data_metal[m].as_ref().unwrap().contents() as *mut u8,
                        byte_len,
                    );
                }
            }
            #[cfg(feature = "metal")]
            (Metal(m), CPU) => {
                let m = m as usize;
                // Copy from Metal buffer to CPU
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.data_metal[m].as_ref().unwrap().contents() as *const u8,
                        self.data_cpu.as_mut().unwrap().as_mut_ptr() as *mut u8,
                        byte_len,
                    );
                }
            }
            #[cfg(feature = "metal")]
            (Metal(m1), Metal(m2)) => {
                let (m1, m2) = (m1 as usize, m2 as usize);
                assert_ne!(m1, m2);
                // Copy between Metal buffers
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.data_metal[m1].as_ref().unwrap().contents() as *const u8,
                        self.data_metal[m2].as_ref().unwrap().contents() as *mut u8,
                        byte_len,
                    );
                }
            }
            // Cross-device copies between CUDA and Metal not directly supported
            #[cfg(all(feature = "cuda", feature = "metal"))]
            (CUDA(_), Metal(_)) | (Metal(_), CUDA(_)) => {
                panic!(
                    "Direct copy between CUDA and Metal devices is not supported. \
                        Please stage through CPU."
                );
            }
            // Cross-device copies involving HIP and other GPU backends
            #[cfg(all(feature = "hip", feature = "cuda"))]
            (HIP(_), CUDA(_)) | (CUDA(_), HIP(_)) => {
                panic!(
                    "Direct copy between HIP and CUDA devices is not supported. \
                        Please stage through CPU."
                );
            }
            #[cfg(all(feature = "hip", feature = "metal"))]
            (HIP(_), Metal(_)) | (Metal(_), HIP(_)) => {
                panic!(
                    "Direct copy between HIP and Metal devices is not supported. \
                        Please stage through CPU."
                );
            }
        }
        self.valid_flag[device.to_id()] = true;
    }
}

impl<T: UniversalCopy> UVec<T> {
    /// schedule a device to make its data available.
    ///
    /// Guaranteed to only modify the buffer and the validity
    /// bit of the specified device.
    /// (which is useful in the safety of read-schedule
    /// interior mutability.)
    #[inline]
    fn schedule_device_read(&mut self, device: Device) {
        self.get_intl_mut().schedule_device_read(device);
    }

    /// schedule a device to make its data available
    /// THROUGH a read-only reference.
    ///
    /// will acquire a lock if it is necessary.
    /// If you have mutable reference, use the lock-free
    /// `schedule_device_read` instead.
    #[inline]
    fn schedule_device_read_ro(&self, device: Device) {
        // safety guaranteed by the lock, and by the
        // guarantee of `schedule_device_read` that only
        // writes to fields related to the specified device.
        let intl = unsafe { self.get_intl_mut_unsafe() };
        let intl_erased = unsafe { &mut *(self.get_intl_mut_unsafe() as *mut UVecInternal<T>) };
        if intl.valid_flag[device.to_id()] {
            return;
        }
        let locked = intl.read_locks[device.to_id()].lock().unwrap();
        intl_erased.schedule_device_read(device);
        drop(locked);
    }

    /// schedule a device write. invalidates all other ranges.
    #[inline]
    fn schedule_device_write(&mut self, device: Device) {
        let intl = self.get_intl_mut();
        if !intl.valid_flag[device.to_id()] {
            intl.schedule_device_read(device);
        }
        // only this is valid.
        intl.valid_flag[..].fill(false);
        intl.valid_flag[device.to_id()] = true;
    }

    #[inline]
    pub fn get(&self, idx: usize) -> T {
        use Device::*;
        let intl = self.get_intl();
        match intl.device_valid().unwrap() {
            CPU => intl.data_cpu.as_ref().unwrap()[idx],
            #[cfg(feature = "cuda")]
            CUDA(c) => {
                let _context = CUDA(c).get_context();
                let mut ret: [T; 1] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
                intl.data_cuda[c as usize]
                    .as_ref()
                    .unwrap()
                    .index(idx)
                    .copy_to(&mut ret)
                    .unwrap();
                CurrentContext::synchronize().unwrap();
                ret[0]
            }
            #[cfg(feature = "hip")]
            HIP(h) => {
                let _ctx = HIP(h).get_context();
                let mut ret: T = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
                unsafe {
                    let err = super::hip_ffi::hipMemcpy(
                        &mut ret as *mut T as *mut std::os::raw::c_void,
                        intl.data_hip[h as usize].as_ref().unwrap().as_ptr().add(idx) as *const std::os::raw::c_void,
                        std::mem::size_of::<T>(),
                        super::hip_ffi::hipMemcpyDeviceToHost,
                    );
                    super::hip_ffi::check_hip(err, "hipMemcpy D2H get");
                    let err = super::hip_ffi::hipDeviceSynchronize();
                    super::hip_ffi::check_hip(err, "hipDeviceSynchronize after get");
                }
                ret
            }
            #[cfg(feature = "metal")]
            Metal(m) => {
                // Metal shared memory can be directly accessed from CPU
                let buffer = intl.data_metal[m as usize].as_ref().unwrap();
                let ptr = buffer.contents() as *const T;
                unsafe { *ptr.add(idx) }
            }
        }
    }
}

impl<T: UniversalCopy + Zeroable> UVec<T> {
    /// Create a new zeroed universal vector with specific size and
    /// capacity;
    #[inline]
    pub fn new_zeroed_with_capacity(len: usize, capacity: usize, device: Device) -> UVec<T> {
        let mut v: UVec<T> = Default::default();
        let intl = v.get_intl_mut();
        assert!(len <= capacity);
        intl.len = len;
        intl.capacity = capacity;
        intl.alloc_zeroed(device);
        intl.valid_flag[device.to_id()] = true;
        v
    }

    /// Create a new zeroed universal vector with specific size.
    #[inline]
    pub fn new_zeroed(len: usize, device: Device) -> UVec<T> {
        Self::new_zeroed_with_capacity(len, len, device)
    }
}

impl<T: UniversalCopy> UVec<T> {
    /// Get length (size) of this universal vector.
    #[inline]
    pub fn len(&self) -> usize {
        self.get_intl().len
    }

    /// Returns `true` if this universal vector has a length of 0.
    ///
    /// Empty uvec can have no valid devices, and will return nullptr on
    /// `AsRef` or [`AsUPtr`] calls.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.get_intl().len == 0
    }

    /// Get capacity of this vector.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.get_intl().capacity
    }

    /// New empty vector.
    ///
    /// This should only be used as a placeholder. it allocates
    /// nothing. will panic if you try to get any pointer from it.
    #[inline]
    pub fn new() -> UVec<T> {
        Default::default()
    }

    /// Create a new uninitialized universal vector with
    /// specific size and capacity.
    #[inline]
    pub unsafe fn new_uninitialized_with_capacity(
        len: usize,
        capacity: usize,
        device: Device,
    ) -> UVec<T> {
        let mut v: UVec<T> = Default::default();
        let intl = v.get_intl_mut();
        assert!(len <= capacity);
        intl.len = len;
        intl.capacity = capacity;
        intl.alloc_uninitialized(device);
        intl.valid_flag[device.to_id()] = true;
        v
    }

    /// Create a new uninitialized universal vector with
    /// specific size.
    #[inline]
    pub unsafe fn new_uninitialized(len: usize, device: Device) -> UVec<T> {
        Self::new_uninitialized_with_capacity(len, len, device)
    }

    /// Create a new zero length universal vector with specific
    /// initial capacity.
    #[inline]
    pub fn with_capacity(capacity: usize, device: Device) -> UVec<T> {
        unsafe { Self::new_uninitialized_with_capacity(0, capacity, device) }
    }

    /// Force set the length of this vector.
    ///
    /// this is a low-level operation that does not reallocate.
    /// safe only when the new length does not exceed current capacity
    /// and the new visible elements (if any) are not uninitialized.
    ///
    /// See also `std::vec::Vec::set_len`.
    #[inline]
    pub unsafe fn set_len(&mut self, new_len: usize) {
        let intl = self.get_intl_mut();
        assert!(new_len <= intl.capacity);
        intl.len = new_len;
    }

    /// Reserves capacity for at least `additional` more elements
    /// to be inserted in the given `UVec<T>`.
    ///
    /// The collection may reserve more space to speculatively avoid
    /// frequent reallocations. After calling reserve, capacity will be
    /// greater than or equal to self.len() + `additional`.
    /// Does nothing if capacity is already sufficient.
    ///
    /// See also `std::vec::Vec::reserve`. we have an additional arg
    /// `device` specifying, when an re-allocation is necessary, which
    /// device's data needs preserving (often means immediate use).
    ///
    /// The `reserve` and `resize_uninit_[no]preserve` are two distinct
    /// methodologies of reallocation (len-based or capacity-based).
    /// You can choose one at your convenience.
    #[inline]
    pub fn reserve(&mut self, additional: usize, device: Device) {
        let intl = self.get_intl_mut();
        if intl.len + additional <= intl.capacity {
            return;
        }
        intl.capacity = realloc_heuristic(intl.len + additional);
        unsafe {
            intl.realloc_uninit_preserve(device);
        }
    }

    /// Resize the universal vector, but do **not** preserve the
    /// original content.
    /// The potential new elements are **uninitialized**.
    ///
    /// If the current capacity is sufficient, we do not need to
    /// reallocate or do anything else. We just mark the desired
    /// device as valid.
    ///
    /// If the current capacity is insufficient, a reallocation
    /// is needed and all current allocations are dropped.
    /// (we maintain the invariant that all allocated buffers for
    /// all devices must all have the same length (= capacity).)
    #[inline]
    pub unsafe fn resize_uninit_nopreserve(&mut self, len: usize, device: Device) {
        let intl = self.get_intl_mut();
        if intl.capacity < len {
            intl.capacity = realloc_heuristic(len);
            intl.realloc_uninit_nopreserve(device);
        }
        intl.len = len;
    }

    /// Resize the universal vector, and preserve all the
    /// original content.
    /// The potential new elements are **uninitialized**.
    #[inline]
    pub unsafe fn resize_uninit_preserve(&mut self, len: usize, device: Device) {
        if self.get_intl().len != 0 {
            self.schedule_device_read(device);
        }
        let intl = self.get_intl_mut();
        if intl.capacity < len {
            intl.capacity = realloc_heuristic(len);
            intl.realloc_uninit_preserve(device);
        }
        intl.len = len;
        intl.valid_flag.fill(false);
        intl.valid_flag[device.to_id()] = true;
    }

    #[inline]
    pub fn fill(&mut self, value: T, device: Device) {
        self.fill_len(value, self.len(), device);
    }

    #[inline]
    pub fn new_filled(value: T, len: usize, device: Device) -> UVec<T> {
        let mut v = unsafe { Self::new_uninitialized(len, device) };
        v.fill(value, device);
        v
    }
}

impl<T: UniversalCopy> AsRef<[T]> for UVec<T> {
    /// Get a CPU slice reference.
    ///
    /// This COULD fail, actually, when we need to copy from
    /// a GPU value to CPU.
    /// This violates the guideline but we have no choice.
    ///
    /// It will lock only when a copy is needed.
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.schedule_device_read_ro(Device::CPU);
        let intl = self.get_intl();
        &intl.data_cpu.as_ref().unwrap()[..intl.len]
    }
}

impl<T: UniversalCopy> AsMut<[T]> for UVec<T> {
    /// Get a mutable CPU slice reference.
    ///
    /// This COULD fail, actually, when we need to copy from
    /// a GPU value to CPU.
    /// This violates the guideline but we have no choice.
    ///
    /// It is lock-free.
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.schedule_device_write(Device::CPU);
        let intl = self.get_intl_mut();
        &mut intl.data_cpu.as_mut().unwrap()[..intl.len]
    }
}

impl<T: UniversalCopy> Deref for UVec<T> {
    type Target = [T];
    /// `Deref` is now implemented for `UVec` to let you
    /// use it transparently.
    ///
    /// Internally it may fail because it might schedule a
    /// inter-device copy to make the data available on CPU.
    /// But it is thread-safe.
    #[inline]
    fn deref(&self) -> &[T] {
        self.as_ref()
    }
}

impl<T: UniversalCopy> DerefMut for UVec<T> {
    /// `Deref` is now implemented for `UVec` to let you
    /// use it transparently.
    ///
    /// Internally it may fail because it might schedule a
    /// inter-device copy to make the data available on CPU.
    /// But it is thread-safe.
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut()
    }
}

impl<T: UniversalCopy, I> Index<I> for UVec<T>
where
    [T]: Index<I>,
{
    type Output = <[T] as Index<I>>::Output;
    #[inline]
    fn index(&self, i: I) -> &Self::Output {
        self.as_ref().index(i)
    }
}

impl<T: UniversalCopy, I> IndexMut<I> for UVec<T>
where
    [T]: IndexMut<I>,
{
    #[inline]
    fn index_mut(&mut self, i: I) -> &mut Self::Output {
        self.as_mut().index_mut(i)
    }
}

impl<T: UniversalCopy> IntoIterator for UVec<T> {
    type Item = T;
    type IntoIter = <Vec<T> as IntoIterator>::IntoIter;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Vec::from(self).into_iter()
    }
}

impl<'i, T: UniversalCopy> IntoIterator for &'i UVec<T> {
    type Item = &'i T;
    type IntoIter = <&'i [T] as IntoIterator>::IntoIter;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_ref().into_iter()
    }
}

impl<T: UniversalCopy> AsUPtr<T> for UVec<T> {
    #[inline]
    fn as_uptr(&self, device: Device) -> *const T {
        if self.capacity() == 0 {
            return std::ptr::null();
        }
        self.schedule_device_read_ro(device);
        let intl = self.get_intl();
        use Device::*;
        match device {
            CPU => intl.data_cpu.as_ref().unwrap().as_ptr(),
            #[cfg(feature = "cuda")]
            CUDA(c) => intl.data_cuda[c as usize]
                .as_ref()
                .unwrap()
                .as_device_ptr()
                .as_ptr(),
            #[cfg(feature = "hip")]
            HIP(h) => intl.data_hip[h as usize].as_ref().unwrap().as_ptr(),
            #[cfg(feature = "metal")]
            Metal(m) => intl.data_metal[m as usize].as_ref().unwrap().contents() as *const T,
        }
    }
}

impl<T: UniversalCopy> AsUPtrMut<T> for UVec<T> {
    #[inline]
    fn as_mut_uptr(&mut self, device: Device) -> *mut T {
        if self.capacity() == 0 {
            return std::ptr::null_mut();
        }
        self.schedule_device_write(device);
        let intl = self.get_intl_mut();
        use Device::*;
        match device {
            CPU => intl.data_cpu.as_mut().unwrap().as_mut_ptr(),
            #[cfg(feature = "cuda")]
            CUDA(c) => intl.data_cuda[c as usize]
                .as_mut()
                .unwrap()
                .as_device_ptr()
                .as_mut_ptr(),
            #[cfg(feature = "hip")]
            HIP(h) => intl.data_hip[h as usize].as_ref().unwrap().as_mut_ptr(),
            #[cfg(feature = "metal")]
            Metal(m) => intl.data_metal[m as usize].as_ref().unwrap().contents() as *mut T,
        }
    }
}

// although convenient, below gets in the way of automatic type inference.

// impl<T: UniversalCopy, const N: usize> AsUPtr<T> for UVec<[T; N]> {
//     /// convenient way to get flattened pointer
//     #[inline]
//     fn as_uptr(&self, device: Device) -> *const T {
//         AsUPtr::<[T; N]>::as_uptr(self, device) as *const T
//     }
// }

// impl<T: UniversalCopy, const N: usize> AsUPtrMut<T> for UVec<[T; N]> {
//     /// convenient way to get flattened pointer
//     #[inline]
//     fn as_mut_uptr(&mut self, device: Device) -> *mut T {
//         AsUPtrMut::<[T; N]>::as_mut_uptr(self, device) as *mut T
//     }
// }

impl<T: UniversalCopy + Hash> Hash for UVec<T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_ref().hash(state)
    }
}

impl<T: UniversalCopy, U: UniversalCopy> PartialEq<UVec<U>> for UVec<T>
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &UVec<U>) -> bool {
        if self.len() != other.len() {
            return false;
        }
        if self.is_empty() {
            return true;
        }
        self.as_ref() == other.as_ref()
    }
}

impl<T: UniversalCopy + Eq> Eq for UVec<T> {}

impl<T: UniversalCopy> Clone for UVecInternal<T> {
    fn clone(&self) -> Self {
        let valid_flag = self.valid_flag.clone();
        let data_cpu = match valid_flag[Device::CPU.to_id()] {
            true => self.data_cpu.clone(),
            false => None,
        };
        #[cfg(feature = "cuda")]
        let data_cuda = unsafe {
            let mut data_cuda: [Option<DeviceBuffer<T>>; MAX_NUM_CUDA_DEVICES] = Default::default();
            for i in 0..MAX_NUM_CUDA_DEVICES {
                if valid_flag[Device::CUDA(i as u8).to_id()] {
                    let _context = Device::CUDA(i as u8).get_context();
                    let dbuf = alloc_cuda_uninit(self.capacity, i as u8);
                    self.data_cuda[i]
                        .as_ref()
                        .unwrap()
                        .index(..self.len)
                        .copy_to(&mut dbuf.index(..self.len))
                        .unwrap();
                    data_cuda[i] = Some(dbuf);
                }
            }
            data_cuda
        };
        #[cfg(feature = "hip")]
        let data_hip = unsafe {
            let mut data_hip: [Option<HipBuffer<T>>; MAX_NUM_HIP_DEVICES] = Default::default();
            for i in 0..MAX_NUM_HIP_DEVICES {
                if valid_flag[Device::HIP(i as u8).to_id()] {
                    let _ctx = Device::HIP(i as u8).get_context();
                    let new_buf = alloc_hip_uninit(self.capacity, i as u8);
                    let byte_len = self.len * std::mem::size_of::<T>();
                    let err = super::hip_ffi::hipMemcpy(
                        new_buf.as_mut_ptr() as *mut std::os::raw::c_void,
                        self.data_hip[i].as_ref().unwrap().as_ptr() as *const std::os::raw::c_void,
                        byte_len,
                        super::hip_ffi::hipMemcpyDeviceToDevice,
                    );
                    super::hip_ffi::check_hip(err, "hipMemcpy D2D clone");
                    data_hip[i] = Some(new_buf);
                }
            }
            data_hip
        };
        #[cfg(feature = "metal")]
        let data_metal = {
            let mut data_metal: [Option<metal::Buffer>; MAX_NUM_METAL_DEVICES] = Default::default();
            for i in 0..MAX_NUM_METAL_DEVICES {
                if valid_flag[Device::Metal(i as u8).to_id()] {
                    let new_buf = alloc_metal_uninit::<T>(self.capacity, i as u8);
                    let byte_len = self.len * std::mem::size_of::<T>();
                    // Copy from old buffer to new buffer (both in shared memory)
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            self.data_metal[i].as_ref().unwrap().contents() as *const u8,
                            new_buf.contents() as *mut u8,
                            byte_len,
                        );
                    }
                    data_metal[i] = Some(new_buf);
                }
            }
            data_metal
        };
        UVecInternal {
            data_cpu,
            #[cfg(feature = "cuda")]
            data_cuda,
            #[cfg(feature = "hip")]
            data_hip,
            #[cfg(feature = "metal")]
            data_metal,
            valid_flag,
            read_locks: Default::default(),
            len: self.len,
            capacity: self.capacity,
        }
    }
}

impl<T: UniversalCopy> Clone for UVec<T> {
    fn clone(&self) -> Self {
        Self(UnsafeCell::new(self.get_intl().clone()))
    }
}
