//! Memory usage profiling utilities

use crate::Device;
#[cfg(feature = "cuda")]
use cust_raw::cuMemGetInfo_v2;
use memory_stats::memory_stats;
use size::Size;

/// Returns CPU memory usage of current process.
///
/// the memory is the size of the physical residence set (working set).
/// obtained using `memory_stats` crate.
///
/// note this is NOT the peak.
pub fn cpu_physical_mem() -> usize {
    memory_stats().unwrap().physical_mem
}

/// Returns CUDA device used memory and total memory.
///
/// obtained using [cuMemGetInfo](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0).
///
/// not this is NOT the peak, and also NOT only the memory used
/// by current process but other processes as well.
#[cfg(feature = "cuda")]
pub fn cuda_used_total_mem(cuda_device_id: u8) -> (usize, usize) {
    let (mut free, mut total) = (0, 0);
    let _context = Device::CUDA(cuda_device_id).get_context();
    unsafe {
        let ret = cuMemGetInfo_v2(&mut free, &mut total);
        assert_eq!(ret, cust_raw::cudaError_enum::CUDA_SUCCESS);
    }
    (total - free, total)
}

/// Returns Metal device recommended working set size.
///
/// On Apple Silicon, Metal uses unified memory so this returns
/// the recommended working set size hint from the Metal device.
#[cfg(feature = "metal")]
pub fn metal_recommended_working_set(metal_device_id: u8) -> usize {
    use crate::METAL_DEVICES;
    METAL_DEVICES[metal_device_id as usize].recommended_max_working_set_size() as usize
}

/// Get memory used on the specified universal device.
pub fn device_mem_used(device: Device) -> usize {
    match device {
        Device::CPU => cpu_physical_mem(),
        #[cfg(feature = "cuda")]
        Device::CUDA(cuid) => cuda_used_total_mem(cuid).0,
        // HIP does not expose a simple memory query in our minimal FFI;
        // return CPU physical memory as an approximation.
        #[cfg(feature = "hip")]
        Device::HIP(_) => cpu_physical_mem(),
        // Metal uses unified memory on Apple Silicon, so GPU memory
        // is part of system memory. Return CPU physical mem as approximation.
        #[cfg(feature = "metal")]
        Device::Metal(_) => cpu_physical_mem(),
    }
}

/// Print in the logging interface the memory usage of all
/// managed heterogeneous devices.
pub fn log_memory_stats() {
    let cpu_mem = cpu_physical_mem();
    clilog::info!("cpu memory footprint: {}", Size::from_bytes(cpu_mem));
    #[cfg(feature = "cuda")]
    {
        for cuid in 0..*crate::NUM_CUDA_DEVICES {
            let (cuda_used, cuda_total) = cuda_used_total_mem(cuid.try_into().unwrap());
            clilog::info!(
                "cuda device {} memory usage: {} / {}",
                cuid,
                Size::from_bytes(cuda_used),
                Size::from_bytes(cuda_total)
            );
        }
    }
    #[cfg(feature = "hip")]
    {
        let num_hip = *crate::NUM_HIP_DEVICES;
        clilog::info!("hip devices detected: {}", num_hip);
    }
    #[cfg(feature = "metal")]
    {
        for mid in 0..*crate::NUM_METAL_DEVICES {
            let recommended = metal_recommended_working_set(mid.try_into().unwrap());
            // Metal uses unified memory, so we report the recommended working set
            clilog::info!(
                "metal device {} recommended working set: {}",
                mid,
                Size::from_bytes(recommended)
            );
        }
    }
}
