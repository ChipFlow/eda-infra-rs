//! this build script exports the csrc dir to dependents.

fn main() {
    println!("Building the C source files of ulib..");
    println!("cargo:rerun-if-changed=csrc");

    let mut cl_cpp = ucc::cl_cpp_openmp();
    cl_cpp.file("csrc/memfill.cpp");
    cl_cpp.compile("ulibc");
    println!("cargo:rustc-link-lib=static=ulibc");

    #[cfg(feature = "cuda")]
    let cl_cuda = {
        let mut cl_cuda = ucc::cl_cuda();
        cl_cuda.ccbin(false);
        cl_cuda.debug(false).opt_level(3).file("csrc/memfill.cu");
        cl_cuda.compile("ulibcu");
        println!("cargo:rustc-link-lib=static=ulibcu");
        println!("cargo:rustc-link-lib=dylib=cudart");
        cl_cuda
    };

    #[cfg(feature = "hip")]
    let cl_hip = {
        let mut cl_hip = ucc::cl_hip();
        cl_hip.debug(false).opt_level(3).file("csrc/memfill.hip.cpp");
        cl_hip.compile("ulibhip");
        println!("cargo:rustc-link-lib=static=ulibhip");
        // On AMD backend, HIP runtime is in libamdhip64; on NVIDIA backend,
        // hipcc links against cudart. The hip_ffi_* wrapper functions compiled
        // into ulibhip handle both cases — link the appropriate runtime.
        if std::env::var("HIP_PLATFORM").as_deref() == Ok("nvidia") {
            println!("cargo:rustc-link-lib=dylib=cudart");
            // CUDA toolkit may be in a non-standard path.
            let cuda_path = std::env::var("CUDA_PATH")
                .unwrap_or_else(|_| "/usr/local/cuda".to_string());
            println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
            println!("cargo:rustc-link-search=native={}/lib", cuda_path);
        } else {
            println!("cargo:rustc-link-lib=dylib=amdhip64");
            let rocm_path = std::env::var("ROCM_PATH")
                .unwrap_or_else(|_| "/opt/rocm".to_string());
            println!("cargo:rustc-link-search=native={}/lib", rocm_path);
        }
        println!("cargo:rerun-if-env-changed=HIP_PLATFORM");
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
        cl_hip
    };

    ucc::bindgen(
        [
            "csrc/memfill.cpp",
            #[cfg(feature = "cuda")]
            "csrc/memfill.cu",
            #[cfg(feature = "hip")]
            "csrc/memfill.hip.cpp",
        ],
        "memfill.rs",
    );

    ucc::export_csrc();
    ucc::make_compile_commands(&[
        &cl_cpp,
        #[cfg(feature = "cuda")]
        &cl_cuda,
        #[cfg(feature = "hip")]
        &cl_hip,
    ]);
}
