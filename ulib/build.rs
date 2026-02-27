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
        println!("cargo:rustc-link-lib=dylib=amdhip64");
        // Add ROCm library search path (default /opt/rocm/lib).
        let rocm_path = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
        println!("cargo:rustc-link-search=native={}/lib", rocm_path);
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
