//! compiler collection

use cc::Build;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// private util to add version definitions to the compiler.
fn add_definitions(builder: &mut Build) {
    macro_rules! add_definition {
        ($($def:ident),+) => {$(
            #[allow(non_snake_case)]
            let $def = env::var(stringify!($def)).ok();
            builder.define(stringify!($def),
                           $def.as_ref().map(|s| s.as_str()));
            println!("cargo:rerun-if-env-changed={}", stringify!($def));
        )+}
    }
    add_definition! {
        CARGO_PKG_NAME,
        CARGO_PKG_VERSION,
        CARGO_PKG_VERSION_MAJOR,
        CARGO_PKG_VERSION_MINOR,
        CARGO_PKG_VERSION_PATCH
    }
    builder.define("UCC_VERSION", env!("CARGO_PKG_VERSION"));
}

/// initialize a `cc` compiler with openmp support.
pub fn cl_cpp_openmp() -> Build {
    let mut builder = Build::new();
    println!("cargo:rerun-if-env-changed=CC");
    if cfg!(target_os = "macos") {
        if !env::var("CC").is_ok() {
            // if CC env var is not set, we try some common
            // overrides.
            if Path::new("/opt/homebrew/opt/llvm/bin/clang").exists() {
                // on macos m1, we use the homebrew clang compiler.
                // this supports openmp.
                //
                // this also supports some static linking of openmp,
                // but according to the openmp docs it is not recommended.
                builder.compiler("/opt/homebrew/opt/llvm/bin/clang");
                println!("cargo:rustc-link-search=/opt/homebrew/opt/llvm/lib");
                println!("cargo:rustc-link-search=/opt/homebrew/lib");
            }
        }
        // on macos, the library is omp.
        println!("cargo:rustc-link-lib=dylib=omp");
    } else {
        // on linux, the library is gomp.
        // static linking is also available but not very straightforward,
        // as the libgomp.a is hidden somewhere. also not preferred.
        println!("cargo:rustc-link-lib=dylib=gomp");
    }
    builder
        .cpp(true)
        .flag("-Wall")
        .flag("-fopenmp")
        .flag("-std=c++14")
        .out_dir(
            env::var_os("OUT_DIR")
                .map(|v| {
                    let mut v = PathBuf::from(v);
                    v.push("ucc_cpp");
                    v
                })
                .unwrap(),
        );
    add_definitions(&mut builder);
    builder
}

/// initialize a cuda compiler with given code generation options.
///
/// - if `gencode` is specified, gencode is applied to generate SASS code
///   for the specific capabilities.
/// - if `ptx_arch` is specified, arch is applied to attach PTX code
///   for at least the given capabilities.
///
/// it is ALWAYS suggested to specify `ptx_arch` as it makes the code
/// compatible to a wide range of GPU targets.
pub fn cl_cuda_arch(gencode: Option<&[u32]>, ptx_arch: Option<u32>) -> Build {
    let mut builder_cuda = Build::new();
    builder_cuda
        .cuda(true)
        .flag("-Xcompiler")
        .flag("-Wall")
        .flag("-std=c++14");
    for arch in gencode.unwrap_or(&[]) {
        builder_cuda
            .flag("-gencode")
            .flag(&format!("arch=compute_{arch},code=sm_{arch}"));
    }
    if let Some(ptx_arch) = ptx_arch {
        builder_cuda.flag(&format!("-arch=compute_{ptx_arch}"));
        builder_cuda.flag(&format!("-code=sm_{ptx_arch},compute_{ptx_arch}"));
    }
    builder_cuda.out_dir(
        env::var_os("OUT_DIR")
            .map(|v| {
                let mut v = PathBuf::from(v);
                v.push("ucc_cuda");
                v
            })
            .unwrap(),
    );
    add_definitions(&mut builder_cuda);
    builder_cuda
}

/// a shorthand for frequently-used cuda compiler options, can be controlled
/// by environment variables.
///
/// if nothing is given in environment variables, we will generate PTX for
/// cc5.0, and SASS for cc8.0 and cc7.0.
///
/// if `UCC_CUDA_PTX` is set, ptx is set to the given version (empty means
/// no ptx is generated).
/// if `UCC_CUDA_GENCODE` is set, gencode is set to given versions (comma
/// separated).
///
/// if you want direct control, see [`cl_cuda_arch`].
pub fn cl_cuda() -> Build {
    println!("cargo:rerun-if-env-changed=UCC_CUDA_PTX");
    println!("cargo:rerun-if-env-changed=UCC_CUDA_GENCODE");
    let ptx_arch = match env::var("UCC_CUDA_PTX") {
        Ok(v) => {
            if v.is_empty() {
                None
            } else {
                Some(v.parse().unwrap())
            }
        }
        Err(_) => Some(50),
    };
    let gencode = match env::var("UCC_CUDA_GENCODE") {
        Ok(v) => {
            if v.is_empty() {
                None
            } else {
                Some(v.split(',').map(|i| i.parse().unwrap()).collect::<Vec<_>>())
            }
        }
        Err(_) => Some(vec![80, 70]),
    };
    cl_cuda_arch(gencode.as_deref(), ptx_arch)
}

/// Metal shader build configuration.
///
/// This struct provides a builder pattern for compiling Metal shaders
/// into metallib files that can be loaded at runtime.
pub struct MetalBuild {
    files: Vec<PathBuf>,
    include_dirs: Vec<PathBuf>,
    defines: Vec<(String, Option<String>)>,
    out_dir: PathBuf,
    std_version: String,
    macos_version_min: String,
}

impl MetalBuild {
    /// Create a new Metal build configuration.
    pub fn new() -> Self {
        let out_dir = env::var_os("OUT_DIR")
            .map(|v| {
                let mut v = PathBuf::from(v);
                v.push("ucc_metal");
                v
            })
            .unwrap_or_else(|| PathBuf::from("target/ucc_metal"));

        Self {
            files: Vec::new(),
            include_dirs: Vec::new(),
            defines: Vec::new(),
            out_dir,
            std_version: "metal3.0".to_string(),
            macos_version_min: "14.0".to_string(),
        }
    }

    /// Add a Metal shader file to compile.
    pub fn file(&mut self, path: impl AsRef<Path>) -> &mut Self {
        let path = path.as_ref();
        println!("cargo:rerun-if-changed={}", path.display());
        self.files.push(path.to_path_buf());
        self
    }

    /// Add an include directory.
    pub fn include(&mut self, path: impl AsRef<Path>) -> &mut Self {
        self.include_dirs.push(path.as_ref().to_path_buf());
        self
    }

    /// Add a preprocessor definition.
    pub fn define(&mut self, name: &str, value: Option<&str>) -> &mut Self {
        self.defines
            .push((name.to_string(), value.map(|s| s.to_string())));
        self
    }

    /// Set the Metal shader language version.
    pub fn std_version(&mut self, version: &str) -> &mut Self {
        self.std_version = version.to_string();
        self
    }

    /// Set the minimum macOS deployment target.
    pub fn macos_version_min(&mut self, version: &str) -> &mut Self {
        self.macos_version_min = version.to_string();
        self
    }

    /// Set the output directory.
    pub fn out_dir(&mut self, path: impl AsRef<Path>) -> &mut Self {
        self.out_dir = path.as_ref().to_path_buf();
        self
    }

    /// Compile the Metal shaders into a metallib file.
    ///
    /// Returns the path to the generated metallib file.
    /// Also sets the METALLIB_PATH environment variable for the crate.
    pub fn compile(&self, lib_name: &str) -> PathBuf {
        fs::create_dir_all(&self.out_dir).expect("failed to create output directory");

        let mut air_files = Vec::new();

        // Compile each .metal file to .air (Apple Intermediate Representation)
        for metal_file in &self.files {
            let file_stem = metal_file
                .file_stem()
                .expect("metal file has no stem")
                .to_str()
                .expect("invalid file stem");

            let air_file = self.out_dir.join(format!("{}.air", file_stem));

            let mut cmd = Command::new("xcrun");
            cmd.arg("-sdk")
                .arg("macosx")
                .arg("metal")
                .arg("-c")
                .arg(format!("-std={}", self.std_version))
                .arg(format!("-mmacosx-version-min={}", self.macos_version_min));

            // Add include directories
            for inc in &self.include_dirs {
                cmd.arg("-I").arg(inc);
            }

            // Add defines
            for (name, value) in &self.defines {
                match value {
                    Some(v) => cmd.arg(format!("-D{}={}", name, v)),
                    None => cmd.arg(format!("-D{}", name)),
                };
            }

            cmd.arg("-o").arg(&air_file).arg(metal_file);

            let status = cmd.status().expect("failed to run metal compiler");

            if !status.success() {
                panic!("metal compiler failed for {}", metal_file.display());
            }

            air_files.push(air_file);
        }

        // Link all .air files into a single .metallib
        let metallib_file = self.out_dir.join(format!("{}.metallib", lib_name));

        let mut cmd = Command::new("xcrun");
        cmd.arg("-sdk")
            .arg("macosx")
            .arg("metallib")
            .arg("-o")
            .arg(&metallib_file);

        for air_file in &air_files {
            cmd.arg(air_file);
        }

        let status = cmd.status().expect("failed to run metallib linker");

        if !status.success() {
            panic!("metallib linking failed");
        }

        // Export the metallib path for the crate to use
        println!("cargo:rustc-env=METALLIB_PATH={}", metallib_file.display());

        metallib_file
    }
}

impl Default for MetalBuild {
    fn default() -> Self {
        Self::new()
    }
}

/// Initialize a Metal shader compiler.
///
/// This returns a MetalBuild struct that can be used to configure
/// and compile Metal shader files (.metal) into a metallib.
///
/// # Example
///
/// ```ignore
/// let metallib_path = ucc::cl_metal()
///     .file("csrc/kernel.metal")
///     .include("csrc")
///     .compile("my_shaders");
/// ```
pub fn cl_metal() -> MetalBuild {
    MetalBuild::new()
}

/// Initialize a HIP compiler (`hipcc`) for GPU targets.
///
/// Uses `hipcc` to compile `.hip.cpp` files. Supports both AMD (default)
/// and NVIDIA backends. When `HIP_PLATFORM=nvidia`, hipcc wraps nvcc and
/// AMD-specific `--offload-arch` flags are skipped.
///
/// Target architectures can be controlled via the `UCC_HIP_TARGETS` environment
/// variable (comma-separated, e.g. `gfx1030,gfx1100` for AMD, or `sm_80,sm_89`
/// for NVIDIA). Set to `none` to skip architecture flags entirely. Defaults to
/// `gfx1030` (RDNA2) and `gfx1100` (RDNA3) on AMD, or no arch flags on NVIDIA.
///
/// # Example
///
/// ```ignore
/// let mut cl = ucc::cl_hip();
/// cl.debug(false).opt_level(3);
/// cl.file("csrc/kernel.hip.cpp");
/// cl.compile("myhip");
/// ```
pub fn cl_hip() -> Build {
    println!("cargo:rerun-if-env-changed=UCC_HIP_TARGETS");
    println!("cargo:rerun-if-env-changed=HIP_PLATFORM");

    let is_nvidia = env::var("HIP_PLATFORM")
        .map(|v| v == "nvidia")
        .unwrap_or(false);

    let targets = match env::var("UCC_HIP_TARGETS") {
        // Explicit "none" means skip arch flags entirely.
        Ok(v) if v.trim().eq_ignore_ascii_case("none") => vec![],
        Ok(v) if !v.is_empty() => v.split(',').map(|s| s.trim().to_string()).collect::<Vec<_>>(),
        _ => {
            if is_nvidia {
                // On NVIDIA backend, let nvcc use default compute capability.
                vec![]
            } else {
                // Default AMD targets: RDNA2 + RDNA3.
                vec!["gfx1030".to_string(), "gfx1100".to_string()]
            }
        }
    };

    let mut builder = Build::new();
    if is_nvidia {
        // When HIP targets NVIDIA, hipcc wraps nvcc. Use .cuda(true) so the
        // cc crate wraps host-compiler flags (like -ffunction-sections) with
        // -Xcompiler, which nvcc requires.
        builder
            .cuda(true)
            .compiler("hipcc")
            .flag("-Xcompiler")
            .flag("-Wall")
            .flag("-std=c++14");
    } else {
        // On AMD, hipcc is a clang wrapper — GCC-like flags work directly.
        builder
            .cpp(true)
            .compiler("hipcc")
            .flag("-Wall")
            .flag("-std=c++14");
    }

    for target in &targets {
        if is_nvidia {
            // NVIDIA targets use -gencode syntax via hipcc.
            builder
                .flag("-gencode")
                .flag(&format!(
                    "arch=compute_{t},code=sm_{t}",
                    t = target.trim_start_matches("sm_")
                ));
        } else {
            builder.flag(&format!("--offload-arch={}", target));
        }
    }

    builder.out_dir(
        env::var_os("OUT_DIR")
            .map(|v| {
                let mut v = PathBuf::from(v);
                v.push("ucc_hip");
                v
            })
            .unwrap(),
    );
    add_definitions(&mut builder);
    builder
}
