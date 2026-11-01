//! Generates the libibverbs FFI bindings from the system rdma-core headers.
//!
//! The headers come from the `libibverbs-dev` (Debian, Ubuntu) or `rdma-core`
//! (Arch, Fedora) package. pkg-config supplies the include and link paths when
//! it knows `libibverbs`. Otherwise the compiler's default search path applies.
//!
//! bindgen 0.72.1 or newer is required. With libclang 22 or newer, older
//! bindgen emits a one-byte opaque struct for every struct that another struct
//! names before its definition, e.g. `ibv_mr` and `ibv_qp`. The layout
//! assertions then fail to compile.

use std::env;
use std::path::PathBuf;

const HEADER: &str = "infiniband/verbs.h";

fn main() {
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_SYSROOT_DIR");

    let include_paths = match pkg_config::Config::new().probe("libibverbs") {
        // pkg-config also prints the link flags for libibverbs.
        Ok(lib) => lib.include_paths,
        Err(err) => {
            println!(
                "cargo:warning=pkg-config did not find libibverbs ({err}). \
                 Linking -libverbs from the default search path."
            );
            println!("cargo:rustc-link-lib=ibverbs");
            Vec::new()
        }
    };

    let mut builder = bindgen::Builder::default()
        .header_contents("nexar_rdma_ibverbs.h", &format!("#include <{HEADER}>\n"))
        .rust_target(bindgen::RustTarget::stable(89, 0).expect("Rust 1.89 is a stable target"))
        .rust_edition(bindgen::RustEdition::Edition2024)
        .clang_args(
            include_paths
                .iter()
                .map(|path| format!("-I{}", path.display())),
        )
        .allowlist_function("ibv_.*")
        .allowlist_function("_ibv_.*")
        .allowlist_type("ibv_.*")
        .allowlist_var("IBV_LINK_LAYER_.*")
        .bitfield_enum("ibv_access_flags")
        .bitfield_enum("ibv_qp_attr_mask")
        .bitfield_enum("ibv_wc_flags")
        .bitfield_enum("ibv_send_flags")
        .bitfield_enum("ibv_port_cap_flags")
        .constified_enum_module("ibv_qp_type")
        .constified_enum_module("ibv_qp_state")
        .constified_enum_module("ibv_port_state")
        .constified_enum_module("ibv_wc_opcode")
        .constified_enum_module("ibv_wr_opcode")
        .constified_enum_module("ibv_wc_status")
        .derive_default(true)
        .derive_debug(true)
        .prepend_enum_name(false)
        .size_t_is_usize(true)
        .generate_comments(false)
        .layout_tests(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));

    // Header search paths from the environment, e.g. a non-default rdma-core prefix.
    println!("cargo:rerun-if-env-changed=NEXAR_RDMA_INCLUDE_DIR");
    if let Ok(dir) = env::var("NEXAR_RDMA_INCLUDE_DIR") {
        builder = builder.clang_arg(format!("-I{dir}"));
    }

    let bindings = builder.generate().unwrap_or_else(|err| {
        panic!(
            "bindgen could not generate bindings for <{HEADER}>: {err}. \
             Install the libibverbs headers (`libibverbs-dev` on Debian/Ubuntu, \
             `rdma-core` on Arch/Fedora) and libclang, or set NEXAR_RDMA_INCLUDE_DIR \
             to the directory that holds `infiniband/verbs.h`."
        )
    });

    let out = PathBuf::from(env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
    let path = out.join("ibverbs.rs");
    bindings
        .write_to_file(&path)
        .unwrap_or_else(|err| panic!("cannot write {}: {err}", path.display()));
}
