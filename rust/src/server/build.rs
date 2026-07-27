// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#[path = "build_support/openengine_provenance.rs"]
mod openengine_provenance;

use std::path::PathBuf;
use std::process::Command;

use openengine_provenance::{
    SchemaSource, parse_bsr_module, resolve_local_source, schema_proto_root,
};

const OPENENGINE_PROTO_ROOT_ENV: &str = "OPENENGINE_PROTO_ROOT";
const OPENENGINE_BSR_MODULE_ENV: &str = "OPENENGINE_BSR_MODULE";
const OPENENGINE_SCHEMA_RELEASE_ENV: &str = "OPENENGINE_SCHEMA_RELEASE";
const OPENENGINE_SOURCE_COMMIT: &str = "57cd5033554cd22ab9645ae6c17f34d7fa9f5bb0";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let proto_dir = format!("{manifest_dir}/../../proto");

    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .protoc_arg("--experimental_allow_proto3_optional") // be compatible with old compilers
        .compile_protos(&[format!("{proto_dir}/vllm_grpc.proto")], &[proto_dir])?;

    println!("cargo:rerun-if-env-changed={OPENENGINE_PROTO_ROOT_ENV}");
    println!("cargo:rerun-if-env-changed={OPENENGINE_BSR_MODULE_ENV}");
    println!("cargo:rerun-if-env-changed={OPENENGINE_SCHEMA_RELEASE_ENV}");
    if std::env::var_os("CARGO_FEATURE_OPENENGINE").is_some() {
        let source = openengine_schema_source()?;
        println!(
            "cargo:rustc-env=OPENENGINE_SCHEMA_RELEASE={}",
            source.release
        );
        let entrypoint = source.proto_root.join("openengine/v1/openengine.proto");
        println!("cargo:rerun-if-changed={}", source.proto_root.display());
        tonic_prost_build::configure()
            .build_server(true)
            .build_client(true)
            .protoc_arg("--experimental_allow_proto3_optional")
            .compile_protos(&[entrypoint], &[source.proto_root])?;
    }

    Ok(())
}

fn openengine_schema_source() -> Result<SchemaSource, Box<dyn std::error::Error>> {
    let local_root = std::env::var_os(OPENENGINE_PROTO_ROOT_ENV);
    let bsr_module = std::env::var(OPENENGINE_BSR_MODULE_ENV).ok();
    if local_root.is_some() && bsr_module.is_some() {
        return Err(format!(
            "set only one of {OPENENGINE_PROTO_ROOT_ENV} or {OPENENGINE_BSR_MODULE_ENV}"
        )
        .into());
    }
    if let Some(root) = local_root {
        let explicit_release = std::env::var(OPENENGINE_SCHEMA_RELEASE_ENV).ok();
        return resolve_local_source(
            PathBuf::from(root),
            OPENENGINE_SOURCE_COMMIT,
            explicit_release.as_deref(),
        )
        .map_err(Into::into);
    }

    let module = bsr_module.ok_or_else(|| {
        format!(
            "the `openengine` feature requires either {OPENENGINE_PROTO_ROOT_ENV} \
             for a local checkout or {OPENENGINE_BSR_MODULE_ENV} for an immutable \
             `buf.build/openengine/openengine:<commit>` input"
        )
    })?;
    let schema_release = parse_bsr_module(&module)?;

    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR"));
    let export_dir = out_dir.join("openengine-schema");
    let status = Command::new("buf")
        .args(["export", &module, "--output"])
        .arg(&export_dir)
        .status()
        .map_err(|error| {
            format!(
                "failed to run Buf for {OPENENGINE_BSR_MODULE_ENV}={module}: {error}; \
                 install Buf or set {OPENENGINE_PROTO_ROOT_ENV}"
            )
        })?;
    if !status.success() {
        return Err(format!("`buf export {module}` failed with {status}").into());
    }
    Ok(SchemaSource {
        proto_root: schema_proto_root(export_dir)?,
        release: schema_release,
    })
}
