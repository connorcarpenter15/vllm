// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::path::{Path, PathBuf};
use std::process::Command;

const OPENENGINE_PROTO_ROOT_ENV: &str = "OPENENGINE_PROTO_ROOT";
const OPENENGINE_BSR_MODULE_ENV: &str = "OPENENGINE_BSR_MODULE";
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
    if std::env::var_os("CARGO_FEATURE_OPENENGINE").is_some() {
        let (openengine_proto_root, schema_release) = openengine_proto_root()?;
        println!("cargo:rustc-env=OPENENGINE_SCHEMA_RELEASE={schema_release}");
        let entrypoint = openengine_proto_root.join("openengine/v1/openengine.proto");
        if !entrypoint.is_file() {
            return Err(format!(
                "OpenEngine schema entrypoint not found at {}; \
                 {OPENENGINE_PROTO_ROOT_ENV} must name the schema's `proto` directory",
                entrypoint.display()
            )
            .into());
        }
        println!("cargo:rerun-if-changed={}", openengine_proto_root.display());
        tonic_prost_build::configure()
            .build_server(true)
            .build_client(true)
            .protoc_arg("--experimental_allow_proto3_optional")
            .compile_protos(&[entrypoint], &[openengine_proto_root])?;
    }

    Ok(())
}

fn openengine_proto_root() -> Result<(PathBuf, String), Box<dyn std::error::Error>> {
    if let Some(root) = std::env::var_os(OPENENGINE_PROTO_ROOT_ENV) {
        return Ok((
            normalize_proto_root(PathBuf::from(root)),
            OPENENGINE_SOURCE_COMMIT.to_string(),
        ));
    }

    let module = std::env::var(OPENENGINE_BSR_MODULE_ENV).map_err(|_| {
        format!(
            "the `openengine` feature requires either {OPENENGINE_PROTO_ROOT_ENV} \
             for a local checkout or {OPENENGINE_BSR_MODULE_ENV} for an immutable \
             `buf.build/openengine/openengine:<commit>` input"
        )
    })?;
    if !module.starts_with("buf.build/openengine/openengine:") {
        return Err(format!(
            "{OPENENGINE_BSR_MODULE_ENV} must be an immutable \
             `buf.build/openengine/openengine:<commit>` input, got `{module}`"
        )
        .into());
    }
    let schema_release = module
        .rsplit_once(':')
        .map(|(_, commit)| commit)
        .filter(|commit| !commit.is_empty())
        .ok_or_else(|| format!("{OPENENGINE_BSR_MODULE_ENV} is missing its immutable commit"))?
        .to_string();

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
    Ok((normalize_proto_root(export_dir), schema_release))
}

fn normalize_proto_root(root: PathBuf) -> PathBuf {
    let nested = root.join("proto");
    if Path::new(&nested).join("openengine/v1/openengine.proto").is_file() {
        nested
    } else {
        root
    }
}
