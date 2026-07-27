// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

const BSR_MODULE_PREFIX: &str = "buf.build/openengine/openengine:";

pub(crate) struct SchemaSource {
    pub(crate) proto_root: PathBuf,
    pub(crate) release: String,
}

pub(crate) fn parse_bsr_module(module: &str) -> Result<String, String> {
    let commit = module.strip_prefix(BSR_MODULE_PREFIX).ok_or_else(|| {
        format!(
            "OPENENGINE_BSR_MODULE must be `{BSR_MODULE_PREFIX}<32-lowercase-hex-commit>`, \
             got `{module}`"
        )
    })?;
    if !is_lower_hex(commit, 32) {
        return Err(format!(
            "OPENENGINE_BSR_MODULE must end in exactly 32 lowercase hexadecimal characters; \
             mutable labels such as `main` are not allowed, got `{module}`"
        ));
    }
    Ok(commit.to_string())
}

pub(crate) fn resolve_local_source(
    root: PathBuf,
    expected_git_commit: &str,
    explicit_release: Option<&str>,
) -> Result<SchemaSource, String> {
    if !is_lower_hex(expected_git_commit, 40) {
        return Err("the pinned OpenEngine source commit must be exactly 40 lowercase hex".into());
    }
    let proto_root = schema_proto_root(root)?;
    let git = Command::new("git")
        .arg("-C")
        .arg(&proto_root)
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .map_err(|error| format!("failed to inspect OpenEngine Git metadata: {error}"))?;
    if !git.status.success() {
        let release = explicit_release.ok_or_else(|| {
            "OPENENGINE_PROTO_ROOT has no Git metadata; \
             set OPENENGINE_SCHEMA_RELEASE to its exact 32-hex BSR or 40-hex source commit"
                .to_string()
        })?;
        validate_release_identity(release)?;
        return Ok(SchemaSource {
            proto_root,
            release: release.to_string(),
        });
    }

    let git_root = parse_path_output(&git.stdout, "OpenEngine Git root")?;
    let git_root = git_root.canonicalize().map_err(|error| {
        format!(
            "failed to canonicalize OpenEngine Git root {}: {error}",
            git_root.display()
        )
    })?;
    let head = checked_git_stdout(&git_root, &["rev-parse", "--verify", "HEAD"])?;
    if head != expected_git_commit {
        return Err(format!(
            "OPENENGINE_PROTO_ROOT is at Git commit `{head}`, expected `{expected_git_commit}`"
        ));
    }
    if let Some(release) = explicit_release {
        validate_release_identity(release)?;
        if release != head {
            return Err(format!(
                "OPENENGINE_SCHEMA_RELEASE `{release}` does not match OpenEngine Git HEAD `{head}`"
            ));
        }
    }

    let relative_proto = proto_root.strip_prefix(&git_root).map_err(|_| {
        format!(
            "OpenEngine proto root {} is outside Git root {}",
            proto_root.display(),
            git_root.display()
        )
    })?;
    let relative_proto = if relative_proto.as_os_str().is_empty() {
        OsStr::new(".")
    } else {
        relative_proto.as_os_str()
    };
    let status = Command::new("git")
        .arg("-C")
        .arg(&git_root)
        .args(["status", "--porcelain=v1", "--untracked-files=all", "--"])
        .arg(relative_proto)
        .output()
        .map_err(|error| format!("failed to inspect OpenEngine proto cleanliness: {error}"))?;
    if !status.status.success() {
        return Err(format!(
            "failed to inspect OpenEngine proto cleanliness: {}",
            String::from_utf8_lossy(&status.stderr).trim()
        ));
    }
    if !status.stdout.is_empty() {
        return Err(format!(
            "OPENENGINE_PROTO_ROOT must be clean at `{expected_git_commit}`; \
             commit or remove changes under {}",
            proto_root.display()
        ));
    }

    Ok(SchemaSource {
        proto_root,
        release: head,
    })
}

pub(crate) fn schema_proto_root(root: PathBuf) -> Result<PathBuf, String> {
    let nested = root.join("proto");
    let proto_root = if nested.join("openengine/v1/openengine.proto").is_file() {
        nested
    } else {
        root
    };
    let entrypoint = proto_root.join("openengine/v1/openengine.proto");
    if !entrypoint.is_file() {
        return Err(format!(
            "OpenEngine schema entrypoint not found at {}; \
             OPENENGINE_PROTO_ROOT must name the repository or its `proto` directory",
            entrypoint.display()
        ));
    }
    proto_root.canonicalize().map_err(|error| {
        format!(
            "failed to canonicalize OpenEngine proto root {}: {error}",
            proto_root.display()
        )
    })
}

fn validate_release_identity(release: &str) -> Result<(), String> {
    if is_lower_hex(release, 32) || is_lower_hex(release, 40) {
        Ok(())
    } else {
        Err(format!(
            "OPENENGINE_SCHEMA_RELEASE must be an exact 32-hex BSR or 40-hex source commit, \
             got `{release}`"
        ))
    }
}

fn is_lower_hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn parse_path_output(stdout: &[u8], name: &str) -> Result<PathBuf, String> {
    let value = std::str::from_utf8(stdout)
        .map_err(|error| format!("{name} is not valid UTF-8: {error}"))?
        .trim();
    if value.is_empty() {
        return Err(format!("{name} is empty"));
    }
    Ok(Path::new(value).to_path_buf())
}

fn checked_git_stdout(root: &Path, args: &[&str]) -> Result<String, String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .map_err(|error| format!("failed to run `git {}`: {error}", args.join(" ")))?;
    if !output.status.success() {
        return Err(format!(
            "`git {}` failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8(output.stdout)
        .map(|value| value.trim().to_string())
        .map_err(|error| format!("`git {}` returned invalid UTF-8: {error}", args.join(" ")))
}
