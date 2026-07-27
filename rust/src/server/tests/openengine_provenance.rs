// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#[path = "../build_support/openengine_provenance.rs"]
mod provenance;

use std::fs;
use std::path::Path;
use std::process::Command;

use provenance::{parse_bsr_module, resolve_local_source};

const SOURCE_COMMIT: &str = "57cd5033554cd22ab9645ae6c17f34d7fa9f5bb0";

#[test]
fn bsr_module_requires_exact_immutable_commit() {
    let commit = "0123456789abcdef0123456789abcdef";
    assert_eq!(
        parse_bsr_module(&format!("buf.build/openengine/openengine:{commit}")).unwrap(),
        commit
    );
    for invalid in [
        "buf.build/openengine/openengine:main",
        "buf.build/openengine/openengine:0123456789abcdef0123456789abcde",
        "buf.build/openengine/openengine:0123456789abcdef0123456789abcdef0",
        "buf.build/openengine/openengine:0123456789ABCDEF0123456789ABCDEF",
        "buf.build/other/openengine:0123456789abcdef0123456789abcdef",
    ] {
        assert!(parse_bsr_module(invalid).is_err(), "{invalid} was accepted");
    }
}

#[test]
fn metadata_free_source_requires_explicit_exact_identity() {
    let root = tempfile::tempdir().unwrap();
    write_schema(root.path());

    let error = resolve_local_source(root.path().to_path_buf(), SOURCE_COMMIT, None)
        .err()
        .expect("metadata-free source must require an identity");
    assert!(error.contains("OPENENGINE_SCHEMA_RELEASE"));

    let source = resolve_local_source(
        root.path().to_path_buf(),
        SOURCE_COMMIT,
        Some(SOURCE_COMMIT),
    )
    .expect("exact source identity");
    assert_eq!(source.release, SOURCE_COMMIT);
    assert!(resolve_local_source(root.path().to_path_buf(), SOURCE_COMMIT, Some("main")).is_err());
}

#[test]
fn git_source_requires_pinned_head_and_clean_proto() {
    let root = tempfile::tempdir().unwrap();
    write_schema(root.path());
    git(root.path(), &["init", "--quiet"]);
    git(root.path(), &["add", "proto"]);
    git(
        root.path(),
        &[
            "-c",
            "user.name=OpenEngine Test",
            "-c",
            "user.email=openengine@example.com",
            "commit",
            "--quiet",
            "-m",
            "test schema",
        ],
    );
    let head = git(root.path(), &["rev-parse", "HEAD"]);

    let source = resolve_local_source(root.path().to_path_buf(), &head, None)
        .expect("clean pinned Git source");
    assert_eq!(source.release, head);

    let wrong_head = "0000000000000000000000000000000000000000";
    let error = resolve_local_source(root.path().to_path_buf(), wrong_head, None)
        .err()
        .expect("wrong Git HEAD must fail");
    assert!(error.contains("expected"));

    fs::write(
        root.path().join("proto/openengine/v1/openengine.proto"),
        "syntax = \"proto3\";\n// dirty\n",
    )
    .unwrap();
    let error = resolve_local_source(root.path().to_path_buf(), &head, None)
        .err()
        .expect("dirty proto must fail");
    assert!(error.contains("must be clean"));
}

fn write_schema(root: &Path) {
    let schema_dir = root.join("proto/openengine/v1");
    fs::create_dir_all(&schema_dir).unwrap();
    fs::write(
        schema_dir.join("openengine.proto"),
        "syntax = \"proto3\";\n",
    )
    .unwrap();
}

fn git(root: &Path, args: &[&str]) -> String {
    let output = Command::new("git").arg("-C").arg(root).args(args).output().unwrap();
    assert!(
        output.status.success(),
        "git {} failed: {}",
        args.join(" "),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap().trim().to_string()
}
