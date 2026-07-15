mod support;

use std::{
    io::Write,
    process::{Command, Stdio},
};

use nana_tracking_protocol::{CanonicalCodec, StructureFeatures};
use ntp_conformance::CertificationReport;
use support::{descriptor, frame};

#[test]
fn cli_certifies_canonical_binary_streams() {
    let descriptor = descriptor(36, StructureFeatures::BASIC_REQUIRED, &[37]);
    let result = frame(&descriptor, 1, 0, 1, 1_000_000_000);
    let mut input = CanonicalCodec::encode(&descriptor).unwrap();
    input.extend(CanonicalCodec::encode(&result).unwrap());
    let output = run_cli(&["--output", "json"], &input);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: CertificationReport = serde_json::from_slice(&output.stdout).unwrap();
    assert!(report.passed);
}

#[test]
fn cli_certifies_diagnostic_json_lines_without_treating_them_as_wire_data() {
    let descriptor = descriptor(41, StructureFeatures::SPATIAL_REQUIRED, &[57]);
    let result = frame(&descriptor, 1, 0, 1, 1_000_000_000);
    let descriptor_event = serde_json::json!({"kind": "descriptor", "value": descriptor});
    let result_event = serde_json::json!({"kind": "result", "value": result});
    let input = format!("{descriptor_event}\n{result_event}\n");
    let output = run_cli(
        &["--input-format", "jsonl", "--output", "json"],
        input.as_bytes(),
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: CertificationReport = serde_json::from_slice(&output.stdout).unwrap();
    assert!(report.passed);
}

#[test]
fn cli_returns_input_error_for_result_before_descriptor() {
    let descriptor = descriptor(36, StructureFeatures::BASIC_REQUIRED, &[]);
    let result = frame(&descriptor, 1, 0, 1, 1_000_000_000);
    let event = serde_json::json!({"kind": "result", "value": result});
    let output = run_cli(
        &["--input-format", "jsonl", "--output", "json"],
        format!("{event}\n").as_bytes(),
    );
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn cli_processes_long_binary_streams_incrementally() {
    let descriptor = descriptor(36, StructureFeatures::BASIC_REQUIRED, &[]);
    let mut input = CanonicalCodec::encode(&descriptor).unwrap();
    for sequence in 1..=256 {
        let result = frame(
            &descriptor,
            1,
            0,
            sequence,
            1_000_000_000 + sequence * 1_000_000,
        );
        input.extend(CanonicalCodec::encode(&result).unwrap());
    }
    let output = run_cli(&["--output", "json"], &input);
    assert!(output.status.success());
    let report: CertificationReport = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report.frames_seen, 256);
}

#[test]
fn cli_rejects_oversized_length_before_allocating_the_payload() {
    let mut header = Vec::from(*b"NTP1");
    header.extend([1, 1, 0, 0]);
    header.extend(u32::MAX.to_le_bytes());
    let output = run_cli(&["--output", "json"], &header);
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("limit"));
}

fn run_cli(arguments: &[&str], input: &[u8]) -> std::process::Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_ntp-conformance"))
        .args(arguments)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(input).unwrap();
    child.wait_with_output().unwrap()
}
