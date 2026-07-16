use std::{
    env,
    fs::File,
    io::{self, BufRead, BufReader, Read},
    process::ExitCode,
};

use nana_tracking_protocol::{CanonicalCodec, NanaTrackingDescriptor, NanaTrackingResult};
use ntp_conformance::{CertificationReport, ConformanceOptions, ConformanceValidator};
use serde::Deserialize;

#[derive(Clone, Copy)]
enum InputFormat {
    Binary,
    JsonLines,
}

#[derive(Clone, Copy)]
enum OutputFormat {
    Text,
    Json,
}

#[derive(Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
enum JsonEvent {
    Descriptor(NanaTrackingDescriptor),
    Result(Box<NanaTrackingResult>),
}

enum Event {
    Descriptor(NanaTrackingDescriptor),
    Result(Box<NanaTrackingResult>),
}

const HEADER_LENGTH: usize = 12;
const MAX_MESSAGE_BYTES: usize = 64 * 1024 * 1024;

#[derive(Default)]
struct CertificationStream {
    descriptor: Option<NanaTrackingDescriptor>,
    validator: Option<ConformanceValidator>,
}

fn main() -> ExitCode {
    match run() {
        Ok(passed) => {
            if passed {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(error) => {
            eprintln!("ntp-conformance: {error}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<bool, String> {
    let (path, input_format, output_format) = parse_args()?;
    let report = if path == "-" {
        let stdin = io::stdin();
        certify(stdin.lock(), input_format)?
    } else {
        let file =
            File::open(&path).map_err(|error| format!("could not read {path:?}: {error}"))?;
        certify(file, input_format)?
    };
    match output_format {
        OutputFormat::Text => print!("{report}"),
        OutputFormat::Json => println!(
            "{}",
            serde_json::to_string_pretty(&report)
                .map_err(|error| format!("could not serialize report: {error}"))?
        ),
    }
    Ok(report.passed)
}

impl CertificationStream {
    fn accept(&mut self, event: Event) -> Result<(), String> {
        match event {
            Event::Descriptor(value) => {
                match &self.descriptor {
                    Some(previous) if previous != &value => {
                        return Err(
                            "descriptor changed inside one certification stream; start a new run"
                                .into(),
                        );
                    }
                    _ => {}
                }
                self.descriptor = Some(value.clone());
                if self.validator.is_none() {
                    self.validator = Some(ConformanceValidator::new(
                        value,
                        ConformanceOptions::default(),
                    ));
                }
            }
            Event::Result(value) => self
                .validator
                .as_mut()
                .ok_or_else(|| "result appeared before a descriptor".to_string())?
                .validate_frame(&value),
        }
        Ok(())
    }

    fn finish(self) -> Result<CertificationReport, String> {
        Ok(self
            .validator
            .ok_or_else(|| "stream did not contain a descriptor".to_string())?
            .finish())
    }
}

fn parse_args() -> Result<(String, InputFormat, OutputFormat), String> {
    let mut path = "-".to_string();
    let mut input = InputFormat::Binary;
    let mut output = OutputFormat::Text;
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--input-format" => {
                input = match arguments.next().as_deref() {
                    Some("binary") => InputFormat::Binary,
                    Some("jsonl") => InputFormat::JsonLines,
                    Some(other) => return Err(format!("unknown input format {other:?}")),
                    None => return Err("--input-format requires binary or jsonl".into()),
                };
            }
            "--output" => {
                output = match arguments.next().as_deref() {
                    Some("text") => OutputFormat::Text,
                    Some("json") => OutputFormat::Json,
                    Some(other) => return Err(format!("unknown output format {other:?}")),
                    None => return Err("--output requires text or json".into()),
                };
            }
            "-h" | "--help" => {
                println!(
                    "Usage: ntp-conformance [--input-format binary|jsonl] \
                     [--output text|json] [PATH|-]"
                );
                std::process::exit(0);
            }
            value if value.starts_with('-') && value != "-" => {
                return Err(format!("unknown option {value:?}"));
            }
            value => path = value.to_string(),
        }
    }
    Ok((path, input, output))
}

fn certify(reader: impl Read, input_format: InputFormat) -> Result<CertificationReport, String> {
    let mut stream = CertificationStream::default();
    match input_format {
        InputFormat::Binary => consume_binary(reader, &mut stream)?,
        InputFormat::JsonLines => consume_json_lines(reader, &mut stream)?,
    }
    stream.finish()
}

fn consume_binary(mut reader: impl Read, stream: &mut CertificationStream) -> Result<(), String> {
    let mut offset = 0_u64;
    loop {
        let mut header = [0_u8; HEADER_LENGTH];
        let first = reader
            .read(&mut header[..1])
            .map_err(|error| format!("could not read NTP header at byte {offset}: {error}"))?;
        if first == 0 {
            break;
        }
        reader
            .read_exact(&mut header[1..])
            .map_err(|error| format!("truncated NTP header at byte {offset}: {error}"))?;
        let payload_length = u32::from_le_bytes(
            header[8..12]
                .try_into()
                .expect("fixed header length was checked"),
        ) as usize;
        let frame_length = HEADER_LENGTH
            .checked_add(payload_length)
            .ok_or_else(|| "NTP frame length overflow".to_string())?;
        if frame_length > MAX_MESSAGE_BYTES {
            return Err(format!(
                "NTP message at byte {offset} is {frame_length} bytes; limit is {MAX_MESSAGE_BYTES}"
            ));
        }
        let mut frame = vec![0_u8; frame_length];
        frame[..HEADER_LENGTH].copy_from_slice(&header);
        reader
            .read_exact(&mut frame[HEADER_LENGTH..])
            .map_err(|error| format!("truncated NTP payload at byte {offset}: {error}"))?;
        let event = match header[4] {
            1 => Event::Descriptor(
                CanonicalCodec::decode(&frame)
                    .map_err(|error| format!("invalid descriptor at byte {offset}: {error}"))?,
            ),
            2 => Event::Result(Box::new(
                CanonicalCodec::decode(&frame)
                    .map_err(|error| format!("invalid result at byte {offset}: {error}"))?,
            )),
            kind => return Err(format!("unknown NTP message kind {kind} at byte {offset}")),
        };
        stream.accept(event)?;
        offset += u64::try_from(frame_length).expect("message limit fits u64");
    }
    Ok(())
}

fn consume_json_lines(reader: impl Read, stream: &mut CertificationStream) -> Result<(), String> {
    let mut reader = BufReader::new(reader);
    let mut line = String::new();
    let mut line_number = 0_u64;
    loop {
        line.clear();
        let length = Read::by_ref(&mut reader)
            .take(u64::try_from(MAX_MESSAGE_BYTES).expect("message limit fits u64") + 1)
            .read_line(&mut line)
            .map_err(|error| format!("could not read JSONL line {}: {error}", line_number + 1))?;
        if length == 0 {
            break;
        }
        line_number += 1;
        if length > MAX_MESSAGE_BYTES {
            return Err(format!(
                "JSONL event on line {line_number} is {length} bytes; limit is {MAX_MESSAGE_BYTES}"
            ));
        }
        if line.trim().is_empty() {
            continue;
        }
        let event: JsonEvent = serde_json::from_str(&line)
            .map_err(|error| format!("invalid JSON event on line {line_number}: {error}"))?;
        stream.accept(match event {
            JsonEvent::Descriptor(value) => Event::Descriptor(value),
            JsonEvent::Result(value) => Event::Result(value),
        })?;
    }
    Ok(())
}
