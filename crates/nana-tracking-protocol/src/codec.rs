use alloc::{vec, vec::Vec};
use core::fmt;

use crate::{
    capability::{NanaTrackingDescriptor, StructureFeatures, TrackingFeatures, TrackingProfile},
    revision::{ContractRevisions, ProtocolVersion, Revision},
    signal::{STABLE_SIGNAL_COUNT, SignalBitSet, SignalId},
    types::{
        CoordinateSpace, Direction3, EyeGeometry, FaceLandmark, LengthBasis, NanaGeometryResult,
        NanaRigResult, NanaSkeletonResult, NanaTrackingQuality, NanaTrackingResult, Pose,
        Position3, Quaternion, RegionQuality, SessionId, SideMap, SignalSample, SignalState,
        Tracked, Vec3,
    },
    validate::{ContractError, Validate},
};

const MAGIC: [u8; 4] = *b"NTP1";
const HEADER_LEN: usize = 12;
const DESCRIPTOR_KIND: u8 = 1;
const RESULT_KIND: u8 = 2;
const BLOCK_VERSION: u16 = 1;
const WIRE_MAJOR: u8 = 1;
const WIRE_MINOR: u8 = 0;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CodecError {
    UnexpectedEnd,
    InvalidMagic,
    WrongMessageKind,
    IncompatibleVersion,
    InvalidLength,
    MissingField(u16),
    DuplicateField(u16),
    InvalidEnum(&'static str, u8),
    InvalidSignalId,
    TooManyItems(&'static str),
    Contract(ContractError),
}

impl fmt::Display for CodecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEnd => formatter.write_str("unexpected end of NTP payload"),
            Self::InvalidMagic => formatter.write_str("invalid NTP magic"),
            Self::WrongMessageKind => formatter.write_str("wrong NTP message kind"),
            Self::IncompatibleVersion => formatter.write_str("incompatible NTP wire version"),
            Self::InvalidLength => formatter.write_str("invalid NTP field length"),
            Self::MissingField(tag) => write!(formatter, "missing required NTP field {tag}"),
            Self::DuplicateField(tag) => write!(formatter, "duplicate NTP field {tag}"),
            Self::InvalidEnum(name, value) => write!(formatter, "invalid {name} value {value}"),
            Self::InvalidSignalId => formatter.write_str("Signal ID 0 is reserved"),
            Self::TooManyItems(name) => write!(formatter, "too many {name}"),
            Self::Contract(error) => write!(formatter, "decoded contract is invalid: {error}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CodecError {}

impl From<ContractError> for CodecError {
    fn from(value: ContractError) -> Self {
        Self::Contract(value)
    }
}

pub trait WireEncode {
    /// Encode a validated value using canonical NTP binary framing.
    ///
    /// # Errors
    ///
    /// Returns an error when the value violates its contract or exceeds a wire limit.
    fn encode_wire(&self) -> Result<Vec<u8>, CodecError>;
}

pub trait WireDecode: Sized {
    /// Decode and validate a canonical NTP binary message.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed, incompatible, incomplete, or invalid input.
    fn decode_wire(bytes: &[u8]) -> Result<Self, CodecError>;
}

pub struct CanonicalCodec;

impl CanonicalCodec {
    /// Encode a value with its `WireEncode` implementation.
    ///
    /// # Errors
    ///
    /// Propagates validation and encoding failures.
    pub fn encode<T: WireEncode>(value: &T) -> Result<Vec<u8>, CodecError> {
        value.encode_wire()
    }

    /// Decode a value with its `WireDecode` implementation.
    ///
    /// # Errors
    ///
    /// Propagates framing, compatibility, and contract failures.
    pub fn decode<T: WireDecode>(bytes: &[u8]) -> Result<T, CodecError> {
        T::decode_wire(bytes)
    }
}

#[derive(Default)]
struct Writer(Vec<u8>);

impl Writer {
    fn u8(&mut self, value: u8) {
        self.0.push(value);
    }

    fn u16(&mut self, value: u16) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    fn f32(&mut self, value: f32) {
        self.u32(if value == 0.0 { 0 } else { value.to_bits() });
    }

    fn bytes(&mut self, value: &[u8]) {
        self.0.extend_from_slice(value);
    }

    fn tlv(&mut self, tag: u16, encode: impl FnOnce(&mut Self)) -> Result<(), CodecError> {
        let mut body = Self::default();
        encode(&mut body);
        let length = u32::try_from(body.0.len()).map_err(|_| CodecError::InvalidLength)?;
        self.u16(tag);
        self.u32(length);
        self.bytes(&body.0);
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn remaining(self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], CodecError> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or(CodecError::InvalidLength)?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(CodecError::UnexpectedEnd)?;
        self.offset = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, CodecError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, CodecError> {
        Ok(u16::from_le_bytes(
            self.take(2)?.try_into().expect("two bytes"),
        ))
    }

    fn u32(&mut self) -> Result<u32, CodecError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("four bytes"),
        ))
    }

    fn u64(&mut self) -> Result<u64, CodecError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("eight bytes"),
        ))
    }

    fn f32(&mut self) -> Result<f32, CodecError> {
        Ok(f32::from_bits(self.u32()?))
    }

    fn tlv(&mut self) -> Result<Option<(u16, Reader<'a>)>, CodecError> {
        if self.remaining() == 0 {
            return Ok(None);
        }
        let tag = self.u16()?;
        let length = usize::try_from(self.u32()?).map_err(|_| CodecError::InvalidLength)?;
        Ok(Some((tag, Self::new(self.take(length)?))))
    }
}

fn frame(kind: u8, payload: &Writer) -> Result<Vec<u8>, CodecError> {
    let length = u32::try_from(payload.0.len()).map_err(|_| CodecError::InvalidLength)?;
    let mut result = Writer::default();
    result.bytes(&MAGIC);
    result.u8(kind);
    result.u8(WIRE_MAJOR);
    result.u8(WIRE_MINOR);
    result.u8(0);
    result.u32(length);
    result.bytes(&payload.0);
    Ok(result.0)
}

fn payload(bytes: &[u8], expected_kind: u8) -> Result<Reader<'_>, CodecError> {
    if bytes.len() < HEADER_LEN {
        return Err(CodecError::UnexpectedEnd);
    }
    let mut reader = Reader::new(bytes);
    if reader.take(4)? != MAGIC {
        return Err(CodecError::InvalidMagic);
    }
    if reader.u8()? != expected_kind {
        return Err(CodecError::WrongMessageKind);
    }
    let major = reader.u8()?;
    let _minor = reader.u8()?;
    let _reserved = reader.u8()?;
    if major != WIRE_MAJOR {
        return Err(CodecError::IncompatibleVersion);
    }
    let length = usize::try_from(reader.u32()?).map_err(|_| CodecError::InvalidLength)?;
    if length != reader.remaining() {
        return Err(CodecError::InvalidLength);
    }
    Ok(Reader::new(reader.take(length)?))
}

fn encode_revision(writer: &mut Writer, value: Revision) {
    writer.u16(value.major);
    writer.u16(value.minor);
    writer.u16(value.patch);
}

fn decode_revision(reader: &mut Reader<'_>) -> Result<Revision, CodecError> {
    Ok(Revision {
        major: reader.u16()?,
        minor: reader.u16()?,
        patch: reader.u16()?,
    })
}

impl WireEncode for NanaTrackingDescriptor {
    fn encode_wire(&self) -> Result<Vec<u8>, CodecError> {
        self.validate()?;
        let mut payload = Writer::default();
        payload.tlv(1, |writer| {
            writer.u16(self.revisions.protocol.major);
            writer.u16(self.revisions.protocol.minor);
            writer.u32(self.revisions.schema_revision);
            encode_revision(writer, self.revisions.signal_registry);
            encode_revision(writer, self.revisions.normalization);
            encode_revision(writer, self.revisions.calibration);
            encode_revision(writer, self.revisions.features);
        })?;
        payload.tlv(2, |writer| writer.u8(self.guaranteed_profile as u8))?;
        let signal_count = u32::try_from(self.supported_signals.iter().count())
            .map_err(|_| CodecError::TooManyItems("signals"))?;
        payload.tlv(3, |writer| {
            writer.u32(signal_count);
            for id in self.supported_signals.iter() {
                writer.u16(id.get());
            }
        })?;
        payload.tlv(4, |writer| writer.u64(self.supported_structures.bits()))?;
        payload.tlv(5, |writer| writer.u64(self.features.bits()))?;
        frame(DESCRIPTOR_KIND, &payload)
    }
}

impl WireDecode for NanaTrackingDescriptor {
    fn decode_wire(bytes: &[u8]) -> Result<Self, CodecError> {
        let mut payload = payload(bytes, DESCRIPTOR_KIND)?;
        let mut revisions = None;
        let mut profile = None;
        let mut signals = None;
        let mut structures = None;
        let mut features = None;
        while let Some((tag, mut field)) = payload.tlv()? {
            match tag {
                1 => {
                    set_once(
                        &mut revisions,
                        tag,
                        ContractRevisions {
                            protocol: ProtocolVersion {
                                major: field.u16()?,
                                minor: field.u16()?,
                            },
                            schema_revision: field.u32()?,
                            signal_registry: decode_revision(&mut field)?,
                            normalization: decode_revision(&mut field)?,
                            calibration: decode_revision(&mut field)?,
                            features: decode_revision(&mut field)?,
                        },
                    )?;
                }
                2 => set_once(&mut profile, tag, decode_profile(field.u8()?)?)?,
                3 => {
                    let count =
                        usize::try_from(field.u32()?).map_err(|_| CodecError::InvalidLength)?;
                    if count > usize::from(u16::MAX) {
                        return Err(CodecError::TooManyItems("signals"));
                    }
                    if count > field.remaining() / 2 {
                        return Err(CodecError::InvalidLength);
                    }
                    let mut value = SignalBitSet::new();
                    let mut previous = 0;
                    for _ in 0..count {
                        let raw = field.u16()?;
                        if raw == 0 || raw <= previous {
                            return Err(CodecError::InvalidSignalId);
                        }
                        previous = raw;
                        value.insert(SignalId::new(raw).ok_or(CodecError::InvalidSignalId)?);
                    }
                    set_once(&mut signals, tag, value)?;
                }
                4 => set_once(&mut structures, tag, StructureFeatures(field.u64()?))?,
                5 => set_once(&mut features, tag, TrackingFeatures(field.u64()?))?,
                _ => {}
            }
        }
        let value = Self {
            revisions: revisions.ok_or(CodecError::MissingField(1))?,
            guaranteed_profile: profile.ok_or(CodecError::MissingField(2))?,
            supported_signals: signals.ok_or(CodecError::MissingField(3))?,
            supported_structures: structures.ok_or(CodecError::MissingField(4))?,
            features: features.ok_or(CodecError::MissingField(5))?,
        };
        value.validate()?;
        Ok(value)
    }
}

fn set_once<T>(slot: &mut Option<T>, tag: u16, value: T) -> Result<(), CodecError> {
    if slot.replace(value).is_some() {
        Err(CodecError::DuplicateField(tag))
    } else {
        Ok(())
    }
}

fn decode_profile(value: u8) -> Result<TrackingProfile, CodecError> {
    match value {
        0 => Ok(TrackingProfile::Partial),
        1 => Ok(TrackingProfile::Basic),
        2 => Ok(TrackingProfile::Spatial),
        3 => Ok(TrackingProfile::Full),
        other => Err(CodecError::InvalidEnum("tracking profile", other)),
    }
}

fn decode_state(value: u8) -> Result<SignalState, CodecError> {
    match value {
        0 => Ok(SignalState::Observed),
        1 => Ok(SignalState::Fused),
        2 => Ok(SignalState::Predicted),
        3 => Ok(SignalState::Occluded),
        4 => Ok(SignalState::OutOfFrame),
        5 => Ok(SignalState::TrackingLost),
        6 => Ok(SignalState::Unsupported),
        other => Err(CodecError::InvalidEnum("signal state", other)),
    }
}

fn decode_space(value: u8) -> Result<CoordinateSpace, CodecError> {
    match value {
        0 => Ok(CoordinateSpace::Camera),
        1 => Ok(CoordinateSpace::TorsoLocal),
        2 => Ok(CoordinateSpace::HeadLocal),
        other => Err(CodecError::InvalidEnum("coordinate space", other)),
    }
}

fn decode_basis(value: u8) -> Result<LengthBasis, CodecError> {
    match value {
        0 => Ok(LengthBasis::Metric),
        1 => Ok(LengthBasis::HeadRelative),
        2 => Ok(LengthBasis::TorsoRelative),
        other => Err(CodecError::InvalidEnum("length basis", other)),
    }
}

fn encode_vec3(writer: &mut Writer, value: Vec3) {
    writer.f32(value.x);
    writer.f32(value.y);
    writer.f32(value.z);
}

fn decode_vec3(reader: &mut Reader<'_>) -> Result<Vec3, CodecError> {
    Ok(Vec3 {
        x: reader.f32()?,
        y: reader.f32()?,
        z: reader.f32()?,
    })
}

fn encode_position(writer: &mut Writer, value: &Position3) {
    writer.u8(value.space as u8);
    writer.u8(value.length_basis as u8);
    encode_vec3(writer, value.value);
}

fn decode_position(reader: &mut Reader<'_>) -> Result<Position3, CodecError> {
    Ok(Position3 {
        space: decode_space(reader.u8()?)?,
        length_basis: decode_basis(reader.u8()?)?,
        value: decode_vec3(reader)?,
    })
}

fn encode_direction(writer: &mut Writer, value: &Direction3) {
    writer.u8(value.space as u8);
    encode_vec3(writer, value.value);
}

fn decode_direction(reader: &mut Reader<'_>) -> Result<Direction3, CodecError> {
    Ok(Direction3 {
        space: decode_space(reader.u8()?)?,
        value: decode_vec3(reader)?,
    })
}

fn encode_pose(writer: &mut Writer, value: &Pose) {
    writer.u8(value.parent_space as u8);
    writer.u8(value.length_basis as u8);
    encode_vec3(writer, value.position);
    let orientation = value.orientation_xyzw.canonicalized();
    writer.f32(orientation.x);
    writer.f32(orientation.y);
    writer.f32(orientation.z);
    writer.f32(orientation.w);
}

fn decode_pose(reader: &mut Reader<'_>) -> Result<Pose, CodecError> {
    Ok(Pose {
        parent_space: decode_space(reader.u8()?)?,
        length_basis: decode_basis(reader.u8()?)?,
        position: decode_vec3(reader)?,
        orientation_xyzw: Quaternion {
            x: reader.f32()?,
            y: reader.f32()?,
            z: reader.f32()?,
            w: reader.f32()?,
        },
    })
}

fn encode_tracked<T>(
    writer: &mut Writer,
    tracked: &Tracked<T>,
    encode: impl FnOnce(&mut Writer, &T),
) {
    writer.u8(tracked.state as u8);
    writer.u8(u8::from(tracked.value.is_some()));
    writer.f32(tracked.confidence);
    writer.u64(tracked.sample_capture_timestamp_ns);
    writer.u64(tracked.prediction_horizon_ns);
    if let Some(value) = &tracked.value {
        encode(writer, value);
    }
}

fn decode_tracked<T>(
    reader: &mut Reader<'_>,
    decode: impl FnOnce(&mut Reader<'_>) -> Result<T, CodecError>,
) -> Result<Tracked<T>, CodecError> {
    let state = decode_state(reader.u8()?)?;
    let has_value = match reader.u8()? {
        0 => false,
        1 => true,
        other => return Err(CodecError::InvalidEnum("value presence", other)),
    };
    let confidence = reader.f32()?;
    let sample_capture_timestamp_ns = reader.u64()?;
    let prediction_horizon_ns = reader.u64()?;
    let value = if has_value {
        Some(decode(reader)?)
    } else {
        None
    };
    Ok(Tracked {
        value,
        confidence,
        state,
        sample_capture_timestamp_ns,
        prediction_horizon_ns,
    })
}

fn decode_f32(reader: &mut Reader<'_>) -> Result<f32, CodecError> {
    reader.f32()
}

fn encode_sample(writer: &mut Writer, sample: &SignalSample) {
    writer.u8(sample.state as u8);
    writer.u8(u8::from(sample.value.is_some()));
    writer.f32(sample.confidence);
    writer.u64(sample.sample_capture_timestamp_ns);
    writer.u64(sample.prediction_horizon_ns);
    if let Some(value) = sample.value {
        writer.f32(value);
    }
}

fn decode_sample(reader: &mut Reader<'_>) -> Result<SignalSample, CodecError> {
    let state = decode_state(reader.u8()?)?;
    let has_value = match reader.u8()? {
        0 => false,
        1 => true,
        other => return Err(CodecError::InvalidEnum("value presence", other)),
    };
    let confidence = reader.f32()?;
    let sample_capture_timestamp_ns = reader.u64()?;
    let prediction_horizon_ns = reader.u64()?;
    let value = if has_value { Some(reader.f32()?) } else { None };
    Ok(SignalSample {
        value,
        confidence,
        state,
        sample_capture_timestamp_ns,
        prediction_horizon_ns,
    })
}

fn encode_rig(writer: &mut Writer, rig: &NanaRigResult) -> Result<(), CodecError> {
    writer.u16(BLOCK_VERSION);
    let active_count = rig
        .iter()
        .filter(|(_, sample)| **sample != SignalSample::unsupported())
        .count();
    writer.u16(u16::try_from(active_count).map_err(|_| CodecError::TooManyItems("rig signals"))?);
    for (id, sample) in rig
        .iter()
        .filter(|(_, sample)| **sample != SignalSample::unsupported())
    {
        let mut entry = Writer::default();
        encode_sample(&mut entry, sample);
        writer.u16(id.get());
        writer.u16(u16::try_from(entry.0.len()).map_err(|_| CodecError::InvalidLength)?);
        writer.bytes(&entry.0);
    }
    Ok(())
}

fn decode_rig(reader: &mut Reader<'_>) -> Result<NanaRigResult, CodecError> {
    if reader.u16()? != BLOCK_VERSION {
        return Err(CodecError::IncompatibleVersion);
    }
    let count = usize::from(reader.u16()?);
    if count > reader.remaining() / 4 {
        return Err(CodecError::InvalidLength);
    }
    let mut slots = vec![SignalSample::unsupported(); STABLE_SIGNAL_COUNT].into_boxed_slice();
    let mut previous_id = 0;
    for _ in 0..count {
        let raw_id = reader.u16()?;
        if raw_id == 0 || raw_id <= previous_id {
            return Err(CodecError::InvalidSignalId);
        }
        previous_id = raw_id;
        let length = usize::from(reader.u16()?);
        let mut entry = Reader::new(reader.take(length)?);
        if let Some(slot) = SignalId::new(raw_id).and_then(SignalId::stable_slot) {
            slots[slot] = decode_sample(&mut entry)?;
        }
        // Unknown additive IDs and compatible trailing entry fields are intentionally skipped.
    }
    Ok(NanaRigResult::from_slots(slots))
}

fn encode_geometry(writer: &mut Writer, geometry: &NanaGeometryResult) -> Result<(), CodecError> {
    writer.u16(BLOCK_VERSION);
    encode_tracked(writer, &geometry.head_camera_pose, encode_pose);
    for eye in [&geometry.eyes.left, &geometry.eyes.right] {
        encode_tracked(writer, &eye.origin_head, encode_position);
        encode_tracked(writer, &eye.direction_head, encode_direction);
    }
    encode_tracked(writer, &geometry.look_at_camera, encode_position);
    writer.u8(geometry.face_geometry_state as u8);
    writer.u16(
        u16::try_from(geometry.face_landmarks.len())
            .map_err(|_| CodecError::TooManyItems("face landmarks"))?,
    );
    for landmark in &geometry.face_landmarks {
        writer.u16(landmark.semantic_id);
        encode_tracked(writer, &landmark.position_head, encode_position);
    }
    Ok(())
}

fn decode_geometry(reader: &mut Reader<'_>) -> Result<NanaGeometryResult, CodecError> {
    if reader.u16()? != BLOCK_VERSION {
        return Err(CodecError::IncompatibleVersion);
    }
    let head_camera_pose = decode_tracked(reader, decode_pose)?;
    let left = EyeGeometry {
        origin_head: decode_tracked(reader, decode_position)?,
        direction_head: decode_tracked(reader, decode_direction)?,
    };
    let right = EyeGeometry {
        origin_head: decode_tracked(reader, decode_position)?,
        direction_head: decode_tracked(reader, decode_direction)?,
    };
    let look_at_camera = decode_tracked(reader, decode_position)?;
    let face_geometry_state = decode_state(reader.u8()?)?;
    let landmark_count = usize::from(reader.u16()?);
    if landmark_count > reader.remaining() / 24 {
        return Err(CodecError::InvalidLength);
    }
    let mut face_landmarks = Vec::with_capacity(landmark_count);
    for _ in 0..landmark_count {
        face_landmarks.push(FaceLandmark {
            semantic_id: reader.u16()?,
            position_head: decode_tracked(reader, decode_position)?,
        });
    }
    Ok(NanaGeometryResult {
        head_camera_pose,
        eyes: SideMap { left, right },
        look_at_camera,
        face_geometry_state,
        face_landmarks,
    })
}

fn encode_skeleton(writer: &mut Writer, skeleton: &NanaSkeletonResult) {
    writer.u16(BLOCK_VERSION);
    encode_tracked(writer, &skeleton.torso_camera_pose, encode_pose);
    for tracked in [
        &skeleton.shoulder.left,
        &skeleton.shoulder.right,
        &skeleton.elbow.left,
        &skeleton.elbow.right,
        &skeleton.wrist.left,
        &skeleton.wrist.right,
    ] {
        encode_tracked(writer, tracked, encode_pose);
    }
    for tracked in [
        &skeleton.upper_arm_direction_torso.left,
        &skeleton.upper_arm_direction_torso.right,
        &skeleton.forearm_direction_torso.left,
        &skeleton.forearm_direction_torso.right,
    ] {
        encode_tracked(writer, tracked, encode_direction);
    }
    for tracked in [
        &skeleton.upper_arm_twist.left,
        &skeleton.upper_arm_twist.right,
        &skeleton.forearm_twist.left,
        &skeleton.forearm_twist.right,
    ] {
        encode_tracked(writer, tracked, |writer, value| writer.f32(*value));
    }
}

fn decode_skeleton(reader: &mut Reader<'_>) -> Result<NanaSkeletonResult, CodecError> {
    if reader.u16()? != BLOCK_VERSION {
        return Err(CodecError::IncompatibleVersion);
    }
    let torso_camera_pose = decode_tracked(reader, decode_pose)?;
    let shoulder = SideMap {
        left: decode_tracked(reader, decode_pose)?,
        right: decode_tracked(reader, decode_pose)?,
    };
    let elbow = SideMap {
        left: decode_tracked(reader, decode_pose)?,
        right: decode_tracked(reader, decode_pose)?,
    };
    let wrist = SideMap {
        left: decode_tracked(reader, decode_pose)?,
        right: decode_tracked(reader, decode_pose)?,
    };
    let upper_arm_direction_torso = SideMap {
        left: decode_tracked(reader, decode_direction)?,
        right: decode_tracked(reader, decode_direction)?,
    };
    let forearm_direction_torso = SideMap {
        left: decode_tracked(reader, decode_direction)?,
        right: decode_tracked(reader, decode_direction)?,
    };
    let upper_arm_twist = SideMap {
        left: decode_tracked(reader, decode_f32)?,
        right: decode_tracked(reader, decode_f32)?,
    };
    let forearm_twist = SideMap {
        left: decode_tracked(reader, decode_f32)?,
        right: decode_tracked(reader, decode_f32)?,
    };
    Ok(NanaSkeletonResult {
        torso_camera_pose,
        shoulder,
        elbow,
        wrist,
        upper_arm_direction_torso,
        forearm_direction_torso,
        upper_arm_twist,
        forearm_twist,
    })
}

fn encode_region(writer: &mut Writer, value: RegionQuality) {
    writer.f32(value.confidence);
    writer.u8(value.state as u8);
}

fn decode_region(reader: &mut Reader<'_>) -> Result<RegionQuality, CodecError> {
    Ok(RegionQuality {
        confidence: reader.f32()?,
        state: decode_state(reader.u8()?)?,
    })
}

fn encode_quality(writer: &mut Writer, quality: &NanaTrackingQuality) {
    writer.u16(BLOCK_VERSION);
    writer.f32(quality.overall_confidence);
    for region in [
        quality.face,
        quality.eyes,
        quality.torso,
        quality.arm.left,
        quality.arm.right,
        quality.auricle.left,
        quality.auricle.right,
    ] {
        encode_region(writer, region);
    }
    encode_revision(writer, quality.stabilization_revision);
}

fn decode_quality(reader: &mut Reader<'_>) -> Result<NanaTrackingQuality, CodecError> {
    if reader.u16()? != BLOCK_VERSION {
        return Err(CodecError::IncompatibleVersion);
    }
    Ok(NanaTrackingQuality {
        overall_confidence: reader.f32()?,
        face: decode_region(reader)?,
        eyes: decode_region(reader)?,
        torso: decode_region(reader)?,
        arm: SideMap {
            left: decode_region(reader)?,
            right: decode_region(reader)?,
        },
        auricle: SideMap {
            left: decode_region(reader)?,
            right: decode_region(reader)?,
        },
        stabilization_revision: decode_revision(reader)?,
    })
}

impl WireEncode for NanaTrackingResult {
    fn encode_wire(&self) -> Result<Vec<u8>, CodecError> {
        self.validate()?;
        let mut payload = Writer::default();
        payload.tlv(1, |writer| {
            writer.bytes(&self.session_id.0);
            writer.u32(self.generation);
            writer.u64(self.sequence);
            writer.u64(self.capture_timestamp_ns);
            writer.u64(self.produced_timestamp_ns);
        })?;
        let mut rig = Writer::default();
        encode_rig(&mut rig, &self.rig)?;
        payload.tlv(2, |writer| writer.bytes(&rig.0))?;
        let mut geometry = Writer::default();
        encode_geometry(&mut geometry, &self.geometry)?;
        payload.tlv(3, |writer| writer.bytes(&geometry.0))?;
        payload.tlv(4, |writer| encode_skeleton(writer, &self.skeleton))?;
        payload.tlv(5, |writer| encode_quality(writer, &self.quality))?;
        frame(RESULT_KIND, &payload)
    }
}

impl WireDecode for NanaTrackingResult {
    fn decode_wire(bytes: &[u8]) -> Result<Self, CodecError> {
        let mut payload = payload(bytes, RESULT_KIND)?;
        let mut envelope = None;
        let mut rig = None;
        let mut geometry = None;
        let mut skeleton = None;
        let mut quality = None;
        while let Some((tag, mut field)) = payload.tlv()? {
            match tag {
                1 => {
                    let mut session = [0; 16];
                    session.copy_from_slice(field.take(16)?);
                    set_once(
                        &mut envelope,
                        tag,
                        (
                            SessionId(session),
                            field.u32()?,
                            field.u64()?,
                            field.u64()?,
                            field.u64()?,
                        ),
                    )?;
                }
                2 => set_once(&mut rig, tag, decode_rig(&mut field)?)?,
                3 => set_once(&mut geometry, tag, decode_geometry(&mut field)?)?,
                4 => set_once(&mut skeleton, tag, decode_skeleton(&mut field)?)?,
                5 => set_once(&mut quality, tag, decode_quality(&mut field)?)?,
                _ => {}
            }
        }
        let (session_id, generation, sequence, capture_timestamp_ns, produced_timestamp_ns) =
            envelope.ok_or(CodecError::MissingField(1))?;
        let result = Self {
            session_id,
            generation,
            sequence,
            capture_timestamp_ns,
            produced_timestamp_ns,
            rig: rig.ok_or(CodecError::MissingField(2))?,
            geometry: geometry.ok_or(CodecError::MissingField(3))?,
            skeleton: skeleton.ok_or(CodecError::MissingField(4))?,
            quality: quality.ok_or(CodecError::MissingField(5))?,
        };
        result.validate()?;
        Ok(result)
    }
}
