import Foundation

public enum NTPCodecError: Error, Equatable {
  case unexpectedEnd
  case invalidMagic
  case wrongMessageKind
  case incompatibleVersion
  case invalidLength
  case missingField(UInt16)
  case duplicateField(UInt16)
  case invalidEnum(String, UInt8)
  case invalidSignalID
  case invalidContract(String)
}

public enum NTPCanonicalCodec {
  public static func encode(_ descriptor: NTPDescriptor) throws -> Data {
    try validate(descriptor)
    var payload = NTPWriter()
    try payload.tlv(1) { writer in
      writer.u16(descriptor.revisions.protocolVersion.major)
      writer.u16(descriptor.revisions.protocolVersion.minor)
      writer.u32(descriptor.revisions.schemaRevision)
      writer.revision(descriptor.revisions.signalRegistry)
      writer.revision(descriptor.revisions.normalization)
      writer.revision(descriptor.revisions.calibration)
      writer.revision(descriptor.revisions.features)
    }
    try payload.tlv(2) { $0.u8(descriptor.guaranteedProfile.rawValue) }
    try payload.tlv(3) { writer in
      writer.u32(UInt32(descriptor.supportedSignals.count))
      for signal in descriptor.supportedSignals {
        writer.u16(signal)
      }
    }
    try payload.tlv(4) { $0.u64(descriptor.supportedStructures.rawValue) }
    try payload.tlv(5) { $0.u64(descriptor.features.rawValue) }
    return try frame(kind: 1, payload: payload.data)
  }

  public static func decodeDescriptor(_ data: Data) throws -> NTPDescriptor {
    var payload = try payload(data, expectedKind: 1)
    var revisions: NTPContractRevisions?
    var profile: NTPTrackingProfile?
    var signals: [UInt16]?
    var structures: NTPStructureFeatures?
    var features: NTPTrackingFeatures?
    while let (tag, fieldValue) = try payload.tlv() {
      var field = fieldValue
      switch tag {
      case 1:
        try requireUnset(revisions, tag)
        revisions = NTPContractRevisions(
          protocolVersion: NTPProtocolVersion(major: try field.u16(), minor: try field.u16()),
          schemaRevision: try field.u32(),
          signalRegistry: try field.revision(),
          normalization: try field.revision(),
          calibration: try field.revision(),
          features: try field.revision()
        )
      case 2:
        try requireUnset(profile, tag)
        let raw = try field.u8()
        guard let value = NTPTrackingProfile(rawValue: raw) else {
          throw NTPCodecError.invalidEnum("tracking profile", raw)
        }
        profile = value
      case 3:
        try requireUnset(signals, tag)
        let count = Int(try field.u32())
        guard count <= field.remaining / 2 else { throw NTPCodecError.invalidLength }
        var decoded: [UInt16] = []
        decoded.reserveCapacity(count)
        var previous: UInt16 = 0
        for _ in 0..<count {
          let signal = try field.u16()
          guard signal != 0, signal > previous else { throw NTPCodecError.invalidSignalID }
          decoded.append(signal)
          previous = signal
        }
        signals = decoded
      case 4:
        try requireUnset(structures, tag)
        structures = NTPStructureFeatures(rawValue: try field.u64())
      case 5:
        try requireUnset(features, tag)
        features = NTPTrackingFeatures(rawValue: try field.u64())
      default:
        continue
      }
    }
    guard let revisions else { throw NTPCodecError.missingField(1) }
    guard let profile else { throw NTPCodecError.missingField(2) }
    guard let signals else { throw NTPCodecError.missingField(3) }
    guard let structures else { throw NTPCodecError.missingField(4) }
    guard let features else { throw NTPCodecError.missingField(5) }
    let descriptor = NTPDescriptor(
      revisions: revisions,
      guaranteedProfile: profile,
      supportedSignals: signals,
      supportedStructures: structures,
      features: features
    )
    try validate(descriptor)
    return descriptor
  }

  public static func encode(_ result: NTPTrackingResult) throws -> Data {
    try validate(result)
    var payload = NTPWriter()
    try payload.tlv(1) { writer in
      writer.bytes(result.sessionID)
      writer.u32(result.generation)
      writer.u64(result.sequence)
      writer.u64(result.captureTimestampNs)
      writer.u64(result.producedTimestampNs)
    }
    try payload.tlv(2) { writer in
      writer.u16(1)
      let active = result.rig.filter { $0.value != .unsupported }.sorted { $0.key < $1.key }
      guard active.count <= Int(UInt16.max) else { throw NTPCodecError.invalidLength }
      writer.u16(UInt16(active.count))
      for (signalID, sample) in active {
        var entry = NTPWriter()
        entry.sample(sample)
        guard entry.data.count <= Int(UInt16.max) else { throw NTPCodecError.invalidLength }
        writer.u16(signalID)
        writer.u16(UInt16(entry.data.count))
        writer.data.append(entry.data)
      }
    }
    try payload.tlv(3) { try encodeGeometry(result.geometry, into: &$0) }
    try payload.tlv(4) { encodeSkeleton(result.skeleton, into: &$0) }
    try payload.tlv(5) { encodeQuality(result.quality, into: &$0) }
    return try frame(kind: 2, payload: payload.data)
  }

  public static func decodeResult(_ data: Data) throws -> NTPTrackingResult {
    var payload = try payload(data, expectedKind: 2)
    var envelope: ([UInt8], UInt32, UInt64, UInt64, UInt64)?
    var rig: [UInt16: NTPSignalSample]?
    var geometry: NTPGeometryResult?
    var skeleton: NTPSkeletonResult?
    var quality: NTPTrackingQuality?
    while let (tag, fieldValue) = try payload.tlv() {
      var field = fieldValue
      switch tag {
      case 1:
        try requireUnset(envelope, tag)
        envelope = (
          try field.bytes(count: 16),
          try field.u32(),
          try field.u64(),
          try field.u64(),
          try field.u64()
        )
      case 2:
        try requireUnset(rig, tag)
        rig = try decodeRig(&field)
      case 3:
        try requireUnset(geometry, tag)
        geometry = try decodeGeometry(&field)
      case 4:
        try requireUnset(skeleton, tag)
        skeleton = try decodeSkeleton(&field)
      case 5:
        try requireUnset(quality, tag)
        quality = try decodeQuality(&field)
      default:
        continue
      }
    }
    guard let envelope else { throw NTPCodecError.missingField(1) }
    guard let rig else { throw NTPCodecError.missingField(2) }
    guard let geometry else { throw NTPCodecError.missingField(3) }
    guard let skeleton else { throw NTPCodecError.missingField(4) }
    guard let quality else { throw NTPCodecError.missingField(5) }
    let result = NTPTrackingResult(
      sessionID: envelope.0,
      generation: envelope.1,
      sequence: envelope.2,
      captureTimestampNs: envelope.3,
      producedTimestampNs: envelope.4,
      rig: rig,
      geometry: geometry,
      skeleton: skeleton,
      quality: quality
    )
    try validate(result)
    return result
  }
}

private struct NTPWriter {
  var data = Data()

  mutating func u8(_ value: UInt8) {
    data.append(value)
  }

  mutating func u16(_ value: UInt16) {
    bytes(withUnsafeBytes(of: value.littleEndian, Array.init))
  }

  mutating func u32(_ value: UInt32) {
    bytes(withUnsafeBytes(of: value.littleEndian, Array.init))
  }

  mutating func u64(_ value: UInt64) {
    bytes(withUnsafeBytes(of: value.littleEndian, Array.init))
  }

  mutating func f32(_ value: Float) {
    u32(value == 0 ? 0 : value.bitPattern)
  }

  mutating func bytes(_ value: [UInt8]) {
    data.append(contentsOf: value)
  }

  mutating func revision(_ value: NTPRevision) {
    u16(value.major)
    u16(value.minor)
    u16(value.patch)
  }

  mutating func tlv(_ tag: UInt16, encode: (inout NTPWriter) throws -> Void) throws {
    var body = NTPWriter()
    try encode(&body)
    guard body.data.count <= Int(UInt32.max) else { throw NTPCodecError.invalidLength }
    u16(tag)
    u32(UInt32(body.data.count))
    data.append(body.data)
  }

  mutating func vector(_ value: NTPVector3) {
    f32(value.x)
    f32(value.y)
    f32(value.z)
  }

  mutating func position(_ value: NTPPosition3) {
    u8(value.space.rawValue)
    u8(value.lengthBasis.rawValue)
    vector(value.value)
  }

  mutating func direction(_ value: NTPDirection3) {
    u8(value.space.rawValue)
    vector(value.value)
  }

  mutating func pose(_ value: NTPPose) {
    u8(value.parentSpace.rawValue)
    u8(value.lengthBasis.rawValue)
    vector(value.position)
    let orientation = value.orientationXYZW.canonicalized
    f32(orientation.x)
    f32(orientation.y)
    f32(orientation.z)
    f32(orientation.w)
  }

  mutating func sample(_ value: NTPSignalSample) {
    u8(value.state.rawValue)
    u8(value.value == nil ? 0 : 1)
    f32(value.confidence)
    u64(value.sampleCaptureTimestampNs)
    u64(value.predictionHorizonNs)
    if let sample = value.value {
      f32(sample)
    }
  }
}

private struct NTPReader {
  let data: [UInt8]
  var offset = 0

  init(_ data: Data) {
    self.data = Array(data)
  }

  init(_ data: [UInt8]) {
    self.data = data
  }

  var remaining: Int { data.count - offset }

  mutating func bytes(count: Int) throws -> [UInt8] {
    guard count >= 0, count <= remaining else { throw NTPCodecError.unexpectedEnd }
    defer { offset += count }
    return Array(data[offset..<(offset + count)])
  }

  mutating func u8() throws -> UInt8 {
    try bytes(count: 1)[0]
  }

  mutating func u16() throws -> UInt16 {
    let value = try bytes(count: 2)
    return UInt16(value[0]) | UInt16(value[1]) << 8
  }

  mutating func u32() throws -> UInt32 {
    let value = try bytes(count: 4)
    return value.enumerated().reduce(0) { $0 | UInt32($1.element) << UInt32($1.offset * 8) }
  }

  mutating func u64() throws -> UInt64 {
    let value = try bytes(count: 8)
    return value.enumerated().reduce(0) { $0 | UInt64($1.element) << UInt64($1.offset * 8) }
  }

  mutating func f32() throws -> Float {
    Float(bitPattern: try u32())
  }

  mutating func revision() throws -> NTPRevision {
    NTPRevision(major: try u16(), minor: try u16(), patch: try u16())
  }

  mutating func tlv() throws -> (UInt16, NTPReader)? {
    guard remaining > 0 else { return nil }
    let tag = try u16()
    let length = Int(try u32())
    return (tag, NTPReader(try bytes(count: length)))
  }

  mutating func state() throws -> NTPSignalState {
    let raw = try u8()
    guard let state = NTPSignalState(rawValue: raw) else {
      throw NTPCodecError.invalidEnum("signal state", raw)
    }
    return state
  }

  mutating func valuePresence() throws -> Bool {
    let raw = try u8()
    guard raw == 0 || raw == 1 else {
      throw NTPCodecError.invalidEnum("value presence", raw)
    }
    return raw == 1
  }

  mutating func coordinateSpace() throws -> NTPCoordinateSpace {
    let raw = try u8()
    guard let value = NTPCoordinateSpace(rawValue: raw) else {
      throw NTPCodecError.invalidEnum("coordinate space", raw)
    }
    return value
  }

  mutating func lengthBasis() throws -> NTPLengthBasis {
    let raw = try u8()
    guard let value = NTPLengthBasis(rawValue: raw) else {
      throw NTPCodecError.invalidEnum("length basis", raw)
    }
    return value
  }

  mutating func vector() throws -> NTPVector3 {
    NTPVector3(x: try f32(), y: try f32(), z: try f32())
  }

  mutating func position() throws -> NTPPosition3 {
    NTPPosition3(space: try coordinateSpace(), lengthBasis: try lengthBasis(), value: try vector())
  }

  mutating func direction() throws -> NTPDirection3 {
    NTPDirection3(space: try coordinateSpace(), value: try vector())
  }

  mutating func pose() throws -> NTPPose {
    NTPPose(
      parentSpace: try coordinateSpace(),
      lengthBasis: try lengthBasis(),
      position: try vector(),
      orientationXYZW: NTPQuaternion(x: try f32(), y: try f32(), z: try f32(), w: try f32())
    )
  }

  mutating func sample() throws -> NTPSignalSample {
    let state = try state()
    let hasValue = try valuePresence()
    let confidence = try f32()
    let timestamp = try u64()
    let horizon = try u64()
    return NTPSignalSample(
      value: hasValue ? try f32() : nil,
      confidence: confidence,
      state: state,
      sampleCaptureTimestampNs: timestamp,
      predictionHorizonNs: horizon
    )
  }
}

private func frame(kind: UInt8, payload: Data) throws -> Data {
  guard payload.count <= Int(UInt32.max) else { throw NTPCodecError.invalidLength }
  var writer = NTPWriter()
  writer.bytes(Array("NTP1".utf8))
  writer.u8(kind)
  writer.u8(1)
  writer.u8(0)
  writer.u8(0)
  writer.u32(UInt32(payload.count))
  writer.data.append(payload)
  return writer.data
}

private func payload(_ data: Data, expectedKind: UInt8) throws -> NTPReader {
  var reader = NTPReader(data)
  guard try reader.bytes(count: 4) == Array("NTP1".utf8) else {
    throw NTPCodecError.invalidMagic
  }
  guard try reader.u8() == expectedKind else { throw NTPCodecError.wrongMessageKind }
  guard try reader.u8() == 1 else { throw NTPCodecError.incompatibleVersion }
  _ = try reader.u8()
  _ = try reader.u8()
  let length = Int(try reader.u32())
  guard length == reader.remaining else { throw NTPCodecError.invalidLength }
  return NTPReader(try reader.bytes(count: length))
}

private func requireUnset<T>(_ value: T?, _ tag: UInt16) throws {
  if value != nil { throw NTPCodecError.duplicateField(tag) }
}

private func decodeRig(_ reader: inout NTPReader) throws -> [UInt16: NTPSignalSample] {
  guard try reader.u16() == 1 else { throw NTPCodecError.incompatibleVersion }
  let count = Int(try reader.u16())
  guard count <= reader.remaining / 4 else { throw NTPCodecError.invalidLength }
  var rig: [UInt16: NTPSignalSample] = [:]
  var previous: UInt16 = 0
  for _ in 0..<count {
    let signalID = try reader.u16()
    guard signalID != 0, signalID > previous else { throw NTPCodecError.invalidSignalID }
    previous = signalID
    let length = Int(try reader.u16())
    var entry = NTPReader(try reader.bytes(count: length))
    if signalID <= 88 {
      rig[signalID] = try entry.sample()
    }
  }
  return rig
}

private func encodeTracked<Value: Equatable & Sendable>(
  _ tracked: NTPTracked<Value>,
  into writer: inout NTPWriter,
  encodeValue: (Value, inout NTPWriter) -> Void
) {
  writer.u8(tracked.state.rawValue)
  writer.u8(tracked.value == nil ? 0 : 1)
  writer.f32(tracked.confidence)
  writer.u64(tracked.sampleCaptureTimestampNs)
  writer.u64(tracked.predictionHorizonNs)
  if let value = tracked.value {
    encodeValue(value, &writer)
  }
}

private func decodeTracked<Value: Equatable & Sendable>(
  _ reader: inout NTPReader,
  decodeValue: (inout NTPReader) throws -> Value
) throws -> NTPTracked<Value> {
  let state = try reader.state()
  let hasValue = try reader.valuePresence()
  let confidence = try reader.f32()
  let timestamp = try reader.u64()
  let horizon = try reader.u64()
  return NTPTracked(
    value: hasValue ? try decodeValue(&reader) : nil,
    confidence: confidence,
    state: state,
    sampleCaptureTimestampNs: timestamp,
    predictionHorizonNs: horizon
  )
}

private func encodeEye(_ eye: NTPEyeGeometry, into writer: inout NTPWriter) {
  encodeTracked(eye.originHead, into: &writer) { $1.position($0) }
  encodeTracked(eye.directionHead, into: &writer) { $1.direction($0) }
}

private func decodeEye(_ reader: inout NTPReader) throws -> NTPEyeGeometry {
  NTPEyeGeometry(
    originHead: try decodeTracked(&reader) { try $0.position() },
    directionHead: try decodeTracked(&reader) { try $0.direction() }
  )
}

private func encodeGeometry(_ geometry: NTPGeometryResult, into writer: inout NTPWriter) throws {
  writer.u16(1)
  encodeTracked(geometry.headCameraPose, into: &writer) { $1.pose($0) }
  encodeEye(geometry.leftEye, into: &writer)
  encodeEye(geometry.rightEye, into: &writer)
  encodeTracked(geometry.lookAtCamera, into: &writer) { $1.position($0) }
  writer.u8(geometry.faceGeometryState.rawValue)
  guard geometry.faceLandmarks.count <= Int(UInt16.max) else {
    throw NTPCodecError.invalidLength
  }
  writer.u16(UInt16(geometry.faceLandmarks.count))
  for landmark in geometry.faceLandmarks {
    writer.u16(landmark.semanticID)
    encodeTracked(landmark.positionHead, into: &writer) { $1.position($0) }
  }
}

private func decodeGeometry(_ reader: inout NTPReader) throws -> NTPGeometryResult {
  guard try reader.u16() == 1 else { throw NTPCodecError.incompatibleVersion }
  let head = try decodeTracked(&reader) { try $0.pose() }
  let left = try decodeEye(&reader)
  let right = try decodeEye(&reader)
  let lookAt = try decodeTracked(&reader) { try $0.position() }
  let state = try reader.state()
  let count = Int(try reader.u16())
  guard count <= reader.remaining / 24 else { throw NTPCodecError.invalidLength }
  var landmarks: [NTPFaceLandmark] = []
  landmarks.reserveCapacity(count)
  for _ in 0..<count {
    landmarks.append(
      NTPFaceLandmark(
        semanticID: try reader.u16(),
        positionHead: try decodeTracked(&reader) { try $0.position() }
      ))
  }
  return NTPGeometryResult(
    headCameraPose: head,
    leftEye: left,
    rightEye: right,
    lookAtCamera: lookAt,
    faceGeometryState: state,
    faceLandmarks: landmarks
  )
}

private func encodeSkeleton(_ skeleton: NTPSkeletonResult, into writer: inout NTPWriter) {
  writer.u16(1)
  encodeTracked(skeleton.torsoCameraPose, into: &writer) { $1.pose($0) }
  for tracked in [
    skeleton.shoulder.left, skeleton.shoulder.right,
    skeleton.elbow.left, skeleton.elbow.right,
    skeleton.wrist.left, skeleton.wrist.right,
  ] {
    encodeTracked(tracked, into: &writer) { $1.pose($0) }
  }
  for tracked in [
    skeleton.upperArmDirectionTorso.left, skeleton.upperArmDirectionTorso.right,
    skeleton.forearmDirectionTorso.left, skeleton.forearmDirectionTorso.right,
  ] {
    encodeTracked(tracked, into: &writer) { $1.direction($0) }
  }
  for tracked in [
    skeleton.upperArmTwist.left, skeleton.upperArmTwist.right,
    skeleton.forearmTwist.left, skeleton.forearmTwist.right,
  ] {
    encodeTracked(tracked, into: &writer) { $1.f32($0) }
  }
}

private func decodeSkeleton(_ reader: inout NTPReader) throws -> NTPSkeletonResult {
  guard try reader.u16() == 1 else { throw NTPCodecError.incompatibleVersion }
  let torso = try decodeTracked(&reader) { try $0.pose() }
  let shoulder = NTPSideMap(
    left: try decodeTracked(&reader) { try $0.pose() },
    right: try decodeTracked(&reader) { try $0.pose() }
  )
  let elbow = NTPSideMap(
    left: try decodeTracked(&reader) { try $0.pose() },
    right: try decodeTracked(&reader) { try $0.pose() }
  )
  let wrist = NTPSideMap(
    left: try decodeTracked(&reader) { try $0.pose() },
    right: try decodeTracked(&reader) { try $0.pose() }
  )
  let upperDirection = NTPSideMap(
    left: try decodeTracked(&reader) { try $0.direction() },
    right: try decodeTracked(&reader) { try $0.direction() }
  )
  let forearmDirection = NTPSideMap(
    left: try decodeTracked(&reader) { try $0.direction() },
    right: try decodeTracked(&reader) { try $0.direction() }
  )
  let upperTwist = NTPSideMap(
    left: try decodeTracked(&reader) { try $0.f32() },
    right: try decodeTracked(&reader) { try $0.f32() }
  )
  let forearmTwist = NTPSideMap(
    left: try decodeTracked(&reader) { try $0.f32() },
    right: try decodeTracked(&reader) { try $0.f32() }
  )
  return NTPSkeletonResult(
    torsoCameraPose: torso,
    shoulder: shoulder,
    elbow: elbow,
    wrist: wrist,
    upperArmDirectionTorso: upperDirection,
    forearmDirectionTorso: forearmDirection,
    upperArmTwist: upperTwist,
    forearmTwist: forearmTwist
  )
}

private func encodeRegion(_ region: NTPRegionQuality, into writer: inout NTPWriter) {
  writer.f32(region.confidence)
  writer.u8(region.state.rawValue)
}

private func decodeRegion(_ reader: inout NTPReader) throws -> NTPRegionQuality {
  NTPRegionQuality(confidence: try reader.f32(), state: try reader.state())
}

private func encodeQuality(_ quality: NTPTrackingQuality, into writer: inout NTPWriter) {
  writer.u16(1)
  writer.f32(quality.overallConfidence)
  for region in [
    quality.face, quality.eyes, quality.torso,
    quality.arm.left, quality.arm.right,
    quality.auricle.left, quality.auricle.right,
  ] {
    encodeRegion(region, into: &writer)
  }
  writer.revision(quality.stabilizationRevision)
}

private func decodeQuality(_ reader: inout NTPReader) throws -> NTPTrackingQuality {
  guard try reader.u16() == 1 else { throw NTPCodecError.incompatibleVersion }
  return NTPTrackingQuality(
    overallConfidence: try reader.f32(),
    face: try decodeRegion(&reader),
    eyes: try decodeRegion(&reader),
    torso: try decodeRegion(&reader),
    arm: NTPSideMap(left: try decodeRegion(&reader), right: try decodeRegion(&reader)),
    auricle: NTPSideMap(left: try decodeRegion(&reader), right: try decodeRegion(&reader)),
    stabilizationRevision: try reader.revision()
  )
}

private func validate(_ descriptor: NTPDescriptor) throws {
  guard descriptor.revisions.protocolVersion.major == 1 else {
    throw NTPCodecError.invalidContract("unsupported protocol major")
  }
  guard !descriptor.supportedSignals.contains(0),
    descriptor.supportedSignals == descriptor.supportedSignals.sorted(),
    Set(descriptor.supportedSignals).count == descriptor.supportedSignals.count
  else {
    throw NTPCodecError.invalidSignalID
  }
  let inferred: NTPTrackingProfile
  let signals = Set(descriptor.supportedSignals)
  if (1...76).allSatisfy(signals.contains)
    && descriptor.supportedStructures.contains([.spatialRequired, .bodySkeleton])
  {
    inferred = .full
  } else if (1...41).allSatisfy(signals.contains)
    && descriptor.supportedStructures.contains(.spatialRequired)
  {
    inferred = .spatial
  } else if (1...36).allSatisfy(signals.contains)
    && descriptor.supportedStructures.contains(.headGeometry)
  {
    inferred = .basic
  } else {
    inferred = .partial
  }
  guard inferred == descriptor.guaranteedProfile else {
    throw NTPCodecError.invalidContract("guaranteed profile does not match capabilities")
  }
}

private func validate(_ result: NTPTrackingResult) throws {
  guard result.sessionID.count == 16 else {
    throw NTPCodecError.invalidContract("session ID must contain 16 bytes")
  }
  guard result.producedTimestampNs >= result.captureTimestampNs else {
    throw NTPCodecError.invalidContract("produced timestamp precedes capture")
  }
  for (signalID, sample) in result.rig {
    guard (1...88).contains(signalID) else { throw NTPCodecError.invalidSignalID }
    try validate(sample, producedTimestampNs: result.producedTimestampNs)
    if let value = sample.value {
      guard validSignalValue(value, signalID: signalID) else {
        throw NTPCodecError.invalidContract("signal value outside registry range")
      }
    }
  }
  try validate(result.geometry.headCameraPose, producedTimestampNs: result.producedTimestampNs)
  if let pose = result.geometry.headCameraPose.value {
    guard pose.parentSpace == .camera else {
      throw NTPCodecError.invalidContract("head pose must be camera-relative")
    }
    try validate(pose)
  }
  try validate(result.geometry.leftEye.originHead, producedTimestampNs: result.producedTimestampNs)
  try validate(
    result.geometry.leftEye.directionHead, producedTimestampNs: result.producedTimestampNs)
  try validate(result.geometry.rightEye.originHead, producedTimestampNs: result.producedTimestampNs)
  try validate(
    result.geometry.rightEye.directionHead, producedTimestampNs: result.producedTimestampNs)
  try validate(result.geometry.lookAtCamera, producedTimestampNs: result.producedTimestampNs)
  for eye in [result.geometry.leftEye, result.geometry.rightEye] {
    if let origin = eye.originHead.value {
      guard origin.space == .headLocal, origin.lengthBasis == .headRelative else {
        throw NTPCodecError.invalidContract("eye origin must be head-relative")
      }
      try validate(origin.value)
    }
    if let direction = eye.directionHead.value {
      guard direction.space == .headLocal else {
        throw NTPCodecError.invalidContract("eye direction must be head-local")
      }
      try validateUnit(direction.value)
    }
  }
  if let lookAt = result.geometry.lookAtCamera.value {
    guard lookAt.space == .camera else {
      throw NTPCodecError.invalidContract("look-at point must be camera-relative")
    }
    try validate(lookAt.value)
  }
  var previousLandmark: UInt16 = 0
  for landmark in result.geometry.faceLandmarks {
    guard landmark.semanticID > previousLandmark else {
      throw NTPCodecError.invalidContract("face landmarks must have increasing nonzero IDs")
    }
    previousLandmark = landmark.semanticID
    try validate(landmark.positionHead, producedTimestampNs: result.producedTimestampNs)
    if let position = landmark.positionHead.value {
      guard position.space == .headLocal, position.lengthBasis == .headRelative else {
        throw NTPCodecError.invalidContract("face landmark must be head-relative")
      }
      try validate(position.value)
    }
  }
  try validate(result.skeleton, producedTimestampNs: result.producedTimestampNs)
  guard validConfidence(result.quality.overallConfidence) else {
    throw NTPCodecError.invalidContract("invalid overall confidence")
  }
  for region in [
    result.quality.face, result.quality.eyes, result.quality.torso,
    result.quality.arm.left, result.quality.arm.right,
    result.quality.auricle.left, result.quality.auricle.right,
  ] {
    guard validConfidence(region.confidence),
      region.state != .unsupported || region.confidence == 0
    else {
      throw NTPCodecError.invalidContract("invalid region quality")
    }
  }
}

private func validate(_ sample: NTPSignalSample, producedTimestampNs: UInt64) throws {
  guard validConfidence(sample.confidence), sample.state.carriesValue == (sample.value != nil)
  else {
    throw NTPCodecError.invalidContract("invalid signal state/value")
  }
  try validateStateShell(
    state: sample.state,
    confidence: sample.confidence,
    timestamp: sample.sampleCaptureTimestampNs,
    predictionHorizonNs: sample.predictionHorizonNs,
    producedTimestampNs: producedTimestampNs
  )
}

private func validate<Value>(_ tracked: NTPTracked<Value>, producedTimestampNs: UInt64) throws {
  guard tracked.state.carriesValue == (tracked.value != nil) else {
    throw NTPCodecError.invalidContract("invalid tracked state/value")
  }
  try validateStateShell(
    state: tracked.state,
    confidence: tracked.confidence,
    timestamp: tracked.sampleCaptureTimestampNs,
    predictionHorizonNs: tracked.predictionHorizonNs,
    producedTimestampNs: producedTimestampNs
  )
}

private func validate(
  _ skeleton: NTPSkeletonResult, producedTimestampNs: UInt64
) throws {
  try validate(skeleton.torsoCameraPose, producedTimestampNs: producedTimestampNs)
  if let torso = skeleton.torsoCameraPose.value {
    guard torso.parentSpace == .camera else {
      throw NTPCodecError.invalidContract("torso pose must be camera-relative")
    }
    try validate(torso)
  }
  for joint in [
    skeleton.shoulder.left, skeleton.shoulder.right,
    skeleton.elbow.left, skeleton.elbow.right,
    skeleton.wrist.left, skeleton.wrist.right,
  ] {
    try validate(joint, producedTimestampNs: producedTimestampNs)
    if let pose = joint.value {
      guard pose.parentSpace == .torsoLocal else {
        throw NTPCodecError.invalidContract("body joint must be torso-local")
      }
      if let torso = skeleton.torsoCameraPose.value, pose.lengthBasis != torso.lengthBasis {
        throw NTPCodecError.invalidContract("body skeleton uses mixed length bases")
      }
      try validate(pose)
    }
  }
  for direction in [
    skeleton.upperArmDirectionTorso.left, skeleton.upperArmDirectionTorso.right,
    skeleton.forearmDirectionTorso.left, skeleton.forearmDirectionTorso.right,
  ] {
    try validate(direction, producedTimestampNs: producedTimestampNs)
    if let value = direction.value {
      guard value.space == .torsoLocal else {
        throw NTPCodecError.invalidContract("arm direction must be torso-local")
      }
      try validateUnit(value.value)
    }
  }
  for twist in [
    skeleton.upperArmTwist.left, skeleton.upperArmTwist.right,
    skeleton.forearmTwist.left, skeleton.forearmTwist.right,
  ] {
    try validate(twist, producedTimestampNs: producedTimestampNs)
    if let value = twist.value,
      !value.isFinite || value < -Float.pi || value >= Float.pi
    {
      throw NTPCodecError.invalidContract("arm twist is outside the angle range")
    }
  }
}

private func validate(_ value: NTPVector3) throws {
  guard value.x.isFinite, value.y.isFinite, value.z.isFinite else {
    throw NTPCodecError.invalidContract("vector contains a non-finite component")
  }
}

private func validateUnit(_ value: NTPVector3) throws {
  try validate(value)
  let normSquared = value.x * value.x + value.y * value.y + value.z * value.z
  guard abs(normSquared - 1) <= 2.0e-4 else {
    throw NTPCodecError.invalidContract("direction is not a unit vector")
  }
}

private func validate(_ value: NTPPose) throws {
  try validate(value.position)
  let quaternion = value.orientationXYZW
  let normSquared =
    quaternion.x * quaternion.x + quaternion.y * quaternion.y + quaternion.z * quaternion.z
    + quaternion.w * quaternion.w
  guard normSquared.isFinite, abs(normSquared - 1) <= 2.0e-4 else {
    throw NTPCodecError.invalidContract("pose quaternion is not normalized")
  }
}

private func validateStateShell(
  state: NTPSignalState,
  confidence: Float,
  timestamp: UInt64,
  predictionHorizonNs: UInt64,
  producedTimestampNs: UInt64
) throws {
  guard validConfidence(confidence) else {
    throw NTPCodecError.invalidContract("invalid confidence")
  }
  if state == .unsupported && (confidence != 0 || timestamp != 0 || predictionHorizonNs != 0) {
    throw NTPCodecError.invalidContract("unsupported state carries metadata")
  }
  if (state == .predicted) != (predictionHorizonNs != 0) {
    throw NTPCodecError.invalidContract("invalid prediction horizon")
  }
  if state != .unsupported && timestamp > producedTimestampNs {
    throw NTPCodecError.invalidContract("sample timestamp is in the future")
  }
}

private func validConfidence(_ value: Float) -> Bool {
  value.isFinite && (0...1).contains(value)
}

private func signalRange(_ signalID: UInt16) -> ClosedRange<Float> {
  switch signalID {
  case 9, 10, 13...17, 28, 30, 33...36, 41, 70, 75, 79, 80:
    return 0...1
  case 37, 39:
    return -1.2...1.2
  case 38, 40:
    return -0.8...0.8
  default:
    return -1...1
  }
}

private func validSignalValue(_ value: Float, signalID: UInt16) -> Bool {
  guard value.isFinite else { return false }
  if (45...47).contains(signalID) || (51...53).contains(signalID) {
    return value >= -Float.pi && value < Float.pi
  }
  return signalRange(signalID).contains(value)
}
