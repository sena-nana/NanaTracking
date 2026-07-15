import Foundation

public enum NTPSpatialProducerError: Error, Equatable {
  case invalidSessionID
  case invalidDescriptor
  case incompleteSignalSet(UInt16)
  case mismatchedCaptureTimestamp
  case incompleteSpatialGeometry
  case incompleteSpatialQuality
  case qualityCapabilityMismatch
  case unassignedFaceLandmarks
  case sequenceOverflow
  case generationOverflow
}

public struct NTPSpatialPayload: Equatable, Sendable {
  public let rig: [UInt16: NTPSignalSample]
  public let geometry: NTPGeometryResult
  public let quality: NTPTrackingQuality

  public init(
    rig: [UInt16: NTPSignalSample],
    geometry: NTPGeometryResult,
    quality: NTPTrackingQuality
  ) {
    self.rig = rig
    self.geometry = geometry
    self.quality = quality
  }
}

/// Session-aware final producer for already-normalized Spatial NTP values.
///
/// Raw ARKit names, transforms, and topology never enter this boundary. A caller must first apply
/// the versioned normalization and same-capture fusion contract, then submit all 41 stable Spatial
/// signals and the normalized Spatial geometry structures together.
public actor NTPSpatialProducer {
  public nonisolated let descriptor: NTPDescriptor

  private var sessionID: [UInt8]
  private var generation: UInt32
  private var nextSequence: UInt64

  public init(
    sessionID: [UInt8],
    generation: UInt32 = 0,
    descriptor: NTPDescriptor = .spatialV1
  ) throws {
    guard sessionID.count == 16 else { throw NTPSpatialProducerError.invalidSessionID }
    guard descriptor.guaranteedProfile == .spatial,
      descriptor.supportedStructures.contains(.spatialRequired),
      !descriptor.supportedStructures.contains(.bodySkeleton),
      descriptor.supportedSignals.allSatisfy({ $0 <= 88 }),
      (UInt16(1)...UInt16(41)).allSatisfy(descriptor.supportedSignals.contains)
    else {
      throw NTPSpatialProducerError.invalidDescriptor
    }
    _ = try NTPCanonicalCodec.encode(descriptor)
    self.sessionID = sessionID
    self.descriptor = descriptor
    self.generation = generation
    nextSequence = 0
  }

  public nonisolated func encodeDescriptor() throws -> Data {
    try NTPCanonicalCodec.encode(descriptor)
  }

  /// Increments the stream generation and resets sequence numbering after a producer reconfigure.
  public func reconfigure() throws {
    guard generation < UInt32.max else { throw NTPSpatialProducerError.generationOverflow }
    generation += 1
    nextSequence = 0
  }

  /// Starts a new session without reusing generation or sequence state from the previous session.
  public func beginSession(_ newSessionID: [UInt8]) throws {
    guard newSessionID.count == 16 else { throw NTPSpatialProducerError.invalidSessionID }
    sessionID = newSessionID
    generation = 0
    nextSequence = 0
  }

  public func encode(
    captureTimestampNs: UInt64,
    producedTimestampNs: UInt64,
    payload: NTPSpatialPayload
  ) throws -> Data {
    try validateSpatialPayload(
      payload,
      descriptor: descriptor,
      captureTimestampNs: captureTimestampNs
    )
    guard nextSequence < UInt64.max else { throw NTPSpatialProducerError.sequenceOverflow }
    let result = NTPTrackingResult(
      sessionID: sessionID,
      generation: generation,
      sequence: nextSequence,
      captureTimestampNs: captureTimestampNs,
      producedTimestampNs: producedTimestampNs,
      rig: payload.rig,
      geometry: payload.geometry,
      quality: payload.quality
    )
    let encoded = try NTPCanonicalCodec.encode(result)
    nextSequence += 1
    return encoded
  }
}

private func validateSpatialPayload(
  _ payload: NTPSpatialPayload, descriptor: NTPDescriptor,
  captureTimestampNs: UInt64
) throws {
  for signalID in descriptor.supportedSignals {
    guard let sample = payload.rig[signalID], sample.state != .unsupported else {
      throw NTPSpatialProducerError.incompleteSignalSet(signalID)
    }
    guard sample.sampleCaptureTimestampNs == captureTimestampNs else {
      throw NTPSpatialProducerError.mismatchedCaptureTimestamp
    }
  }
  if payload.rig.values.contains(where: {
    $0.state != .unsupported && $0.sampleCaptureTimestampNs != captureTimestampNs
  }) {
    throw NTPSpatialProducerError.mismatchedCaptureTimestamp
  }
  let declaredSignals = Set(descriptor.supportedSignals)
  if payload.rig.contains(where: {
    $0.value.state != .unsupported && !declaredSignals.contains($0.key)
  }) {
    throw NTPSpatialProducerError.invalidDescriptor
  }
  let spatialStates = [
    payload.geometry.headCameraPose.state,
    payload.geometry.leftEye.originHead.state,
    payload.geometry.leftEye.directionHead.state,
    payload.geometry.rightEye.originHead.state,
    payload.geometry.rightEye.directionHead.state,
    payload.geometry.lookAtCamera.state,
    payload.geometry.faceGeometryState,
  ]
  guard spatialStates.allSatisfy({ $0 != .unsupported }) else {
    throw NTPSpatialProducerError.incompleteSpatialGeometry
  }
  guard payload.geometry.faceLandmarks.isEmpty else {
    throw NTPSpatialProducerError.unassignedFaceLandmarks
  }
  for timestamp in [
    payload.geometry.headCameraPose.sampleCaptureTimestampNs,
    payload.geometry.leftEye.originHead.sampleCaptureTimestampNs,
    payload.geometry.leftEye.directionHead.sampleCaptureTimestampNs,
    payload.geometry.rightEye.originHead.sampleCaptureTimestampNs,
    payload.geometry.rightEye.directionHead.sampleCaptureTimestampNs,
    payload.geometry.lookAtCamera.sampleCaptureTimestampNs,
  ] where timestamp != captureTimestampNs {
    throw NTPSpatialProducerError.mismatchedCaptureTimestamp
  }
  guard payload.quality.face.state != .unsupported,
    payload.quality.eyes.state != .unsupported
  else {
    throw NTPSpatialProducerError.incompleteSpatialQuality
  }
  let signals = Set(descriptor.supportedSignals)
  for (supported, region) in [
    ((42...53).contains(where: signals.contains), payload.quality.torso),
    ([63, 65, 67, 68, 69, 70, 71].contains(where: signals.contains), payload.quality.arm.left),
    ([64, 66, 72, 73, 74, 75, 76].contains(where: signals.contains), payload.quality.arm.right),
    ([57, 59, 61, 81].contains(where: signals.contains), payload.quality.auricle.left),
    ([58, 60, 62, 82].contains(where: signals.contains), payload.quality.auricle.right),
  ] where supported == (region.state == .unsupported) {
    throw NTPSpatialProducerError.qualityCapabilityMismatch
  }
}

/// Bounded latest-frame-only worker for RGB inference from a real-time capture callback.
///
/// `submit` only swaps one pending value under a lock. While inference is busy, newer frames
/// replace the pending frame instead of building an unbounded queue.
public final class NTPLatestFrameWorker<Input: Sendable>: @unchecked Sendable {
  private let lock = NSLock()
  private let queue: DispatchQueue
  private let process: @Sendable (Input) -> Void
  private var pending: Input?
  private var running = false

  public init(
    label: String = "org.nanatracking.rgb-inference",
    qos: DispatchQoS = .userInteractive,
    process: @escaping @Sendable (Input) -> Void
  ) {
    queue = DispatchQueue(label: label, qos: qos)
    self.process = process
  }

  public func submit(_ input: Input) {
    lock.lock()
    pending = input
    let shouldStart = !running
    if shouldStart {
      running = true
    }
    lock.unlock()
    if shouldStart {
      queue.async { [self] in drain() }
    }
  }

  private func drain() {
    while true {
      lock.lock()
      guard let input = pending else {
        running = false
        lock.unlock()
        return
      }
      pending = nil
      lock.unlock()
      process(input)
    }
  }
}

/// Bounded latest-frame worker for capture stages that durably write or upload asynchronously.
///
/// `submit` is synchronous and only replaces one pending value under a lock. Exactly one task
/// drains the slot, so a slow JPEG encoder, filesystem, or network never creates one task per
/// camera frame. `flush` waits for both the active item and the latest pending replacement.
public final class NTPAsyncLatestFrameWorker<Input: Sendable>: @unchecked Sendable {
  private let lock = NSLock()
  private let process: @Sendable (Input) async -> Void
  private var dropped: UInt64 = 0
  private var pending: Input?
  private var running = false
  private var waiters: [CheckedContinuation<Void, Never>] = []

  public init(process: @escaping @Sendable (Input) async -> Void) {
    self.process = process
  }

  public func submit(_ input: Input) {
    let shouldStart = lock.withLock {
      if pending != nil {
        dropped = dropped == UInt64.max ? UInt64.max : dropped + 1
      }
      pending = input
      let shouldStart = !running
      if shouldStart {
        running = true
      }
      return shouldStart
    }
    if shouldStart {
      Task { [self] in await drain() }
    }
  }

  public func droppedCount() -> UInt64 {
    lock.withLock { dropped }
  }

  public func flush() async {
    await withCheckedContinuation { continuation in
      let resumeImmediately = lock.withLock {
        guard running || pending != nil else {
          return true
        }
        waiters.append(continuation)
        return false
      }
      if resumeImmediately {
        continuation.resume()
      }
    }
  }

  private func drain() async {
    while true {
      let (input, completed) = takeNext()
      guard let input else {
        for continuation in completed {
          continuation.resume()
        }
        return
      }
      await process(input)
    }
  }

  private func takeNext() -> (Input?, [CheckedContinuation<Void, Never>]) {
    lock.withLock {
      guard let input = pending else {
        running = false
        let completed = waiters
        waiters.removeAll(keepingCapacity: true)
        return (nil, completed)
      }
      pending = nil
      return (input, [])
    }
  }
}
