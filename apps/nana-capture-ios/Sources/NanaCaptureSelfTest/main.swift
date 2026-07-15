import Foundation
import NanaCaptureCore

enum SelfTestError: Error {
  case protocolVectorNotFound
  case malformedProtocolVector
  case canonicalRoundTripFailed
  case spatialProducerLifecycleFailed
  case latestFramePolicyFailed
  case unexpectedPendingChunks
  case invalidAcknowledgementAccepted
  case corruptReceiverPayloadAccepted
  case controlLifecycleFailed
  case controlContractRoundTripFailed
}

private final class LatestFrameProbe: @unchecked Sendable {
  let firstStarted = DispatchSemaphore(value: 0)
  let releaseFirst = DispatchSemaphore(value: 0)
  let completed = DispatchSemaphore(value: 0)
  private let lock = NSLock()
  private var values: [Int] = []

  func process(_ value: Int) {
    if value == 1 {
      firstStarted.signal()
      releaseFirst.wait()
    }
    lock.lock()
    values.append(value)
    let isComplete = values.count == 2
    lock.unlock()
    if isComplete { completed.signal() }
  }

  func snapshot() -> [Int] {
    lock.lock()
    defer { lock.unlock() }
    return values
  }
}

private func blockingWait(
  _ semaphore: DispatchSemaphore, timeout: DispatchTime
) -> DispatchTimeoutResult {
  semaphore.wait(timeout: timeout)
}

@main
struct NanaCaptureSelfTest {
  static func main() async throws {
    try canonicalProtocolRoundTrip()
    try latestFrameOnlyPolicy()
    try await asyncLatestFrameOnlyPolicy()
    try await spatialProducerLifecycle()
    let root = FileManager.default.temporaryDirectory.appending(
      path: "nana-capture-\(UUID().uuidString)",
      directoryHint: .isDirectory
    )
    defer { try? FileManager.default.removeItem(at: root) }
    let source = try LocalChunkRecorder(root: root.appending(path: "source"))
    let first = try await source.writeChunk(
      chunkID: "chunk-0",
      takeID: "take-1",
      kind: .arkit,
      sequenceStart: 0,
      sequenceEnd: 3,
      captureTimestampStartNs: 100,
      captureTimestampEndNs: 130,
      payload: Data("first".utf8)
    )
    let second = try await source.writeChunk(
      chunkID: "chunk-1",
      takeID: "take-1",
      kind: .arkit,
      sequenceStart: 4,
      sequenceEnd: 7,
      captureTimestampStartNs: 140,
      captureTimestampEndNs: 170,
      payload: Data("second".utf8)
    )
    guard
      second.relativePath
        == LocalChunkRecorder.relativePath(
          chunkID: second.chunkID,
          takeID: second.takeID,
          kind: second.kind,
          sequenceStart: second.sequenceStart,
          sequenceEnd: second.sequenceEnd
        )
    else {
      throw SelfTestError.unexpectedPendingChunks
    }
    try await source.acknowledge(
      ChunkAcknowledgement(chunkID: first.chunkID, sha256: first.sha256)
    )
    let reopened = try LocalChunkRecorder(root: root.appending(path: "source"))
    guard try await reopened.pendingChunks() == [second] else {
      throw SelfTestError.unexpectedPendingChunks
    }
    do {
      try await reopened.acknowledge(
        ChunkAcknowledgement(
          chunkID: second.chunkID,
          sha256: String(repeating: "0", count: 64)
        )
      )
      throw SelfTestError.invalidAcknowledgementAccepted
    } catch ChunkRecorderError.acknowledgementDigestMismatch {
      // Expected: the sender retains the chunk for retry.
    }

    let receiver = try LocalChunkRecorder(root: root.appending(path: "receiver"))
    do {
      try await receiver.receiveChunk(second, payload: Data("corrupt".utf8))
      throw SelfTestError.corruptReceiverPayloadAccepted
    } catch ChunkRecorderError.descriptorMismatch {
      // Expected: no durable receiver record was created.
    }
    guard try await receiver.chunks().isEmpty else {
      throw SelfTestError.corruptReceiverPayloadAccepted
    }
    _ = try await receiver.receiveChunk(second, payload: Data("second".utf8))

    let lifecycle = StudioCaptureLifecycle()
    let startCommand = StudioControlCommand(
      sessionID: "session-1",
      revision: 1,
      action: .start,
      takeID: "take-1",
      actionScriptID: "basic-v1"
    )
    let encodedCommand = try JSONEncoder().encode(startCommand)
    guard try JSONDecoder().decode(StudioControlCommand.self, from: encodedCommand) == startCommand
    else {
      throw SelfTestError.controlContractRoundTripFailed
    }
    try await lifecycle.apply(startCommand)
    try await lifecycle.apply(
      StudioControlCommand(
        sessionID: "session-1",
        revision: 2,
        action: .pause,
        takeID: "take-1"
      )
    )
    try await lifecycle.apply(
      StudioControlCommand(
        sessionID: "session-1",
        revision: 3,
        action: .start,
        takeID: "take-1",
        actionScriptID: "basic-v1"
      )
    )
    try await lifecycle.apply(
      StudioControlCommand(
        sessionID: "session-1",
        revision: 4,
        action: .stop,
        takeID: "take-1"
      )
    )
    try await lifecycle.apply(
      StudioControlCommand(
        sessionID: "session-1",
        revision: 5,
        action: .end
      )
    )
    guard await lifecycle.state == .complete, await lifecycle.revision == 5 else {
      throw SelfTestError.controlLifecycleFailed
    }
    print(
      "NanaCaptureSelfTest passed: NTP cross-language round-trip, latest-only inference, "
        + "producer generations, restart, receiver verification, and control lifecycle"
    )
  }

  private static func asyncLatestFrameOnlyPolicy() async throws {
    let probe = LatestFrameProbe()
    let worker = NTPAsyncLatestFrameWorker<Int> { value in
      probe.process(value)
    }
    worker.submit(1)
    let started = await Task.detached {
      blockingWait(probe.firstStarted, timeout: .now() + 2)
    }.value
    guard started == .success else {
      throw SelfTestError.latestFramePolicyFailed
    }
    worker.submit(2)
    worker.submit(3)
    probe.releaseFirst.signal()
    await worker.flush()
    guard probe.snapshot() == [1, 3], worker.droppedCount() == 1 else {
      throw SelfTestError.latestFramePolicyFailed
    }
  }

  private static func canonicalProtocolRoundTrip() throws {
    let vector = try String(contentsOf: protocolVectorURL(), encoding: .utf8)
    let entries: [(String, Data)] = try vector.split(separator: "\n").compactMap {
      line -> (String, Data)? in
      guard !line.hasPrefix("#") else { return nil }
      let parts = line.split(separator: "=", maxSplits: 1)
      guard parts.count == 2 else { return nil }
      return (String(parts[0]), try decodeHex(parts[1]))
    }
    let fields = Dictionary(uniqueKeysWithValues: entries)
    guard let descriptorBytes = fields["descriptor"], let resultBytes = fields["result"] else {
      throw SelfTestError.malformedProtocolVector
    }
    let descriptor = try NTPCanonicalCodec.decodeDescriptor(descriptorBytes)
    let result = try NTPCanonicalCodec.decodeResult(resultBytes)
    guard try NTPCanonicalCodec.encode(descriptor) == descriptorBytes,
      try NTPCanonicalCodec.encode(result) == resultBytes
    else {
      throw SelfTestError.canonicalRoundTripFailed
    }
  }

  private static func spatialProducerLifecycle() async throws {
    let timestamp: UInt64 = 2_000_000_000
    let rig = Dictionary(
      uniqueKeysWithValues: (UInt16(1)...UInt16(42)).map { signalID in
        (
          signalID,
          NTPSignalSample(
            value: 0,
            confidence: 0.9,
            state: .fused,
            sampleCaptureTimestampNs: timestamp
          )
        )
      })
    let trackedPose = NTPTracked(
      value: NTPPose(
        parentSpace: .camera,
        lengthBasis: .headRelative,
        position: .zero,
        orientationXYZW: .identity
      ),
      confidence: 0.9,
      state: .fused,
      sampleCaptureTimestampNs: timestamp
    )
    func origin(_ x: Float) -> NTPTracked<NTPPosition3> {
      NTPTracked(
        value: NTPPosition3(
          space: .headLocal,
          lengthBasis: .headRelative,
          value: NTPVector3(x: x, y: 0, z: 0)
        ),
        confidence: 0.9,
        state: .fused,
        sampleCaptureTimestampNs: timestamp
      )
    }
    let direction = NTPTracked(
      value: NTPDirection3(space: .headLocal, value: NTPVector3(x: 0, y: 0, z: 1)),
      confidence: 0.9,
      state: .fused,
      sampleCaptureTimestampNs: timestamp
    )
    let lookAt = NTPTracked(
      value: NTPPosition3(
        space: .camera,
        lengthBasis: .headRelative,
        value: NTPVector3(x: 0, y: 0, z: 1)
      ),
      confidence: 0.9,
      state: .fused,
      sampleCaptureTimestampNs: timestamp
    )
    let geometry = NTPGeometryResult(
      headCameraPose: trackedPose,
      leftEye: NTPEyeGeometry(originHead: origin(-0.15), directionHead: direction),
      rightEye: NTPEyeGeometry(originHead: origin(0.15), directionHead: direction),
      lookAtCamera: lookAt,
      faceGeometryState: .fused
    )
    let quality = NTPTrackingQuality(
      overallConfidence: 0.9,
      face: NTPRegionQuality(confidence: 0.9, state: .fused),
      eyes: NTPRegionQuality(confidence: 0.9, state: .fused),
      torso: NTPRegionQuality(confidence: 0.9, state: .fused),
      arm: NTPSideMap(left: .unsupported, right: .unsupported),
      auricle: NTPSideMap(left: .unsupported, right: .unsupported)
    )
    let payload = NTPSpatialPayload(rig: rig, geometry: geometry, quality: quality)
    let descriptor = NTPDescriptor(
      guaranteedProfile: .spatial,
      supportedSignals: Array(UInt16(1)...UInt16(42)),
      supportedStructures: .spatialRequired
    )
    let producer = try NTPSpatialProducer(
      sessionID: [UInt8](repeating: 7, count: 16), descriptor: descriptor)
    let firstBytes = try await producer.encode(
      captureTimestampNs: timestamp,
      producedTimestampNs: timestamp + 1_000_000,
      payload: payload
    )
    let first = try NTPCanonicalCodec.decodeResult(firstBytes)
    try await producer.reconfigure()
    let second = try NTPCanonicalCodec.decodeResult(
      await producer.encode(
        captureTimestampNs: timestamp,
        producedTimestampNs: timestamp + 1_000_000,
        payload: payload
      ))
    guard first.generation == 0, first.sequence == 0, first.rig[42]?.value == 0,
      second.generation == 1, second.sequence == 0
    else {
      throw SelfTestError.spatialProducerLifecycleFailed
    }
    if let argument = CommandLine.arguments.firstIndex(of: "--emit-spatial-stream"),
      CommandLine.arguments.indices.contains(argument + 1)
    {
      var stream = try producer.encodeDescriptor()
      stream.append(firstBytes)
      try stream.write(
        to: URL(fileURLWithPath: CommandLine.arguments[argument + 1]), options: .atomic)
    }
  }

  private static func latestFrameOnlyPolicy() throws {
    let probe = LatestFrameProbe()
    let worker = NTPLatestFrameWorker<Int> { probe.process($0) }
    worker.submit(1)
    guard probe.firstStarted.wait(timeout: .now() + 2) == .success else {
      throw SelfTestError.latestFramePolicyFailed
    }
    worker.submit(2)
    worker.submit(3)
    probe.releaseFirst.signal()
    guard probe.completed.wait(timeout: .now() + 2) == .success,
      probe.snapshot() == [1, 3]
    else {
      throw SelfTestError.latestFramePolicyFailed
    }
  }

  private static func protocolVectorURL() throws -> URL {
    let relative = "crates/ntp-conformance/tests/vectors/protocol-golden-v1.hex"
    var root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    for _ in 0..<6 {
      let candidate = root.appending(path: relative)
      if FileManager.default.fileExists(atPath: candidate.path()) {
        return candidate
      }
      root.deleteLastPathComponent()
    }
    throw SelfTestError.protocolVectorNotFound
  }

  private static func decodeHex(_ value: Substring) throws -> Data {
    guard value.count.isMultiple(of: 2) else { throw SelfTestError.malformedProtocolVector }
    var bytes: [UInt8] = []
    bytes.reserveCapacity(value.count / 2)
    var index = value.startIndex
    while index < value.endIndex {
      let end = value.index(index, offsetBy: 2)
      guard let byte = UInt8(value[index..<end], radix: 16) else {
        throw SelfTestError.malformedProtocolVector
      }
      bytes.append(byte)
      index = end
    }
    return Data(bytes)
  }
}
