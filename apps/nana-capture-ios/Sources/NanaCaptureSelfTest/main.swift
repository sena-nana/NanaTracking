import Foundation
import NanaCaptureCore

enum SelfTestError: Error {
  case protocolVectorNotFound
  case malformedProtocolVector
  case canonicalRoundTripFailed
  case spatialProducerLifecycleFailed
  case latestFramePolicyFailed
  case duplicateJournalAccepted
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
    let chunksJournal = root.appending(path: "source/.capture-state/chunks.jsonl")
    let tornHandle = try FileHandle(forWritingTo: chunksJournal)
    try tornHandle.seekToEnd()
    try tornHandle.write(contentsOf: Data(#"{"chunk_id":"torn"#.utf8))
    try tornHandle.close()
    let reopened = try LocalChunkRecorder(root: root.appending(path: "source"))
    guard try await reopened.pendingChunks() == [second] else {
      throw SelfTestError.unexpectedPendingChunks
    }
    guard try Data(contentsOf: chunksJournal).last == 0x0A else {
      throw SelfTestError.unexpectedPendingChunks
    }
    let duplicateRoot = root.appending(path: "duplicate-source")
    try FileManager.default.copyItem(at: root.appending(path: "source"), to: duplicateRoot)
    let duplicateJournal = duplicateRoot.appending(path: ".capture-state/chunks.jsonl")
    let duplicateHandle = try FileHandle(forWritingTo: duplicateJournal)
    try duplicateHandle.seekToEnd()
    var duplicateLine = try JSONEncoder().encode(first)
    duplicateLine.append(0x0A)
    try duplicateHandle.write(contentsOf: duplicateLine)
    try duplicateHandle.close()
    do {
      _ = try LocalChunkRecorder(root: duplicateRoot)
      throw SelfTestError.duplicateJournalAccepted
    } catch ChunkRecorderError.duplicateChunkID {
      // Expected: corrupt duplicate metadata is rejected without a Dictionary runtime trap.
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
        + "exact-frame Spatial fusion, producer generations, restart, receiver verification, "
        + "and control lifecycle"
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
    let sessionID = [UInt8](repeating: 7, count: 16)
    let referenceDescriptor = NTPDescriptor(
      guaranteedProfile: .spatial,
      supportedSignals: Array(UInt16(1)...UInt16(41)),
      supportedStructures: .spatialRequired
    )
    let supplementDescriptor = NTPDescriptor(
      guaranteedProfile: .spatial,
      supportedSignals: Array(UInt16(1)...UInt16(42)),
      supportedStructures: .spatialRequired
    )
    let fusionPlan = try NTPSpatialFusionPlan(
      referenceDescriptor: referenceDescriptor,
      extensionDescriptor: supplementDescriptor
    )
    let referenceRig = Dictionary(
      uniqueKeysWithValues: (UInt16(1)...UInt16(41)).map { signalID in
        let value: Float? = signalID == 41 ? nil : (signalID == 37 ? 0.2 : 0)
        return (
          signalID,
          NTPSignalSample(
            value: value,
            confidence: signalID == 41 ? 0.2 : 0.7,
            state: value == nil ? .occluded : .observed,
            sampleCaptureTimestampNs: timestamp
          )
        )
      })
    let supplementRig = Dictionary(
      uniqueKeysWithValues: (UInt16(1)...UInt16(42)).map { signalID in
        let value: Float =
          switch signalID {
          case 1: 0.02
          case 37: -0.5
          case 41: 0.7
          case 42: 0.4
          default: 0
          }
        return (
          signalID,
          NTPSignalSample(
            value: value,
            confidence: 0.9,
            state: .observed,
            sampleCaptureTimestampNs: timestamp
          )
        )
      })
    func pose(_ x: Float) -> NTPTracked<NTPPose> {
      NTPTracked(
        value: NTPPose(
          parentSpace: .camera,
          lengthBasis: .headRelative,
          position: NTPVector3(x: x, y: 0, z: 0),
          orientationXYZW: .identity
        ),
        confidence: 0.9,
        state: .observed,
        sampleCaptureTimestampNs: timestamp
      )
    }
    func origin(_ x: Float) -> NTPTracked<NTPPosition3> {
      NTPTracked(
        value: NTPPosition3(
          space: .headLocal,
          lengthBasis: .headRelative,
          value: NTPVector3(x: x, y: 0, z: 0)
        ),
        confidence: 0.9,
        state: .observed,
        sampleCaptureTimestampNs: timestamp
      )
    }
    let direction = NTPTracked(
      value: NTPDirection3(space: .headLocal, value: NTPVector3(x: 0, y: 0, z: 1)),
      confidence: 0.9,
      state: .observed,
      sampleCaptureTimestampNs: timestamp
    )
    let lookAt = NTPTracked(
      value: NTPPosition3(
        space: .camera,
        lengthBasis: .headRelative,
        value: NTPVector3(x: 0, y: 0, z: 1)
      ),
      confidence: 0.9,
      state: .observed,
      sampleCaptureTimestampNs: timestamp
    )
    func geometry(_ x: Float) -> NTPGeometryResult {
      NTPGeometryResult(
        headCameraPose: pose(x),
        leftEye: NTPEyeGeometry(originHead: origin(-0.15), directionHead: direction),
        rightEye: NTPEyeGeometry(originHead: origin(0.15), directionHead: direction),
        lookAtCamera: lookAt,
        faceGeometryState: .observed
      )
    }
    func quality(torso: NTPRegionQuality) -> NTPTrackingQuality {
      NTPTrackingQuality(
        overallConfidence: 0.9,
        face: NTPRegionQuality(confidence: 0.9, state: .observed),
        eyes: NTPRegionQuality(confidence: 0.9, state: .observed),
        torso: torso,
        arm: NTPSideMap(left: .unsupported, right: .unsupported),
        auricle: NTPSideMap(left: .unsupported, right: .unsupported)
      )
    }
    let reference = NTPTrackingResult(
      sessionID: sessionID,
      generation: 0,
      sequence: 0,
      captureTimestampNs: timestamp,
      producedTimestampNs: timestamp + 1_000_000,
      rig: referenceRig,
      geometry: geometry(0),
      quality: quality(torso: .unsupported)
    )
    let supplement = NTPTrackingResult(
      sessionID: sessionID,
      generation: 0,
      sequence: 0,
      captureTimestampNs: timestamp,
      producedTimestampNs: timestamp + 2_000_000,
      rig: supplementRig,
      geometry: geometry(0.5),
      quality: quality(torso: NTPRegionQuality(confidence: 0.9, state: .observed))
    )
    let fused = try fusionPlan.fuse(reference: reference, supplement: supplement)
    guard fused.rig[1]?.value == 0, fused.rig[1]?.state == .fused,
      fused.rig[37]?.value == 0.2, fused.rig[41]?.value == 0.7,
      fused.rig[42]?.value == 0.4, fused.quality.torso.state == .observed,
      fused.geometry.headCameraPose.value?.position.x == 0,
      fused.geometry.headCameraPose.state == .fused
    else {
      throw SelfTestError.spatialProducerLifecycleFailed
    }
    let mismatchedSupplement = NTPTrackingResult(
      sessionID: sessionID,
      generation: 0,
      sequence: 1,
      captureTimestampNs: timestamp,
      producedTimestampNs: supplement.producedTimestampNs,
      rig: supplement.rig,
      geometry: supplement.geometry,
      quality: supplement.quality
    )
    do {
      _ = try fusionPlan.fuse(reference: reference, supplement: mismatchedSupplement)
      throw SelfTestError.spatialProducerLifecycleFailed
    } catch NTPSpatialFusionError.sequenceMismatch {
      // Expected: arrival order cannot join distinct capture identities.
    }
    let producer = try NTPSpatialProducer(
      sessionID: sessionID, descriptor: fusionPlan.descriptor)
    let firstBytes = try await producer.encode(fused)
    let first = try NTPCanonicalCodec.decodeResult(firstBytes)
    try await producer.reconfigure()
    let nextGeneration = NTPTrackingResult(
      sessionID: sessionID,
      generation: 1,
      sequence: 0,
      captureTimestampNs: timestamp,
      producedTimestampNs: fused.producedTimestampNs,
      rig: fused.rig,
      geometry: fused.geometry,
      quality: fused.quality
    )
    let second = try NTPCanonicalCodec.decodeResult(
      await producer.encode(nextGeneration))
    guard first.generation == 0, first.sequence == 0, first.rig[42]?.value == 0.4,
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
