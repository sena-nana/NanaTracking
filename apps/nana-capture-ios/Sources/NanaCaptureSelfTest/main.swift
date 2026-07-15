import Foundation
import NanaCaptureCore

enum SelfTestError: Error {
  case unexpectedPendingChunks
  case invalidAcknowledgementAccepted
  case corruptReceiverPayloadAccepted
  case controlLifecycleFailed
  case controlContractRoundTripFailed
}

@main
struct NanaCaptureSelfTest {
  static func main() async throws {
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
      "NanaCaptureSelfTest passed: restart, receiver verification, and control lifecycle"
    )
  }
}
