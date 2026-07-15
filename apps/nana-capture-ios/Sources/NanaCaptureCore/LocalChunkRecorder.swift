import CryptoKit
import Darwin
import Foundation

public enum CaptureChunkKind: String, Codable, Sendable {
  case rgb
  case depth
  case arkit
  case geometry
  case camera
}

public struct CaptureChunk: Codable, Equatable, Sendable {
  public let chunkID: String
  public let takeID: String
  public let kind: CaptureChunkKind
  public let relativePath: String
  public let sequenceStart: UInt64
  public let sequenceEnd: UInt64
  public let captureTimestampStartNs: UInt64
  public let captureTimestampEndNs: UInt64
  public let byteLength: Int
  public let sha256: String

  enum CodingKeys: String, CodingKey {
    case chunkID = "chunk_id"
    case takeID = "take_id"
    case kind
    case relativePath = "relative_path"
    case sequenceStart = "sequence_start"
    case sequenceEnd = "sequence_end"
    case captureTimestampStartNs = "capture_timestamp_start_ns"
    case captureTimestampEndNs = "capture_timestamp_end_ns"
    case byteLength = "byte_length"
    case sha256
  }
}

public struct ChunkAcknowledgement: Codable, Equatable, Sendable {
  public let chunkID: String
  public let sha256: String

  public init(chunkID: String, sha256: String) {
    self.chunkID = chunkID
    self.sha256 = sha256
  }

  enum CodingKeys: String, CodingKey {
    case chunkID = "chunk_id"
    case sha256
  }
}

public enum ChunkRecorderError: Error, Equatable {
  case emptyPayload
  case invalidRange
  case duplicateChunkID
  case descriptorMismatch
  case unknownAcknowledgement
  case acknowledgementDigestMismatch
}

public actor LocalChunkRecorder {
  private let root: URL
  private let state: URL
  private let chunksJournal: URL
  private let acknowledgementsJournal: URL
  private let encoder: JSONEncoder
  private var chunkList: [CaptureChunk]
  private var chunksByID: [String: CaptureChunk]
  private var chunkPaths: Set<String>
  private var acknowledgementList: [ChunkAcknowledgement]
  private var acknowledgementsByID: [String: ChunkAcknowledgement]

  public init(root: URL) throws {
    self.root = root.standardizedFileURL
    state = self.root.appending(path: ".capture-state", directoryHint: .isDirectory)
    chunksJournal = state.appending(path: "chunks.jsonl")
    acknowledgementsJournal = state.appending(path: "acknowledged.jsonl")
    encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    try FileManager.default.createDirectory(at: state, withIntermediateDirectories: true)
    chunkList = try Self.loadJournal(chunksJournal, as: CaptureChunk.self)
    chunksByID = Dictionary(uniqueKeysWithValues: chunkList.map { ($0.chunkID, $0) })
    chunkPaths = Set(chunkList.map(\.relativePath))
    acknowledgementList = try Self.loadJournal(
      acknowledgementsJournal,
      as: ChunkAcknowledgement.self
    )
    acknowledgementsByID = Dictionary(
      uniqueKeysWithValues: acknowledgementList.map { ($0.chunkID, $0) }
    )
    guard chunksByID.count == chunkList.count, chunkPaths.count == chunkList.count else {
      throw ChunkRecorderError.duplicateChunkID
    }
    guard acknowledgementsByID.count == acknowledgementList.count else {
      throw ChunkRecorderError.duplicateChunkID
    }
  }

  @discardableResult
  public func writeChunk(
    chunkID: String,
    takeID: String,
    kind: CaptureChunkKind,
    sequenceStart: UInt64,
    sequenceEnd: UInt64,
    captureTimestampStartNs: UInt64,
    captureTimestampEndNs: UInt64,
    payload: Data
  ) throws -> CaptureChunk {
    guard !payload.isEmpty else { throw ChunkRecorderError.emptyPayload }
    guard sequenceEnd >= sequenceStart,
      captureTimestampEndNs >= captureTimestampStartNs
    else { throw ChunkRecorderError.invalidRange }
    let relativePath = [
      "chunks", takeID, kind.rawValue,
      String(format: "%020llu-%020llu-%@.bin", sequenceStart, sequenceEnd, chunkID),
    ].joined(separator: "/")
    let chunk = CaptureChunk(
      chunkID: chunkID,
      takeID: takeID,
      kind: kind,
      relativePath: relativePath,
      sequenceStart: sequenceStart,
      sequenceEnd: sequenceEnd,
      captureTimestampStartNs: captureTimestampStartNs,
      captureTimestampEndNs: captureTimestampEndNs,
      byteLength: payload.count,
      sha256: Self.digest(payload)
    )
    return try receiveChunk(chunk, payload: payload)
  }

  @discardableResult
  public func receiveChunk(_ chunk: CaptureChunk, payload: Data) throws -> CaptureChunk {
    guard payload.count == chunk.byteLength, Self.digest(payload) == chunk.sha256 else {
      throw ChunkRecorderError.descriptorMismatch
    }
    let expectedPath = [
      "chunks", chunk.takeID, chunk.kind.rawValue,
      String(
        format: "%020llu-%020llu-%@.bin",
        chunk.sequenceStart,
        chunk.sequenceEnd,
        chunk.chunkID
      ),
    ].joined(separator: "/")
    guard chunk.relativePath == expectedPath,
      chunk.sequenceEnd >= chunk.sequenceStart,
      chunk.captureTimestampEndNs >= chunk.captureTimestampStartNs,
      Self.safeIdentifier(chunk.chunkID),
      Self.safeIdentifier(chunk.takeID)
    else {
      throw ChunkRecorderError.descriptorMismatch
    }
    if let existing = chunksByID[chunk.chunkID] {
      guard existing == chunk else { throw ChunkRecorderError.duplicateChunkID }
      return existing
    }
    guard !chunkPaths.contains(chunk.relativePath) else {
      throw ChunkRecorderError.duplicateChunkID
    }
    let destination = root.appending(path: chunk.relativePath)
    try durableWrite(payload, to: destination)
    try append(chunk, to: chunksJournal)
    chunkList.append(chunk)
    chunksByID[chunk.chunkID] = chunk
    chunkPaths.insert(chunk.relativePath)
    return chunk
  }

  public func acknowledge(_ acknowledgement: ChunkAcknowledgement) throws {
    guard let chunk = chunksByID[acknowledgement.chunkID] else {
      throw ChunkRecorderError.unknownAcknowledgement
    }
    guard chunk.sha256 == acknowledgement.sha256 else {
      throw ChunkRecorderError.acknowledgementDigestMismatch
    }
    if let previous = acknowledgementsByID[acknowledgement.chunkID] {
      guard previous == acknowledgement else {
        throw ChunkRecorderError.acknowledgementDigestMismatch
      }
      return
    }
    try append(acknowledgement, to: acknowledgementsJournal)
    acknowledgementList.append(acknowledgement)
    acknowledgementsByID[acknowledgement.chunkID] = acknowledgement
  }

  public func chunks() throws -> [CaptureChunk] {
    chunkList
  }

  public func acknowledgements() throws -> [ChunkAcknowledgement] {
    acknowledgementList
  }

  public func pendingChunks() throws -> [CaptureChunk] {
    let acknowledged = Set(try acknowledgements().map(\.chunkID))
    return try chunks().filter { !acknowledged.contains($0.chunkID) }
  }

  public func chunkURL(for chunk: CaptureChunk) throws -> URL {
    guard chunksByID[chunk.chunkID] == chunk else {
      throw ChunkRecorderError.descriptorMismatch
    }
    return root.appending(path: chunk.relativePath)
  }

  public func synchronizePending(using client: CaptureStudioClient) async throws -> Int {
    var synchronized = 0
    for chunk in try pendingChunks() {
      let acknowledgement = try await client.uploadChunk(
        chunk,
        file: try chunkURL(for: chunk)
      )
      try acknowledge(acknowledgement)
      synchronized += 1
    }
    return synchronized
  }

  public nonisolated static func digest(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }

  private nonisolated static func safeIdentifier(_ value: String) -> Bool {
    value.range(
      of: #"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"#,
      options: .regularExpression
    ) != nil
  }

  private func append<T: Encodable>(_ value: T, to journal: URL) throws {
    var line = try encoder.encode(value)
    line.append(0x0A)
    let created = !FileManager.default.fileExists(atPath: journal.path)
    if created {
      FileManager.default.createFile(atPath: journal.path, contents: nil)
    }
    let handle = try FileHandle(forWritingTo: journal)
    try handle.seekToEnd()
    try handle.write(contentsOf: line)
    try handle.synchronize()
    try handle.close()
    if created {
      try synchronizeDirectory(journal.deletingLastPathComponent())
    }
  }

  private static func loadJournal<T: Decodable>(_ journal: URL, as type: T.Type) throws -> [T] {
    guard FileManager.default.fileExists(atPath: journal.path) else { return [] }
    let decoder = JSONDecoder()
    return try Data(contentsOf: journal)
      .split(separator: 0x0A)
      .map { try decoder.decode(type, from: Data($0)) }
  }

  private func durableWrite(_ data: Data, to destination: URL) throws {
    try FileManager.default.createDirectory(
      at: destination.deletingLastPathComponent(),
      withIntermediateDirectories: true
    )
    let temporary = destination.deletingLastPathComponent().appending(
      path: ".\(destination.lastPathComponent).\(UUID().uuidString)"
    )
    FileManager.default.createFile(atPath: temporary.path, contents: nil)
    do {
      let handle = try FileHandle(forWritingTo: temporary)
      try handle.write(contentsOf: data)
      try handle.synchronize()
      try handle.close()
      if FileManager.default.fileExists(atPath: destination.path) {
        try FileManager.default.removeItem(at: destination)
      }
      try FileManager.default.moveItem(at: temporary, to: destination)
      try synchronizeDirectory(destination.deletingLastPathComponent())
    } catch {
      try? FileManager.default.removeItem(at: temporary)
      throw error
    }
  }

  private func synchronizeDirectory(_ directory: URL) throws {
    let descriptor = Darwin.open(directory.path, O_RDONLY)
    guard descriptor >= 0 else {
      throw CocoaError(.fileWriteUnknown)
    }
    defer { Darwin.close(descriptor) }
    guard Darwin.fsync(descriptor) == 0 else {
      throw CocoaError(.fileWriteUnknown)
    }
  }
}
