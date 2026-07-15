import CryptoKit
import Foundation

public enum StudioControlAction: String, Codable, Sendable {
  case start
  case pause
  case stop
  case retake
  case end
}

public struct StudioControlCommand: Codable, Equatable, Sendable {
  public let schemaVersion: String
  public let sessionID: String
  public let revision: UInt64
  public let commandID: String
  public let action: StudioControlAction
  public let takeID: String?
  public let actionScriptID: String?
  public let retakeOf: String?
  public let issuedAtNs: UInt64

  public init(
    sessionID: String,
    revision: UInt64,
    action: StudioControlAction,
    takeID: String? = nil,
    actionScriptID: String? = nil,
    retakeOf: String? = nil,
    issuedAtNs: UInt64 = 0
  ) {
    schemaVersion = "nana-capture-control/1.0.0"
    self.sessionID = sessionID
    self.revision = revision
    commandID = "\(sessionID)-\(String(format: "%020llu", revision))"
    self.action = action
    self.takeID = takeID
    self.actionScriptID = actionScriptID
    self.retakeOf = retakeOf
    self.issuedAtNs = issuedAtNs
  }

  enum CodingKeys: String, CodingKey {
    case schemaVersion = "schema_version"
    case sessionID = "session_id"
    case revision
    case commandID = "command_id"
    case action
    case takeID = "take_id"
    case actionScriptID = "action_script_id"
    case retakeOf = "retake_of"
    case issuedAtNs = "issued_at_ns"
  }
}

public struct StudioCommandAcknowledgement: Codable, Equatable, Sendable {
  public let sessionID: String
  public let revision: UInt64
  public let commandID: String
  public let deviceID: String
  public let appliedAtNs: UInt64

  enum CodingKeys: String, CodingKey {
    case sessionID = "session_id"
    case revision
    case commandID = "command_id"
    case deviceID = "device_id"
    case appliedAtNs = "applied_at_ns"
  }
}

public struct StudioPreviewMetadata: Codable, Equatable, Sendable {
  public let sessionID: String
  public let takeID: String
  public let sequence: UInt64
  public let captureTimestampNs: UInt64
  public let byteLength: Int

  public init(
    sessionID: String,
    takeID: String,
    sequence: UInt64,
    captureTimestampNs: UInt64,
    byteLength: Int
  ) {
    self.sessionID = sessionID
    self.takeID = takeID
    self.sequence = sequence
    self.captureTimestampNs = captureTimestampNs
    self.byteLength = byteLength
  }

  enum CodingKeys: String, CodingKey {
    case sessionID = "session_id"
    case takeID = "take_id"
    case sequence
    case captureTimestampNs = "capture_timestamp_ns"
    case byteLength = "byte_length"
  }
}

public struct StudioQualitySample: Codable, Equatable, Sendable {
  public struct Point: Codable, Equatable, Sendable {
    public let x: Float
    public let y: Float

    public init(x: Float, y: Float) {
      self.x = x
      self.y = y
    }
  }

  public let sessionID: String
  public let takeID: String
  public let sequence: UInt64
  public let captureTimestampNs: UInt64
  public let luminance: Float
  public let clippedFraction: Float
  public let occludedFraction: Float
  public let trackingState: String
  public let faceMesh: [Point]
  public let parameters: [String: Float]

  public init(
    sessionID: String,
    takeID: String,
    sequence: UInt64,
    captureTimestampNs: UInt64,
    luminance: Float,
    clippedFraction: Float,
    occludedFraction: Float,
    trackingState: String,
    faceMesh: [Point] = [],
    parameters: [String: Float] = [:]
  ) {
    self.sessionID = sessionID
    self.takeID = takeID
    self.sequence = sequence
    self.captureTimestampNs = captureTimestampNs
    self.luminance = luminance
    self.clippedFraction = clippedFraction
    self.occludedFraction = occludedFraction
    self.trackingState = trackingState
    self.faceMesh = faceMesh
    self.parameters = parameters
  }

  enum CodingKeys: String, CodingKey {
    case sessionID = "session_id"
    case takeID = "take_id"
    case sequence
    case captureTimestampNs = "capture_timestamp_ns"
    case luminance
    case clippedFraction = "clipped_fraction"
    case occludedFraction = "occluded_fraction"
    case trackingState = "tracking_state"
    case faceMesh = "face_mesh"
    case parameters
  }
}

private struct StudioQualityResult: Decodable {
  let acceptable: Bool
}

public enum StudioTransportError: Error, Equatable {
  case invalidResponse
  case server(status: Int, message: String)
  case acknowledgementMismatch
  case fileMismatch
  case commandRevisionMismatch
  case invalidTransition
}

public actor StudioCaptureLifecycle {
  public enum State: String, Sendable {
    case ready
    case recording
    case paused
    case stopped
    case complete
  }

  public private(set) var state: State = .ready
  public private(set) var revision: UInt64 = 0
  public private(set) var currentTakeID: String?

  public init() {}

  public func apply(_ command: StudioControlCommand) throws {
    guard command.revision == revision + 1 else {
      throw StudioTransportError.commandRevisionMismatch
    }
    switch command.action {
    case .start:
      guard command.takeID != nil, command.actionScriptID != nil else {
        throw StudioTransportError.invalidTransition
      }
      if state == .paused, command.takeID != currentTakeID {
        throw StudioTransportError.invalidTransition
      }
      guard state == .ready || state == .paused || state == .stopped else {
        throw StudioTransportError.invalidTransition
      }
      currentTakeID = command.takeID
      state = .recording
    case .pause:
      guard state == .recording, command.takeID == currentTakeID else {
        throw StudioTransportError.invalidTransition
      }
      state = .paused
    case .stop:
      guard state == .recording || state == .paused, command.takeID == currentTakeID else {
        throw StudioTransportError.invalidTransition
      }
      state = .stopped
    case .retake:
      guard state == .ready || state == .stopped,
        command.takeID != nil,
        command.retakeOf != nil,
        command.actionScriptID != nil
      else {
        throw StudioTransportError.invalidTransition
      }
      currentTakeID = command.takeID
      state = .recording
    case .end:
      guard state == .ready || state == .stopped else {
        throw StudioTransportError.invalidTransition
      }
      currentTakeID = nil
      state = .complete
    }
    revision = command.revision
  }
}

public actor CaptureStudioClient {
  private let baseURL: URL
  private let token: String?
  private let session: URLSession
  private let encoder: JSONEncoder
  private let decoder: JSONDecoder

  public init(baseURL: URL, token: String? = nil, session: URLSession = .shared) {
    self.baseURL = baseURL
    self.token = token
    self.session = session
    encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    decoder = JSONDecoder()
  }

  public func commands(after revision: UInt64) async throws -> [StudioControlCommand] {
    var components = URLComponents(
      url: endpoint("api/commands"),
      resolvingAgainstBaseURL: false
    )!
    components.queryItems = [URLQueryItem(name: "after", value: String(revision))]
    var request = URLRequest(url: components.url!)
    request.httpMethod = "GET"
    authorize(&request)
    let data = try await perform(request)
    return try decoder.decode([StudioControlCommand].self, from: data)
  }

  public func applyPendingCommands(
    lifecycle: StudioCaptureLifecycle,
    deviceID: String,
    handler: @Sendable (StudioControlCommand) async throws -> Void
  ) async throws -> Int {
    let currentRevision = await lifecycle.revision
    let pending = try await commands(after: currentRevision)
    for command in pending {
      try await handler(command)
      try await lifecycle.apply(command)
      try await acknowledge(
        StudioCommandAcknowledgement(
          sessionID: command.sessionID,
          revision: command.revision,
          commandID: command.commandID,
          deviceID: deviceID,
          appliedAtNs: DispatchTime.now().uptimeNanoseconds
        )
      )
    }
    return pending.count
  }

  public func acknowledge(_ acknowledgement: StudioCommandAcknowledgement) async throws {
    let _: StudioCommandAcknowledgement = try await sendJSON(
      acknowledgement,
      to: "api/command-ack"
    )
  }

  @discardableResult
  public func publishQuality(_ sample: StudioQualitySample) async throws -> Bool {
    let result: StudioQualityResult = try await sendJSON(sample, to: "api/quality")
    return result.acceptable
  }

  public func publishPreview(_ jpeg: Data, metadata: StudioPreviewMetadata) async throws {
    guard jpeg.count == metadata.byteLength else { throw StudioTransportError.fileMismatch }
    var request = URLRequest(url: endpoint("api/preview"))
    request.httpMethod = "PUT"
    request.httpBody = jpeg
    request.setValue("image/jpeg", forHTTPHeaderField: "Content-Type")
    request.setValue(try header(metadata), forHTTPHeaderField: "X-Nana-Preview")
    authorize(&request)
    _ = try await perform(request)
  }

  public func uploadChunk(_ chunk: CaptureChunk, file: URL) async throws
    -> ChunkAcknowledgement
  {
    let attributes = try FileManager.default.attributesOfItem(atPath: file.path)
    guard let byteLength = attributes[.size] as? NSNumber,
      byteLength.intValue == chunk.byteLength,
      try Self.fileDigest(file) == chunk.sha256
    else {
      throw StudioTransportError.fileMismatch
    }
    var request = URLRequest(url: endpoint("api/chunks/\(chunk.chunkID)"))
    request.httpMethod = "PUT"
    request.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
    request.setValue(String(chunk.byteLength), forHTTPHeaderField: "Content-Length")
    request.setValue(try header(chunk), forHTTPHeaderField: "X-Nana-Chunk")
    authorize(&request)
    let (data, response) = try await session.upload(for: request, fromFile: file)
    try validate(response, data: data)
    let acknowledgement = try decoder.decode(ChunkAcknowledgement.self, from: data)
    guard acknowledgement.chunkID == chunk.chunkID, acknowledgement.sha256 == chunk.sha256 else {
      throw StudioTransportError.acknowledgementMismatch
    }
    return acknowledgement
  }

  private func sendJSON<Value: Encodable, Result: Decodable>(
    _ value: Value,
    to path: String
  ) async throws -> Result {
    var request = URLRequest(url: endpoint(path))
    request.httpMethod = "POST"
    request.httpBody = try encoder.encode(value)
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    authorize(&request)
    return try decoder.decode(Result.self, from: try await perform(request))
  }

  private func perform(_ request: URLRequest) async throws -> Data {
    let (data, response) = try await session.data(for: request)
    try validate(response, data: data)
    return data
  }

  private func validate(_ response: URLResponse, data: Data) throws {
    guard let response = response as? HTTPURLResponse else {
      throw StudioTransportError.invalidResponse
    }
    guard (200..<300).contains(response.statusCode) else {
      throw StudioTransportError.server(
        status: response.statusCode,
        message: String(data: data, encoding: .utf8) ?? ""
      )
    }
  }

  private func endpoint(_ path: String) -> URL {
    path.split(separator: "/").reduce(baseURL) { partial, component in
      partial.appending(path: String(component))
    }
  }

  private func authorize(_ request: inout URLRequest) {
    if let token {
      request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
    }
  }

  private func header<Value: Encodable>(_ value: Value) throws -> String {
    try encoder.encode(value).base64EncodedString()
      .replacingOccurrences(of: "+", with: "-")
      .replacingOccurrences(of: "/", with: "_")
      .replacingOccurrences(of: "=", with: "")
  }

  private nonisolated static func fileDigest(_ file: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: file)
    defer { try? handle.close() }
    var digest = SHA256()
    while let data = try handle.read(upToCount: 1024 * 1024), !data.isEmpty {
      digest.update(data: data)
    }
    return digest.finalize().map { String(format: "%02x", $0) }.joined()
  }
}
