#if os(iOS) && canImport(ARKit)
  import ARKit
  import CoreImage
  import Foundation
  import simd

  public struct CaptureIdentity: Sendable {
    public let subjectID: String
    public let sessionID: String
    public let takeID: String
    public let actionScriptID: String
    public let consentRecordID: String
    public let deviceID: String

    public init(
      subjectID: String,
      sessionID: String,
      takeID: String,
      actionScriptID: String,
      consentRecordID: String,
      deviceID: String
    ) {
      self.subjectID = subjectID
      self.sessionID = sessionID
      self.takeID = takeID
      self.actionScriptID = actionScriptID
      self.consentRecordID = consentRecordID
      self.deviceID = deviceID
    }
  }

  public struct ExposureMetadata: Codable, Sendable {
    public let exposureDurationNs: UInt64
    public let iso: Float
    public let frameDurationNs: UInt64

    public init(exposureDurationNs: UInt64, iso: Float, frameDurationNs: UInt64) {
      self.exposureDurationNs = exposureDurationNs
      self.iso = iso
      self.frameDurationNs = frameDurationNs
    }
  }

  public struct CaptureConditions: Codable, Sendable {
    public let lighting: String
    public let occlusions: [String]

    public init(lighting: String, occlusions: [String] = []) {
      self.lighting = lighting
      self.occlusions = occlusions
    }
  }

  public struct DepthPayload: Sendable {
    public let data: Data
    public let confidence: Float

    public init(data: Data, confidence: Float) {
      self.data = data
      self.confidence = confidence
    }
  }

  private struct RGBRecord: Codable {
    let uri: String
    let width: Int
    let height: Int
    let exposureDurationNs: UInt64
    let iso: Float
    let frameDurationNs: UInt64
  }

  private struct CameraRecord: Codable {
    let intrinsics: [Float]
    let distortionModel: String
    let distortionCoefficients: [Float]
  }

  private struct ConditionsRecord: Codable {
    let lighting: String
    let occlusions: [String]
  }

  private struct RawARKitRecord: Codable {
    let schemaVersion = "nana-raw-arkit-frame/1.0.0"
    let recordID: String
    let subjectID: String
    let sessionID: String
    let takeID: String
    let deviceID: String
    let actionScriptID: String
    let consentRecordID: String
    let captureTimestampNs: UInt64
    let sequence: UInt64
    let rgb: RGBRecord
    let camera: CameraRecord
    let blendshapes: [String: Float]
    let headTransformColumnMajor: [Float]
    let leftEyeTransformColumnMajor: [Float]
    let rightEyeTransformColumnMajor: [Float]
    let faceGeometryURI: String
    let depthURI: String?
    let depthConfidence: Float
    let trackingState: String
    let conditions: ConditionsRecord
  }

  public enum ARKitCaptureError: Error {
    case missingFaceAnchor
    case jpegEncodingFailed
  }

  public actor ARKitCapturePipeline {
    private let recorder: LocalChunkRecorder
    private let identity: CaptureIdentity
    private let context = CIContext(options: [.cacheIntermediates: false])
    private let encoder: JSONEncoder

    public init(recorder: LocalChunkRecorder, identity: CaptureIdentity) {
      self.recorder = recorder
      self.identity = identity
      encoder = JSONEncoder()
      encoder.keyEncodingStrategy = .convertToSnakeCase
      encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    }

    public func capture(
      frame: ARFrame,
      sequence: UInt64,
      exposure: ExposureMetadata,
      conditions: CaptureConditions,
      depth: DepthPayload? = nil
    ) async throws -> [CaptureChunk] {
      guard let face = frame.anchors.compactMap({ $0 as? ARFaceAnchor }).first else {
        throw ARKitCaptureError.missingFaceAnchor
      }
      let timestampNs = UInt64(frame.timestamp * 1_000_000_000.0)
      let recordID = "\(identity.sessionID)-\(String(format: "%020llu", sequence))"
      let rgbRelative = "chunks/\(identity.takeID)/rgb/\(recordID).jpg"
      let geometryRelative = "chunks/\(identity.takeID)/geometry/\(recordID).json"
      let depthRelative = depth.map { _ in "chunks/\(identity.takeID)/depth/\(recordID).bin" }
      let image = CIImage(cvPixelBuffer: frame.capturedImage)
      let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
      guard
        let jpeg = context.jpegRepresentation(
          of: image,
          colorSpace: colorSpace,
          options: [.lossyCompressionQuality: 0.95]
        )
      else {
        throw ARKitCaptureError.jpegEncodingFailed
      }
      let geometry = try encoder.encode([
        "vertices": face.geometry.vertices.flatMap { [$0.x, $0.y, $0.z] },
        "texture_coordinates": face.geometry.textureCoordinates.flatMap { [$0.x, $0.y] },
        "triangle_indices": face.geometry.triangleIndices.map(Float.init),
      ])
      let raw = RawARKitRecord(
        recordID: recordID,
        subjectID: identity.subjectID,
        sessionID: identity.sessionID,
        takeID: identity.takeID,
        deviceID: identity.deviceID,
        actionScriptID: identity.actionScriptID,
        consentRecordID: identity.consentRecordID,
        captureTimestampNs: timestampNs,
        sequence: sequence,
        rgb: RGBRecord(
          uri: rgbRelative,
          width: CVPixelBufferGetWidth(frame.capturedImage),
          height: CVPixelBufferGetHeight(frame.capturedImage),
          exposureDurationNs: exposure.exposureDurationNs,
          iso: exposure.iso,
          frameDurationNs: exposure.frameDurationNs
        ),
        camera: CameraRecord(
          intrinsics: flatten(frame.camera.intrinsics),
          distortionModel: "none",
          distortionCoefficients: []
        ),
        blendshapes: Dictionary(
          uniqueKeysWithValues: face.blendShapes.map { ($0.key.rawValue, $0.value.floatValue) }
        ),
        headTransformColumnMajor: flatten(face.transform),
        leftEyeTransformColumnMajor: flatten(face.leftEyeTransform),
        rightEyeTransformColumnMajor: flatten(face.rightEyeTransform),
        faceGeometryURI: geometryRelative,
        depthURI: depthRelative,
        depthConfidence: depth?.confidence ?? 0.0,
        trackingState: trackingState(frame.camera.trackingState),
        conditions: ConditionsRecord(
          lighting: conditions.lighting,
          occlusions: conditions.occlusions
        )
      )
      var chunks: [CaptureChunk] = []
      chunks.append(
        try await recorder.writeChunk(
          chunkID: "\(recordID)-rgb",
          takeID: identity.takeID,
          kind: .rgb,
          sequenceStart: sequence,
          sequenceEnd: sequence,
          captureTimestampStartNs: timestampNs,
          captureTimestampEndNs: timestampNs,
          payload: jpeg
        ))
      chunks.append(
        try await recorder.writeChunk(
          chunkID: "\(recordID)-geometry",
          takeID: identity.takeID,
          kind: .geometry,
          sequenceStart: sequence,
          sequenceEnd: sequence,
          captureTimestampStartNs: timestampNs,
          captureTimestampEndNs: timestampNs,
          payload: geometry
        ))
      if let depth {
        chunks.append(
          try await recorder.writeChunk(
            chunkID: "\(recordID)-depth",
            takeID: identity.takeID,
            kind: .depth,
            sequenceStart: sequence,
            sequenceEnd: sequence,
            captureTimestampStartNs: timestampNs,
            captureTimestampEndNs: timestampNs,
            payload: depth.data
          ))
      }
      chunks.append(
        try await recorder.writeChunk(
          chunkID: "\(recordID)-arkit",
          takeID: identity.takeID,
          kind: .arkit,
          sequenceStart: sequence,
          sequenceEnd: sequence,
          captureTimestampStartNs: timestampNs,
          captureTimestampEndNs: timestampNs,
          payload: try encoder.encode(raw)
        ))
      return chunks
    }
  }

  private func flatten(_ matrix: simd_float4x4) -> [Float] {
    (0..<4).flatMap { column in
      (0..<4).map { row in matrix[column][row] }
    }
  }

  private func flatten(_ matrix: simd_float3x3) -> [Float] {
    (0..<3).flatMap { column in
      (0..<3).map { row in matrix[column][row] }
    }
  }

  private func trackingState(_ state: ARCamera.TrackingState) -> String {
    switch state {
    case .normal: "normal"
    case .limited: "limited"
    case .notAvailable: "not_available"
    }
  }
#endif
