#if os(iOS) && canImport(ARKit)
  @preconcurrency import ARKit
  @preconcurrency import AVFoundation
  import CoreImage
  import CoreVideo
  import Foundation
  import ImageIO
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
    public let captureTimestampNs: UInt64
    public let width: Int
    public let height: Int
    public let pixelFormat: String
    public let confidence: Float
    public let confidenceSource: String
    public let accuracy: String
    public let quality: String
    public let filtered: Bool

    public init(
      data: Data,
      captureTimestampNs: UInt64,
      width: Int,
      height: Int,
      pixelFormat: String = "float32-le-meters",
      confidence: Float,
      confidenceSource: String,
      accuracy: String,
      quality: String,
      filtered: Bool
    ) {
      self.data = data
      self.captureTimestampNs = captureTimestampNs
      self.width = width
      self.height = height
      self.pixelFormat = pixelFormat
      self.confidence = confidence
      self.confidenceSource = confidenceSource
      self.accuracy = accuracy
      self.quality = quality
      self.filtered = filtered
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
    let schemaVersion = "nana-raw-arkit-frame/1.1.0"
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
    let depthCaptureTimestampNs: UInt64?
    let depthWidth: Int?
    let depthHeight: Int?
    let depthPixelFormat: String?
    let depthConfidenceSource: String
    let depthAccuracy: String?
    let depthQuality: String?
    let depthFiltered: Bool?
    let trackingState: String
    let conditions: ConditionsRecord
  }

  public enum ARKitCaptureError: Error {
    case cameraPermissionDenied
    case captureIdentityMismatch
    case depthEncodingFailed
    case exposureMetadataUnavailable
    case invalidControlCommand
    case missingFaceAnchor
    case jpegEncodingFailed
    case sequenceOverflow
    case unsupportedDevice
  }

  public actor ARKitCapturePipeline {
    private let recorder: LocalChunkRecorder
    private var identity: CaptureIdentity
    private let context = CIContext(options: [.cacheIntermediates: false])
    private let encoder: JSONEncoder

    public init(recorder: LocalChunkRecorder, identity: CaptureIdentity) {
      self.recorder = recorder
      self.identity = identity
      encoder = JSONEncoder()
      encoder.keyEncodingStrategy = .convertToSnakeCase
      encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    }

    public func setIdentity(_ identity: CaptureIdentity) {
      self.identity = identity
    }

    public func capture(
      frame: ARFrame,
      sequence: UInt64,
      exposure: ExposureMetadata,
      conditions: CaptureConditions,
      depth: DepthPayload? = nil
    ) async throws -> [CaptureChunk] {
      let identity = identity
      guard let face = frame.anchors.compactMap({ $0 as? ARFaceAnchor }).first else {
        throw ARKitCaptureError.missingFaceAnchor
      }
      let timestampNs = UInt64(frame.timestamp * 1_000_000_000.0)
      let recordID = "\(identity.sessionID)-\(String(format: "%020llu", sequence))"
      let rgbRelative = LocalChunkRecorder.relativePath(
        chunkID: "\(recordID)-rgb",
        takeID: identity.takeID,
        kind: .rgb,
        sequenceStart: sequence,
        sequenceEnd: sequence
      )
      let geometryRelative = LocalChunkRecorder.relativePath(
        chunkID: "\(recordID)-geometry",
        takeID: identity.takeID,
        kind: .geometry,
        sequenceStart: sequence,
        sequenceEnd: sequence
      )
      let depthRelative = depth.map { _ in
        LocalChunkRecorder.relativePath(
          chunkID: "\(recordID)-depth",
          takeID: identity.takeID,
          kind: .depth,
          sequenceStart: sequence,
          sequenceEnd: sequence
        )
      }
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
        depthCaptureTimestampNs: depth?.captureTimestampNs,
        depthWidth: depth?.width,
        depthHeight: depth?.height,
        depthPixelFormat: depth?.pixelFormat,
        depthConfidenceSource: depth?.confidenceSource ?? "unavailable",
        depthAccuracy: depth?.accuracy,
        depthQuality: depth?.quality,
        depthFiltered: depth?.filtered,
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
            captureTimestampStartNs: depth.captureTimestampNs,
            captureTimestampEndNs: depth.captureTimestampNs,
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

  private struct PendingARFrame: @unchecked Sendable {
    let frame: ARFrame
    let sequence: UInt64
    let frameDurationNs: UInt64
  }

  /// Real `ARSession` driver for the durable capture pipeline.
  ///
  /// The delegate callback only assigns a sequence and replaces one pending frame. Image/depth
  /// encoding and durable writes run on a single asynchronous worker. Studio commands can use
  /// `apply` as the handler passed to `CaptureStudioClient.applyPendingCommands`.
  public final class ARKitCaptureSessionController: NSObject, ARSessionDelegate,
    @unchecked Sendable
  {
    public typealias ErrorHandler = @Sendable (Error) -> Void

    private let errorHandler: ErrorHandler
    private let captureDevice: AVCaptureDevice?
    private let lock = NSLock()
    private let pipeline: ARKitCapturePipeline
    private let session: ARSession
    private var acceptingFrames = false
    private var frameDurationNs: UInt64 = 0
    private var identity: CaptureIdentity
    private var nextSequence: UInt64 = 0
    private var worker: NTPAsyncLatestFrameWorker<PendingARFrame>!

    public init(
      recorder: LocalChunkRecorder,
      identity: CaptureIdentity,
      session: ARSession = ARSession(),
      errorHandler: @escaping ErrorHandler
    ) {
      self.identity = identity
      self.session = session
      self.errorHandler = errorHandler
      captureDevice = AVCaptureDevice.default(
        .builtInTrueDepthCamera,
        for: .video,
        position: .front
      )
      pipeline = ARKitCapturePipeline(recorder: recorder, identity: identity)
      super.init()
      worker = NTPAsyncLatestFrameWorker { [weak self] pending in
        await self?.process(pending)
      }
      self.session.delegate = self
      self.session.delegateQueue = DispatchQueue(
        label: "org.nanatracking.capture.arkit-delegate",
        qos: .userInteractive
      )
    }

    public func apply(_ command: StudioControlCommand) async throws {
      let sessionID = lock.withLock { identity.sessionID }
      guard command.sessionID == sessionID else {
        throw ARKitCaptureError.captureIdentityMismatch
      }
      switch command.action {
      case .start, .retake:
        guard let takeID = command.takeID, let actionScriptID = command.actionScriptID else {
          throw ARKitCaptureError.invalidControlCommand
        }
        try await start(
          takeID: takeID, actionScriptID: actionScriptID, reset: command.action == .retake)
      case .pause:
        try validateTake(command.takeID)
        await pause()
      case .stop:
        try validateTake(command.takeID)
        await stop()
      case .end:
        await stop()
      }
    }

    public func start(
      takeID: String,
      actionScriptID: String,
      reset: Bool = false
    ) async throws {
      guard ARFaceTrackingConfiguration.isSupported else {
        throw ARKitCaptureError.unsupportedDevice
      }
      guard await Self.cameraAccessGranted() else {
        throw ARKitCaptureError.cameraPermissionDenied
      }
      lock.withLock { acceptingFrames = false }
      session.pause()
      await worker.flush()
      let previous = lock.withLock { identity }
      let updated = CaptureIdentity(
        subjectID: previous.subjectID,
        sessionID: previous.sessionID,
        takeID: takeID,
        actionScriptID: actionScriptID,
        consentRecordID: previous.consentRecordID,
        deviceID: previous.deviceID
      )
      await pipeline.setIdentity(updated)
      lock.withLock {
        identity = updated
        acceptingFrames = true
      }
      let configuration = ARFaceTrackingConfiguration()
      configuration.isLightEstimationEnabled = true
      configuration.maximumNumberOfTrackedFaces = 1
      let framesPerSecond = max(1, configuration.videoFormat.framesPerSecond)
      let durationNs = 1_000_000_000 / UInt64(framesPerSecond)
      lock.withLock { frameDurationNs = durationNs }
      let options: ARSession.RunOptions = reset ? [.resetTracking, .removeExistingAnchors] : []
      session.run(configuration, options: options)
    }

    public func pause() async {
      lock.withLock { acceptingFrames = false }
      session.pause()
      await worker.flush()
    }

    public func stop() async {
      await pause()
    }

    public func droppedFrameCount() -> UInt64 {
      worker.droppedCount()
    }

    public func session(_ session: ARSession, didUpdate frame: ARFrame) {
      let submission: (UInt64, UInt64)? = lock.withLock {
        guard acceptingFrames else { return nil }
        guard nextSequence < UInt64.max else {
          acceptingFrames = false
          return nil
        }
        let sequence = nextSequence
        nextSequence += 1
        return (sequence, frameDurationNs)
      }
      guard let (sequence, frameDurationNs) = submission else {
        if lock.withLock({ nextSequence == UInt64.max }) {
          errorHandler(ARKitCaptureError.sequenceOverflow)
        }
        return
      }
      worker.submit(
        PendingARFrame(
          frame: frame,
          sequence: sequence,
          frameDurationNs: frameDurationNs
        )
      )
    }

    public func session(_ session: ARSession, didFailWithError error: Error) {
      lock.withLock { acceptingFrames = false }
      errorHandler(error)
    }

    public func sessionWasInterrupted(_ session: ARSession) {
      lock.withLock { acceptingFrames = false }
    }

    private func process(_ pending: PendingARFrame) async {
      do {
        _ = try await pipeline.capture(
          frame: pending.frame,
          sequence: pending.sequence,
          exposure: try exposureMetadata(
            pending.frame,
            frameDurationNs: pending.frameDurationNs,
            captureDevice: captureDevice
          ),
          conditions: captureConditions(pending.frame),
          depth: try depthPayload(pending.frame)
        )
      } catch {
        errorHandler(error)
      }
    }

    private func validateTake(_ takeID: String?) throws {
      guard takeID == lock.withLock({ identity.takeID }) else {
        throw ARKitCaptureError.captureIdentityMismatch
      }
    }

    private static func cameraAccessGranted() async -> Bool {
      switch AVCaptureDevice.authorizationStatus(for: .video) {
      case .authorized:
        true
      case .notDetermined:
        await withCheckedContinuation { continuation in
          AVCaptureDevice.requestAccess(for: .video) { granted in
            continuation.resume(returning: granted)
          }
        }
      case .denied, .restricted:
        false
      @unknown default:
        false
      }
    }
  }

  private func exposureMetadata(
    _ frame: ARFrame,
    frameDurationNs: UInt64,
    captureDevice: AVCaptureDevice?
  ) throws -> ExposureMetadata {
    if #available(iOS 18.0, *),
      let exposureSeconds = frame.exifData[kCGImagePropertyExifExposureTime as String] as? NSNumber,
      let isoValues = frame.exifData[kCGImagePropertyExifISOSpeedRatings as String] as? [NSNumber],
      let iso = isoValues.first
    {
      return ExposureMetadata(
        exposureDurationNs: UInt64(max(0, exposureSeconds.doubleValue) * 1_000_000_000),
        iso: iso.floatValue,
        frameDurationNs: frameDurationNs
      )
    }
    if let captureDevice {
      let exposureSeconds = CMTimeGetSeconds(captureDevice.exposureDuration)
      guard exposureSeconds.isFinite, exposureSeconds >= 0, captureDevice.iso.isFinite else {
        throw ARKitCaptureError.exposureMetadataUnavailable
      }
      return ExposureMetadata(
        exposureDurationNs: UInt64(exposureSeconds * 1_000_000_000),
        iso: captureDevice.iso,
        frameDurationNs: frameDurationNs
      )
    }
    throw ARKitCaptureError.exposureMetadataUnavailable
  }

  private func captureConditions(_ frame: ARFrame) -> CaptureConditions {
    let lighting: String
    if let intensity = frame.lightEstimate?.ambientIntensity {
      lighting = intensity < 250 ? "low" : intensity > 1_500 ? "bright" : "normal"
    } else {
      lighting = "unavailable"
    }
    return CaptureConditions(lighting: lighting)
  }

  private func depthPayload(_ frame: ARFrame) throws -> DepthPayload? {
    guard let captured = frame.capturedDepthData else { return nil }
    guard
      captured.availableDepthDataTypes.contains(
        NSNumber(value: kCVPixelFormatType_DepthFloat32)
      )
    else {
      throw ARKitCaptureError.depthEncodingFailed
    }
    let depth =
      captured.depthDataType == kCVPixelFormatType_DepthFloat32
      ? captured
      : captured.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
    let pixelBuffer = depth.depthDataMap
    guard CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly) == kCVReturnSuccess else {
      throw ARKitCaptureError.depthEncodingFailed
    }
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
    let width = CVPixelBufferGetWidth(pixelBuffer)
    let height = CVPixelBufferGetHeight(pixelBuffer)
    let rowBytes = width * MemoryLayout<Float32>.stride
    let sourceRowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
    guard sourceRowBytes >= rowBytes, let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
      throw ARKitCaptureError.depthEncodingFailed
    }
    var data = Data(capacity: rowBytes * height)
    for row in 0..<height {
      data.append(
        base.advanced(by: row * sourceRowBytes).assumingMemoryBound(to: UInt8.self),
        count: rowBytes
      )
    }
    return DepthPayload(
      data: data,
      captureTimestampNs: UInt64(max(0, frame.capturedDepthDataTimestamp) * 1_000_000_000),
      width: width,
      height: height,
      confidence: 0.0,
      confidenceSource: "unavailable",
      accuracy: depth.depthDataAccuracy == .absolute ? "absolute" : "relative",
      quality: depth.depthDataQuality == .high ? "high" : "low",
      filtered: depth.isDepthDataFiltered
    )
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
