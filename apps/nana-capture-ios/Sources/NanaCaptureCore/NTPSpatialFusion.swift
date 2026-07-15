import Foundation

public struct NTPSpatialFusionPolicy: Equatable, Sendable {
  public let agreementTolerance: Float
  public let confidenceSwitchMargin: Float

  public init(
    agreementTolerance: Float = 0.08,
    confidenceSwitchMargin: Float = 0.12
  ) {
    self.agreementTolerance = agreementTolerance
    self.confidenceSwitchMargin = confidenceSwitchMargin
  }
}

public enum NTPSpatialFusionError: Error, Equatable {
  case invalidPolicy
  case revisionMismatch
  case sessionMismatch
  case generationMismatch
  case sequenceMismatch
  case captureTimestampMismatch
  case referenceContract(NTPSpatialProducerError)
  case referenceCodec(NTPCodecError)
  case extensionContract(NTPSpatialProducerError)
  case extensionCodec(NTPCodecError)
  case outputContract(NTPSpatialProducerError)
  case outputCodec(NTPCodecError)
}

/// Prevalidated same-capture fusion plan for normalized Spatial NTP results.
///
/// Sensor SDK fields and model tensors stay outside this boundary. Descriptor union and policy
/// validation happen once when a stream generation is configured; each frame is then matched by
/// session, generation, sequence, and capture timestamp before any values are combined.
public struct NTPSpatialFusionPlan: Sendable {
  public let descriptor: NTPDescriptor

  private let extensionDescriptor: NTPDescriptor
  private let policy: NTPSpatialFusionPolicy
  private let referenceDescriptor: NTPDescriptor

  public init(
    referenceDescriptor: NTPDescriptor,
    extensionDescriptor: NTPDescriptor,
    policy: NTPSpatialFusionPolicy = NTPSpatialFusionPolicy()
  ) throws {
    guard policy.agreementTolerance.isFinite,
      policy.agreementTolerance >= 0,
      policy.confidenceSwitchMargin.isFinite,
      (0...1).contains(policy.confidenceSwitchMargin)
    else {
      throw NTPSpatialFusionError.invalidPolicy
    }
    try Self.validateDescriptor(referenceDescriptor, source: .reference)
    try Self.validateDescriptor(extensionDescriptor, source: .supplement)
    guard referenceDescriptor.revisions == extensionDescriptor.revisions else {
      throw NTPSpatialFusionError.revisionMismatch
    }
    let supportedSignals = Array(
      Set(referenceDescriptor.supportedSignals).union(extensionDescriptor.supportedSignals)
    ).sorted()
    let output = NTPDescriptor(
      revisions: referenceDescriptor.revisions,
      guaranteedProfile: .spatial,
      supportedSignals: supportedSignals,
      supportedStructures: NTPStructureFeatures(
        rawValue: referenceDescriptor.supportedStructures.rawValue
          | extensionDescriptor.supportedStructures.rawValue
      ),
      features: NTPTrackingFeatures(
        rawValue: referenceDescriptor.features.rawValue | extensionDescriptor.features.rawValue
      )
    )
    try Self.validateDescriptor(output, source: .output)
    self.referenceDescriptor = referenceDescriptor
    self.extensionDescriptor = extensionDescriptor
    self.policy = policy
    descriptor = output
  }

  public func fuse(
    reference: NTPTrackingResult,
    supplement: NTPTrackingResult
  ) throws -> NTPTrackingResult {
    try Self.validateResult(reference, descriptor: referenceDescriptor, source: .reference)
    try Self.validateResult(supplement, descriptor: extensionDescriptor, source: .supplement)
    guard reference.sessionID == supplement.sessionID else {
      throw NTPSpatialFusionError.sessionMismatch
    }
    guard reference.generation == supplement.generation else {
      throw NTPSpatialFusionError.generationMismatch
    }
    guard reference.sequence == supplement.sequence else {
      throw NTPSpatialFusionError.sequenceMismatch
    }
    guard reference.captureTimestampNs == supplement.captureTimestampNs else {
      throw NTPSpatialFusionError.captureTimestampMismatch
    }

    var rig: [UInt16: NTPSignalSample] = [:]
    rig.reserveCapacity(descriptor.supportedSignals.count)
    for signalID in descriptor.supportedSignals {
      rig[signalID] = fuseSignal(
        signalID: signalID,
        reference: reference.rig[signalID] ?? .unsupported,
        supplement: supplement.rig[signalID] ?? .unsupported,
        policy: policy
      )
    }
    let output = NTPTrackingResult(
      sessionID: reference.sessionID,
      generation: reference.generation,
      sequence: reference.sequence,
      captureTimestampNs: reference.captureTimestampNs,
      producedTimestampNs: max(
        reference.producedTimestampNs,
        supplement.producedTimestampNs
      ),
      rig: rig,
      geometry: fuseGeometry(reference.geometry, supplement.geometry),
      quality: fuseQuality(reference.quality, supplement.quality)
    )
    try Self.validateResult(output, descriptor: descriptor, source: .output)
    return output
  }

  private enum Source {
    case reference
    case supplement
    case output
  }

  private static func validateDescriptor(
    _ descriptor: NTPDescriptor, source: Source
  ) throws {
    do {
      try validateSpatialDescriptor(descriptor)
    } catch let error as NTPSpatialProducerError {
      switch source {
      case .reference: throw NTPSpatialFusionError.referenceContract(error)
      case .supplement: throw NTPSpatialFusionError.extensionContract(error)
      case .output: throw NTPSpatialFusionError.outputContract(error)
      }
    } catch let error as NTPCodecError {
      switch source {
      case .reference: throw NTPSpatialFusionError.referenceCodec(error)
      case .supplement: throw NTPSpatialFusionError.extensionCodec(error)
      case .output: throw NTPSpatialFusionError.outputCodec(error)
      }
    }
  }

  private static func validateResult(
    _ result: NTPTrackingResult,
    descriptor: NTPDescriptor,
    source: Source
  ) throws {
    do {
      try validateSpatialResult(result, descriptor: descriptor)
    } catch let error as NTPSpatialProducerError {
      switch source {
      case .reference: throw NTPSpatialFusionError.referenceContract(error)
      case .supplement: throw NTPSpatialFusionError.extensionContract(error)
      case .output: throw NTPSpatialFusionError.outputContract(error)
      }
    } catch let error as NTPCodecError {
      switch source {
      case .reference: throw NTPSpatialFusionError.referenceCodec(error)
      case .supplement: throw NTPSpatialFusionError.extensionCodec(error)
      case .output: throw NTPSpatialFusionError.outputCodec(error)
      }
    }
  }
}

private func fusedConfidence(_ reference: Float, _ supplement: Float) -> Float {
  min(max(1 - (1 - reference) * (1 - supplement), 0), 1)
}

private func fuseSignal(
  signalID: UInt16,
  reference: NTPSignalSample,
  supplement: NTPSignalSample,
  policy: NTPSpatialFusionPolicy
) -> NTPSignalSample {
  switch (reference.value, supplement.value) {
  case (let referenceValue?, let supplementValue?):
    let agrees = abs(referenceValue - supplementValue) <= policy.agreementTolerance
    let gazePrefersReference = (37...40).contains(signalID)
    let chooseSupplement =
      !gazePrefersReference && !agrees
      && supplement.confidence >= reference.confidence + policy.confidenceSwitchMargin
    let selected = chooseSupplement ? supplement : reference
    return NTPSignalSample(
      value: selected.value ?? referenceValue,
      confidence: fusedConfidence(reference.confidence, supplement.confidence),
      state: .fused,
      sampleCaptureTimestampNs: min(
        reference.sampleCaptureTimestampNs,
        supplement.sampleCaptureTimestampNs
      )
    )
  case (_?, nil):
    return reference
  case (nil, _?):
    return supplement
  case (nil, nil):
    return chooseUnavailable(reference, supplement)
  }
}

private func chooseUnavailable(
  _ reference: NTPSignalSample, _ supplement: NTPSignalSample
) -> NTPSignalSample {
  if reference.state == .unsupported
    || (supplement.state != .unsupported && supplement.confidence > reference.confidence)
  {
    return supplement
  }
  return reference
}

private func chooseUnavailable<Value: Equatable & Sendable>(
  _ reference: NTPTracked<Value>, _ supplement: NTPTracked<Value>
) -> NTPTracked<Value> {
  if reference.state == .unsupported
    || (supplement.state != .unsupported && supplement.confidence > reference.confidence)
  {
    return supplement
  }
  return reference
}

private func fuseTracked<Value: Equatable & Sendable>(
  _ reference: NTPTracked<Value>, _ supplement: NTPTracked<Value>
) -> NTPTracked<Value> {
  switch (reference.value, supplement.value) {
  case (let value?, _?):
    return NTPTracked(
      value: value,
      confidence: fusedConfidence(reference.confidence, supplement.confidence),
      state: .fused,
      sampleCaptureTimestampNs: min(
        reference.sampleCaptureTimestampNs,
        supplement.sampleCaptureTimestampNs
      )
    )
  case (_?, nil):
    return reference
  case (nil, _?):
    return supplement
  case (nil, nil):
    return chooseUnavailable(reference, supplement)
  }
}

private func fuseGeometry(
  _ reference: NTPGeometryResult, _ supplement: NTPGeometryResult
) -> NTPGeometryResult {
  NTPGeometryResult(
    headCameraPose: fuseTracked(reference.headCameraPose, supplement.headCameraPose),
    leftEye: NTPEyeGeometry(
      originHead: fuseTracked(reference.leftEye.originHead, supplement.leftEye.originHead),
      directionHead: fuseTracked(
        reference.leftEye.directionHead,
        supplement.leftEye.directionHead
      )
    ),
    rightEye: NTPEyeGeometry(
      originHead: fuseTracked(reference.rightEye.originHead, supplement.rightEye.originHead),
      directionHead: fuseTracked(
        reference.rightEye.directionHead,
        supplement.rightEye.directionHead
      )
    ),
    lookAtCamera: fuseTracked(reference.lookAtCamera, supplement.lookAtCamera),
    faceGeometryState: fuseState(
      reference.faceGeometryState,
      supplement.faceGeometryState
    ),
    faceLandmarks: reference.faceGeometryState != .unsupported
      ? reference.faceLandmarks : supplement.faceLandmarks
  )
}

private func fuseState(
  _ reference: NTPSignalState, _ supplement: NTPSignalState
) -> NTPSignalState {
  if reference == .unsupported { return supplement }
  if supplement == .unsupported { return reference }
  if reference.carriesValue && supplement.carriesValue { return .fused }
  return reference
}

private func fuseRegion(
  _ reference: NTPRegionQuality, _ supplement: NTPRegionQuality
) -> NTPRegionQuality {
  NTPRegionQuality(
    confidence: fusedConfidence(reference.confidence, supplement.confidence),
    state: fuseState(reference.state, supplement.state)
  )
}

private func fuseQuality(
  _ reference: NTPTrackingQuality, _ supplement: NTPTrackingQuality
) -> NTPTrackingQuality {
  NTPTrackingQuality(
    overallConfidence: fusedConfidence(
      reference.overallConfidence,
      supplement.overallConfidence
    ),
    face: fuseRegion(reference.face, supplement.face),
    eyes: fuseRegion(reference.eyes, supplement.eyes),
    torso: fuseRegion(reference.torso, supplement.torso),
    arm: NTPSideMap(
      left: fuseRegion(reference.arm.left, supplement.arm.left),
      right: fuseRegion(reference.arm.right, supplement.arm.right)
    ),
    auricle: NTPSideMap(
      left: fuseRegion(reference.auricle.left, supplement.auricle.left),
      right: fuseRegion(reference.auricle.right, supplement.auricle.right)
    ),
    stabilizationRevision: reference.stabilizationRevision
  )
}
