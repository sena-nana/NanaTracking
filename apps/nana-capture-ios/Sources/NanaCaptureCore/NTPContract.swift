import Foundation

public struct NTPProtocolVersion: Equatable, Sendable {
  public let major: UInt16
  public let minor: UInt16

  public init(major: UInt16, minor: UInt16) {
    self.major = major
    self.minor = minor
  }

  public static let v1 = NTPProtocolVersion(major: 1, minor: 0)
}

public struct NTPRevision: Equatable, Sendable {
  public let major: UInt16
  public let minor: UInt16
  public let patch: UInt16

  public init(major: UInt16, minor: UInt16, patch: UInt16) {
    self.major = major
    self.minor = minor
    self.patch = patch
  }

  public static let v1 = NTPRevision(major: 1, minor: 0, patch: 0)
}

public struct NTPContractRevisions: Equatable, Sendable {
  public let protocolVersion: NTPProtocolVersion
  public let schemaRevision: UInt32
  public let signalRegistry: NTPRevision
  public let normalization: NTPRevision
  public let calibration: NTPRevision
  public let features: NTPRevision

  public init(
    protocolVersion: NTPProtocolVersion,
    schemaRevision: UInt32,
    signalRegistry: NTPRevision,
    normalization: NTPRevision,
    calibration: NTPRevision,
    features: NTPRevision
  ) {
    self.protocolVersion = protocolVersion
    self.schemaRevision = schemaRevision
    self.signalRegistry = signalRegistry
    self.normalization = normalization
    self.calibration = calibration
    self.features = features
  }

  public static let ntpV1 = NTPContractRevisions(
    protocolVersion: .v1,
    schemaRevision: 1,
    signalRegistry: .v1,
    normalization: .v1,
    calibration: .v1,
    features: .v1
  )
}

public enum NTPTrackingProfile: UInt8, Equatable, Sendable {
  case partial = 0
  case basic = 1
  case spatial = 2
  case full = 3
}

public struct NTPStructureFeatures: OptionSet, Equatable, Sendable {
  public let rawValue: UInt64

  public init(rawValue: UInt64) {
    self.rawValue = rawValue
  }

  public static let headGeometry = NTPStructureFeatures(rawValue: 1 << 0)
  public static let eyeGeometry = NTPStructureFeatures(rawValue: 1 << 1)
  public static let lookAtPoint = NTPStructureFeatures(rawValue: 1 << 2)
  public static let faceGeometry = NTPStructureFeatures(rawValue: 1 << 3)
  public static let bodySkeleton = NTPStructureFeatures(rawValue: 1 << 4)
  public static let spatialRequired: NTPStructureFeatures = [
    .headGeometry, .eyeGeometry, .lookAtPoint, .faceGeometry,
  ]
}

public struct NTPTrackingFeatures: OptionSet, Equatable, Sendable {
  public let rawValue: UInt64

  public init(rawValue: UInt64) {
    self.rawValue = rawValue
  }

  public static let metricCoordinates = NTPTrackingFeatures(rawValue: 1 << 0)
  public static let denseFaceMesh = NTPTrackingFeatures(rawValue: 1 << 1)
  public static let auricleLocalGeometry = NTPTrackingFeatures(rawValue: 1 << 2)
  public static let wristPose = NTPTrackingFeatures(rawValue: 1 << 3)
}

public struct NTPDescriptor: Equatable, Sendable {
  public let revisions: NTPContractRevisions
  public let guaranteedProfile: NTPTrackingProfile
  public let supportedSignals: [UInt16]
  public let supportedStructures: NTPStructureFeatures
  public let features: NTPTrackingFeatures

  public init(
    revisions: NTPContractRevisions = .ntpV1,
    guaranteedProfile: NTPTrackingProfile,
    supportedSignals: [UInt16],
    supportedStructures: NTPStructureFeatures,
    features: NTPTrackingFeatures = []
  ) {
    self.revisions = revisions
    self.guaranteedProfile = guaranteedProfile
    self.supportedSignals = supportedSignals
    self.supportedStructures = supportedStructures
    self.features = features
  }

  public static let spatialV1 = NTPDescriptor(
    guaranteedProfile: .spatial,
    supportedSignals: Array(1...41),
    supportedStructures: .spatialRequired
  )
}

public enum NTPSignalState: UInt8, Equatable, Sendable {
  case observed = 0
  case fused = 1
  case predicted = 2
  case occluded = 3
  case outOfFrame = 4
  case trackingLost = 5
  case unsupported = 6

  public var carriesValue: Bool {
    self == .observed || self == .fused || self == .predicted
  }
}

public enum NTPCoordinateSpace: UInt8, Equatable, Sendable {
  case camera = 0
  case torsoLocal = 1
  case headLocal = 2
}

public enum NTPLengthBasis: UInt8, Equatable, Sendable {
  case metric = 0
  case headRelative = 1
  case torsoRelative = 2
}

public struct NTPVector3: Equatable, Sendable {
  public let x: Float
  public let y: Float
  public let z: Float

  public init(x: Float, y: Float, z: Float) {
    self.x = x
    self.y = y
    self.z = z
  }

  public static let zero = NTPVector3(x: 0, y: 0, z: 0)
}

public struct NTPQuaternion: Equatable, Sendable {
  public let x: Float
  public let y: Float
  public let z: Float
  public let w: Float

  public init(x: Float, y: Float, z: Float, w: Float) {
    self.x = x
    self.y = y
    self.z = z
    self.w = w
  }

  public static let identity = NTPQuaternion(x: 0, y: 0, z: 0, w: 1)

  var canonicalized: NTPQuaternion {
    let firstNonzero = [x, y, z].first { $0 != 0 }
    if w < 0 || (w == 0 && (firstNonzero ?? 0) < 0) {
      return NTPQuaternion(x: -x, y: -y, z: -z, w: -w)
    }
    return self
  }
}

public struct NTPPosition3: Equatable, Sendable {
  public let space: NTPCoordinateSpace
  public let lengthBasis: NTPLengthBasis
  public let value: NTPVector3

  public init(space: NTPCoordinateSpace, lengthBasis: NTPLengthBasis, value: NTPVector3) {
    self.space = space
    self.lengthBasis = lengthBasis
    self.value = value
  }
}

public struct NTPDirection3: Equatable, Sendable {
  public let space: NTPCoordinateSpace
  public let value: NTPVector3

  public init(space: NTPCoordinateSpace, value: NTPVector3) {
    self.space = space
    self.value = value
  }
}

public struct NTPPose: Equatable, Sendable {
  public let parentSpace: NTPCoordinateSpace
  public let lengthBasis: NTPLengthBasis
  public let position: NTPVector3
  public let orientationXYZW: NTPQuaternion

  public init(
    parentSpace: NTPCoordinateSpace,
    lengthBasis: NTPLengthBasis,
    position: NTPVector3,
    orientationXYZW: NTPQuaternion
  ) {
    self.parentSpace = parentSpace
    self.lengthBasis = lengthBasis
    self.position = position
    self.orientationXYZW = orientationXYZW
  }
}

public struct NTPTracked<Value: Equatable & Sendable>: Equatable, Sendable {
  public let value: Value?
  public let confidence: Float
  public let state: NTPSignalState
  public let sampleCaptureTimestampNs: UInt64
  public let predictionHorizonNs: UInt64

  public init(
    value: Value?,
    confidence: Float,
    state: NTPSignalState,
    sampleCaptureTimestampNs: UInt64,
    predictionHorizonNs: UInt64 = 0
  ) {
    self.value = value
    self.confidence = confidence
    self.state = state
    self.sampleCaptureTimestampNs = sampleCaptureTimestampNs
    self.predictionHorizonNs = predictionHorizonNs
  }

  public static var unsupported: NTPTracked<Value> {
    NTPTracked(
      value: nil,
      confidence: 0,
      state: .unsupported,
      sampleCaptureTimestampNs: 0
    )
  }
}

public struct NTPSignalSample: Equatable, Sendable {
  public let value: Float?
  public let confidence: Float
  public let state: NTPSignalState
  public let sampleCaptureTimestampNs: UInt64
  public let predictionHorizonNs: UInt64

  public init(
    value: Float?,
    confidence: Float,
    state: NTPSignalState,
    sampleCaptureTimestampNs: UInt64,
    predictionHorizonNs: UInt64 = 0
  ) {
    self.value = value
    self.confidence = confidence
    self.state = state
    self.sampleCaptureTimestampNs = sampleCaptureTimestampNs
    self.predictionHorizonNs = predictionHorizonNs
  }

  public static let unsupported = NTPSignalSample(
    value: nil,
    confidence: 0,
    state: .unsupported,
    sampleCaptureTimestampNs: 0
  )
}

public struct NTPEyeGeometry: Equatable, Sendable {
  public let originHead: NTPTracked<NTPPosition3>
  public let directionHead: NTPTracked<NTPDirection3>

  public init(
    originHead: NTPTracked<NTPPosition3>, directionHead: NTPTracked<NTPDirection3>
  ) {
    self.originHead = originHead
    self.directionHead = directionHead
  }

  public static let unsupported = NTPEyeGeometry(
    originHead: .unsupported,
    directionHead: .unsupported
  )
}

public struct NTPFaceLandmark: Equatable, Sendable {
  public let semanticID: UInt16
  public let positionHead: NTPTracked<NTPPosition3>

  public init(semanticID: UInt16, positionHead: NTPTracked<NTPPosition3>) {
    self.semanticID = semanticID
    self.positionHead = positionHead
  }
}

public struct NTPGeometryResult: Equatable, Sendable {
  public let headCameraPose: NTPTracked<NTPPose>
  public let leftEye: NTPEyeGeometry
  public let rightEye: NTPEyeGeometry
  public let lookAtCamera: NTPTracked<NTPPosition3>
  public let faceGeometryState: NTPSignalState
  public let faceLandmarks: [NTPFaceLandmark]

  public init(
    headCameraPose: NTPTracked<NTPPose>,
    leftEye: NTPEyeGeometry,
    rightEye: NTPEyeGeometry,
    lookAtCamera: NTPTracked<NTPPosition3>,
    faceGeometryState: NTPSignalState,
    faceLandmarks: [NTPFaceLandmark] = []
  ) {
    self.headCameraPose = headCameraPose
    self.leftEye = leftEye
    self.rightEye = rightEye
    self.lookAtCamera = lookAtCamera
    self.faceGeometryState = faceGeometryState
    self.faceLandmarks = faceLandmarks
  }

  public static let unsupported = NTPGeometryResult(
    headCameraPose: .unsupported,
    leftEye: .unsupported,
    rightEye: .unsupported,
    lookAtCamera: .unsupported,
    faceGeometryState: .unsupported
  )
}

public struct NTPSideMap<Value: Equatable & Sendable>: Equatable, Sendable {
  public let left: Value
  public let right: Value

  public init(left: Value, right: Value) {
    self.left = left
    self.right = right
  }
}

public struct NTPSkeletonResult: Equatable, Sendable {
  public let torsoCameraPose: NTPTracked<NTPPose>
  public let shoulder: NTPSideMap<NTPTracked<NTPPose>>
  public let elbow: NTPSideMap<NTPTracked<NTPPose>>
  public let wrist: NTPSideMap<NTPTracked<NTPPose>>
  public let upperArmDirectionTorso: NTPSideMap<NTPTracked<NTPDirection3>>
  public let forearmDirectionTorso: NTPSideMap<NTPTracked<NTPDirection3>>
  public let upperArmTwist: NTPSideMap<NTPTracked<Float>>
  public let forearmTwist: NTPSideMap<NTPTracked<Float>>

  public init(
    torsoCameraPose: NTPTracked<NTPPose>,
    shoulder: NTPSideMap<NTPTracked<NTPPose>>,
    elbow: NTPSideMap<NTPTracked<NTPPose>>,
    wrist: NTPSideMap<NTPTracked<NTPPose>>,
    upperArmDirectionTorso: NTPSideMap<NTPTracked<NTPDirection3>>,
    forearmDirectionTorso: NTPSideMap<NTPTracked<NTPDirection3>>,
    upperArmTwist: NTPSideMap<NTPTracked<Float>>,
    forearmTwist: NTPSideMap<NTPTracked<Float>>
  ) {
    self.torsoCameraPose = torsoCameraPose
    self.shoulder = shoulder
    self.elbow = elbow
    self.wrist = wrist
    self.upperArmDirectionTorso = upperArmDirectionTorso
    self.forearmDirectionTorso = forearmDirectionTorso
    self.upperArmTwist = upperArmTwist
    self.forearmTwist = forearmTwist
  }

  public static let unsupported = NTPSkeletonResult(
    torsoCameraPose: .unsupported,
    shoulder: NTPSideMap(left: .unsupported, right: .unsupported),
    elbow: NTPSideMap(left: .unsupported, right: .unsupported),
    wrist: NTPSideMap(left: .unsupported, right: .unsupported),
    upperArmDirectionTorso: NTPSideMap(left: .unsupported, right: .unsupported),
    forearmDirectionTorso: NTPSideMap(left: .unsupported, right: .unsupported),
    upperArmTwist: NTPSideMap(left: .unsupported, right: .unsupported),
    forearmTwist: NTPSideMap(left: .unsupported, right: .unsupported)
  )
}

public struct NTPRegionQuality: Equatable, Sendable {
  public let confidence: Float
  public let state: NTPSignalState

  public init(confidence: Float, state: NTPSignalState) {
    self.confidence = confidence
    self.state = state
  }

  public static let unsupported = NTPRegionQuality(confidence: 0, state: .unsupported)
}

public struct NTPTrackingQuality: Equatable, Sendable {
  public let overallConfidence: Float
  public let face: NTPRegionQuality
  public let eyes: NTPRegionQuality
  public let torso: NTPRegionQuality
  public let arm: NTPSideMap<NTPRegionQuality>
  public let auricle: NTPSideMap<NTPRegionQuality>
  public let stabilizationRevision: NTPRevision

  public init(
    overallConfidence: Float,
    face: NTPRegionQuality,
    eyes: NTPRegionQuality,
    torso: NTPRegionQuality,
    arm: NTPSideMap<NTPRegionQuality>,
    auricle: NTPSideMap<NTPRegionQuality>,
    stabilizationRevision: NTPRevision = .v1
  ) {
    self.overallConfidence = overallConfidence
    self.face = face
    self.eyes = eyes
    self.torso = torso
    self.arm = arm
    self.auricle = auricle
    self.stabilizationRevision = stabilizationRevision
  }

  public static let unsupported = NTPTrackingQuality(
    overallConfidence: 0,
    face: .unsupported,
    eyes: .unsupported,
    torso: .unsupported,
    arm: NTPSideMap(left: .unsupported, right: .unsupported),
    auricle: NTPSideMap(left: .unsupported, right: .unsupported)
  )
}

public struct NTPTrackingResult: Equatable, Sendable {
  public let sessionID: [UInt8]
  public let generation: UInt32
  public let sequence: UInt64
  public let captureTimestampNs: UInt64
  public let producedTimestampNs: UInt64
  public let rig: [UInt16: NTPSignalSample]
  public let geometry: NTPGeometryResult
  public let skeleton: NTPSkeletonResult
  public let quality: NTPTrackingQuality

  public init(
    sessionID: [UInt8],
    generation: UInt32,
    sequence: UInt64,
    captureTimestampNs: UInt64,
    producedTimestampNs: UInt64,
    rig: [UInt16: NTPSignalSample],
    geometry: NTPGeometryResult,
    skeleton: NTPSkeletonResult = .unsupported,
    quality: NTPTrackingQuality
  ) {
    self.sessionID = sessionID
    self.generation = generation
    self.sequence = sequence
    self.captureTimestampNs = captureTimestampNs
    self.producedTimestampNs = producedTimestampNs
    self.rig = rig
    self.geometry = geometry
    self.skeleton = skeleton
    self.quality = quality
  }
}
