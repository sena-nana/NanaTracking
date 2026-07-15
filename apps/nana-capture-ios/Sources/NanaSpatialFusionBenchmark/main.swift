import Foundation
import NanaCaptureCore

private enum BenchmarkError: Error {
  case invalidOutput
}

private struct BenchmarkReport: Codable {
  let schema: String
  let smokeOnly: Bool
  let configuration: String
  let warmupIterations: Int
  let measuredIterationsPerRun: Int
  let runs: Int
  let nanosecondsPerFusion: [Double]
  let minimumNanosecondsPerFusion: Double
  let medianNanosecondsPerFusion: Double
  let maximumNanosecondsPerFusion: Double
  let checksum: Double
}

private func makeGeometry(timestamp: UInt64, headX: Float) -> NTPGeometryResult {
  func tracked<Value: Equatable & Sendable>(_ value: Value) -> NTPTracked<Value> {
    NTPTracked(
      value: value,
      confidence: 0.9,
      state: .observed,
      sampleCaptureTimestampNs: timestamp
    )
  }
  let direction = tracked(
    NTPDirection3(space: .headLocal, value: NTPVector3(x: 0, y: 0, z: 1)))
  return NTPGeometryResult(
    headCameraPose: tracked(
      NTPPose(
        parentSpace: .camera,
        lengthBasis: .headRelative,
        position: NTPVector3(x: headX, y: 0, z: 0),
        orientationXYZW: .identity
      )),
    leftEye: NTPEyeGeometry(
      originHead: tracked(
        NTPPosition3(
          space: .headLocal,
          lengthBasis: .headRelative,
          value: NTPVector3(x: -0.15, y: 0, z: 0)
        )),
      directionHead: direction
    ),
    rightEye: NTPEyeGeometry(
      originHead: tracked(
        NTPPosition3(
          space: .headLocal,
          lengthBasis: .headRelative,
          value: NTPVector3(x: 0.15, y: 0, z: 0)
        )),
      directionHead: direction
    ),
    lookAtCamera: tracked(
      NTPPosition3(
        space: .camera,
        lengthBasis: .headRelative,
        value: NTPVector3(x: 0, y: 0, z: 1)
      )),
    faceGeometryState: .observed
  )
}

private func makeQuality(torso: NTPRegionQuality) -> NTPTrackingQuality {
  NTPTrackingQuality(
    overallConfidence: 0.9,
    face: NTPRegionQuality(confidence: 0.9, state: .observed),
    eyes: NTPRegionQuality(confidence: 0.9, state: .observed),
    torso: torso,
    arm: NTPSideMap(left: .unsupported, right: .unsupported),
    auricle: NTPSideMap(left: .unsupported, right: .unsupported)
  )
}

private func makeResult(
  sessionID: [UInt8], timestamp: UInt64, signalRange: ClosedRange<UInt16>,
  signalValue: Float, headX: Float, torso: NTPRegionQuality
) -> NTPTrackingResult {
  let rig = Dictionary(
    uniqueKeysWithValues: signalRange.map { signalID in
      (
        signalID,
        NTPSignalSample(
          value: signalID == 42 ? 0.4 : signalValue,
          confidence: 0.9,
          state: .observed,
          sampleCaptureTimestampNs: timestamp
        )
      )
    })
  return NTPTrackingResult(
    sessionID: sessionID,
    generation: 0,
    sequence: 0,
    captureTimestampNs: timestamp,
    producedTimestampNs: timestamp + 1_000_000,
    rig: rig,
    geometry: makeGeometry(timestamp: timestamp, headX: headX),
    quality: makeQuality(torso: torso)
  )
}

@main
struct NanaSpatialFusionBenchmark {
  static func main() throws {
    let timestamp: UInt64 = 2_000_000_000
    let sessionID = [UInt8](repeating: 7, count: 16)
    let plan = try NTPSpatialFusionPlan(
      referenceDescriptor: NTPDescriptor(
        guaranteedProfile: .spatial,
        supportedSignals: Array(UInt16(1)...UInt16(41)),
        supportedStructures: .spatialRequired
      ),
      extensionDescriptor: NTPDescriptor(
        guaranteedProfile: .spatial,
        supportedSignals: Array(UInt16(1)...UInt16(42)),
        supportedStructures: .spatialRequired
      )
    )
    let reference = makeResult(
      sessionID: sessionID,
      timestamp: timestamp,
      signalRange: UInt16(1)...UInt16(41),
      signalValue: 0,
      headX: 0,
      torso: .unsupported
    )
    let supplement = makeResult(
      sessionID: sessionID,
      timestamp: timestamp,
      signalRange: UInt16(1)...UInt16(42),
      signalValue: 0.02,
      headX: 0.5,
      torso: NTPRegionQuality(confidence: 0.9, state: .observed)
    )

    let warmupIterations = 500
    for _ in 0..<warmupIterations {
      _ = try plan.fuse(reference: reference, supplement: supplement)
    }

    let runs = 5
    let iterations = 10_000
    var checksum = 0.0
    var samples: [Double] = []
    samples.reserveCapacity(runs)
    for _ in 0..<runs {
      let start = DispatchTime.now().uptimeNanoseconds
      for _ in 0..<iterations {
        let output = try plan.fuse(reference: reference, supplement: supplement)
        checksum += Double(output.rig[42]?.value ?? -1)
      }
      let elapsed = DispatchTime.now().uptimeNanoseconds - start
      samples.append(Double(elapsed) / Double(iterations))
    }
    guard checksum > 0 else { throw BenchmarkError.invalidOutput }
    let sorted = samples.sorted()
    let report = BenchmarkReport(
      schema: "nanatracking.swift-spatial-fusion-benchmark/1.0.0",
      smokeOnly: true,
      configuration: "Swift release, prevalidated descriptor union, 41+42 Spatial signals",
      warmupIterations: warmupIterations,
      measuredIterationsPerRun: iterations,
      runs: runs,
      nanosecondsPerFusion: samples,
      minimumNanosecondsPerFusion: sorted[0],
      medianNanosecondsPerFusion: sorted[sorted.count / 2],
      maximumNanosecondsPerFusion: sorted[sorted.count - 1],
      checksum: checksum
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    print(String(decoding: try encoder.encode(report), as: UTF8.self))
  }
}
