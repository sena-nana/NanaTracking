// swift-tools-version: 6.2

import PackageDescription

let package = Package(
  name: "NanaCaptureIOS",
  platforms: [
    .iOS(.v17),
    .macOS(.v14),
  ],
  products: [
    .library(name: "NanaCaptureCore", targets: ["NanaCaptureCore"]),
    .executable(name: "NanaCaptureSelfTest", targets: ["NanaCaptureSelfTest"]),
    .executable(
      name: "NanaCaptureSchedulingBenchmark",
      targets: ["NanaCaptureSchedulingBenchmark"]
    ),
  ],
  targets: [
    .target(name: "NanaCaptureCore"),
    .executableTarget(name: "NanaCaptureSelfTest", dependencies: ["NanaCaptureCore"]),
    .executableTarget(
      name: "NanaCaptureSchedulingBenchmark",
      dependencies: ["NanaCaptureCore"]
    ),
  ]
)
