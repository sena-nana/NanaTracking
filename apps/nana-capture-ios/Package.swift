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
  ],
  targets: [
    .target(name: "NanaCaptureCore"),
    .executableTarget(name: "NanaCaptureSelfTest", dependencies: ["NanaCaptureCore"]),
  ]
)
