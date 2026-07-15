import Dispatch
import Foundation
import NanaCaptureCore

private final class SchedulingProbe: @unchecked Sendable {
  let firstStarted = DispatchSemaphore(value: 0)
  let releaseFirst = DispatchSemaphore(value: 0)
  private let lock = NSLock()
  private var processed: [Int] = []

  func process(_ value: Int) {
    if value == 0 {
      firstStarted.signal()
      releaseFirst.wait()
    }
    lock.withLock { processed.append(value) }
  }

  func snapshot() -> [Int] {
    lock.withLock { processed }
  }
}

private enum BenchmarkError: Error {
  case firstFrameDidNotStart
  case latestOnlyContractFailed
}

private func blockingWait(
  _ semaphore: DispatchSemaphore, timeout: DispatchTime
) -> DispatchTimeoutResult {
  semaphore.wait(timeout: timeout)
}

@main
private struct NanaCaptureSchedulingBenchmark {
  static func main() async throws {
    let iterations = 200_000
    let probe = SchedulingProbe()
    let worker = NTPAsyncLatestFrameWorker<Int> { value in
      probe.process(value)
    }
    worker.submit(0)
    let started = await Task.detached {
      blockingWait(probe.firstStarted, timeout: .now() + 2)
    }.value
    guard started == .success else { throw BenchmarkError.firstFrameDidNotStart }

    let began = DispatchTime.now().uptimeNanoseconds
    for value in 1...iterations {
      worker.submit(value)
    }
    let elapsed = DispatchTime.now().uptimeNanoseconds - began
    probe.releaseFirst.signal()
    await worker.flush()

    let processed = probe.snapshot()
    guard processed == [0, iterations], worker.droppedCount() == UInt64(iterations - 1) else {
      throw BenchmarkError.latestOnlyContractFailed
    }
    let nanosecondsPerSubmit = Double(elapsed) / Double(iterations)
    print(
      "{\"dropped\":\(worker.droppedCount()),\"elapsed_ns\":\(elapsed),"
        + "\"iterations\":\(iterations),\"nanoseconds_per_submit\":\(nanosecondsPerSubmit),"
        + "\"processed\":\(processed)}"
    )
  }
}
