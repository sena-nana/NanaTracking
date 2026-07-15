"""Isolated Python 3.14 subinterpreter broker.

PyTorch 2.11 autograd must not share a process that has created subinterpreters,
so this module deliberately contains no ML or third-party imports.
"""

import json
import sys
from typing import Any

from nana_tracking.data.executors import run_interpreter_pool


def main() -> None:
    request: dict[str, Any] = json.loads(sys.stdin.read())
    try:
        values = run_interpreter_pool(
            [int(value) for value in request["values"]],
            workers=int(request["workers"]),
            buffersize=int(request["buffersize"]),
            rounds=int(request["rounds"]),
        )
        response = {"ok": True, "values": values}
    except Exception as error:  # The broker must serialize worker failures to the parent.
        response = {
            "ok": False,
            "error_type": type(error).__name__,
            "message": str(error),
        }
    sys.stdout.write(json.dumps(response))


if __name__ == "__main__":
    main()
