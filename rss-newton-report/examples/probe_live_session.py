"""End-to-end RSS agent-loop probe. Requires a running local live server.

The probe temporarily edits its owned scene file, tests failed-edit recovery,
then restores the original. Never point it at someone else's active scene.
"""

import argparse
import ast
import json
import socket
import subprocess
import time
from pathlib import Path


def request(sock, cmd, **kwargs):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(300)
        client.connect(str(sock))
        stream = client.makefile("rwb")
        stream.write((json.dumps({"cmd": cmd, **kwargs}) + "\n").encode())
        stream.flush()
        return json.loads(stream.readline())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-session", default="")
    args = parser.parse_args()
    sock = args.run_dir / "ctl.sock"

    def call(cmd, **kw):
        return request(sock, cmd, **kw)

    results = {}

    def chapter(title, description):
        if args.video_session:
            subprocess.run(
                [
                    "playwright-cli",
                    "-s=" + args.video_session,
                    "video-chapter",
                    title,
                    "--description=" + description,
                    "--duration=3500",
                ],
                check=True,
            )

    def wait_for(predicate):
        started = time.monotonic()
        while time.monotonic() - started < 300:
            value = call("status")
            if predicate(value):
                return value
            time.sleep(0.3)
        raise TimeoutError(value)

    def camera():
        response = call(
            "eval",
            code=(
                "{k: {'position': [float(v) for v in c.camera.position], "
                "'look_at': [float(v) for v in c.camera.look_at]} "
                "for k,c in server.get_clients().items()}"
            ),
        )
        assert response["ok"], response
        return ast.literal_eval(response["value"])

    assert call("pause")["ok"]
    initial = call("status")
    assert initial["finite"] and initial["paused"], initial
    path = Path(initial["file"])
    original = path.read_text()
    before_camera = camera()
    assert before_camera, "Keep a browser connected to verify camera persistence"
    results["before"] = initial
    try:
        chapter(
            "Agent API: advance exactly 12 steps",
            "The coding agent uses the local JSON socket. It does not need to manipulate the browser canvas.",
        )
        start = initial["step"]
        assert call("run", steps=12)["ok"]
        stepped = wait_for(lambda s: s["paused"] and s["step"] >= start + 12)
        assert stepped["step"] == start + 12, stepped
        results["exact_steps"] = {"pass": True, "start": start, "end": stepped["step"]}
        chapter(
            "An invalid edit keeps the previous scene alive",
            "Save a deliberate syntax error. The server reports it while preserving the existing model and browser.",
        )
        path.write_text(original + "\nthis is deliberately invalid python !!!\n")
        failed = wait_for(lambda s: bool(s["last_error"]))
        assert failed["pid"] == initial["pid"] and failed["names"] == initial["names"] and failed["finite"]
        results["bad_edit_preserved_scene"] = {"pass": True, "error": failed["last_error"]}
        time.sleep(2)
        chapter(
            "Fix the script: move and recolor the cube",
            "A valid save rebuilds the scene in the same process. The camera stays in place.",
        )
        changed = original.replace(
            "at=(0.0, 0.17, 0.12), color=(0.15, 0.65, 0.9)", "at=(0.15, 0.14, 0.17), color=(0.75, 0.22, 0.85)"
        )
        assert changed != original
        t0 = time.monotonic()
        path.write_text(changed)
        reloaded = wait_for(lambda s: not s["last_error"] and abs(s["positions"]["cube"][0] - 0.75) < 0.001)
        after_camera = camera()
        assert reloaded["pid"] == initial["pid"]
        assert after_camera == before_camera, (before_camera, after_camera)
        results["valid_reload"] = {
            "pass": True,
            "wall_seconds": round(time.monotonic() - t0, 3),
            "build_seconds": reloaded["load_seconds"],
            "state": reloaded,
            "camera_before": before_camera,
            "camera_after": after_camera,
        }
        time.sleep(3)
        snapshot = call(
            "snapshot", camera="overview", size="640x480", out=str(args.output.parent / "agent-tiled-snapshot.png")
        )
        assert snapshot["ok"] and snapshot["width"] == 640 and snapshot["height"] == 480, snapshot
        results["snapshot"] = snapshot
    finally:
        path.write_text(original)
        restored = wait_for(lambda s: not s["last_error"] and abs(s["positions"]["cube"][0] - 0.6) < 0.001)
        call("pause")
        results["restored"] = {"pass": True, "pid": restored["pid"]}
        args.output.write_text(json.dumps(results, indent=2))
    print(
        "PASS: exact stepping, failed-edit recovery, valid reload, same process and camera, tiled snapshot, restoration"
    )


if __name__ == "__main__":
    main()
