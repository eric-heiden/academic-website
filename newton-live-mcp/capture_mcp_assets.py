"""Capture real task assets through the official MCP client, outside timed trials.

Run in the Newton environment with the optional MCP SDK installed. A submitted
final configuration is required; this script never chooses or tunes parameters.
Connection credentials and subprocess logs stay in a temporary directory.
"""

import argparse
import asyncio
import base64
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def sanitize(value):
    """Remove local artifact paths from otherwise public capture metadata."""
    if isinstance(value, dict):
        return {
            key: sanitize(item)
            for key, item in value.items()
            if key
            not in {
                "token",
                "directory",
                "raw_artifact",
                "artifact_directory",
                "recording_path",
            }
        }
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    return value


async def capture(args, private):
    camera = {
        "panda": {"eye": [1.2, -1.2, 0.85], "target": [0, 0, 0.45], "up": [0, 0, 1]},
        "allegro": {"eye": [0.4, -0.45, 0.53], "target": [0, 0, 0.32], "up": [0, 0, 1]},
        "hug": {"eye": [0.45, -0.4, 0.38], "target": [0, 0, 0.07], "up": [0, 0, 1]},
    }[args.scenario] | {"width": 640, "height": 480}
    camera["eye"] = [
        target + args.camera_scale * (eye - target)
        for eye, target in zip(camera["eye"], camera["target"], strict=True)
    ]
    connection = private / "session.json"
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "newton.mcp", "--connect", str(connection), "--profile", "code"],
        cwd=str(args.repo),
    )
    evidence = {
        "purpose": "Actual MCP asset observation and recording demonstration; outside timed agent trials",
        "scenario": args.scenario,
        "variant": args.variant,
        "source_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "client": "official Python MCP SDK",
        "profile": "code",
        "calls": [],
    }
    async with (
        stdio_client(parameters) as (read, write),
        ClientSession(read, write) as client,
    ):
        initialized = await client.initialize()
        evidence["protocol_version"] = initialized.protocolVersion
        evidence["server"] = initialized.serverInfo.model_dump()
        evidence["tools"] = [tool.name for tool in (await client.list_tools()).tools]

        async def call(name, arguments):
            start = time.perf_counter()
            response = await client.call_tool(name, arguments)
            if response.isError:
                raise RuntimeError(str(response.content))
            result = response.structuredContent
            evidence["calls"].append(
                {
                    "tool": name,
                    "arguments": arguments,
                    "elapsed_seconds": time.perf_counter() - start,
                    "result": sanitize(result),
                }
            )
            return response, result

        await call("newton_describe", {})
        recording = camera | {"action": "start", "every_steps": 50, "max_frames": 31}
        code = (
            "session.dispatch('reset')\n"
            f"session.dispatch('record', {recording!r})\n"
            "session.dispatch('step', {'count': 1500})\n"
            "result = {'metrics': session.scenario.metrics(), 'recording': session.dispatch('record', {'action': 'stop'})}\n"
        )
        _, rolled = await call("newton_execute", {"code": code})
        for suffix, options in {
            "color": {},
            "albedo": {"channel": "albedo"},
            "camera-two": {
                "eye": [-camera["eye"][0], camera["eye"][1], camera["eye"][2]]
            },
        }.items():
            response, _ = await call("newton_observe", camera | options)
            images = [item for item in response.content if item.type == "image"]
            assert len(images) == 1 and images[0].mimeType == "image/png"
            (args.output / "assets" / f"{args.scenario}-mcp-{suffix}.png").write_bytes(
                base64.b64decode(images[0].data)
            )

        # Copy only the requested recording sequence, never the connection file.
        recording_directory = Path(rolled["result"]["recording"]["directory"])
        manifest = json.loads((recording_directory / "manifest.json").read_text())
        target = args.output / "assets" / f"{args.scenario}-mcp-recording"
        target.mkdir(exist_ok=True)
        for frame in recording_directory.glob("frame-*.png"):
            shutil.copy2(frame, target / frame.name)
        (target / "manifest.json").write_text(
            json.dumps(sanitize(manifest), indent=2) + "\n"
        )
        evidence["recording"] = sanitize(manifest)
    (args.output / "data" / f"{args.scenario}-mcp-evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--scenario", choices=("panda", "allegro", "hug"), required=True
    )
    parser.add_argument("--variant", type=int, choices=(0, 1), default=0)
    parser.add_argument("--camera-scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.repo, args.config, args.output = (
        args.repo.resolve(),
        args.config.resolve(),
        args.output.resolve(),
    )
    (args.output / "assets").mkdir(parents=True, exist_ok=True)
    (args.output / "data").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="newton-mcp-capture-") as directory:
        private = Path(directory)
        with (private / "server.log").open("w") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "tools.mcp_evaluation.rollout",
                    "--scenario",
                    args.scenario,
                    "--variant",
                    str(args.variant),
                    "--config",
                    str(args.config),
                    "--live",
                    "--connection-file",
                    str(private / "session.json"),
                    "--output",
                    str(private / "metrics.json"),
                ],
                cwd=args.repo,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + 60
                while not (private / "server_ready.json").exists():
                    if process.poll() is not None:
                        raise RuntimeError(
                            "Simulation server exited before becoming ready"
                        )
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Simulation server did not become ready")
                    time.sleep(0.1)
                asyncio.run(capture(args, private))
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    print(
        json.dumps(
            {
                "scenario": args.scenario,
                "evidence": f"data/{args.scenario}-mcp-evidence.json",
            }
        )
    )


if __name__ == "__main__":
    main()
