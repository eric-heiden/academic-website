"""Capture a second camera pose and albedo with the public Newton MCP API."""

import base64
import json
import sys
from pathlib import Path

import newton
import warp as wp
from newton.mcp import SimulationSession
from newton.solvers import SolverXPBD


def main():
    output = Path(__file__).parent
    builder = newton.ModelBuilder()
    builder.add_ground_plane(color=(0.25, 0.28, 0.34))
    sphere = builder.add_body(
        xform=wp.transform(wp.vec3(-0.7, 0, 0.48), wp.quat_identity())
    )
    builder.add_shape_sphere(sphere, radius=0.5, color=(0.94, 0.24, 0.1))
    box = builder.add_body(
        xform=wp.transform(wp.vec3(0.6, 0, 0.45), wp.quat_identity())
    )
    builder.add_shape_box(box, hx=0.4, hy=0.4, hz=0.45, color=(0.12, 0.58, 0.95))
    capsule = builder.add_body(
        xform=wp.transform(wp.vec3(0, 0.9, 0.7), wp.quat_identity())
    )
    builder.add_shape_capsule(
        capsule, radius=0.2, half_height=0.5, color=(0.95, 0.7, 0.12)
    )
    model = builder.finalize(device="cpu")
    session = SimulationSession(
        model, SolverXPBD(model), artifact_directory=output / "data"
    )
    camera = {
        "width": 384,
        "height": 256,
        "eye": [3, -4, 2.8],
        "target": [0, 0.15, 0.4],
        "up": [0, 0, 1],
        "fov_y": 48,
    }
    evidence = {"device": "cpu", "viewer": False, "captures": {}}
    try:
        session.dispatch("collide")
        for name, options in {
            "sensor-albedo": {"channel": "albedo"},
            "sensor-camera-two": {"eye": [-3, -3.2, 2.0], "channel": "color"},
        }.items():
            result = session.dispatch("observe", camera | options)
            png = base64.b64decode(result.pop("image_base64"))
            (output / "assets" / f"{name}.png").write_bytes(png)
            evidence["captures"][name] = result | {"png_bytes": len(png)}
        evidence["pyglet_imported"] = "pyglet" in sys.modules
        assert not evidence["pyglet_imported"]
        (output / "data" / "rendering-additional.json").write_text(
            json.dumps(evidence, indent=2) + "\n"
        )
        print(
            json.dumps(
                {"captures": list(evidence["captures"]), "pyglet_imported": False}
            )
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()
