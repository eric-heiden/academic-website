#!/usr/bin/env python3
"""Probe a Unitree G1 URDF against an installed SuperDex source build.

The probe records three independently checkable boundaries: what the URDF
declares, what the SuperDex runtime importer retains, and what happens during a
short gravity-driven simulation. JSON is printed to stdout and may also be
written with ``--output``.

Example, from a pinned Project SuperDex checkout::

    CC=clang-17 CXX=clang++-17 uv sync --extra core
    uv run /path/to/probe_superdex_g1.py \
      --urdf /path/to/unitree_g1/urdf/g1_29dof.urdf \
      --output superdex-g1-probe.json

Creating the articulated actor can take several minutes because raw collision
meshes are processed during import.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import superdex.physics as sdp
import superdex.robotics as sdr


DEFAULT_STEPS = 10
MAX_STEPS = 100
TIME_STEP_S = 0.002
START_HEIGHT_M = 0.85
GRAVITY_M_S2 = 9.81


def bounded_steps(value: str) -> int:
    """Parse a deliberately short, bounded smoke-test length."""
    try:
        steps = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("steps must be an integer") from error
    if not 1 <= steps <= MAX_STEPS:
        raise argparse.ArgumentTypeError(f"steps must be between 1 and {MAX_STEPS}")
    return steps


def local_name(tag: str) -> str:
    """Return an XML tag without an optional namespace."""
    return tag.rsplit("}", 1)[-1]


def direct_children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if local_name(child.tag) == name]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_urdf(path: Path) -> dict[str, Any]:
    """Summarize geometry and joints directly from the input URDF."""
    root = ET.parse(path).getroot()
    links = direct_children(root, "link")
    joints = direct_children(root, "joint")

    geometry_counts: dict[str, Counter[str]] = {
        "visual": Counter(),
        "collision": Counter(),
    }
    geometry_by_link: dict[str, dict[str, list[dict[str, str]]]] = {}
    package_uris: list[str] = []

    for link in links:
        link_name = link.get("name", "")
        entry = {"visual": [], "collision": []}
        for role in ("visual", "collision"):
            for element in direct_children(link, role):
                geometries = direct_children(element, "geometry")
                if not geometries:
                    continue
                for shape in geometries[0]:
                    shape_type = local_name(shape.tag)
                    if shape_type not in {"mesh", "box", "sphere", "cylinder"}:
                        continue
                    record: dict[str, str] = {"type": shape_type}
                    for key, value in sorted(shape.attrib.items()):
                        record[key] = value
                    entry[role].append(record)
                    geometry_counts[role][shape_type] += 1
                    if shape_type == "mesh":
                        uri = shape.get("filename", "")
                        if uri.startswith("package://"):
                            package_uris.append(uri)
        geometry_by_link[link_name] = entry

    primitive_types = {"box", "sphere", "cylinder"}
    primitive_collision_links = sorted(
        name
        for name, entry in geometry_by_link.items()
        if any(item["type"] in primitive_types for item in entry["collision"])
    )
    primitive_only_collision_links = sorted(
        name
        for name, entry in geometry_by_link.items()
        if entry["collision"]
        and all(item["type"] in primitive_types for item in entry["collision"])
    )

    return {
        "robot_name": root.get("name"),
        "sha256": sha256_file(path),
        "link_count": len(links),
        "joint_count": len(joints),
        "joint_types": dict(sorted(Counter(joint.get("type", "") for joint in joints).items())),
        "geometry_counts": {
            role: dict(sorted(counts.items())) for role, counts in geometry_counts.items()
        },
        "package_uri_count": len(package_uris),
        "package_uris_unique": sorted(set(package_uris)),
        "primitive_collision_links": primitive_collision_links,
        "primitive_only_collision_links": primitive_only_collision_links,
        "geometry_by_link": geometry_by_link,
    }


def run_git(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def find_git_root(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def distribution_source(name: str) -> tuple[str, Path | None, dict[str, Any] | None]:
    """Return distribution version, editable path, and direct-url metadata."""
    distribution = importlib.metadata.distribution(name)
    direct_url_text = distribution.read_text("direct_url.json")
    direct_url = json.loads(direct_url_text) if direct_url_text else None
    source_path = None
    if direct_url and str(direct_url.get("url", "")).startswith("file:"):
        parsed = urlparse(direct_url["url"])
        source_path = Path(unquote(parsed.path)).resolve()
    return distribution.version, source_path, direct_url


def source_build_metadata() -> dict[str, Any]:
    physics_version, source_path, direct_url = distribution_source("superdex-physics")
    robotics_version, _, robotics_direct_url = distribution_source("superdex-robotics")

    search_start = source_path or Path(sdp.__file__).resolve()
    repo = find_git_root(search_start)
    result: dict[str, Any] = {
        "superdex_physics_version": physics_version,
        "superdex_robotics_version": robotics_version,
        "physics_module": str(Path(sdp.__file__).resolve()),
        "physics_direct_url": direct_url,
        "robotics_direct_url": robotics_direct_url,
        "uses_double_precision": bool(sdp.uses_double_precision()),
    }
    if repo is None:
        result["source_revision"] = None
        return result

    result.update(
        {
            "source_repository": str(repo),
            "source_revision": run_git(repo, "rev-parse", "HEAD"),
            "source_revision_date": run_git(repo, "show", "-s", "--format=%cI", "HEAD"),
            "source_revision_subject": run_git(repo, "show", "-s", "--format=%s", "HEAD"),
        }
    )

    caches = list((repo / "build-fp32").glob("*/CMakeCache.txt"))
    if caches:
        cache = max(caches, key=lambda path: path.stat().st_mtime_ns)
        values: dict[str, str] = {}
        wanted = {
            "CMAKE_C_COMPILER",
            "CMAKE_CXX_COMPILER",
            "CMAKE_BUILD_TYPE",
            "MOCHI_USE_CUDA",
        }
        for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line or line.startswith(("#", "//")) or "=" not in line:
                continue
            key_with_type, value = line.split("=", 1)
            key = key_with_type.split(":", 1)[0]
            if key in wanted:
                values[key] = value
        result["cmake_cache"] = str(cache)
        result["cmake_values"] = values
        compiler = values.get("CMAKE_CXX_COMPILER")
        if compiler:
            try:
                result["compiler_version"] = subprocess.check_output(
                    [compiler, "--version"], text=True, stderr=subprocess.STDOUT
                ).splitlines()[0]
            except (OSError, subprocess.CalledProcessError):
                pass
    return result


def prefab_metadata(prefab: Any, urdf: dict[str, Any]) -> dict[str, Any]:
    links: list[dict[str, Any]] = []
    for index in range(len(prefab.links)):
        link = prefab.links[index]
        shape_file = str(link.shape_file or "")
        render_model_file = str(link.render_model_file or "")
        links.append(
            {
                "index": index,
                "name": link.name,
                "shape_file": shape_file,
                "shape_file_is_absolute": bool(shape_file and Path(shape_file).is_absolute()),
                "shape_file_exists": bool(shape_file and Path(shape_file).exists()),
                "render_model_file": render_model_file,
                "render_model_file_is_absolute": bool(
                    render_model_file and Path(render_model_file).is_absolute()
                ),
                "render_model_file_exists": bool(
                    render_model_file and Path(render_model_file).exists()
                ),
            }
        )

    loaded_shape_links = {link["name"] for link in links if link["shape_file"]}
    primitive_links = set(urdf["primitive_collision_links"])
    primitive_only_links = set(urdf["primitive_only_collision_links"])
    joint_types = Counter(prefab.joints[index].type.name for index in range(len(prefab.joints)))

    return {
        "name": prefab.name,
        "link_count": len(prefab.links),
        "joint_count": len(prefab.joints),
        "joint_types": dict(sorted(joint_types.items())),
        "root_joint_name": prefab.joints[0].name if len(prefab.joints) else None,
        "root_joint_type": prefab.joints[0].type.name if len(prefab.joints) else None,
        "default_pose_value_count": len(prefab.default_pose),
        "links_with_collision_shape": len(loaded_shape_links),
        "links_with_render_model": sum(bool(link["render_model_file"]) for link in links),
        "all_loaded_shape_paths_absolute": all(
            link["shape_file_is_absolute"] for link in links if link["shape_file"]
        ),
        "all_loaded_shape_paths_exist": all(
            link["shape_file_exists"] for link in links if link["shape_file"]
        ),
        "primitive_collision_links_with_loaded_shape": sorted(primitive_links & loaded_shape_links),
        "primitive_only_collision_links_without_loaded_shape": sorted(
            primitive_only_links - loaded_shape_links
        ),
        "links": links,
    }


def json_float(value: float) -> float | str:
    value = float(value)
    if math.isfinite(value):
        return value
    if math.isnan(value):
        return "nan"
    return "inf" if value > 0 else "-inf"


def transform_translation(transform: Any) -> list[float | str]:
    return [json_float(value) for value in np.asarray(transform.translation, dtype=float)]


def run_simulation(prefab: Any, steps: int, logs: list[dict[str, Any]]) -> dict[str, Any]:
    """Instantiate the imported prefab and perform a short gravity smoke test."""
    log_counts: dict[tuple[str, str, str, int], int] = {}
    log_lock = threading.Lock()

    def log_callback(channel: Any, message: str, file: str, line: int) -> None:
        key = (channel.name, message.strip(), file, int(line))
        with log_lock:
            log_counts[key] = log_counts.get(key, 0) + 1

    initialized = False
    scene = None
    bot = None
    result: dict[str, Any] = {
        "requested_steps": steps,
        "completed_steps": 0,
        "time_step_s": TIME_STEP_S,
        "start_height_m": START_HEIGHT_M,
        "gravity_m_s2": [0.0, 0.0, -GRAVITY_M_S2],
    }

    try:
        sdp.initialize(num_worker_threads=0)
        initialized = True
        sdp.set_log_callback(log_callback)

        scene = sdp.create_scene("SuperDex G1 import probe")
        scene.set_gravity([0.0, 0.0, -GRAVITY_M_S2])
        plane_shape = sdp.create_plane_shape(normal=[0.0, 0.0, 1.0], distance=0.0)
        scene.create_rigid_actor(name="ground", shape=plane_shape, is_static=True)

        prefab.world_from_root = sdp.TransformRT(translation=[0.0, 0.0, START_HEIGHT_M])
        robotics_context = sdr.create_context()
        create_started = time.perf_counter()
        bot = sdr.create_bot(scene, prefab, robotics_context)
        result["actor_creation_wall_time_s"] = round(time.perf_counter() - create_started, 6)

        actor = bot.get_articulated_actor()
        result["runtime_dof_count"] = actor.get_num_dofs()
        result["runtime_nested_link_count"] = len(actor.get_nested_link_actors())
        result["initial_root_translation_m"] = transform_translation(actor.get_root_transform())

        step_records: list[dict[str, Any]] = []
        for index in range(steps):
            step_started = time.perf_counter()
            scene.step(TIME_STEP_S)
            stats = scene.get_solver_stats()
            root_translation = transform_translation(actor.get_root_transform())
            step_records.append(
                {
                    "step": index + 1,
                    "scene_time_s": json_float(scene.get_total_simulation_time()),
                    "root_translation_m": root_translation,
                    "solver_convergence_status": stats.convergence_status.name,
                    "solver_residual_norm": json_float(stats.residual_norm),
                    "max_non_linear_iterations": stats.max_non_linear_iters,
                    "max_line_search_iterations": stats.max_line_search_iters,
                    "wall_time_s": round(time.perf_counter() - step_started, 6),
                }
            )
            result["completed_steps"] = index + 1

        pose = sdp.DynamicArrayReal(actor.get_num_dofs())
        actor.get_articulated_pose(pose)
        pose_array = np.asarray(pose, dtype=float)
        initial_root = np.asarray(result["initial_root_translation_m"], dtype=float)
        final_root = np.asarray(step_records[-1]["root_translation_m"], dtype=float)
        result.update(
            {
                "step_records": step_records,
                "final_root_translation_m": step_records[-1]["root_translation_m"],
                "root_displacement_m": json_float(np.linalg.norm(final_root - initial_root)),
                "root_vertical_displacement_m": json_float(final_root[2] - initial_root[2]),
                "articulated_pose_all_finite": bool(np.isfinite(pose_array).all()),
                "convergence_status_counts": dict(
                    sorted(Counter(record["solver_convergence_status"] for record in step_records).items())
                ),
            }
        )
    except Exception as error:  # Preserve a machine-readable partial result.
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        if bot is not None and scene is not None:
            try:
                sdr.destroy_bot(scene, bot)
            except Exception as error:
                result["cleanup_error"] = {"type": type(error).__name__, "message": str(error)}
        if initialized:
            sdp.shutdown()

        with log_lock:
            for (channel, message, file, line), count in log_counts.items():
                logs.append(
                    {
                        "channel": channel,
                        "message": message,
                        "source_file": file,
                        "source_line": line,
                        "count": count,
                    }
                )

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True, help="Path to the G1 URDF")
    parser.add_argument(
        "--steps",
        type=bounded_steps,
        default=DEFAULT_STEPS,
        help=f"Simulation steps (default: {DEFAULT_STEPS}; maximum: {MAX_STEPS})",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    urdf_path = args.urdf.expanduser().resolve()
    if not urdf_path.is_file():
        parser.error(f"URDF does not exist: {urdf_path}")

    started = time.perf_counter()
    document: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "probe": "SuperDex Unitree G1 URDF import and bounded gravity simulation",
        "host": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "source_build": source_build_metadata(),
        "input": {"urdf": str(urdf_path)},
    }

    urdf = inspect_urdf(urdf_path)
    document["urdf"] = urdf
    prefab = sdr.load_bot_prefab_from_urdf_file(str(urdf_path))
    document["imported_prefab"] = prefab_metadata(prefab, urdf)
    logs: list[dict[str, Any]] = []
    document["simulation"] = run_simulation(prefab, args.steps, logs)
    document["superdex_logs"] = logs
    document["total_wall_time_s"] = round(time.perf_counter() - started, 6)

    encoded = json.dumps(document, indent=2, sort_keys=False, allow_nan=False) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)
    return 1 if "error" in document["simulation"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
