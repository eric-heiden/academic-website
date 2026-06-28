from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
from fractions import Fraction
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit


SCHEMA_VERSION = 1
REQUIRED_SECTION_IDS = (
    "scope",
    "semantics",
    "literature",
    "grammar",
    "protocol",
    "analytic",
    "path-tracing",
    "contact-3d",
    "opaque-mesh",
    "gradient-quality",
    "optimization",
    "performance",
    "limitations",
    "reproducibility",
)
REQUIRED_METHOD_IDS = (
    "crisp_ad",
    "crisp_fd",
    "smoothed_pathwise",
    "score",
    "smoothed_crn_fd",
    "soft_ad",
    "straight_through_ad",
    "residual_control_variate",
)
REQUIRED_SCENARIO_FAMILIES = (
    "analytic",
    "triangle_2d",
    "collision_2d",
    "path_tracer",
    "contact_3d",
    "opaque_mesh",
)
REQUIRED_LITERATURE_URLS = (
    "https://arxiv.org/abs/2109.05143",
    "https://arxiv.org/abs/2310.03585",
    "https://github.com/DiscoGrad/DiscoGrad",
    "https://github.com/a-paulus/softtorch",
    "https://arxiv.org/abs/2603.08824",
)
EXPECTED_REFERENCE_CELLS = (
    "triangle_2d:edge_midpoints",
    "collision_2d:pinball_bank:start_0",
    "collision_2d:pinball_bank:start_1",
    "collision_2d:pinball_bank:start_2",
    "collision_2d:crowded_table:start_0",
    "collision_2d:crowded_table:start_1",
    "collision_2d:crowded_table:start_2",
    "path_tracer:initial_parameters",
    "contact_3d:initial_launch_velocity",
    "opaque_mesh:camera_parameters",
)
RAW_NAMES = (
    "analytic.json",
    "triangle_2d.json",
    "collision_2d.json",
    "path_tracer_gradients.json",
    "path_tracer_optimization.json",
    "contact_3d_gradients.json",
    "contact_3d_optimization.json",
    "opaque_mesh.json",
    "performance.json",
    "references.json",
)
PLOT_NAMES = (
    "analytic_gates.json",
    "gradient_quality.json",
    "bias_variance.json",
    "optimization.json",
    "validity.json",
    "performance.json",
)
FIGURE_NAMES = (
    "analytic_gates.png",
    "triangle_edge_slices.png",
    "collision_2d.png",
    "path_tracer_recovery.png",
    "path_tracer_gradient_quality.png",
    "contact_3d_trajectories.png",
    "contact_3d_gradient_quality.png",
    "opaque_mesh_boundary.png",
    "bias_variance.png",
    "optimization.png",
    "performance.png",
)
IMAGE_NAMES = ("path_target.png", "path_initial.png", "path_recovered.png")
EXPECTED_FILES = frozenset(
    ("data/summary.json",)
    + tuple(f"data/raw/{name}" for name in RAW_NAMES)
    + tuple(f"data/plot_data/{name}" for name in PLOT_NAMES)
    + tuple(f"assets/figures/{name}" for name in FIGURE_NAMES)
    + tuple(f"assets/images/{name}" for name in IMAGE_NAMES)
)
LEGACY_FILES = frozenset(
    {
        "results.json",
        "benchmark_speed.png",
        "collision_convergence.png",
        "collision_trajectories.png",
        "gradient_variance.png",
        "triangle_gradients.png",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
_WINDOWS_PATH_RE = re.compile(r"(?:^|\s)[A-Za-z]:[\\/]")
_EMBEDDED_POSIX_PATH_RE = re.compile(r"(?<![A-Za-z0-9:/])/(?!/)[^\s\"'<>]+")
_PROGRAM_MODULE = "warp.examples.optim.example_program_smoothing"
_COMMAND_PREFIXES = (
    ("uv", "run", "--extra", "examples", "-m", _PROGRAM_MODULE),
    ("uv", "run", "-m", _PROGRAM_MODULE),
    ("python", "-m", _PROGRAM_MODULE),
)


def _css_has_external_or_active_content(value: str) -> bool:
    if "\\" in value:
        return True
    without_comments = re.sub(r"/\*.*?\*/", "", value, flags=re.DOTALL)
    normalized = re.sub(r"\s+", "", without_comments).casefold()
    return any(
        token in normalized
        for token in (
            "url(",
            "@import",
            "image-set(",
            "-webkit-image-set(",
            "expression(",
            "javascript:",
            "behavior:",
            "-moz-binding:",
        )
    )


class ValidationError(ValueError):
    """Raised when a publication artifact violates its evidence contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValidationError(f"non-finite JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _validate_finite_tree(value: Any, *, location: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValidationError(f"non-finite number at {location}")
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_finite_tree(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_finite_tree(child, location=f"{location}[{index}]")


def load_json_finite(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValidationError(f"cannot read UTF-8 JSON {path}: {error}") from error
    try:
        value = json.loads(
            text, parse_constant=_reject_constant, object_pairs_hook=_unique_object
        )
    except ValidationError:
        raise
    except json.JSONDecodeError as error:
        raise ValidationError(f"invalid JSON in {path}: {error}") from error
    _validate_finite_tree(value, location=path.as_posix())
    return value


def _load_json_bytes_finite(data: bytes, *, location: str) -> Any:
    try:
        text = data.decode("utf-8")
    except UnicodeError as error:
        raise ValidationError(
            f"cannot decode UTF-8 JSON {location}: {error}"
        ) from error
    try:
        value = json.loads(
            text, parse_constant=_reject_constant, object_pairs_hook=_unique_object
        )
    except ValidationError:
        raise
    except json.JSONDecodeError as error:
        raise ValidationError(f"invalid JSON in {location}: {error}") from error
    _validate_finite_tree(value, location=location)
    return value


def _safe_relative_path(root: Path, raw: Any) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValidationError("manifest file names must be nonempty strings")
    if "\\" in raw:
        raise ValidationError(f"manifest path must use POSIX separators: {raw!r}")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValidationError(f"unsafe manifest path {raw!r}")
    candidate = root.joinpath(*pure.parts)
    cursor = root
    for part in pure.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValidationError(f"symlink is not allowed in publication path {raw!r}")
    try:
        candidate.resolve(strict=False).relative_to(root.resolve())
    except ValueError as error:
        raise ValidationError(f"manifest path escapes report root: {raw!r}") from error
    return candidate


def _relative_parts(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise ValidationError(
            f"artifact path must be a nonempty relative POSIX path: {raw!r}"
        )
    pure = PurePosixPath(raw)
    if (
        pure.is_absolute()
        or pure.as_posix() != raw
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValidationError(f"unsafe manifest path {raw!r}")
    return pure.parts


def _read_artifact_at(root_descriptor: int, relative: str) -> bytes:
    parts = _relative_parts(relative)
    directory_flags = (
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_BINARY", 0)
    )
    directory_descriptor = os.dup(root_descriptor)
    file_descriptor: int | None = None
    try:
        try:
            for component in parts[:-1]:
                next_descriptor = os.open(
                    component, directory_flags, dir_fd=directory_descriptor
                )
                os.close(directory_descriptor)
                directory_descriptor = next_descriptor
            file_descriptor = os.open(
                parts[-1], file_flags, dir_fd=directory_descriptor
            )
        except OSError as error:
            raise ValidationError(
                f"artifact is missing, non-regular, or traverses a symlink: {relative}"
            ) from error
        if not stat.S_ISREG(os.fstat(file_descriptor).st_mode):
            raise ValidationError(f"artifact is not a regular file: {relative}")
        chunks: list[bytes] = []
        while block := os.read(file_descriptor, 1024 * 1024):
            chunks.append(block)
        return b"".join(chunks)
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        os.close(directory_descriptor)


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"{name} must be a nonnegative integer")
    return value


def _finite_number(value: Any, *, name: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be finite")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValidationError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValidationError(f"{name} must be finite")
    if nonnegative and result < 0.0:
        raise ValidationError(f"{name} must be nonnegative")
    return result


def _finite_vector(value: Any, *, name: str, nonnegative: bool = False) -> list[float]:
    if not isinstance(value, list):
        raise ValidationError(f"{name} must be a JSON array")
    return [
        _finite_number(item, name=f"{name}[{index}]", nonnegative=nonnegative)
        for index, item in enumerate(value)
    ]


def _assert_no_local_path(value: Any, *, location: str) -> None:
    if isinstance(value, str):
        stripped = value.strip()
        is_web_url = (
            stripped.startswith(("https://", "http://")) and " " not in stripped
        )
        if (
            stripped.casefold().startswith("file:")
            or stripped.startswith(("/", "\\\\"))
            or _WINDOWS_PATH_RE.search(value)
            or (not is_web_url and _EMBEDDED_POSIX_PATH_RE.search(value))
        ):
            raise ValidationError(f"local filesystem path leaked at {location}")
    elif isinstance(value, dict):
        for key, child in value.items():
            _assert_no_local_path(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_local_path(child, location=f"{location}[{index}]")


def validate_manifest(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(root)
    if not (
        os.name == "posix"
        and hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
        and os.open in os.supports_dir_fd
    ):
        raise ValidationError(
            "secure publication validation requires descriptor-relative O_NOFOLLOW support"
        )
    root_flags = (
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        root_descriptor = os.open(os.fspath(root), root_flags)
    except OSError as error:
        raise ValidationError(
            "report root is missing, not a directory, or is a symbolic link"
        ) from error
    try:
        manifest_bytes = _read_artifact_at(root_descriptor, "data/manifest.json")
        manifest = _load_json_bytes_finite(
            manifest_bytes, location="data/manifest.json"
        )
    except ValidationError as error:
        os.close(root_descriptor)
        if "data/manifest.json" not in str(error):
            raise ValidationError(f"data/manifest.json is invalid: {error}") from error
        raise
    if not isinstance(manifest, dict):
        os.close(root_descriptor)
        raise ValidationError("manifest must be a JSON object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        os.close(root_descriptor)
        raise ValidationError("manifest schema_version must be 1")
    if manifest.get("tier") != "report":
        os.close(root_descriptor)
        raise ValidationError("manifest tier must be 'report'")
    files = manifest.get("files")
    if not isinstance(files, dict):
        os.close(root_descriptor)
        raise ValidationError("manifest files must be an object")
    actual_names = frozenset(files)
    if actual_names != EXPECTED_FILES:
        os.close(root_descriptor)
        missing = sorted(EXPECTED_FILES - actual_names)
        extra = sorted(actual_names - EXPECTED_FILES)
        raise ValidationError(
            f"manifest must declare exactly 31 artifacts; missing={missing}, undeclared={extra}"
        )

    loaded: dict[str, Any] = {}
    try:
        for relative, descriptor in files.items():
            if not isinstance(descriptor, dict) or set(descriptor) != {
                "sha256",
                "bytes",
            }:
                raise ValidationError(f"invalid manifest descriptor for {relative}")
            digest = descriptor.get("sha256")
            if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                raise ValidationError(f"invalid SHA-256 for {relative}")
            byte_count = _nonnegative_int(
                descriptor.get("bytes"), name=f"{relative}.bytes"
            )
            data = _read_artifact_at(root_descriptor, relative)
            if len(data) != byte_count:
                raise ValidationError(f"byte count mismatch for {relative}")
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValidationError(f"SHA-256 mismatch for {relative}")
            if relative.endswith(".json"):
                loaded[relative] = _load_json_bytes_finite(data, location=relative)
    finally:
        os.close(root_descriptor)

    on_disk: set[str] = set()
    for parent in (root / "data", root / "assets"):
        for directory, directory_names, file_names in os.walk(
            parent, followlinks=False
        ):
            directory_path = Path(directory)
            for name in directory_names:
                if (directory_path / name).is_symlink():
                    raise ValidationError(
                        f"symlink is not allowed in publication tree: {(directory_path / name).relative_to(root)}"
                    )
            for name in file_names:
                path = directory_path / name
                if path.is_symlink():
                    raise ValidationError(
                        f"symlink is not allowed in publication tree: {path.relative_to(root)}"
                    )
                relative = path.relative_to(root).as_posix()
                if relative != "data/manifest.json":
                    on_disk.add(relative)
    undeclared = sorted(on_disk - EXPECTED_FILES)
    if undeclared:
        raise ValidationError(f"undeclared publication artifacts: {undeclared}")
    missing_on_disk = sorted(EXPECTED_FILES - on_disk)
    if missing_on_disk:
        raise ValidationError(f"manifest artifacts missing on disk: {missing_on_disk}")
    _assert_no_local_path(manifest, location="manifest")
    for relative, payload in loaded.items():
        _assert_no_local_path(payload, location=relative)
    return manifest, loaded


def _dataset_rows(payload: Any, *, relative: str) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValidationError(f"{relative} must use schema_version 1")
    if not isinstance(payload.get("dataset"), str) or not payload["dataset"]:
        raise ValidationError(f"{relative} has no dataset name")
    rows = payload.get("rows")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValidationError(f"{relative}.rows must be an array of objects")
    return rows


def _validate_variance_fields(
    row: dict[str, Any], *, row_id: str, dimension: int
) -> None:
    for stem in ("contribution_variance", "gradient_variance"):
        flag = row.get(f"{stem}_available")
        value = row.get(stem)
        if not isinstance(flag, bool):
            raise ValidationError(f"{row_id}.{stem}_available must be boolean")
        if flag:
            vector = _finite_vector(value, name=f"{row_id}.{stem}", nonnegative=True)
            if len(vector) != dimension:
                raise ValidationError(f"{row_id}.{stem} dimension mismatch")
        elif value is not None:
            raise ValidationError(f"{row_id}.{stem} must be null when unavailable")
    low_raw = row.get("ci_low")
    high_raw = row.get("ci_high")
    if low_raw is None or high_raw is None:
        if low_raw is not None or high_raw is not None:
            raise ValidationError(
                f"{row_id} confidence bounds must both be null or arrays"
            )
        if row["gradient_variance_available"] and row["independent_contributions"] >= 2:
            raise ValidationError(
                f"{row_id} confidence interval is required when mean variance is available"
            )
        return
    low = _finite_vector(low_raw, name=f"{row_id}.ci_low")
    high = _finite_vector(high_raw, name=f"{row_id}.ci_high")
    if not row["gradient_variance_available"]:
        raise ValidationError(
            f"{row_id} confidence interval requires gradient variance"
        )
    if len(low) != len(high) or any(first > second for first, second in zip(low, high)):
        raise ValidationError(f"{row_id} confidence interval has invalid bounds")
    if len(low) != dimension:
        raise ValidationError(f"{row_id} confidence interval dimension mismatch")


def _validate_method_row(row: dict[str, Any]) -> None:
    row_id = row.get("row_id")
    if not isinstance(row_id, str) or not row_id:
        raise ValidationError("method row has no row_id")
    method = row.get("method")
    if method not in REQUIRED_METHOD_IDS:
        raise ValidationError(f"{row_id} has unknown method {method!r}")
    samples = _nonnegative_int(row.get("samples"), name=f"{row_id}.samples")
    if samples == 0:
        raise ValidationError(f"{row_id}.samples must be positive")
    counts = {}
    for field in (
        "forward_executions",
        "backward_executions",
        "independent_contributions",
        "parameter_perturbations",
    ):
        counts[field] = _nonnegative_int(row.get(field), name=f"{row_id}.{field}")
    _finite_number(row.get("wall_time"), name=f"{row_id}.wall_time", nonnegative=True)
    gradient = _finite_vector(row.get("gradient"), name=f"{row_id}.gradient")
    if not gradient:
        raise ValidationError(f"{row_id}.gradient must be nonempty")
    dimension = len(gradient)
    antithetic = row.get("antithetic")
    if not isinstance(antithetic, bool):
        raise ValidationError(f"{row_id}.antithetic must be boolean")
    if antithetic and samples % 2:
        raise ValidationError(f"{row_id} antithetic samples must be positive and even")
    _validate_variance_fields(row, row_id=row_id, dimension=dimension)

    hard = row.get("hard_forward_executions")
    soft = row.get("soft_forward_executions")
    independent = samples // 2 if antithetic else samples
    expected_underlying = {
        "crisp_ad": "pathwise",
        "crisp_fd": "finite_difference",
        "smoothed_pathwise": "pathwise",
        "score": "score_function",
        "smoothed_crn_fd": "finite_difference",
        "soft_ad": "pathwise",
        "straight_through_ad": "pathwise",
        "residual_control_variate": "control_variate",
    }[method]
    if row.get("underlying_method") not in {None, expected_underlying}:
        raise ValidationError(f"{row_id}.underlying_method disagrees with method")
    if expected_underlying == "score_function":
        expected = (independent, samples, samples, 0, None, None)
    elif expected_underlying == "pathwise":
        expected = (independent, samples, samples, samples, None, None)
    elif expected_underlying == "finite_difference":
        expected = (1, 2 * dimension, samples * (1 + 2 * dimension), 0, None, None)
    else:
        expected = (independent, samples, 2 * samples, samples, samples, samples)

    if expected_underlying == "control_variate":
        hard_count = _nonnegative_int(hard, name=f"{row_id}.hard_forward_executions")
        soft_count = _nonnegative_int(soft, name=f"{row_id}.soft_forward_executions")
    else:
        hard_count = hard
        soft_count = soft
    if expected_underlying != "control_variate" and (
        hard is not None or soft is not None
    ):
        raise ValidationError(
            f"{row_id} hard/soft forward fields are reserved for residual_control_variate"
        )
    actual = (
        counts["independent_contributions"],
        counts["parameter_perturbations"],
        counts["forward_executions"],
        counts["backward_executions"],
        hard_count,
        soft_count,
    )
    if actual != expected:
        raise ValidationError(
            f"{row_id} estimator accounting mismatch: expected {expected}, got {actual}"
        )


def _reference_interval(
    values: list[list[float]], record: Any, *, name: str
) -> tuple[list[float], list[float], list[float]]:
    if not isinstance(record, dict):
        raise ValidationError(f"{name} interval record must be an object")
    dimension = len(values[0])
    replicates = len(values)
    means: list[float] = []
    variances: list[float] = []
    mean_variances: list[float] = []
    for index in range(dimension):
        components = [Fraction.from_float(row[index]) for row in values]
        exact_total = sum(components, Fraction())
        exact_mean = exact_total / replicates
        exact_variance = sum(
            ((value - exact_mean) ** 2 for value in components),
            Fraction(),
        ) / (replicates - 1)
        exact_mean_variance = exact_variance / replicates
        converted: list[float] = []
        for statistic, exact in (
            ("mean", exact_mean),
            ("variance", exact_variance),
            ("variance of the mean", exact_mean_variance),
        ):
            try:
                binary64 = float(exact)
            except OverflowError as error:
                raise ValidationError(
                    f"{name} has unrepresentable {statistic}"
                ) from error
            if not math.isfinite(binary64) or (exact != 0 and binary64 == 0.0):
                raise ValidationError(f"{name} has unrepresentable {statistic}")
            converted.append(binary64)
        means.append(converted[0])
        variances.append(converted[1])
        mean_variances.append(converted[2])
    raw_half_widths = [3.182446305284263 * math.sqrt(value) for value in mean_variances]
    ci_low = [mean - width for mean, width in zip(means, raw_half_widths)]
    ci_high = [mean + width for mean, width in zip(means, raw_half_widths)]
    half_widths = [0.5 * (high - low) for low, high in zip(ci_low, ci_high)]
    expected_vectors = {
        "mean": means,
        "variance": variances,
        "mean_variance": mean_variances,
        "half_width": half_widths,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }
    parsed: dict[str, list[float]] = {}
    for field, expected in expected_vectors.items():
        parsed[field] = _finite_vector(record.get(field), name=f"{name}.{field}")
        if len(parsed[field]) != dimension or any(
            not math.isclose(actual, target, rel_tol=1.0e-10, abs_tol=1.0e-12)
            for actual, target in zip(parsed[field], expected)
        ):
            raise ValidationError(f"{name}.{field} disagrees with stored replicates")
    if record.get("replicates") != replicates:
        raise ValidationError(f"{name}.replicates is inconsistent")
    if record.get("degrees_of_freedom") != replicates - 1:
        raise ValidationError(f"{name}.degrees_of_freedom is inconsistent")
    if _finite_number(record.get("confidence"), name=f"{name}.confidence") != 0.95:
        raise ValidationError(f"{name}.confidence must be 0.95")
    return parsed["mean"], parsed["ci_low"], parsed["ci_high"]


def _validate_reference_rows(
    rows: list[dict[str, Any]],
    required_cells: Any,
    *,
    config: dict[str, Any],
    source_commit: str,
    device: str,
) -> None:
    if not isinstance(required_cells, list) or any(
        not isinstance(cell, str) or not cell for cell in required_cells
    ):
        raise ValidationError(
            "reference_required_cells must be a list of nonempty strings"
        )
    by_cell: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        cell = row.get("cell_id")
        if isinstance(cell, str):
            by_cell.setdefault(cell, []).append(row)
    for cell in required_cells:
        accepted = [
            row
            for row in by_cell.get(cell, ())
            if isinstance(row.get("accepted"), dict)
            and row["accepted"].get("references") is True
        ]
        if not accepted:
            raise ValidationError(f"required cell {cell!r} has no accepted reference")
        if len(accepted) != 1:
            raise ValidationError(
                f"required cell {cell!r} must have exactly one accepted reference"
            )
        for row in accepted:
            if row.get("tier") != "report" or row.get("device") != device:
                raise ValidationError(
                    f"accepted reference {cell!r} has wrong tier/device"
                )
            if row.get("source_commit") != source_commit:
                raise ValidationError(
                    f"accepted reference {cell!r} has wrong source_commit"
                )
            parameters = _finite_vector(
                row.get("parameters"), name=f"{cell}.parameters"
            )
            if not parameters:
                raise ValidationError(
                    f"accepted reference {cell!r} has empty parameters"
                )
            dimension = len(parameters)
            sigma = _finite_vector(row.get("sigma"), name=f"{cell}.sigma")
            h = _finite_vector(row.get("h"), name=f"{cell}.h")
            h_half = _finite_vector(row.get("h_half"), name=f"{cell}.h_half")
            if any(len(vector) != dimension for vector in (sigma, h, h_half)) or any(
                step <= 0.0 for step in sigma + h + h_half
            ):
                raise ValidationError(
                    f"accepted reference {cell!r} has invalid vector dimensions"
                )
            if any(first != 2.0 * second for first, second in zip(h, h_half)):
                raise ValidationError(
                    f"accepted reference {cell!r} must store exact h and h/2"
                )

            matrices: dict[str, list[list[float]]] = {}
            for field in ("g_h", "g_h2", "score"):
                raw_matrix = row.get(field)
                if not isinstance(raw_matrix, list) or len(raw_matrix) != 4:
                    raise ValidationError(
                        f"accepted reference {cell!r} needs four {field} replicates"
                    )
                matrix = [
                    _finite_vector(vector, name=f"{cell}.{field}[{index}]")
                    for index, vector in enumerate(raw_matrix)
                ]
                if any(len(vector) != dimension for vector in matrix):
                    raise ValidationError(
                        f"accepted reference {cell!r} {field} dimension mismatch"
                    )
                matrices[field] = matrix
            paired = [
                [first - second for first, second in zip(g_h, g_h2)]
                for g_h, g_h2 in zip(matrices["g_h"], matrices["g_h2"])
            ]
            if any(not math.isfinite(value) for vector in paired for value in vector):
                raise ValidationError(
                    f"accepted reference {cell!r} paired differences are non-finite"
                )
            intervals = row.get("intervals")
            if not isinstance(intervals, dict):
                raise ValidationError(
                    f"accepted reference {cell!r} intervals must be an object"
                )
            h_mean, h_low, h_high = _reference_interval(
                matrices["g_h"],
                intervals.get("g_h"),
                name=f"{cell}.intervals.g_h",
            )
            h2_mean, h2_low, h2_high = _reference_interval(
                matrices["g_h2"],
                intervals.get("g_h2"),
                name=f"{cell}.intervals.g_h2",
            )
            _score_mean, score_low, score_high = _reference_interval(
                matrices["score"],
                intervals.get("score"),
                name=f"{cell}.intervals.score",
            )
            _paired_mean, paired_low, paired_high = _reference_interval(
                paired,
                intervals.get("paired_h_minus_h2"),
                name=f"{cell}.intervals.paired_h_minus_h2",
            )
            reference_gradient = _finite_vector(
                row.get("reference_gradient"), name=f"{cell}.reference_gradient"
            )
            if reference_gradient != h2_mean:
                raise ValidationError(
                    f"accepted reference {cell!r} reference_gradient must equal g_h2 mean"
                )

            overlap = [
                max(first, third) <= min(second, fourth)
                for first, second, third, fourth in zip(
                    h2_low, h2_high, score_low, score_high
                )
            ]
            paired_consistency = [
                low <= 0.0 <= high for low, high in zip(paired_low, paired_high)
            ]
            marginal_consistency = [
                abs(first - second)
                <= (first_high - first_low + second_high - second_low) / 2.0
                for first, second, first_low, first_high, second_low, second_high in zip(
                    h_mean, h2_mean, h_low, h_high, h2_low, h2_high
                )
            ]
            if not all(overlap) or not all(paired_consistency):
                raise ValidationError(
                    f"accepted reference {cell!r} fails overlap or paired step consistency"
                )
            diagnostics = row.get("diagnostics")
            expected_diagnostics = {
                "overlap_components": overlap,
                "marginal_step_components": marginal_consistency,
                "paired_step_components": paired_consistency,
            }
            if not isinstance(diagnostics, dict) or any(
                diagnostics.get(name) != expected
                for name, expected in expected_diagnostics.items()
            ):
                raise ValidationError(
                    f"accepted reference {cell!r} diagnostics disagree with replicates"
                )
            accepted_flags = row["accepted"]
            required_flags = {
                "references": True,
                "fd_score_overlap": all(overlap),
                "step_consistency": all(paired_consistency),
                "marginal_step_consistency": all(marginal_consistency),
                "paired_step_consistency": all(paired_consistency),
                "replicate_count_sufficient": True,
                "smoke_only": False,
            }
            if any(
                accepted_flags.get(name) is not value
                for name, value in required_flags.items()
            ):
                raise ValidationError(
                    f"accepted reference {cell!r} acceptance flags are inconsistent"
                )

            counts = row.get("counts")
            if not isinstance(counts, dict):
                raise ValidationError(
                    f"accepted reference {cell!r} counts must be an object"
                )
            sample_field = (
                "contact_reference_samples"
                if cell.startswith("contact_3d:")
                else "path_reference_samples"
            )
            samples = _nonnegative_int(
                config.get(sample_field), name=f"config.{sample_field}"
            )
            replicates = _nonnegative_int(
                config.get("reference_seed_sets"), name="config.reference_seed_sets"
            )
            if replicates != 4 or samples == 0 or samples % 2:
                raise ValidationError(
                    "report reference configuration must use four replicates and even samples"
                )
            expected_single = 4 * dimension * samples * replicates
            expected_counts = {
                "samples": samples,
                "replicates": replicates,
                "h_forward_executions": expected_single,
                "h2_forward_executions": expected_single,
                "five_point_forward_executions": 2 * expected_single,
                "score_forward_executions": samples * replicates,
                "forward_executions": 2 * expected_single + samples * replicates,
            }
            if any(
                counts.get(name) != expected
                for name, expected in expected_counts.items()
            ):
                raise ValidationError(
                    f"accepted reference {cell!r} forward execution counts are inconsistent"
                )

            seeds = row.get("seeds")
            seed_names = (
                "five_point_outer",
                "five_point_inner",
                "score_outer",
                "score_inner",
            )
            if not isinstance(seeds, dict) or set(seeds) != set(seed_names):
                raise ValidationError(
                    f"accepted reference {cell!r} seed streams are invalid"
                )
            domain_size = (2**31) // 4
            flattened: list[int] = []
            offsets: list[list[int]] = []
            for domain, name in enumerate(seed_names):
                stream = seeds[name]
                if not isinstance(stream, list) or len(stream) != replicates:
                    raise ValidationError(
                        f"accepted reference {cell!r} seed stream {name} has wrong length"
                    )
                parsed = [
                    _nonnegative_int(seed, name=f"{cell}.seeds.{name}")
                    for seed in stream
                ]
                if any(
                    seed > 2**31 - 1 or seed // domain_size != domain for seed in parsed
                ):
                    raise ValidationError(
                        f"accepted reference {cell!r} seed stream {name} has wrong domain"
                    )
                flattened.extend(parsed)
                offsets.append([seed % domain_size for seed in parsed])
            if len(flattened) != len(set(flattened)) or any(
                offset != offsets[0] for offset in offsets[1:]
            ):
                raise ValidationError(
                    f"accepted reference {cell!r} seed streams are not CRN-aligned and disjoint"
                )
            if offsets[0] != [
                (offsets[0][0] + index) % domain_size for index in range(replicates)
            ]:
                raise ValidationError(
                    f"accepted reference {cell!r} seed offsets are not consecutive"
                )


def _validate_producer_command(command: Any, *, device: str) -> None:
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(part, str) or not part for part in command)
    ):
        raise ValidationError("source command must be a nonempty argv array")
    options: list[str] | None = None
    for prefix in _COMMAND_PREFIXES:
        if tuple(command[: len(prefix)]) == prefix:
            options = command[len(prefix) :]
            break
    if options is None:
        raise ValidationError(
            "source command must be a canonical program-smoothing producer invocation"
        )
    values: dict[str, str] = {}
    headless = False
    index = 0
    while index < len(options):
        option = options[index]
        if option == "--headless":
            if headless:
                raise ValidationError("canonical producer command repeats --headless")
            headless = True
            index += 1
            continue
        if option not in {"--device", "--tier", "--output-dir"} or option in values:
            raise ValidationError(
                "source command must be a canonical report producer invocation"
            )
        if index + 1 >= len(options):
            raise ValidationError(f"source command option {option} has no value")
        values[option] = options[index + 1]
        index += 2
    if values != {
        "--device": device,
        "--tier": "report",
        "--output-dir": "$OUTPUT_DIR",
    }:
        raise ValidationError(
            "source command must select the manifest device, report tier, and $OUTPUT_DIR"
        )


def _validate_seed_tree(value: Any, *, name: str) -> None:
    if isinstance(value, bool):
        raise ValidationError(f"{name} seed metadata must contain nonnegative integers")
    if isinstance(value, int):
        if value < 0 or value > 2**31 - 1:
            raise ValidationError(f"{name} seed is outside Warp's signed 32-bit domain")
        return
    if isinstance(value, list):
        if not value:
            raise ValidationError(f"{name} seed sequence must be nonempty")
        for index, child in enumerate(value):
            _validate_seed_tree(child, name=f"{name}[{index}]")
        return
    if isinstance(value, dict):
        if not value or any(not isinstance(key, str) or not key for key in value):
            raise ValidationError(
                f"{name} seed mapping must be nonempty with string keys"
            )
        for key, child in value.items():
            _validate_seed_tree(child, name=f"{name}.{key}")
        return
    raise ValidationError(f"{name} seed metadata has unsupported type")


def _validate_source(
    manifest: dict[str, Any], method_rows: list[dict[str, Any]]
) -> tuple[str, str]:
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ValidationError("manifest source must be an object")
    if source.get("dirty") is not False:
        raise ValidationError("published Warp source must be clean")
    commit = source.get("commit")
    if not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None:
        raise ValidationError("source commit must be a full lowercase Git object ID")
    device = source.get("device")
    if not isinstance(device, str) or not device:
        raise ValidationError("source device metadata is required")
    _validate_producer_command(source.get("command"), device=device)
    for field in ("python", "warp", "platform", "compiler", "cpu_model"):
        if not isinstance(source.get(field), str) or not source[field]:
            raise ValidationError(f"source {field} metadata is required")
    if _nonnegative_int(source.get("cpu_threads"), name="source.cpu_threads") == 0:
        raise ValidationError("source.cpu_threads must be positive")
    if not isinstance(source.get("seeds"), dict) or not source["seeds"]:
        raise ValidationError("source.seeds must be a nonempty object")
    _validate_seed_tree(source["seeds"], name="source.seeds")
    for row in method_rows:
        row_commit = row.get("source_commit")
        if row_commit != commit:
            raise ValidationError(
                f"{row['row_id']} source_commit disagrees with manifest"
            )
        if row.get("device") != device:
            raise ValidationError(f"{row['row_id']} device disagrees with manifest")
    return commit, device


def _validate_performance_row(row: dict[str, Any]) -> None:
    row_id = row.get("row_id")
    if not isinstance(row_id, str) or not row_id:
        raise ValidationError("performance row has no row_id")
    if row.get("method") not in REQUIRED_METHOD_IDS:
        raise ValidationError(f"{row_id} has an unknown performance method")
    if not isinstance(row.get("scenario"), str) or not row["scenario"]:
        raise ValidationError(f"{row_id}.scenario must be a nonempty string")
    if not isinstance(row.get("device"), str) or not row["device"]:
        raise ValidationError(f"{row_id}.device must be a nonempty string")
    if (
        not isinstance(row.get("source_commit"), str)
        or _COMMIT_RE.fullmatch(row["source_commit"]) is None
    ):
        raise ValidationError(f"{row_id}.source_commit must be a full object ID")
    for field in ("forward_executions", "backward_executions"):
        _nonnegative_int(row.get(field), name=f"{row_id}.{field}")
    _finite_number(row.get("wall_time"), name=f"{row_id}.wall_time", nonnegative=True)
    for field in ("cold_compile_time", "warm_median", "warm_iqr"):
        _finite_number(row.get(field), name=f"{row_id}.{field}", nonnegative=True)
    repeats = _nonnegative_int(row.get("warm_repeats"), name=f"{row_id}.warm_repeats")
    if repeats < 5:
        raise ValidationError(f"{row_id}.warm_repeats must be at least five")
    for field in (
        "tracemalloc_peak",
        "rss_delta",
        "warp_allocation_peak",
        "device_free_memory_delta",
    ):
        availability = row.get(f"{field}_available")
        if not isinstance(availability, bool):
            raise ValidationError(f"{row_id}.{field}_available must be boolean")
        value = row.get(field)
        if availability:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValidationError(
                    f"{row_id}.{field} must be an integer when available"
                )
            if field.endswith("peak") and value < 0:
                raise ValidationError(f"{row_id}.{field} must be nonnegative")
        elif value is not None:
            raise ValidationError(f"{row_id}.{field} must be null when unavailable")


def _validate_protocol(
    manifest: dict[str, Any], loaded: dict[str, Any]
) -> tuple[int, str]:
    raw_rows: dict[str, list[dict[str, Any]]] = {
        name: _dataset_rows(loaded[f"data/raw/{name}"], relative=f"data/raw/{name}")
        for name in RAW_NAMES
    }
    plot_rows: dict[str, list[dict[str, Any]]] = {
        name: _dataset_rows(
            loaded[f"data/plot_data/{name}"], relative=f"data/plot_data/{name}"
        )
        for name in PLOT_NAMES
    }
    summary = loaded["data/summary.json"]
    if not isinstance(summary, dict) or summary.get("schema_version") != SCHEMA_VERSION:
        raise ValidationError("data/summary.json must use schema_version 1")

    all_raw_rows = [row for rows in raw_rows.values() for row in rows]
    row_ids: dict[str, dict[str, Any]] = {}
    for row in all_raw_rows:
        row_id = row.get("row_id")
        if not isinstance(row_id, str) or not row_id:
            raise ValidationError("raw row_id must be a nonempty string")
        if row_id in row_ids:
            raise ValidationError(f"duplicate raw row_id {row_id!r}")
        row_ids[row_id] = row

    method_rows = [
        row
        for row in all_raw_rows
        if row.get("method") in REQUIRED_METHOD_IDS and "gradient" in row
    ]
    gradient_methods = {row["method"] for row in method_rows}
    missing_methods = sorted(set(REQUIRED_METHOD_IDS) - gradient_methods)
    if missing_methods:
        raise ValidationError(
            f"gradient records omit required methods: {missing_methods}"
        )
    for row in raw_rows["performance.json"]:
        _validate_performance_row(row)
    performance_methods = {row.get("method") for row in raw_rows["performance.json"]}
    missing_performance = sorted(set(REQUIRED_METHOD_IDS) - performance_methods)
    if missing_performance:
        raise ValidationError(
            f"performance records omit required methods: {missing_performance}"
        )
    families = {row.get("scenario_family") for row in all_raw_rows}
    missing_families = sorted(set(REQUIRED_SCENARIO_FAMILIES) - families)
    if missing_families:
        raise ValidationError(f"raw records omit scenario families: {missing_families}")

    for row in method_rows:
        _validate_method_row(row)
    commit, source_device = _validate_source(manifest, method_rows)
    for row in raw_rows["performance.json"]:
        if row.get("source_commit") != commit:
            raise ValidationError(
                f"{row['row_id']}.source_commit disagrees with manifest"
            )
        if row.get("device") != source_device:
            raise ValidationError(f"{row['row_id']}.device disagrees with manifest")

    applicability = manifest.get("applicability")
    if not isinstance(applicability, list) or not applicability:
        raise ValidationError("manifest applicability matrix must be nonempty")
    required_cell_keys = {
        "scenario",
        "method",
        "samples",
        "stochastic",
        "antithetic",
        "report_required",
        "estimator_only",
        "applicable",
        "transformable",
        "optimization_enabled",
        "reference_required",
        "reason",
    }
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValidationError("manifest config must be an object")
    estimator_seeds = _nonnegative_int(
        config.get("estimator_seeds"), name="config.estimator_seeds"
    )
    if estimator_seeds != 32:
        raise ValidationError("report config.estimator_seeds must be 32")
    smoothing_samples_raw = config.get("smoothing_samples")
    if (
        not isinstance(smoothing_samples_raw, list)
        or not smoothing_samples_raw
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            or value % 2
            for value in smoothing_samples_raw
        )
        or len(smoothing_samples_raw) != len(set(smoothing_samples_raw))
    ):
        raise ValidationError(
            "config.smoothing_samples must contain unique positive even values"
        )
    smoothing_samples = set(smoothing_samples_raw)
    stochastic_groups: dict[tuple[str, str], set[int]] = {}
    applicability_identities: set[tuple[str, str, int]] = set()
    matrix_pairs: set[tuple[str, str]] = set()
    for index, cell in enumerate(applicability):
        if not isinstance(cell, dict) or set(cell) != required_cell_keys:
            raise ValidationError(f"applicability[{index}] has the wrong schema")
        for field in (
            "stochastic",
            "antithetic",
            "report_required",
            "estimator_only",
            "applicable",
            "transformable",
            "optimization_enabled",
            "reference_required",
        ):
            if not isinstance(cell[field], bool):
                raise ValidationError(f"applicability[{index}].{field} must be boolean")
        method = cell["method"]
        if method not in REQUIRED_METHOD_IDS:
            raise ValidationError(f"applicability[{index}] has unknown method")
        samples = _nonnegative_int(
            cell["samples"], name=f"applicability[{index}].samples"
        )
        if samples == 0:
            raise ValidationError(f"applicability[{index}].samples must be positive")
        identity = (cell["scenario"], method, samples)
        if identity in applicability_identities:
            raise ValidationError(f"duplicate applicability cell {identity}")
        applicability_identities.add(identity)
        matrix_pairs.add((cell["scenario"], method))
        if cell["estimator_only"] and cell["transformable"]:
            raise ValidationError(
                f"applicability[{index}] estimator-only cell cannot be transformable"
            )
        if not cell["applicable"]:
            if not isinstance(cell["reason"], str) or not cell["reason"]:
                raise ValidationError(f"inapplicable cell {index} needs a reason")
            continue
        if not cell["report_required"]:
            continue
        matching = [
            row
            for row in method_rows
            if row.get("scenario_family") == cell["scenario"]
            and row.get("method") == method
            and row.get("samples") == samples
        ]
        if not matching:
            raise ValidationError(
                f"report-required applicability cell has no row: {cell['scenario']}/{method}/N={samples}"
            )
        if any(row.get("antithetic") is not cell["antithetic"] for row in matching):
            raise ValidationError(
                f"{cell['scenario']}/{method}/N={samples} antithetic declaration disagrees with rows"
            )
        if cell["stochastic"]:
            stochastic_groups.setdefault((cell["scenario"], method), set()).add(samples)
            raw_seeds = [row.get("outer_seed") for row in matching]
            if any(
                isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
                for seed in raw_seeds
            ):
                raise ValidationError(
                    f"{cell['scenario']}/{method}/N={samples} outer_seed values must be nonnegative integers"
                )
            seeds = set(raw_seeds)
            if len(seeds) != estimator_seeds:
                raise ValidationError(
                    f"{cell['scenario']}/{method}/N={samples} requires 32 distinct outer seeds; found {len(seeds)}"
                )
            strata: dict[tuple[Any, ...], list[int]] = {}
            for row, seed in zip(matching, raw_seeds):
                key = (
                    row.get("start_id"),
                    row.get("target"),
                    row.get("inner_seed"),
                    json.dumps(
                        row.get("config"), sort_keys=True, separators=(",", ":")
                    ),
                )
                strata.setdefault(key, []).append(seed)
            for key, stratum_seeds in strata.items():
                if (
                    len(stratum_seeds) != estimator_seeds
                    or len(set(stratum_seeds)) != estimator_seeds
                ):
                    raise ValidationError(
                        f"{cell['scenario']}/{method}/N={samples} duplicate or incomplete stochastic rows in stratum {key}"
                    )
    expected_matrix_pairs = {
        (scenario, method)
        for scenario in REQUIRED_SCENARIO_FAMILIES
        for method in REQUIRED_METHOD_IDS
    }
    if matrix_pairs != expected_matrix_pairs:
        missing = sorted(expected_matrix_pairs - matrix_pairs)
        extra = sorted(matrix_pairs - expected_matrix_pairs)
        raise ValidationError(
            f"applicability matrix is incomplete; missing={missing}, extra={extra}"
        )
    for (scenario, method), declared_samples in stochastic_groups.items():
        if declared_samples != smoothing_samples:
            missing = sorted(smoothing_samples - declared_samples)
            extra = sorted(declared_samples - smoothing_samples)
            raise ValidationError(
                f"{scenario}/{method} stochastic sample cells disagree with config; missing N={missing}, extra={extra}"
            )

    required_reference_cells = manifest.get("reference_required_cells")
    if (
        not isinstance(required_reference_cells, list)
        or set(required_reference_cells) != set(EXPECTED_REFERENCE_CELLS)
        or len(required_reference_cells) != len(EXPECTED_REFERENCE_CELLS)
    ):
        raise ValidationError(
            "reference_required_cells must declare each canonical publication reference exactly once"
        )
    _validate_reference_rows(
        raw_rows["references.json"],
        required_reference_cells,
        config=config,
        source_commit=commit,
        device=source_device,
    )

    rejected_ids = {
        row["row_id"]
        for row in all_raw_rows
        if (
            row.get("accepted") is False
            or (
                isinstance(row.get("accepted"), dict)
                and row["accepted"].get("references") is False
            )
        )
        and "row_id" in row
    }
    referenced_ids: set[str] = set()
    for name, rows in plot_rows.items():
        for index, row in enumerate(rows):
            source_ids = row.get("source_row_ids")
            if (
                not isinstance(source_ids, list)
                or not source_ids
                or any(not isinstance(item, str) for item in source_ids)
            ):
                raise ValidationError(
                    f"data/plot_data/{name}.rows[{index}] needs nonempty source_row_ids"
                )
            if len(source_ids) != len(set(source_ids)):
                raise ValidationError(
                    f"data/plot_data/{name}.rows[{index}] contains duplicate source_row_ids"
                )
            for source_id in source_ids:
                if source_id not in row_ids:
                    raise ValidationError(
                        f"plot lineage references unknown raw row {source_id!r}"
                    )
                if source_id in rejected_ids and name != "validity.json":
                    raise ValidationError(
                        f"plot lineage references rejected diagnostic row {source_id!r}"
                    )
                referenced_ids.add(source_id)
    summary_schemas = {
        "method_summaries": {
            "scenario",
            "target",
            "method",
            "samples",
            "mean_gradient",
            "relative_error",
            "cosine_similarity",
            "sign_agreement",
            "empirical_bias",
            "empirical_variance",
            "mean_squared_error",
            "source_row_ids",
        },
        "optimization_summaries": {
            "scenario",
            "method",
            "final_hard_loss_mean",
            "final_hard_loss_ci_low",
            "final_hard_loss_ci_high",
            "success_rate",
            "held_out_loss_mean",
            "source_row_ids",
        },
        "performance_summaries": {
            "scenario",
            "method",
            "cold_compile_time",
            "warm_median",
            "warm_iqr",
            "forward_executions",
            "backward_executions",
            "tracemalloc_peak",
            "rss_delta",
            "warp_allocation_peak",
            "device_free_memory_delta",
            "source_row_ids",
        },
        "scenario_validity": {
            "scenario",
            "accepted",
            "metrics",
            "source_row_ids",
        },
    }
    for collection in (
        "method_summaries",
        "optimization_summaries",
        "performance_summaries",
        "scenario_validity",
    ):
        values = summary.get(collection)
        if not isinstance(values, list) or not values:
            raise ValidationError(f"summary.{collection} must be a nonempty array")
        for index, row in enumerate(values):
            if not isinstance(row, dict):
                raise ValidationError(
                    f"summary.{collection}[{index}] must be an object"
                )
            if set(row) != summary_schemas[collection]:
                raise ValidationError(
                    f"summary.{collection}[{index}] has the wrong schema"
                )
            if not isinstance(row.get("scenario"), str) or not row["scenario"]:
                raise ValidationError(
                    f"summary.{collection}[{index}].scenario must be nonempty"
                )
            if collection != "scenario_validity":
                if row.get("method") not in REQUIRED_METHOD_IDS:
                    raise ValidationError(
                        f"summary.{collection}[{index}] has unknown method"
                    )
            if collection == "method_summaries":
                if not isinstance(row.get("target"), str) or not row["target"]:
                    raise ValidationError(
                        f"summary.{collection}[{index}].target must be nonempty"
                    )
                if (
                    _nonnegative_int(
                        row.get("samples"),
                        name=f"summary.{collection}[{index}].samples",
                    )
                    == 0
                ):
                    raise ValidationError("method summary samples must be positive")
                vectors = [
                    _finite_vector(
                        row.get(field),
                        name=f"summary.{collection}[{index}].{field}",
                        nonnegative=field
                        in {"empirical_variance", "mean_squared_error"},
                    )
                    for field in (
                        "mean_gradient",
                        "empirical_bias",
                        "empirical_variance",
                        "mean_squared_error",
                    )
                ]
                if not vectors[0] or any(
                    len(vector) != len(vectors[0]) for vector in vectors
                ):
                    raise ValidationError("method summary gradient dimensions disagree")
                _finite_number(
                    row.get("relative_error"),
                    name="method summary relative_error",
                    nonnegative=True,
                )
                cosine = _finite_number(
                    row.get("cosine_similarity"),
                    name="method summary cosine_similarity",
                )
                sign = _finite_number(
                    row.get("sign_agreement"), name="method summary sign_agreement"
                )
                if not -1.0 <= cosine <= 1.0 or not 0.0 <= sign <= 1.0:
                    raise ValidationError(
                        "method summary similarity metrics are out of range"
                    )
            elif collection == "optimization_summaries":
                low = _finite_number(
                    row.get("final_hard_loss_ci_low"),
                    name="optimization CI low",
                    nonnegative=True,
                )
                mean = _finite_number(
                    row.get("final_hard_loss_mean"),
                    name="optimization mean",
                    nonnegative=True,
                )
                high = _finite_number(
                    row.get("final_hard_loss_ci_high"),
                    name="optimization CI high",
                    nonnegative=True,
                )
                _finite_number(
                    row.get("held_out_loss_mean"),
                    name="optimization held-out loss",
                    nonnegative=True,
                )
                success = _finite_number(
                    row.get("success_rate"), name="optimization success rate"
                )
                if not low <= mean <= high or not 0.0 <= success <= 1.0:
                    raise ValidationError(
                        "optimization summary intervals or success rate are invalid"
                    )
            elif collection == "performance_summaries":
                for field in ("cold_compile_time", "warm_median", "warm_iqr"):
                    _finite_number(
                        row.get(field),
                        name=f"performance summary {field}",
                        nonnegative=True,
                    )
                for field in ("forward_executions", "backward_executions"):
                    _nonnegative_int(
                        row.get(field), name=f"performance summary {field}"
                    )
                for field in (
                    "tracemalloc_peak",
                    "rss_delta",
                    "warp_allocation_peak",
                    "device_free_memory_delta",
                ):
                    if row.get(field) is not None and (
                        isinstance(row[field], bool) or not isinstance(row[field], int)
                    ):
                        raise ValidationError(
                            f"performance summary {field} must be integer or null"
                        )
            elif not isinstance(row.get("accepted"), bool) or not isinstance(
                row.get("metrics"), dict
            ):
                raise ValidationError("scenario validity fields are invalid")
            sources = row.get("source_row_ids")
            if not isinstance(sources, list) or not sources:
                raise ValidationError(
                    f"summary.{collection}[{index}] has orphaned aggregate data"
                )
            if len(sources) != len(set(sources)):
                raise ValidationError(
                    f"summary.{collection}[{index}] contains duplicate source_row_ids"
                )
            for source_id in sources:
                if source_id not in row_ids:
                    raise ValidationError(
                        f"summary lineage references unknown raw row {source_id!r}"
                    )
                if source_id in rejected_ids and collection != "scenario_validity":
                    raise ValidationError(
                        f"summary lineage references rejected diagnostic row {source_id!r}"
                    )
                referenced_ids.add(source_id)

    headline_metrics = summary.get("headline_metrics")
    if not isinstance(headline_metrics, dict) or not headline_metrics:
        raise ValidationError("summary.headline_metrics must be a nonempty object")
    for name, record in headline_metrics.items():
        if not isinstance(record, dict):
            raise ValidationError(
                f"headline metric {name!r} must be an aggregate object"
            )
        source_ids = record.get("source_row_ids")
        if not isinstance(source_ids, list) or not source_ids:
            raise ValidationError(f"headline metric {name!r} has no source_row_ids")
        if len(source_ids) != len(set(source_ids)):
            raise ValidationError(
                f"headline metric {name!r} contains duplicate source_row_ids"
            )
        for source_id in source_ids:
            if source_id not in row_ids:
                raise ValidationError(
                    f"headline metric {name!r} references unknown raw row {source_id!r}"
                )
            if source_id in rejected_ids:
                raise ValidationError(
                    f"headline metric {name!r} references rejected raw row {source_id!r}"
                )
            referenced_ids.add(source_id)

    orphaned_ids = sorted(set(row_ids) - referenced_ids)
    if orphaned_ids:
        raise ValidationError(
            f"raw rows have no plot or aggregate lineage: {orphaned_ids}"
        )

    opaque_rows = raw_rows["opaque_mesh.json"]
    if not opaque_rows:
        raise ValidationError("opaque mesh diagnostics are missing")
    for row in opaque_rows:
        if (
            row.get("transform_status") != "estimator_only"
            or row.get("transformable") is not False
        ):
            raise ValidationError(
                "opaque native/BVH branches must be labeled estimator-only"
            )
    for cell in applicability:
        if cell["scenario"] == "opaque_mesh" and cell["transformable"] is not False:
            raise ValidationError(
                "opaque mesh applicability cannot claim transformed native branches"
            )

    literature = summary.get("literature_urls")
    if not isinstance(literature, list):
        raise ValidationError("summary literature_urls must be an array")
    for url in REQUIRED_LITERATURE_URLS:
        if url not in literature:
            raise ValidationError(f"required literature URL is missing: {url}")

    accepted = manifest.get("accepted")
    if not isinstance(accepted, dict) or any(
        accepted.get(key) is not True
        for key in ("analytic", "references", "scenario_validity")
    ):
        raise ValidationError("manifest acceptance gates must all pass")
    total_rows = len(all_raw_rows) + sum(len(rows) for rows in plot_rows.values())
    return total_rows, commit


def _validated_html_attributes(
    tag: str, attrs: list[tuple[str, str | None]]
) -> dict[str, str | None]:
    validated: list[tuple[str, str | None]] = []
    seen: set[str] = set()
    for name, value in attrs:
        normalized = name.casefold()
        if normalized in seen:
            raise ValidationError(f"duplicate HTML attribute {name!r} on <{tag}>")
        seen.add(normalized)
        validated.append((normalized, value))
    return dict(validated)


class _ReportHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: dict[str, tuple[str, bool]] = {}
        self.duplicate_ids: set[str] = set()
        self.method_text: dict[str, list[str]] = {}
        self.section_text: dict[str, list[str]] = {}
        self.references: list[tuple[str, str, str]] = []
        self.active_content_violations: list[str] = []
        self.source_commit: str | None = None
        self.reproducibility_text: list[str] = []
        self.text_fragments: list[str] = []
        self.attribute_values: list[str] = []
        self._element_stack: list[tuple[str, str | None, bool, bool, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = _validated_html_attributes(tag, attrs)
        element_id = attributes.get("id")
        style = (attributes.get("style") or "").replace(" ", "").casefold()
        hidden = (
            any(element[3] for element in self._element_stack)
            or "hidden" in attributes
            or (attributes.get("aria-hidden") or "").casefold() == "true"
            or "display:none" in style
            or "visibility:hidden" in style
            or tag in {"head", "script", "style", "template"}
        )
        if element_id:
            if element_id in self.ids:
                self.duplicate_ids.add(element_id)
            self.ids[element_id] = (tag, hidden)
            if tag == "section":
                self.section_text.setdefault(element_id, [])
        forbidden_elements = {
            "script",
            "iframe",
            "object",
            "embed",
            "base",
            "link",
            "form",
            "input",
            "button",
            "textarea",
            "select",
            "video",
            "audio",
        }
        if tag in forbidden_elements:
            self.active_content_violations.append(f"forbidden element <{tag}>")
        for attribute, value in attrs:
            lowered = attribute.casefold()
            if lowered.startswith("on") or lowered in {
                "srcset",
                "formaction",
                "xlink:href",
                "action",
                "poster",
                "ping",
                "cite",
                "background",
                "manifest",
                "data",
                "codebase",
                "archive",
                "profile",
                "usemap",
                "srcdoc",
            }:
                self.active_content_violations.append(
                    f"forbidden active attribute {attribute}"
                )
            if lowered == "style" and value:
                if _css_has_external_or_active_content(value):
                    self.active_content_violations.append(
                        "unsafe inline style attribute"
                    )
        if tag == "meta" and "http-equiv" in attributes:
            self.active_content_violations.append("active meta http-equiv control")
        method = attributes.get("data-method")
        if method is not None:
            self.method_text.setdefault(method, [])
        starts_repro = element_id == "reproducibility"
        self._element_stack.append(
            (
                tag,
                method,
                starts_repro,
                hidden,
                element_id if tag == "section" else None,
            )
        )
        self.attribute_values.extend(
            value for value in attributes.values() if isinstance(value, str)
        )
        for attribute in ("src", "href"):
            reference = attributes.get(attribute)
            if reference:
                self.references.append((tag, attribute, reference))
        if tag == "body" and attributes.get("data-source-commit"):
            self.source_commit = attributes["data-source-commit"]

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        match = next(
            (
                index
                for index in range(len(self._element_stack) - 1, -1, -1)
                if self._element_stack[index][0] == tag
            ),
            None,
        )
        if match is None:
            return
        del self._element_stack[match:]

    def handle_data(self, data: str) -> None:
        self.text_fragments.append(data)
        if any(element[0] == "style" for element in self._element_stack):
            if _css_has_external_or_active_content(data):
                self.active_content_violations.append(
                    "unsafe resource or active expression in style element"
                )
        if any(element[3] for element in self._element_stack):
            return
        for _tag, method, _starts_repro, _hidden, _section_id in self._element_stack:
            if method is not None:
                self.method_text.setdefault(method, []).append(data)
        for _tag, _method, _starts_repro, _hidden, section_id in self._element_stack:
            if section_id is not None:
                self.section_text.setdefault(section_id, []).append(data)
        if any(element[2] for element in self._element_stack):
            self.reproducibility_text.append(data)


def _validate_html(root: Path, commit: str) -> None:
    root_flags = (
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        root_descriptor = os.open(os.fspath(root), root_flags)
        try:
            source_bytes = _read_artifact_at(root_descriptor, "index.html")
        finally:
            os.close(root_descriptor)
        source = source_bytes.decode("utf-8")
    except (OSError, UnicodeError, ValidationError) as error:
        raise ValidationError(f"cannot read index.html: {error}") from error
    lower_source = source.casefold()
    for legacy in LEGACY_FILES:
        if legacy.casefold() in lower_source:
            raise ValidationError(f"index.html references legacy artifact {legacy}")
    forbidden_claims = (
        "opaque native branches were transformed",
        "opaque mesh branches were transformed",
        "bvh branches were transformed",
    )
    if any(claim in lower_source for claim in forbidden_claims):
        raise ValidationError(
            "report overclaims transformation of opaque native/BVH branches"
        )

    parser = _ReportHTMLParser()
    try:
        parser.feed(source)
        parser.close()
    except Exception as error:
        raise ValidationError(f"invalid report HTML: {error}") from error
    for index, text in enumerate(parser.text_fragments):
        _assert_no_local_path(text, location=f"index.html text[{index}]")
    for index, value in enumerate(parser.attribute_values):
        _assert_no_local_path(value, location=f"index.html attribute[{index}]")
    if parser.duplicate_ids:
        raise ValidationError(
            f"report contains duplicate IDs: {sorted(parser.duplicate_ids)}"
        )
    if parser.active_content_violations:
        raise ValidationError(
            f"report contains unsafe active HTML: {parser.active_content_violations}"
        )
    missing_sections = [
        section
        for section in REQUIRED_SECTION_IDS
        if parser.ids.get(section) != ("section", False)
    ]
    if missing_sections:
        raise ValidationError(
            f"report is missing required sections: {missing_sections}"
        )
    empty_sections = [
        section
        for section in REQUIRED_SECTION_IDS
        if not " ".join(parser.section_text.get(section, ())).strip()
    ]
    if empty_sections:
        raise ValidationError(
            f"report contains empty required sections: {empty_sections}"
        )
    for method in REQUIRED_METHOD_IDS:
        visible = " ".join(parser.method_text.get(method, ())).strip()
        if not visible:
            raise ValidationError(
                f"report is missing a visible label for method {method}"
            )
    if parser.source_commit != commit:
        raise ValidationError("index.html source commit does not match manifest")
    if commit not in " ".join(parser.reproducibility_text):
        raise ValidationError("reproducibility section omits the exact Warp commit")

    for tag, attribute, raw_reference in parser.references:
        split = urlsplit(raw_reference)
        if split.scheme:
            if split.scheme in {"http", "https"} and attribute == "href" and tag == "a":
                continue
            raise ValidationError(f"unsafe HTML reference scheme in {raw_reference!r}")
        if split.netloc:
            raise ValidationError(
                f"protocol-relative HTML reference is unsafe: {raw_reference!r}"
            )
        if not split.path:
            continue
        if Path(split.path).name in LEGACY_FILES:
            raise ValidationError(
                f"report references legacy artifact {Path(split.path).name}"
            )
        pure = PurePosixPath(split.path)
        if pure.is_absolute() or ".." in pure.parts or "\\" in split.path:
            raise ValidationError(f"unsafe relative asset reference {raw_reference!r}")
        relative = pure.as_posix()
        if relative not in EXPECTED_FILES:
            raise ValidationError(
                f"local HTML reference is not declared by the manifest: {raw_reference}"
            )
        if tag not in {"a", "img", "source"}:
            raise ValidationError(
                f"unexpected local reference element {tag!r} for {raw_reference!r}"
            )


def validate_publication(root: Path) -> dict[str, int]:
    root = Path(root)
    if not root.is_dir():
        raise ValidationError(f"report root does not exist: {root}")
    manifest, loaded = validate_manifest(root)
    rows, commit = _validate_protocol(manifest, loaded)
    _validate_html(root.resolve(), commit)
    return {
        "files": len(EXPECTED_FILES),
        "rows": rows,
        "assets": len(FIGURE_NAMES) + len(IMAGE_NAMES),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the published Warp smoothing report"
    )
    parser.add_argument(
        "root", nargs="?", type=Path, default=Path(__file__).resolve().parent
    )
    arguments = parser.parse_args(argv)
    try:
        result = validate_publication(arguments.root)
    except ValidationError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(
        f"PASS: files={result['files']} rows={result['rows']} assets={result['assets']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
