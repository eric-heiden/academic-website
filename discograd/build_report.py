from __future__ import annotations

import argparse
import binascii
import html
import json
import math
import os
import re
import secrets
import statistics
import stat
import struct
import sys
import zlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any
from urllib.parse import quote, unquote, urlsplit

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from discograd.validate_report import (
    REQUIRED_METHOD_IDS,
    REQUIRED_SECTION_IDS,
    ValidationError,
    _assert_no_local_path,
    _ReportHTMLParser,
    validate_manifest,
)


REPORT_ROOT = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = REPORT_ROOT / "report_template.html"
DEFAULT_OUTPUT = REPORT_ROOT / "index.html"
SUMMARY_FILE = "data/summary.json"
RAW_FILES = (
    "data/raw/analytic.json",
    "data/raw/triangle_2d.json",
    "data/raw/collision_2d.json",
    "data/raw/path_tracer_gradients.json",
    "data/raw/path_tracer_optimization.json",
    "data/raw/contact_3d_gradients.json",
    "data/raw/contact_3d_optimization.json",
    "data/raw/opaque_mesh.json",
    "data/raw/performance.json",
    "data/raw/references.json",
)
PLOT_FILES = (
    "data/plot_data/analytic_gates.json",
    "data/plot_data/gradient_quality.json",
    "data/plot_data/bias_variance.json",
    "data/plot_data/optimization.json",
    "data/plot_data/validity.json",
    "data/plot_data/performance.json",
)
TOKEN_RE = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")
ANY_TOKEN_RE = re.compile(r"\{\{[^{}]*\}\}")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class BuildError(ValueError):
    """Raised when evidence cannot be rendered without weakening its contract."""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(child) for key, child in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(child) for child in value)
    return value


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BuildError(f"{name} must be an object")
    return value


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise BuildError(f"{name} must be an array")
    return value


def _string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise BuildError(f"{name} must be a nonempty string")
    return value


def _optional_string(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    return _string(value, name=name)


def _number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BuildError(f"{name} must be numeric")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as error:
        raise BuildError(f"{name} must be finite numeric data") from error
    if not math.isfinite(converted):
        raise BuildError(f"{name} must be finite")
    return converted


def _optional_number(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    return _number(value, name=name)


def _integer(value: Any, *, name: str, nonnegative: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BuildError(f"{name} must be an integer")
    if nonnegative and value < 0:
        raise BuildError(f"{name} must be nonnegative")
    return value


def _optional_integer(value: Any, *, name: str, nonnegative: bool = True) -> int | None:
    if value is None:
        return None
    return _integer(value, name=name, nonnegative=nonnegative)


def _boolean(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise BuildError(f"{name} must be boolean")
    return value


def _vector(value: Any, *, name: str) -> tuple[float, ...]:
    result = tuple(
        _number(item, name=f"{name}[{index}]")
        for index, item in enumerate(_sequence(value, name=name))
    )
    if not result:
        raise BuildError(f"{name} must be nonempty")
    return result


def _optional_vector(value: Any, *, name: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    return _vector(value, name=name)


def _method(value: Any, *, name: str) -> str:
    result = _string(value, name=name)
    if result not in REQUIRED_METHOD_IDS:
        raise BuildError(f"{name} has unknown method {result!r}")
    return result


def _strings(value: Any, *, name: str, nonempty: bool = False) -> tuple[str, ...]:
    values = tuple(
        _string(item, name=f"{name}[{index}]")
        for index, item in enumerate(_sequence(value, name=name))
    )
    if nonempty and not values:
        raise BuildError(f"{name} must be nonempty")
    return values


def checked_fmean(values: Sequence[float]) -> float:
    converted = tuple(_number(value, name="mean value") for value in values)
    if not converted:
        raise BuildError("finite mean input must be nonempty")
    result = statistics.fmean(converted)
    if not math.isfinite(result):
        raise BuildError("finite mean overflowed")
    return result


def checked_quantile(values: Sequence[float], probability: float) -> float:
    converted = sorted(_number(value, name="quantile value") for value in values)
    if not converted:
        raise BuildError("quantile input must be nonempty")
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise BuildError("quantile probability must lie in [0, 1]")
    position = probability * (len(converted) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return converted[lower]
    weight = position - lower
    result = converted[lower] * (1.0 - weight) + converted[upper] * weight
    if not math.isfinite(result):
        raise BuildError("quantile interpolation overflowed")
    return result


def format_number(value: int | float) -> str:
    converted = _number(value, name="display value")
    if converted == 0.0:
        return "-0" if math.copysign(1.0, converted) < 0.0 else "0"
    magnitude = abs(converted)
    if magnitude < 1.0e-3 or magnitude >= 1.0e4:
        return format(converted, ".3e")
    return format(converted, ".4g")


def _validate_png_bytes(data: bytes, *, name: str) -> None:
    if not data.startswith(PNG_SIGNATURE):
        raise BuildError(f"PNG asset {name!r} has an invalid signature")
    offset = len(PNG_SIGNATURE)
    seen_header = False
    seen_end = False
    compressed = bytearray()
    while offset < len(data):
        if offset + 12 > len(data):
            raise BuildError(f"PNG asset {name!r} has a truncated chunk")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_end = offset + 12 + length
        if chunk_end > len(data):
            raise BuildError(f"PNG asset {name!r} has a truncated chunk payload")
        payload = data[offset + 8 : offset + 8 + length]
        declared_crc = struct.unpack(">I", data[offset + 8 + length : chunk_end])[0]
        actual_crc = binascii.crc32(chunk_type + payload) & 0xFFFFFFFF
        if actual_crc != declared_crc:
            raise BuildError(f"PNG asset {name!r} has an invalid chunk checksum")
        if not seen_header:
            if chunk_type != b"IHDR" or length != 13:
                raise BuildError(f"PNG asset {name!r} does not start with IHDR")
            width, height = struct.unpack(">II", payload[:8])
            if width == 0 or height == 0:
                raise BuildError(f"PNG asset {name!r} has empty dimensions")
            seen_header = True
        elif chunk_type == b"IDAT":
            compressed.extend(payload)
        elif chunk_type == b"IEND":
            if length != 0 or chunk_end != len(data):
                raise BuildError(f"PNG asset {name!r} has an invalid IEND chunk")
            seen_end = True
            offset = chunk_end
            break
        offset = chunk_end
    if not seen_header or not compressed or not seen_end:
        raise BuildError(f"PNG asset {name!r} is missing required image chunks")
    try:
        decoded = zlib.decompress(bytes(compressed))
    except zlib.error as error:
        raise BuildError(f"PNG asset {name!r} has invalid compressed pixels") from error
    if not decoded:
        raise BuildError(f"PNG asset {name!r} decodes to no pixel data")


def _validate_png_assets(root: Path, files: Mapping[str, Any]) -> None:
    for relative in sorted(name for name in files if name.endswith(".png")):
        try:
            data = (root / relative).read_bytes()
        except OSError as error:
            raise BuildError(f"cannot read PNG asset {relative!r}: {error}") from error
        _validate_png_bytes(data, name=relative)


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    schema_version: int
    tier: str
    source: Mapping[str, Any]
    config: Mapping[str, Any]
    accepted: Mapping[str, Any]
    files: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class HeadlineMetric:
    index: int
    name: str
    value: float
    unit: str | None
    source_row_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MethodSummary:
    index: int
    scenario: str
    target: str
    method: str
    samples: int
    mean_gradient: tuple[float, ...]
    relative_error: float
    cosine_similarity: float
    sign_agreement: float
    empirical_bias: tuple[float, ...]
    empirical_variance: tuple[float, ...]
    mean_squared_error: tuple[float, ...]
    source_row_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OptimizationSummary:
    index: int
    scenario: str
    method: str
    final_hard_loss_mean: float
    final_hard_loss_ci_low: float
    final_hard_loss_ci_high: float
    success_rate: float
    held_out_loss_mean: float
    source_row_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PerformanceSummary:
    index: int
    scenario: str
    method: str
    cold_compile_time: float
    warm_median: float
    warm_iqr: float
    forward_executions: int
    backward_executions: int
    tracemalloc_peak: int | None
    rss_delta: int | None
    warp_allocation_peak: int | None
    device_free_memory_delta: int | None
    source_row_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScenarioValidity:
    index: int
    scenario: str
    accepted: bool
    metrics: Mapping[str, Any]
    source_row_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SummaryTables:
    schema_version: int
    method_labels: Mapping[str, str]
    scenario_families: tuple[str, ...]
    literature_urls: tuple[str, ...]
    headline_metrics: tuple[HeadlineMetric, ...]
    method_summaries: tuple[MethodSummary, ...]
    optimization_summaries: tuple[OptimizationSummary, ...]
    performance_summaries: tuple[PerformanceSummary, ...]
    scenario_validity: tuple[ScenarioValidity, ...]


@dataclass(frozen=True, slots=True)
class GradientRow:
    source_file: str
    source_index: int
    row_id: str
    scenario_family: str
    scenario: str
    start_id: str
    target: str
    method: str
    samples: int
    outer_seed: int | None
    inner_seed: int | None
    wall_time: float
    forward_executions: int
    hard_forward_executions: int | None
    soft_forward_executions: int | None
    backward_executions: int
    gradient: tuple[float, ...]
    reference_gradient: tuple[float, ...]
    contribution_variance: tuple[float, ...] | None
    gradient_variance: tuple[float, ...] | None
    ci_low: tuple[float, ...] | None
    ci_high: tuple[float, ...] | None
    raw: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class OptimizationRow:
    source_file: str
    source_index: int
    row_id: str
    scenario_family: str
    scenario: str
    method: str
    final_hard_loss: float
    held_out_loss: float
    success: bool
    raw: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PerformanceRow:
    source_file: str
    source_index: int
    row_id: str
    scenario: str
    method: str
    cold_compile_time: float
    warm_median: float
    warm_iqr: float
    forward_executions: int
    backward_executions: int
    tracemalloc_peak: int | None
    tracemalloc_peak_available: bool
    rss_delta: int | None
    rss_delta_available: bool
    warp_allocation_peak: int | None
    warp_allocation_peak_available: bool
    device_free_memory_delta: int | None
    device_free_memory_delta_available: bool
    device: str
    raw: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ReferenceRow:
    source_file: str
    source_index: int
    row_id: str
    cell_id: str
    counts: Mapping[str, Any]
    reference_gradient: tuple[float, ...]
    raw: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RawRecord:
    source_file: str
    source_index: int
    row_id: str
    raw: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PlotRecord:
    source_file: str
    source_index: int
    plot_id: str
    source_row_ids: tuple[str, ...]
    raw: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PlotDataset:
    source_file: str
    rows: tuple[PlotRecord, ...]


@dataclass(frozen=True, slots=True)
class GradientAggregate:
    scenario_family: str
    scenario: str
    target: str
    method: str
    samples: int
    rows: int
    mean_wall_time: float
    wall_time_q25: float
    wall_time_q75: float


@dataclass(frozen=True, slots=True)
class PerformanceAggregate:
    scenario: str
    method: str
    rows: int
    mean_warm_time: float
    warm_time_q25: float
    warm_time_q75: float


@dataclass(frozen=True, slots=True)
class ReportModel:
    root: Path
    manifest: ManifestRecord
    summary: SummaryTables
    gradient_rows: tuple[GradientRow, ...]
    optimization_rows: tuple[OptimizationRow, ...]
    performance_rows: tuple[PerformanceRow, ...]
    reference_rows: tuple[ReferenceRow, ...]
    plots: Mapping[str, PlotDataset]
    raw_row_ids: frozenset[str]
    raw_records: Mapping[str, RawRecord]
    gradient_aggregates: tuple[GradientAggregate, ...]
    performance_aggregates: tuple[PerformanceAggregate, ...]


def _dataset_rows(payload: Any, *, source_file: str) -> tuple[Mapping[str, Any], ...]:
    mapping = _mapping(payload, name=source_file)
    if mapping.get("schema_version") != 1:
        raise BuildError(f"{source_file} must use schema_version 1")
    _string(mapping.get("dataset"), name=f"{source_file}.dataset")
    rows = _sequence(mapping.get("rows"), name=f"{source_file}.rows")
    return tuple(
        _mapping(row, name=f"{source_file}.rows[{index}]")
        for index, row in enumerate(rows)
    )


def _load_manifest(value: Any) -> ManifestRecord:
    mapping = _mapping(value, name="data/manifest.json")
    return ManifestRecord(
        schema_version=_integer(
            mapping.get("schema_version"), name="manifest.schema_version"
        ),
        tier=_string(mapping.get("tier"), name="manifest.tier"),
        source=_freeze(_mapping(mapping.get("source"), name="manifest.source")),
        config=_freeze(_mapping(mapping.get("config"), name="manifest.config")),
        accepted=_freeze(_mapping(mapping.get("accepted"), name="manifest.accepted")),
        files=_freeze(_mapping(mapping.get("files"), name="manifest.files")),
    )


def _load_summary(value: Any) -> SummaryTables:
    mapping = _mapping(value, name=SUMMARY_FILE)
    labels = _mapping(mapping.get("method_labels"), name="summary.method_labels")
    if set(labels) != set(REQUIRED_METHOD_IDS):
        raise BuildError(
            "summary.method_labels must exactly cover the required methods"
        )
    method_labels = MappingProxyType(
        {
            method: _string(labels[method], name=f"method_labels.{method}")
            for method in REQUIRED_METHOD_IDS
        }
    )

    headline_mapping = _mapping(
        mapping.get("headline_metrics"), name="summary.headline_metrics"
    )
    headline_metrics = tuple(
        HeadlineMetric(
            index=index,
            name=_string(name, name="headline metric name"),
            value=_number(
                _mapping(row, name=f"headline_metrics.{name}").get("value"),
                name=f"headline_metrics.{name}.value",
            ),
            unit=_optional_string(
                _mapping(row, name=f"headline_metrics.{name}").get("unit"),
                name=f"headline_metrics.{name}.unit",
            ),
            source_row_ids=_strings(
                _mapping(row, name=f"headline_metrics.{name}").get("source_row_ids"),
                name=f"headline_metrics.{name}.source_row_ids",
                nonempty=True,
            ),
        )
        for index, (name, row) in enumerate(sorted(headline_mapping.items()))
    )
    if not headline_metrics:
        raise BuildError("summary.headline_metrics must be nonempty")

    method_summaries = []
    for index, raw in enumerate(
        _sequence(mapping.get("method_summaries"), name="summary.method_summaries")
    ):
        row = _mapping(raw, name=f"method_summaries[{index}]")
        method = _method(row.get("method"), name=f"method_summaries[{index}].method")
        method_summaries.append(
            MethodSummary(
                index=index,
                scenario=_string(
                    row.get("scenario"), name=f"method_summaries[{index}].scenario"
                ),
                target=_string(
                    row.get("target"), name=f"method_summaries[{index}].target"
                ),
                method=method,
                samples=_integer(
                    row.get("samples"), name=f"method_summaries[{index}].samples"
                ),
                mean_gradient=_vector(
                    row.get("mean_gradient"),
                    name=f"method_summaries[{index}].mean_gradient",
                ),
                relative_error=_number(
                    row.get("relative_error"),
                    name=f"method_summaries[{index}].relative_error",
                ),
                cosine_similarity=_number(
                    row.get("cosine_similarity"),
                    name=f"method_summaries[{index}].cosine_similarity",
                ),
                sign_agreement=_number(
                    row.get("sign_agreement"),
                    name=f"method_summaries[{index}].sign_agreement",
                ),
                empirical_bias=_vector(
                    row.get("empirical_bias"),
                    name=f"method_summaries[{index}].empirical_bias",
                ),
                empirical_variance=_vector(
                    row.get("empirical_variance"),
                    name=f"method_summaries[{index}].empirical_variance",
                ),
                mean_squared_error=_vector(
                    row.get("mean_squared_error"),
                    name=f"method_summaries[{index}].mean_squared_error",
                ),
                source_row_ids=_strings(
                    row.get("source_row_ids"),
                    name=f"method_summaries[{index}].source_row_ids",
                    nonempty=True,
                ),
            )
        )
    if not method_summaries:
        raise BuildError("summary.method_summaries must be nonempty")

    optimization_summaries = []
    for index, raw in enumerate(
        _sequence(
            mapping.get("optimization_summaries"), name="summary.optimization_summaries"
        )
    ):
        row = _mapping(raw, name=f"optimization_summaries[{index}]")
        optimization_summaries.append(
            OptimizationSummary(
                index=index,
                scenario=_string(
                    row.get("scenario"),
                    name=f"optimization_summaries[{index}].scenario",
                ),
                method=_method(
                    row.get("method"), name=f"optimization_summaries[{index}].method"
                ),
                final_hard_loss_mean=_number(
                    row.get("final_hard_loss_mean"),
                    name=f"optimization_summaries[{index}].final_hard_loss_mean",
                ),
                final_hard_loss_ci_low=_number(
                    row.get("final_hard_loss_ci_low"),
                    name=f"optimization_summaries[{index}].final_hard_loss_ci_low",
                ),
                final_hard_loss_ci_high=_number(
                    row.get("final_hard_loss_ci_high"),
                    name=f"optimization_summaries[{index}].final_hard_loss_ci_high",
                ),
                success_rate=_number(
                    row.get("success_rate"),
                    name=f"optimization_summaries[{index}].success_rate",
                ),
                held_out_loss_mean=_number(
                    row.get("held_out_loss_mean"),
                    name=f"optimization_summaries[{index}].held_out_loss_mean",
                ),
                source_row_ids=_strings(
                    row.get("source_row_ids"),
                    name=f"optimization_summaries[{index}].source_row_ids",
                    nonempty=True,
                ),
            )
        )
    if not optimization_summaries:
        raise BuildError("summary.optimization_summaries must be nonempty")

    performance_summaries = []
    for index, raw in enumerate(
        _sequence(
            mapping.get("performance_summaries"), name="summary.performance_summaries"
        )
    ):
        row = _mapping(raw, name=f"performance_summaries[{index}]")
        performance_summaries.append(
            PerformanceSummary(
                index=index,
                scenario=_string(
                    row.get("scenario"), name=f"performance_summaries[{index}].scenario"
                ),
                method=_method(
                    row.get("method"), name=f"performance_summaries[{index}].method"
                ),
                cold_compile_time=_number(
                    row.get("cold_compile_time"),
                    name=f"performance_summaries[{index}].cold_compile_time",
                ),
                warm_median=_number(
                    row.get("warm_median"),
                    name=f"performance_summaries[{index}].warm_median",
                ),
                warm_iqr=_number(
                    row.get("warm_iqr"), name=f"performance_summaries[{index}].warm_iqr"
                ),
                forward_executions=_integer(
                    row.get("forward_executions"),
                    name=f"performance_summaries[{index}].forward_executions",
                ),
                backward_executions=_integer(
                    row.get("backward_executions"),
                    name=f"performance_summaries[{index}].backward_executions",
                ),
                tracemalloc_peak=_optional_integer(
                    row.get("tracemalloc_peak"),
                    name=f"performance_summaries[{index}].tracemalloc_peak",
                ),
                rss_delta=_optional_integer(
                    row.get("rss_delta"),
                    name=f"performance_summaries[{index}].rss_delta",
                    nonnegative=False,
                ),
                warp_allocation_peak=_optional_integer(
                    row.get("warp_allocation_peak"),
                    name=f"performance_summaries[{index}].warp_allocation_peak",
                ),
                device_free_memory_delta=_optional_integer(
                    row.get("device_free_memory_delta"),
                    name=(f"performance_summaries[{index}].device_free_memory_delta"),
                    nonnegative=False,
                ),
                source_row_ids=_strings(
                    row.get("source_row_ids"),
                    name=f"performance_summaries[{index}].source_row_ids",
                    nonempty=True,
                ),
            )
        )
    if not performance_summaries:
        raise BuildError("summary.performance_summaries must be nonempty")

    validity_rows = []
    for index, raw in enumerate(
        _sequence(mapping.get("scenario_validity"), name="summary.scenario_validity")
    ):
        row = _mapping(raw, name=f"scenario_validity[{index}]")
        accepted = row.get("accepted")
        if not isinstance(accepted, bool):
            raise BuildError(f"scenario_validity[{index}].accepted must be boolean")
        validity_rows.append(
            ScenarioValidity(
                index=index,
                scenario=_string(
                    row.get("scenario"), name=f"scenario_validity[{index}].scenario"
                ),
                accepted=accepted,
                metrics=_freeze(
                    _mapping(
                        row.get("metrics"), name=f"scenario_validity[{index}].metrics"
                    )
                ),
                source_row_ids=_strings(
                    row.get("source_row_ids"),
                    name=f"scenario_validity[{index}].source_row_ids",
                    nonempty=True,
                ),
            )
        )
    if not validity_rows:
        raise BuildError("summary.scenario_validity must be nonempty")

    return SummaryTables(
        schema_version=_integer(
            mapping.get("schema_version"), name="summary.schema_version"
        ),
        method_labels=method_labels,
        scenario_families=_strings(
            mapping.get("scenario_families"),
            name="summary.scenario_families",
            nonempty=True,
        ),
        literature_urls=_strings(
            mapping.get("literature_urls"),
            name="summary.literature_urls",
            nonempty=True,
        ),
        headline_metrics=headline_metrics,
        method_summaries=tuple(method_summaries),
        optimization_summaries=tuple(optimization_summaries),
        performance_summaries=tuple(performance_summaries),
        scenario_validity=tuple(validity_rows),
    )


def _gradient_row(source_file: str, index: int, row: Mapping[str, Any]) -> GradientRow:
    prefix = f"{source_file}.rows[{index}]"
    return GradientRow(
        source_file=source_file,
        source_index=index,
        row_id=_string(row.get("row_id"), name=f"{prefix}.row_id"),
        scenario_family=_string(
            row.get("scenario_family"), name=f"{prefix}.scenario_family"
        ),
        scenario=_string(row.get("scenario"), name=f"{prefix}.scenario"),
        start_id=_string(row.get("start_id"), name=f"{prefix}.start_id"),
        target=_string(row.get("target"), name=f"{prefix}.target"),
        method=_method(row.get("method"), name=f"{prefix}.method"),
        samples=_integer(row.get("samples"), name=f"{prefix}.samples"),
        outer_seed=_optional_integer(
            row.get("outer_seed"), name=f"{prefix}.outer_seed"
        ),
        inner_seed=_optional_integer(
            row.get("inner_seed"), name=f"{prefix}.inner_seed"
        ),
        wall_time=_number(row.get("wall_time"), name=f"{prefix}.wall_time"),
        forward_executions=_integer(
            row.get("forward_executions"), name=f"{prefix}.forward_executions"
        ),
        hard_forward_executions=_optional_integer(
            row.get("hard_forward_executions"), name=f"{prefix}.hard_forward_executions"
        ),
        soft_forward_executions=_optional_integer(
            row.get("soft_forward_executions"), name=f"{prefix}.soft_forward_executions"
        ),
        backward_executions=_integer(
            row.get("backward_executions"), name=f"{prefix}.backward_executions"
        ),
        gradient=_vector(row.get("gradient"), name=f"{prefix}.gradient"),
        reference_gradient=_vector(
            row.get("reference_gradient"), name=f"{prefix}.reference_gradient"
        ),
        contribution_variance=_optional_vector(
            row.get("contribution_variance"),
            name=f"{prefix}.contribution_variance",
        ),
        gradient_variance=_optional_vector(
            row.get("gradient_variance"), name=f"{prefix}.gradient_variance"
        ),
        ci_low=_optional_vector(row.get("ci_low"), name=f"{prefix}.ci_low"),
        ci_high=_optional_vector(row.get("ci_high"), name=f"{prefix}.ci_high"),
        raw=_freeze(row),
    )


def _optimization_row(
    source_file: str, index: int, row: Mapping[str, Any]
) -> OptimizationRow:
    prefix = f"{source_file}.rows[{index}]"
    return OptimizationRow(
        source_file=source_file,
        source_index=index,
        row_id=_string(row.get("row_id"), name=f"{prefix}.row_id"),
        scenario_family=_string(
            row.get("scenario_family"), name=f"{prefix}.scenario_family"
        ),
        scenario=_string(row.get("scenario"), name=f"{prefix}.scenario"),
        method=_method(row.get("method"), name=f"{prefix}.method"),
        final_hard_loss=_number(
            row.get("final_hard_loss"), name=f"{prefix}.final_hard_loss"
        ),
        held_out_loss=_number(row.get("held_out_loss"), name=f"{prefix}.held_out_loss"),
        success=_boolean(row.get("success"), name=f"{prefix}.success"),
        raw=_freeze(row),
    )


def _performance_row(
    source_file: str, index: int, row: Mapping[str, Any]
) -> PerformanceRow:
    prefix = f"{source_file}.rows[{index}]"
    return PerformanceRow(
        source_file=source_file,
        source_index=index,
        row_id=_string(row.get("row_id"), name=f"{prefix}.row_id"),
        scenario=_string(row.get("scenario"), name=f"{prefix}.scenario"),
        method=_method(row.get("method"), name=f"{prefix}.method"),
        cold_compile_time=_number(
            row.get("cold_compile_time"), name=f"{prefix}.cold_compile_time"
        ),
        warm_median=_number(row.get("warm_median"), name=f"{prefix}.warm_median"),
        warm_iqr=_number(row.get("warm_iqr"), name=f"{prefix}.warm_iqr"),
        forward_executions=_integer(
            row.get("forward_executions"), name=f"{prefix}.forward_executions"
        ),
        backward_executions=_integer(
            row.get("backward_executions"), name=f"{prefix}.backward_executions"
        ),
        tracemalloc_peak=_optional_integer(
            row.get("tracemalloc_peak"), name=f"{prefix}.tracemalloc_peak"
        ),
        tracemalloc_peak_available=_boolean(
            row.get("tracemalloc_peak_available"),
            name=f"{prefix}.tracemalloc_peak_available",
        ),
        rss_delta=_optional_integer(
            row.get("rss_delta"), name=f"{prefix}.rss_delta", nonnegative=False
        ),
        rss_delta_available=_boolean(
            row.get("rss_delta_available"), name=f"{prefix}.rss_delta_available"
        ),
        warp_allocation_peak=_optional_integer(
            row.get("warp_allocation_peak"), name=f"{prefix}.warp_allocation_peak"
        ),
        warp_allocation_peak_available=_boolean(
            row.get("warp_allocation_peak_available"),
            name=f"{prefix}.warp_allocation_peak_available",
        ),
        device_free_memory_delta=_optional_integer(
            row.get("device_free_memory_delta"),
            name=f"{prefix}.device_free_memory_delta",
            nonnegative=False,
        ),
        device_free_memory_delta_available=_boolean(
            row.get("device_free_memory_delta_available"),
            name=f"{prefix}.device_free_memory_delta_available",
        ),
        device=_string(row.get("device"), name=f"{prefix}.device"),
        raw=_freeze(row),
    )


def _reference_row(
    source_file: str, index: int, row: Mapping[str, Any]
) -> ReferenceRow:
    prefix = f"{source_file}.rows[{index}]"
    gradient = tuple(
        _number(value, name=f"{prefix}.reference_gradient[{component}]")
        for component, value in enumerate(
            _sequence(
                row.get("reference_gradient"), name=f"{prefix}.reference_gradient"
            )
        )
    )
    if not gradient:
        raise BuildError(f"{prefix}.reference_gradient must be nonempty")
    return ReferenceRow(
        source_file=source_file,
        source_index=index,
        row_id=_string(row.get("row_id"), name=f"{prefix}.row_id"),
        cell_id=_string(row.get("cell_id"), name=f"{prefix}.cell_id"),
        counts=_freeze(_mapping(row.get("counts"), name=f"{prefix}.counts")),
        reference_gradient=gradient,
        raw=_freeze(row),
    )


def _gradient_sort_key(row: GradientRow) -> tuple[Any, ...]:
    return (
        row.scenario_family,
        row.scenario,
        row.target,
        row.method,
        row.samples,
        -1 if row.outer_seed is None else row.outer_seed,
        -1 if row.inner_seed is None else row.inner_seed,
    )


def _aggregate_gradients(rows: Sequence[GradientRow]) -> tuple[GradientAggregate, ...]:
    groups: dict[tuple[str, str, str, str, int], list[GradientRow]] = defaultdict(list)
    for row in rows:
        groups[
            (row.scenario_family, row.scenario, row.target, row.method, row.samples)
        ].append(row)
    aggregates = []
    for key, selected in sorted(groups.items()):
        times = tuple(row.wall_time for row in selected)
        aggregates.append(
            GradientAggregate(
                scenario_family=key[0],
                scenario=key[1],
                target=key[2],
                method=key[3],
                samples=key[4],
                rows=len(selected),
                mean_wall_time=checked_fmean(times),
                wall_time_q25=checked_quantile(times, 0.25),
                wall_time_q75=checked_quantile(times, 0.75),
            )
        )
    return tuple(aggregates)


def _aggregate_performance(
    rows: Sequence[PerformanceRow],
) -> tuple[PerformanceAggregate, ...]:
    groups: dict[tuple[str, str], list[PerformanceRow]] = defaultdict(list)
    for row in rows:
        groups[(row.scenario, row.method)].append(row)
    aggregates = []
    for (scenario, method), selected in sorted(groups.items()):
        times = tuple(row.warm_median for row in selected)
        aggregates.append(
            PerformanceAggregate(
                scenario=scenario,
                method=method,
                rows=len(selected),
                mean_warm_time=checked_fmean(times),
                warm_time_q25=checked_quantile(times, 0.25),
                warm_time_q75=checked_quantile(times, 0.75),
            )
        )
    return tuple(aggregates)


def _assert_close(name: str, declared: float, expected: float) -> None:
    if not math.isclose(declared, expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise BuildError(
            f"summary aggregate mismatch for {name}: declared={declared!r}, "
            f"expected={expected!r}"
        )


def _assert_vector_close(
    name: str, declared: Sequence[float], expected: Sequence[float]
) -> None:
    if len(declared) != len(expected):
        raise BuildError(f"summary aggregate dimension mismatch for {name}")
    for index, (declared_value, expected_value) in enumerate(
        zip(declared, expected, strict=True)
    ):
        _assert_close(f"{name}[{index}]", declared_value, expected_value)


def _mean_vectors(name: str, vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not vectors:
        raise BuildError(f"{name} must aggregate nonempty vectors")
    dimension = len(vectors[0])
    if dimension == 0 or any(len(vector) != dimension for vector in vectors):
        raise BuildError(f"{name} vectors have inconsistent dimensions")
    return tuple(
        checked_fmean(tuple(vector[index] for vector in vectors))
        for index in range(dimension)
    )


def _sample_variance_vectors(
    name: str, vectors: Sequence[Sequence[float]]
) -> tuple[float, ...]:
    if not vectors:
        raise BuildError(f"{name} must aggregate nonempty vectors")
    dimension = len(vectors[0])
    if dimension == 0 or any(len(vector) != dimension for vector in vectors):
        raise BuildError(f"{name} vectors have inconsistent dimensions")
    if len(vectors) == 1:
        return (0.0,) * dimension
    result = []
    for index in range(dimension):
        try:
            value = statistics.variance(vector[index] for vector in vectors)
        except (OverflowError, statistics.StatisticsError) as error:
            raise BuildError(f"{name} variance cannot be represented") from error
        result.append(_number(value, name=f"{name} variance[{index}]"))
    return tuple(result)


def _reference_metrics(
    gradient: Sequence[float], reference: Sequence[float]
) -> tuple[float, float, float]:
    if len(gradient) != len(reference) or not gradient:
        raise BuildError("summary reference metrics have inconsistent dimensions")
    reference_scale = max(abs(value) for value in reference)
    gradient_scale = max(abs(value) for value in gradient)
    if reference_scale == 0.0:
        relative_error = 0.0
        cosine_similarity = 0.0
    elif tuple(gradient) == tuple(reference):
        relative_error = 0.0
        cosine_similarity = 1.0
    else:
        common_scale = max(reference_scale, gradient_scale)
        scaled_gradient = tuple(value / common_scale for value in gradient)
        scaled_reference = tuple(value / common_scale for value in reference)
        difference_norm = math.hypot(
            *(
                first - second
                for first, second in zip(scaled_gradient, scaled_reference, strict=True)
            )
        )
        reference_norm = math.hypot(*scaled_reference)
        if reference_norm == 0.0:
            raise BuildError("summary reference-relative error is unrepresentable")
        relative_error = difference_norm / reference_norm
        if gradient_scale == 0.0:
            cosine_similarity = 0.0
        else:
            gradient_norm = math.hypot(*scaled_gradient)
            cosine_similarity = math.fsum(
                first * second
                for first, second in zip(scaled_gradient, scaled_reference, strict=True)
            ) / (gradient_norm * reference_norm)
            cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
    reference_signs = [index for index, value in enumerate(reference) if value != 0.0]
    sign_agreement = (
        checked_fmean(
            tuple(
                float(
                    math.copysign(1.0, gradient[index])
                    == math.copysign(1.0, reference[index])
                )
                if gradient[index] != 0.0
                else 0.0
                for index in reference_signs
            )
        )
        if reference_signs
        else 0.0
    )
    return relative_error, cosine_similarity, sign_agreement


def _assert_exact_sources(
    name: str, declared: Sequence[str], expected: Sequence[str]
) -> None:
    if len(declared) != len(set(declared)) or set(declared) != set(expected):
        raise BuildError(
            f"summary lineage mismatch for {name}: declared={sorted(declared)!r}, "
            f"expected={sorted(expected)!r}"
        )


def _validate_method_summaries(
    summary: SummaryTables, rows: Sequence[GradientRow]
) -> None:
    groups: dict[tuple[str, str, str, str], list[GradientRow]] = defaultdict(list)
    for row in rows:
        scenario = f"{row.scenario_family}:{row.scenario}:{row.start_id}"
        groups[(scenario, row.target, row.method, str(row.samples))].append(row)
    declared = {
        (row.scenario, row.target, row.method, str(row.samples)): row
        for row in summary.method_summaries
    }
    if len(declared) != len(summary.method_summaries) or set(declared) != set(groups):
        raise BuildError("summary method aggregate strata mismatch raw gradient rows")
    for key, selected in groups.items():
        aggregate = declared[key]
        _assert_exact_sources(
            f"method_summaries[{aggregate.index}]",
            aggregate.source_row_ids,
            tuple(row.row_id for row in selected),
        )
        mean_gradient = _mean_vectors(
            f"method_summaries[{aggregate.index}].mean_gradient",
            tuple(row.gradient for row in selected),
        )
        mean_reference = _mean_vectors(
            f"method_summaries[{aggregate.index}].reference_gradient",
            tuple(row.reference_gradient for row in selected),
        )
        empirical_variance = _sample_variance_vectors(
            f"method_summaries[{aggregate.index}]",
            tuple(row.gradient for row in selected),
        )
        bias = tuple(
            first - second
            for first, second in zip(mean_gradient, mean_reference, strict=True)
        )
        mse = tuple(
            _number(first * first + variance, name="method summary MSE")
            for first, variance in zip(bias, empirical_variance, strict=True)
        )
        relative_error, cosine_similarity, sign_agreement = _reference_metrics(
            mean_gradient, mean_reference
        )
        prefix = f"method_summaries[{aggregate.index}]"
        _assert_vector_close(
            prefix + ".mean_gradient", aggregate.mean_gradient, mean_gradient
        )
        _assert_vector_close(prefix + ".empirical_bias", aggregate.empirical_bias, bias)
        _assert_vector_close(
            prefix + ".empirical_variance",
            aggregate.empirical_variance,
            empirical_variance,
        )
        _assert_vector_close(
            prefix + ".mean_squared_error", aggregate.mean_squared_error, mse
        )
        _assert_close(
            prefix + ".relative_error", aggregate.relative_error, relative_error
        )
        _assert_close(
            prefix + ".cosine_similarity",
            aggregate.cosine_similarity,
            cosine_similarity,
        )
        _assert_close(
            prefix + ".sign_agreement", aggregate.sign_agreement, sign_agreement
        )


def _validate_optimization_summaries(
    summary: SummaryTables, rows: Sequence[OptimizationRow]
) -> None:
    groups: dict[tuple[str, str], list[OptimizationRow]] = defaultdict(list)
    for row in rows:
        groups[(row.scenario_family, row.method)].append(row)
    declared = {
        (row.scenario, row.method): row for row in summary.optimization_summaries
    }
    if len(declared) != len(summary.optimization_summaries) or set(declared) != set(
        groups
    ):
        raise BuildError("summary optimization aggregate strata mismatch raw rows")
    for key, selected in groups.items():
        aggregate = declared[key]
        _assert_exact_sources(
            f"optimization_summaries[{aggregate.index}]",
            aggregate.source_row_ids,
            tuple(row.row_id for row in selected),
        )
        final = tuple(row.final_hard_loss for row in selected)
        prefix = f"optimization_summaries[{aggregate.index}]"
        _assert_close(
            prefix + ".final_hard_loss_mean",
            aggregate.final_hard_loss_mean,
            checked_fmean(final),
        )
        _assert_close(
            prefix + ".final_hard_loss_ci_low",
            aggregate.final_hard_loss_ci_low,
            checked_quantile(final, 0.025),
        )
        _assert_close(
            prefix + ".final_hard_loss_ci_high",
            aggregate.final_hard_loss_ci_high,
            checked_quantile(final, 0.975),
        )
        _assert_close(
            prefix + ".success_rate",
            aggregate.success_rate,
            checked_fmean(tuple(float(row.success) for row in selected)),
        )
        _assert_close(
            prefix + ".held_out_loss_mean",
            aggregate.held_out_loss_mean,
            checked_fmean(tuple(row.held_out_loss for row in selected)),
        )


def _validate_performance_summaries(
    summary: SummaryTables, rows: Sequence[PerformanceRow]
) -> None:
    by_id = {row.row_id: row for row in rows}
    referenced: set[str] = set()
    fields = (
        "cold_compile_time",
        "warm_median",
        "warm_iqr",
        "forward_executions",
        "backward_executions",
        "tracemalloc_peak",
        "rss_delta",
        "warp_allocation_peak",
        "device_free_memory_delta",
    )
    for aggregate in summary.performance_summaries:
        if len(aggregate.source_row_ids) != 1:
            raise BuildError("performance summary lineage must be one-to-one")
        source_id = aggregate.source_row_ids[0]
        source = by_id.get(source_id)
        if source is None or source_id in referenced:
            raise BuildError("performance summary lineage mismatch raw rows")
        referenced.add(source_id)
        if (aggregate.scenario, aggregate.method) != (source.scenario, source.method):
            raise BuildError("performance summary label mismatch raw row")
        for field in (
            "tracemalloc_peak",
            "rss_delta",
            "warp_allocation_peak",
            "device_free_memory_delta",
        ):
            if getattr(source, f"{field}_available") != (
                getattr(source, field) is not None
            ):
                raise BuildError(f"performance availability mismatch for {field}")
        for field in fields:
            declared_value = getattr(aggregate, field)
            expected_value = getattr(source, field)
            if declared_value is None or expected_value is None:
                if declared_value is not expected_value:
                    raise BuildError(f"performance summary mismatch for {field}")
            else:
                _assert_close(
                    f"performance_summaries[{aggregate.index}].{field}",
                    float(declared_value),
                    float(expected_value),
                )
    if referenced != set(by_id):
        raise BuildError("performance summaries do not exactly cover raw rows")


def _accepted_flag(row: Mapping[str, Any]) -> bool:
    accepted = row.get("accepted", True)
    if isinstance(accepted, Mapping):
        return bool(accepted.get("references"))
    return bool(accepted)


def _validate_validity_summaries(
    summary: SummaryTables, raw_records: Mapping[str, RawRecord]
) -> None:
    groups: dict[str, list[RawRecord]] = defaultdict(list)
    for record in raw_records.values():
        row = record.raw
        if (
            row.get("method") in REQUIRED_METHOD_IDS
            or "final_hard_loss" in row
            or "warm_median" in row
        ):
            continue
        key = (
            "references"
            if "cell_id" in row and isinstance(row.get("accepted"), Mapping)
            else str(row.get("scenario_family", "other"))
        )
        groups[key].append(record)
    declared = {row.scenario: row for row in summary.scenario_validity}
    if len(declared) != len(summary.scenario_validity) or set(declared) != set(groups):
        raise BuildError("summary validity aggregate strata mismatch raw rows")
    for scenario, selected in groups.items():
        aggregate = declared[scenario]
        flags = tuple(_accepted_flag(record.raw) for record in selected)
        _assert_exact_sources(
            f"scenario_validity[{aggregate.index}]",
            aggregate.source_row_ids,
            tuple(record.row_id for record in selected),
        )
        if aggregate.accepted is not all(flags):
            raise BuildError("summary validity accepted aggregate mismatch")
        expected_metrics = {
            "row_count": len(selected),
            "accepted_count": sum(flags),
        }
        if dict(aggregate.metrics) != expected_metrics:
            raise BuildError("summary validity metrics aggregate mismatch")


def _validate_headline_metrics(
    summary: SummaryTables,
    raw_records: Mapping[str, RawRecord],
    optimization_rows: Sequence[OptimizationRow],
) -> None:
    declared = {row.name: row for row in summary.headline_metrics}
    if set(declared) != {"analytic_anchor_accepted", "path_best_held_out_loss"}:
        raise BuildError("summary headline aggregate names mismatch producer contract")
    analytic = next(
        (
            record
            for record in raw_records.values()
            if record.source_file == "data/raw/analytic.json"
            and record.raw.get("method") not in REQUIRED_METHOD_IDS
            and record.raw.get("accepted") is True
        ),
        None,
    )
    if analytic is None or not optimization_rows:
        raise BuildError("summary headline aggregates lack canonical raw sources")
    analytic_summary = declared["analytic_anchor_accepted"]
    if analytic_summary.unit != "boolean":
        raise BuildError("summary headline analytic unit mismatch")
    _assert_exact_sources(
        "headline_metrics.analytic_anchor_accepted",
        analytic_summary.source_row_ids,
        (analytic.row_id,),
    )
    _assert_close(
        "headline_metrics.analytic_anchor_accepted.value",
        analytic_summary.value,
        1.0,
    )
    path_rows = [
        row
        for row in optimization_rows
        if row.source_file == "data/raw/path_tracer_optimization.json"
        and row.scenario_family == "path_tracer"
    ]
    if not path_rows:
        raise BuildError("summary path headline lacks path-tracer optimization rows")
    best = min(path_rows, key=lambda row: (row.held_out_loss, row.source_index))
    path_summary = declared["path_best_held_out_loss"]
    if path_summary.unit != "mean_squared_error":
        raise BuildError("summary headline path unit mismatch")
    _assert_exact_sources(
        "headline_metrics.path_best_held_out_loss",
        path_summary.source_row_ids,
        (best.row_id,),
    )
    _assert_close(
        "headline_metrics.path_best_held_out_loss.value",
        path_summary.value,
        best.held_out_loss,
    )


def _validate_summary_aggregates(
    summary: SummaryTables,
    *,
    gradient_rows: Sequence[GradientRow],
    optimization_rows: Sequence[OptimizationRow],
    performance_rows: Sequence[PerformanceRow],
    raw_records: Mapping[str, RawRecord],
) -> None:
    _validate_method_summaries(summary, gradient_rows)
    _validate_optimization_summaries(summary, optimization_rows)
    _validate_performance_summaries(summary, performance_rows)
    _validate_validity_summaries(summary, raw_records)
    _validate_headline_metrics(summary, raw_records, optimization_rows)


def _load_report(root: Path) -> ReportModel:
    root = Path(root)
    try:
        manifest_payload, loaded = validate_manifest(root)
    except ValidationError as error:
        raise BuildError(str(error)) from error

    manifest = _load_manifest(manifest_payload)
    _validate_png_assets(root, manifest.files)
    summary = _load_summary(loaded.get(SUMMARY_FILE))
    raw_row_ids: set[str] = set()
    raw_records: dict[str, RawRecord] = {}
    gradient_rows: list[GradientRow] = []
    optimization_rows: list[OptimizationRow] = []
    performance_rows: list[PerformanceRow] = []
    reference_rows: list[ReferenceRow] = []

    for source_file in RAW_FILES:
        rows = _dataset_rows(loaded.get(source_file), source_file=source_file)
        for index, row in enumerate(rows):
            row_id = _string(
                row.get("row_id"), name=f"{source_file}.rows[{index}].row_id"
            )
            if row_id in raw_row_ids:
                raise BuildError(f"duplicate raw row ID {row_id!r}")
            raw_row_ids.add(row_id)
            raw_records[row_id] = RawRecord(
                source_file=source_file,
                source_index=index,
                row_id=row_id,
                raw=_freeze(row),
            )
            if "gradient" in row:
                gradient_rows.append(_gradient_row(source_file, index, row))
            if source_file.endswith("_optimization.json"):
                optimization_rows.append(_optimization_row(source_file, index, row))
            elif source_file == "data/raw/performance.json":
                performance_rows.append(_performance_row(source_file, index, row))
            elif source_file == "data/raw/references.json":
                reference_rows.append(_reference_row(source_file, index, row))
            elif "method" in row:
                _method(row.get("method"), name=f"{source_file}.rows[{index}].method")

    if set(summary.method_labels) != set(REQUIRED_METHOD_IDS):
        raise BuildError("the report cannot silently omit a method label")
    summary_source_ids = {
        source_id
        for collection in (
            summary.headline_metrics,
            summary.method_summaries,
            summary.optimization_summaries,
            summary.performance_summaries,
            summary.scenario_validity,
        )
        for row in collection
        for source_id in row.source_row_ids
    }
    unknown_summary_ids = summary_source_ids - raw_row_ids
    if unknown_summary_ids:
        raise BuildError(
            f"summary references unknown raw rows: {sorted(unknown_summary_ids)}"
        )

    plots: dict[str, PlotDataset] = {}
    for source_file in PLOT_FILES:
        rows = _dataset_rows(loaded.get(source_file), source_file=source_file)
        if not rows:
            raise BuildError(f"{source_file} must contain plot records")
        plot_records = []
        plot_ids: set[str] = set()
        for index, row in enumerate(rows):
            plot_id = _string(
                row.get("plot_id"), name=f"{source_file}.rows[{index}].plot_id"
            )
            if plot_id in plot_ids:
                raise BuildError(
                    f"{source_file} contains duplicate plot ID {plot_id!r}"
                )
            plot_ids.add(plot_id)
            source_ids = _strings(
                row.get("source_row_ids"),
                name=f"{source_file}.rows[{index}].source_row_ids",
                nonempty=True,
            )
            if len(source_ids) != len(set(source_ids)):
                raise BuildError(f"{source_file} plot {plot_id!r} repeats source rows")
            missing = set(source_ids) - raw_row_ids
            if missing:
                raise BuildError(
                    f"{source_file} references unknown raw rows: {sorted(missing)}"
                )
            plot_records.append(
                PlotRecord(
                    source_file=source_file,
                    source_index=index,
                    plot_id=plot_id,
                    source_row_ids=source_ids,
                    raw=_freeze(row),
                )
            )
        plots[source_file] = PlotDataset(
            source_file=source_file, rows=tuple(plot_records)
        )

    ordered_gradients = tuple(sorted(gradient_rows, key=_gradient_sort_key))
    ordered_optimizations = tuple(
        sorted(
            optimization_rows,
            key=lambda row: (row.scenario_family, row.scenario, row.method, row.row_id),
        )
    )
    ordered_performance = tuple(
        sorted(performance_rows, key=lambda row: (row.scenario, row.method, row.row_id))
    )
    ordered_references = tuple(sorted(reference_rows, key=lambda row: row.cell_id))
    _validate_summary_aggregates(
        summary,
        gradient_rows=ordered_gradients,
        optimization_rows=ordered_optimizations,
        performance_rows=ordered_performance,
        raw_records=raw_records,
    )
    model = ReportModel(
        root=root,
        manifest=manifest,
        summary=summary,
        gradient_rows=ordered_gradients,
        optimization_rows=ordered_optimizations,
        performance_rows=ordered_performance,
        reference_rows=ordered_references,
        plots=MappingProxyType(plots),
        raw_row_ids=frozenset(raw_row_ids),
        raw_records=MappingProxyType(raw_records),
        gradient_aggregates=_aggregate_gradients(ordered_gradients),
        performance_aggregates=_aggregate_performance(ordered_performance),
    )
    _validate_figure_semantics(model)
    return model


def load_report(root: Path) -> ReportModel:
    try:
        return _load_report(Path(root))
    except BuildError:
        raise
    except (
        IndexError,
        KeyError,
        OSError,
        OverflowError,
        TypeError,
        ValueError,
    ) as error:
        raise BuildError(f"cannot load report evidence: {error}") from error


METHOD_DESCRIPTIONS = MappingProxyType(
    {
        "crisp_ad": "Reverse AD of the realized hard execution; discontinuity boundary terms are omitted.",
        "crisp_fd": "Central finite differences of the hard program at a finite step.",
        "smoothed_pathwise": "Pathwise AD under Gaussian parameter perturbations; biased at true jumps.",
        "score": "Whole-program Gaussian score estimator of the smoothed hard objective.",
        "smoothed_crn_fd": "Common-random-number finite differences of the Gaussian-smoothed hard objective.",
        "soft_ad": "Reverse AD of the coherent local soft surrogate.",
        "straight_through_ad": "Crisp-forward surrogate/pseudo-gradient, not the derivative of the hard execution.",
        "residual_control_variate": "Hard residual score correction plus the pathwise soft-surrogate gradient.",
    }
)

FIGURE_SPECS = MappingProxyType(
    {
        "ANALYTIC_FIGURES": (
            (
                "assets/figures/analytic_gates.png",
                "data/plot_data/analytic_gates.json",
                "Closed-form gate checks",
                "Measured hard, soft, and analytic gate behavior.",
            ),
            (
                "assets/figures/triangle_edge_slices.png",
                "data/plot_data/gradient_quality.json",
                "Triangle edge slices",
                "Measured gradients across analytic triangle edge crossings.",
            ),
            (
                "assets/figures/collision_2d.png",
                "data/plot_data/validity.json",
                "Two-dimensional collision validity",
                "Measured trajectory and validity diagnostics for the upgraded collision regressions.",
            ),
        ),
        "PATH_FIGURES": (
            (
                "assets/images/path_target.png",
                "data/plot_data/optimization.json",
                "Target render",
                "Independent target image from the fixed-depth analytic-sphere renderer.",
            ),
            (
                "assets/images/path_initial.png",
                "data/plot_data/optimization.json",
                "Initial render",
                "Initial inverse-rendering image before optimization.",
            ),
            (
                "assets/images/path_recovered.png",
                "data/plot_data/optimization.json",
                "Recovered render",
                "Held-out image rendered from recovered parameters.",
            ),
            (
                "assets/figures/path_tracer_recovery.png",
                "data/plot_data/optimization.json",
                "Path-tracing recovery",
                "Measured held-out inverse-rendering recovery trajectories.",
            ),
            (
                "assets/figures/path_tracer_gradient_quality.png",
                "data/plot_data/gradient_quality.json",
                "Path-tracing gradient quality",
                "Reference-relative gradient measurements for the path tracer.",
            ),
        ),
        "CONTACT_FIGURES": (
            (
                "assets/figures/contact_3d_trajectories.png",
                "data/plot_data/validity.json",
                "Three-dimensional contact trajectories",
                "Measured multi-body impulse, friction, and contact-sequence diagnostics.",
            ),
            (
                "assets/figures/contact_3d_gradient_quality.png",
                "data/plot_data/gradient_quality.json",
                "Contact gradient quality",
                "Reference-relative gradient measurements for the contact scene.",
            ),
        ),
        "OPAQUE_FIGURES": (
            (
                "assets/figures/opaque_mesh_boundary.png",
                "data/plot_data/validity.json",
                "Opaque native-query boundary",
                "Measured estimator-only behavior around native mesh-query visibility changes.",
            ),
        ),
        "QUALITY_FIGURES": (
            (
                "assets/figures/bias_variance.png",
                "data/plot_data/bias_variance.json",
                "Bias and variance",
                "Measured bias, contribution variance, and variance-of-the-mean summaries.",
            ),
        ),
        "OPTIMIZATION_FIGURES": (
            (
                "assets/figures/optimization.png",
                "data/plot_data/optimization.json",
                "Optimization outcomes",
                "Multi-seed hard-loss and held-out optimization outcomes.",
            ),
        ),
        "PERFORMANCE_FIGURES": (
            (
                "assets/figures/performance.png",
                "data/plot_data/performance.json",
                "Measured cost",
                "Cold compilation, warmed execution, and method work counts.",
            ),
        ),
    }
)

FIGURE_FAMILIES = MappingProxyType(
    {
        "assets/figures/analytic_gates.png": "analytic",
        "assets/figures/triangle_edge_slices.png": "triangle_2d",
        "assets/figures/collision_2d.png": "collision_2d",
        "assets/images/path_target.png": "path_tracer",
        "assets/images/path_initial.png": "path_tracer",
        "assets/images/path_recovered.png": "path_tracer",
        "assets/figures/path_tracer_recovery.png": "path_tracer",
        "assets/figures/path_tracer_gradient_quality.png": "path_tracer",
        "assets/figures/contact_3d_trajectories.png": "contact_3d",
        "assets/figures/contact_3d_gradient_quality.png": "contact_3d",
        "assets/figures/opaque_mesh_boundary.png": "opaque_mesh",
    }
)

IMAGE_FIELDS = MappingProxyType(
    {
        "assets/images/path_target.png": "path_target_image",
        "assets/images/path_initial.png": "path_initial_image",
        "assets/images/path_recovered.png": "path_recovered_image",
    }
)


def _plot_record_matches(asset: str, record: PlotRecord) -> bool:
    row = record.raw
    if asset.endswith("analytic_gates.png"):
        return record.plot_id == "analytic-gates"
    if asset.endswith("triangle_edge_slices.png"):
        return row.get("kind") == "triangle_edge_slice" and record.plot_id.startswith(
            "triangle-edge-slice-"
        )
    if asset.endswith("collision_2d.png"):
        return row.get("kind") == "collision_trajectory" and record.plot_id.startswith(
            "collision-"
        )
    if asset.endswith("path_tracer_recovery.png") or asset in IMAGE_FIELDS:
        return (
            record.plot_id == "path-render-comparison"
            and row.get("kind") == "path_render_comparison"
        )
    if asset.endswith("path_tracer_gradient_quality.png"):
        return (
            row.get("kind") == "path_tracer_gradient_quality"
            and record.plot_id.startswith("path-tracer-gradient-quality-")
            and str(row.get("scenario", "")).startswith("path_tracer:")
        )
    if asset.endswith("contact_3d_trajectories.png"):
        return (
            record.plot_id == "contact-3d-trajectories"
            and row.get("kind") == "contact_trajectory"
        )
    if asset.endswith("contact_3d_gradient_quality.png"):
        return (
            row.get("kind") == "contact_3d_gradient_quality"
            and record.plot_id.startswith("contact-3d-gradient-quality-")
            and str(row.get("scenario", "")).startswith("contact_3d:")
        )
    if asset.endswith("opaque_mesh_boundary.png"):
        return (
            record.plot_id == "opaque-mesh-boundary"
            and row.get("kind") == "opaque_boundary"
        )
    if asset.endswith("bias_variance.png"):
        return record.plot_id.startswith("bias-variance-")
    if asset.endswith("optimization.png"):
        return record.plot_id.startswith("optimization-") or record.plot_id == (
            "path-render-comparison"
        )
    if asset.endswith("performance.png"):
        return record.plot_id.startswith("performance-")
    raise BuildError(f"figure {asset!r} has no plot-record selector")


def _select_figure_records(
    model: ReportModel, asset: str, source_file: str
) -> tuple[PlotRecord, ...]:
    dataset = model.plots.get(source_file)
    if dataset is None:
        raise BuildError(f"figure {asset!r} has no plot dataset {source_file!r}")
    selected = tuple(
        record for record in dataset.rows if _plot_record_matches(asset, record)
    )
    if not selected:
        raise BuildError(f"figure {asset!r} has no exact matching plot record")
    expected_family = FIGURE_FAMILIES.get(asset)
    if expected_family is not None:
        for record in selected:
            for source_id in record.source_row_ids:
                raw = model.raw_records[source_id].raw
                if raw.get("scenario_family") != expected_family:
                    raise BuildError(
                        f"figure {asset!r} plot lineage crosses scenario families"
                    )
    image_field = IMAGE_FIELDS.get(asset)
    if image_field is not None and any(
        image_field not in record.raw for record in selected
    ):
        raise BuildError(f"figure {asset!r} plot record is missing {image_field!r}")
    return selected


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise BuildError(f"figure evidence is not canonical JSON: {error}") from error


def _require_plot_records(
    asset: str,
    records: Sequence[PlotRecord],
    expected: Mapping[str, Mapping[str, Any]],
) -> None:
    observed: dict[str, PlotRecord] = {}
    for record in records:
        if record.plot_id in observed:
            raise BuildError(
                f"figure {asset!r} has duplicate plot identity {record.plot_id!r}"
            )
        observed[record.plot_id] = record
    if set(observed) != set(expected):
        raise BuildError(
            f"figure {asset!r} plot identity mismatch: "
            f"observed={sorted(observed)!r}, expected={sorted(expected)!r}"
        )
    for plot_id, expected_row in expected.items():
        if _canonical_json(observed[plot_id].raw) != _canonical_json(expected_row):
            raise BuildError(
                f"figure {asset!r} plot {plot_id!r} semantic field mismatch"
            )


def _insert_unique_semantic_record(
    records: dict[str, Any],
    identity: str,
    value: Any,
    *,
    description: str,
) -> None:
    if identity in records:
        raise BuildError(
            f"duplicate {description} {identity!r}; evidence must map one-to-one"
        )
    records[identity] = value


def _raw_file_records(model: ReportModel, source_file: str) -> tuple[RawRecord, ...]:
    return tuple(
        sorted(
            (
                record
                for record in model.raw_records.values()
                if record.source_file == source_file
            ),
            key=lambda record: record.source_index,
        )
    )


def _flatten_numeric(value: Any, *, name: str) -> tuple[float, ...]:
    if isinstance(value, (tuple, list)):
        result = tuple(
            component
            for index, child in enumerate(value)
            for component in _flatten_numeric(child, name=f"{name}[{index}]")
        )
        if not result:
            raise BuildError(f"{name} must contain image samples")
        return result
    return (_number(value, name=name),)


def _image_mse(first: Any, second: Any, *, name: str) -> float:
    first_values = _flatten_numeric(first, name=f"{name}.first")
    second_values = _flatten_numeric(second, name=f"{name}.second")
    if len(first_values) != len(second_values):
        raise BuildError(f"{name} image dimensions disagree")
    return checked_fmean(
        tuple(
            (first_value - second_value) ** 2
            for first_value, second_value in zip(
                first_values, second_values, strict=True
            )
        )
    )


def _validate_figure_semantics(model: ReportModel) -> None:
    selections: dict[str, tuple[PlotRecord, ...]] = {}
    for specs in FIGURE_SPECS.values():
        for asset, source_file, _title, _caption in specs:
            _insert_unique_semantic_record(
                selections,
                asset,
                _select_figure_records(model, asset, source_file),
                description="fixed asset identity",
            )

    analytic_records = _raw_file_records(model, "data/raw/analytic.json")
    analytic_values = [
        float(record.raw["gradient"][0])
        for record in analytic_records
        if record.raw.get("method") in {"score", "soft_ad", "residual_control_variate"}
        and isinstance(record.raw.get("gradient"), (tuple, list))
        and record.raw["gradient"]
    ]
    _require_plot_records(
        "assets/figures/analytic_gates.png",
        selections["assets/figures/analytic_gates.png"],
        {
            "analytic-gates": {
                "plot_id": "analytic-gates",
                "values": analytic_values or [0.0],
                "source_row_ids": [record.row_id for record in analytic_records],
            }
        },
    )

    triangle_expected = {}
    for record in _raw_file_records(model, "data/raw/triangle_2d.json"):
        row = record.raw
        if row.get("scenario") != "edge_slice":
            continue
        edge = _integer(row.get("edge"), name=f"{record.row_id}.edge")
        samples = _sequence(row.get("rows"), name=f"{record.row_id}.rows")
        plot_id = f"triangle-edge-slice-{edge}"
        _insert_unique_semantic_record(
            triangle_expected,
            plot_id,
            {
                "plot_id": plot_id,
                "kind": "triangle_edge_slice",
                "scenario": "triangle_2d:edge_slice",
                "edge": edge,
                "x_values": [
                    _number(
                        _mapping(sample, name=plot_id).get("signed_offset"),
                        name=plot_id,
                    )
                    for sample in samples
                ],
                "values": [
                    _number(
                        _mapping(sample, name=plot_id).get("analytic_intersection"),
                        name=plot_id,
                    )
                    for sample in samples
                ],
                "hard_values": [
                    _number(
                        _mapping(sample, name=plot_id).get("hard_intersection"),
                        name=plot_id,
                    )
                    for sample in samples
                ],
                "source_row_ids": [record.row_id],
            },
            description="triangle edge identity",
        )
    _require_plot_records(
        "assets/figures/triangle_edge_slices.png",
        selections["assets/figures/triangle_edge_slices.png"],
        triangle_expected,
    )

    collision_expected = {}
    for record in _raw_file_records(model, "data/raw/collision_2d.json"):
        row = record.raw
        if "final_positions" not in row:
            continue
        scenario = _string(row.get("scenario"), name=f"{record.row_id}.scenario")
        start_id = _string(row.get("start_id"), name=f"{record.row_id}.start_id")
        outer_seed = _integer(row.get("outer_seed"), name=f"{record.row_id}.outer_seed")
        plot_id = f"collision-{scenario}-{start_id}-{outer_seed}"
        _insert_unique_semantic_record(
            collision_expected,
            plot_id,
            {
                "plot_id": plot_id,
                "kind": "collision_trajectory",
                "scenario": f"collision_2d:{scenario}",
                "final_positions": _jsonable(row["final_positions"]),
                "values": _jsonable(row.get("losses")),
                "source_row_ids": [record.row_id],
            },
            description="collision trajectory identity",
        )
    _require_plot_records(
        "assets/figures/collision_2d.png",
        selections["assets/figures/collision_2d.png"],
        collision_expected,
    )

    comparison_records: dict[str, RawRecord] = {}
    for record in _raw_file_records(model, "data/raw/path_tracer_gradients.json"):
        if record.raw.get("scenario") != "analytic_five_sphere_render" or not str(
            record.raw.get("role", "")
        ).startswith("comparison_"):
            continue
        role = _string(record.raw.get("role"), name=f"{record.row_id}.role")
        _insert_unique_semantic_record(
            comparison_records,
            role,
            record,
            description="path render role",
        )
    comparison_roles = (
        "comparison_initial",
        "comparison_target",
        "comparison_recovered",
    )
    if set(comparison_records) != set(comparison_roles) or any(
        comparison_records[role].raw.get("accepted") is not True
        or "image" not in comparison_records[role].raw
        for role in comparison_roles
    ):
        raise BuildError("path comparison figure requires three accepted render roles")
    initial_image = comparison_records["comparison_initial"].raw["image"]
    target_image = comparison_records["comparison_target"].raw["image"]
    recovered_image = comparison_records["comparison_recovered"].raw["image"]
    comparison_selection = selections["assets/figures/path_tracer_recovery.png"]
    if len(comparison_selection) != 1:
        raise BuildError("path comparison figure requires exactly one plot record")
    observed_comparison_values = tuple(
        _number(value, name="path comparison value")
        for value in _sequence(
            comparison_selection[0].raw.get("values"),
            name="path comparison values",
        )
    )
    expected_comparison_values = (
        _image_mse(initial_image, target_image, name="path initial MSE"),
        _image_mse(recovered_image, target_image, name="path recovered MSE"),
    )
    if len(observed_comparison_values) != len(expected_comparison_values):
        raise BuildError("path comparison figure has the wrong MSE arity")
    for index, (observed, expected) in enumerate(
        zip(observed_comparison_values, expected_comparison_values, strict=True)
    ):
        _assert_close(f"path comparison MSE[{index}]", observed, expected)
    comparison_expected = {
        "path-render-comparison": {
            "plot_id": "path-render-comparison",
            "kind": "path_render_comparison",
            "scenario": "path_tracer",
            "method": "best_recovered_render",
            "values": list(observed_comparison_values),
            "source_row_ids": [
                comparison_records[role].row_id for role in comparison_roles
            ],
            "path_target_image": _jsonable(target_image),
            "path_initial_image": _jsonable(initial_image),
            "path_recovered_image": _jsonable(recovered_image),
        }
    }
    for asset in (
        "assets/figures/path_tracer_recovery.png",
        "assets/images/path_target.png",
        "assets/images/path_initial.png",
        "assets/images/path_recovered.png",
    ):
        _require_plot_records(asset, selections[asset], comparison_expected)

    for family, prefix, kind, asset in (
        (
            "path_tracer",
            "path-tracer",
            "path_tracer_gradient_quality",
            "assets/figures/path_tracer_gradient_quality.png",
        ),
        (
            "contact_3d",
            "contact-3d",
            "contact_3d_gradient_quality",
            "assets/figures/contact_3d_gradient_quality.png",
        ),
    ):
        expected = {}
        for summary in model.summary.method_summaries:
            if not summary.scenario.startswith(f"{family}:"):
                continue
            plot_id = f"{prefix}-gradient-quality-{summary.index}"
            _insert_unique_semantic_record(
                expected,
                plot_id,
                {
                    "plot_id": plot_id,
                    "kind": kind,
                    "scenario": summary.scenario,
                    "method": summary.method,
                    "values": [summary.relative_error],
                    "source_row_ids": list(summary.source_row_ids),
                },
                description=f"{family} gradient plot identity",
            )
        _require_plot_records(asset, selections[asset], expected)

    contact_expected = {}
    for record in _raw_file_records(model, "data/raw/contact_3d_gradients.json"):
        row = record.raw
        if row.get("scenario") != "three_sphere_floor_ramp" or "positions" not in row:
            continue
        plot_id = "contact-3d-trajectories"
        _insert_unique_semantic_record(
            contact_expected,
            plot_id,
            {
                "plot_id": plot_id,
                "kind": "contact_trajectory",
                "scenario": "contact_3d",
                "positions": _jsonable(row["positions"]),
                "values": [
                    _number(
                        row.get("max_penetration"),
                        name=f"{record.row_id}.max_penetration",
                    ),
                    _number(
                        row.get("max_contact_energy_gain"),
                        name=f"{record.row_id}.max_contact_energy_gain",
                    ),
                ],
                "source_row_ids": [record.row_id],
            },
            description="contact trajectory identity",
        )
    _require_plot_records(
        "assets/figures/contact_3d_trajectories.png",
        selections["assets/figures/contact_3d_trajectories.png"],
        contact_expected,
    )

    opaque_expected = {}
    for record in _raw_file_records(model, "data/raw/opaque_mesh.json"):
        row = record.raw
        if row.get("transform_status") != "estimator_only":
            continue
        boundary = _mapping(row.get("boundary"), name=f"{record.row_id}.boundary")
        plot_id = "opaque-mesh-boundary"
        _insert_unique_semantic_record(
            opaque_expected,
            plot_id,
            {
                "plot_id": plot_id,
                "kind": "opaque_boundary",
                "scenario": "opaque_mesh",
                "values": [
                    _integer(
                        boundary.get("transformed_sites"),
                        name=f"{record.row_id}.transformed_sites",
                    ),
                    _integer(
                        boundary.get("preserved_sites"),
                        name=f"{record.row_id}.preserved_sites",
                    ),
                ],
                "labels": ["transformed", "preserved"],
                "source_row_ids": [record.row_id],
            },
            description="opaque boundary identity",
        )
    _require_plot_records(
        "assets/figures/opaque_mesh_boundary.png",
        selections["assets/figures/opaque_mesh_boundary.png"],
        opaque_expected,
    )

    bias_expected = {}
    for summary in model.summary.method_summaries:
        plot_id = f"bias-variance-{summary.index}"
        _insert_unique_semantic_record(
            bias_expected,
            plot_id,
            {
                "plot_id": plot_id,
                "scenario": summary.scenario,
                "method": summary.method,
                "values": list(summary.mean_squared_error),
                "source_row_ids": list(summary.source_row_ids),
            },
            description="bias-variance plot identity",
        )
    _require_plot_records(
        "assets/figures/bias_variance.png",
        selections["assets/figures/bias_variance.png"],
        bias_expected,
    )

    optimization_expected = {}
    for summary in model.summary.optimization_summaries:
        plot_id = f"optimization-{summary.index}"
        _insert_unique_semantic_record(
            optimization_expected,
            plot_id,
            {
                "plot_id": plot_id,
                "scenario": summary.scenario,
                "method": summary.method,
                "values": [summary.final_hard_loss_mean, summary.held_out_loss_mean],
                "source_row_ids": list(summary.source_row_ids),
            },
            description="optimization plot identity",
        )
    for plot_id, expected_row in comparison_expected.items():
        _insert_unique_semantic_record(
            optimization_expected,
            plot_id,
            expected_row,
            description="optimization plot identity",
        )
    _require_plot_records(
        "assets/figures/optimization.png",
        selections["assets/figures/optimization.png"],
        optimization_expected,
    )

    performance_expected = {}
    for summary in model.summary.performance_summaries:
        plot_id = f"performance-{summary.index}"
        _insert_unique_semantic_record(
            performance_expected,
            plot_id,
            {
                "plot_id": plot_id,
                "scenario": summary.scenario,
                "method": summary.method,
                "values": [
                    summary.cold_compile_time,
                    summary.warm_median,
                    summary.warm_iqr,
                ],
                "source_row_ids": list(summary.source_row_ids),
            },
            description="performance plot identity",
        )
    _require_plot_records(
        "assets/figures/performance.png",
        selections["assets/figures/performance.png"],
        performance_expected,
    )


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _json_pointer_part(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _source_fragment(source_key: str) -> str:
    if not source_key.startswith("/"):
        raise BuildError(f"source key must be a JSON pointer: {source_key!r}")
    return f"#{quote(source_key, safe='')}"


def _source_attributes(source_file: str, source_key: str) -> str:
    return (
        f' data-source-file="{_escape(source_file)}"'
        f' data-source-key="{_source_fragment(source_key)}"'
    )


def _text_cell(value: Any, *, static_key: str | None = None) -> str:
    key = static_key
    if key is None:
        key = re.sub(r"[^a-z0-9]+", "-", str(value).casefold()).strip("-")
    if not key:
        key = "empty-label"
    return f'<td data-static-key="{_escape(key)}">{_escape(value)}</td>'


def _provenance_cell(value: Any, source_file: str, source_key: str) -> str:
    return f"<td{_source_attributes(source_file, source_key)}>{_escape(value)}</td>"


def _numeric_cell(value: int | float, source_file: str, source_key: str) -> str:
    return (
        f"<td{_source_attributes(source_file, source_key)}>{format_number(value)}</td>"
    )


def _format_vector(values: Sequence[int | float]) -> str:
    return "[" + ", ".join(format_number(value) for value in values) + "]"


def _vector_cell(
    values: Sequence[int | float], source_file: str, source_key: str
) -> str:
    return _provenance_cell(_format_vector(values), source_file, source_key)


def _optional_numeric_cell(
    value: int | float | None, source_file: str, source_key: str
) -> str:
    if value is None:
        return _provenance_cell("unavailable", source_file, source_key)
    return _numeric_cell(value, source_file, source_key)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(child) for child in value]
    return value


def _canonical_evidence_display(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return format_number(value)
    if (
        isinstance(value, (tuple, list))
        and value
        and all(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            for item in value
        )
    ):
        return _format_vector(value)
    if value is None:
        return "unavailable"
    if isinstance(value, (Mapping, tuple, list)):
        try:
            return json.dumps(
                _jsonable(value),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise BuildError(
                f"evidence value has no canonical display form: {error}"
            ) from error
    if isinstance(value, str):
        return value
    raise BuildError(
        f"evidence value has unsupported display type {type(value).__name__}"
    )


def _value_cell(value: Any, source_file: str, source_key: str) -> str:
    return _provenance_cell(_canonical_evidence_display(value), source_file, source_key)


def _table(caption: str, headers: Sequence[str], rows: Sequence[str]) -> str:
    header = "".join(f'<th scope="col">{_escape(value)}</th>' for value in headers)
    body = "".join(rows)
    return (
        '<div class="table-scroll"><table>'
        f"<caption>{_escape(caption)}</caption>"
        f"<thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _method_table(model: ReportModel) -> str:
    rows = []
    for method in REQUIRED_METHOD_IDS:
        label = model.summary.method_labels[method]
        rows.append(
            "<tr>"
            f"<td{_source_attributes(SUMMARY_FILE, f'/method_labels/{_json_pointer_part(method)}')}>"
            f'<span class="method" data-method="{_escape(method)}">{_escape(label)}</span></td>'
            f"{_text_cell(METHOD_DESCRIPTIONS[method])}"
            "</tr>"
        )
    return _table(
        "Estimator semantics and applicability", ("Method", "Interpretation"), rows
    )


def _headline_table(model: ReportModel) -> str:
    rows = []
    for metric in model.summary.headline_metrics:
        pointer = f"/headline_metrics/{_json_pointer_part(metric.name)}/value"
        label = metric.name.replace("_", " ")
        rows.append(
            "<tr>"
            f"{_text_cell(label)}"
            f"{_numeric_cell(metric.value, SUMMARY_FILE, pointer)}"
            f"{_value_cell(metric.unit, SUMMARY_FILE, f'/headline_metrics/{_json_pointer_part(metric.name)}/unit')}"
            "</tr>"
        )
    return _table("Headline measurements", ("Metric", "Value", "Unit"), rows)


def _method_summary_table(model: ReportModel) -> str:
    rows = []
    gradients = {row.row_id: row for row in model.gradient_rows}
    selected = sorted(
        model.summary.method_summaries,
        key=lambda row: (row.scenario, row.target, row.method, row.samples),
    )
    for row in selected:
        prefix = f"/method_summaries/{row.index}"
        uncertainty = next(
            (
                gradients[source_id]
                for source_id in row.source_row_ids
                if source_id in gradients
                and gradients[source_id].ci_low is not None
                and gradients[source_id].ci_high is not None
            ),
            None,
        )
        if uncertainty is None:
            contribution_variance = _text_cell("unavailable")
            gradient_variance = _text_cell("unavailable")
            ci_low = _text_cell("unavailable")
            ci_high = _text_cell("unavailable")
        else:
            raw_prefix = f"/rows/{uncertainty.source_index}"
            contribution_variance = (
                _vector_cell(
                    uncertainty.contribution_variance,
                    uncertainty.source_file,
                    raw_prefix + "/contribution_variance",
                )
                if uncertainty.contribution_variance is not None
                else _provenance_cell(
                    "unavailable",
                    uncertainty.source_file,
                    raw_prefix + "/contribution_variance",
                )
            )
            gradient_variance = (
                _vector_cell(
                    uncertainty.gradient_variance,
                    uncertainty.source_file,
                    raw_prefix + "/gradient_variance",
                )
                if uncertainty.gradient_variance is not None
                else _provenance_cell(
                    "unavailable",
                    uncertainty.source_file,
                    raw_prefix + "/gradient_variance",
                )
            )
            ci_low = _vector_cell(
                uncertainty.ci_low or (),
                uncertainty.source_file,
                raw_prefix + "/ci_low",
            )
            ci_high = _vector_cell(
                uncertainty.ci_high or (),
                uncertainty.source_file,
                raw_prefix + "/ci_high",
            )
        rows.append(
            "<tr>"
            f"{_provenance_cell(row.scenario, SUMMARY_FILE, prefix + '/scenario')}"
            f"{_provenance_cell(row.target, SUMMARY_FILE, prefix + '/target')}"
            f"{_provenance_cell(model.summary.method_labels[row.method], SUMMARY_FILE, f'/method_labels/{_json_pointer_part(row.method)}')}"
            f"{_numeric_cell(row.samples, SUMMARY_FILE, prefix + '/samples')}"
            f"{_vector_cell(row.mean_gradient, SUMMARY_FILE, prefix + '/mean_gradient')}"
            f"{_vector_cell(row.empirical_bias, SUMMARY_FILE, prefix + '/empirical_bias')}"
            f"{_vector_cell(row.empirical_variance, SUMMARY_FILE, prefix + '/empirical_variance')}"
            f"{_vector_cell(row.mean_squared_error, SUMMARY_FILE, prefix + '/mean_squared_error')}"
            f"{_numeric_cell(row.relative_error, SUMMARY_FILE, prefix + '/relative_error')}"
            f"{_numeric_cell(row.cosine_similarity, SUMMARY_FILE, prefix + '/cosine_similarity')}"
            f"{_numeric_cell(row.sign_agreement, SUMMARY_FILE, prefix + '/sign_agreement')}"
            f"{contribution_variance}{gradient_variance}{ci_low}{ci_high}"
            "</tr>"
        )
    return _table(
        "Reference-relative gradient summaries",
        (
            "Scenario",
            "Target",
            "Method",
            "Samples",
            "Mean gradient",
            "Empirical bias",
            "Empirical variance",
            "Mean squared error",
            "Relative error",
            "Cosine",
            "Sign agreement",
            "Contribution variance",
            "Variance of mean",
            "CI low",
            "CI high",
        ),
        rows,
    )


def _optimization_table(model: ReportModel) -> str:
    rows = []
    selected = sorted(
        model.summary.optimization_summaries, key=lambda row: (row.scenario, row.method)
    )
    for row in selected:
        prefix = f"/optimization_summaries/{row.index}"
        rows.append(
            "<tr>"
            f"{_provenance_cell(row.scenario, SUMMARY_FILE, prefix + '/scenario')}"
            f"{_provenance_cell(model.summary.method_labels[row.method], SUMMARY_FILE, f'/method_labels/{_json_pointer_part(row.method)}')}"
            f"{_numeric_cell(row.final_hard_loss_mean, SUMMARY_FILE, prefix + '/final_hard_loss_mean')}"
            f"{_numeric_cell(row.final_hard_loss_ci_low, SUMMARY_FILE, prefix + '/final_hard_loss_ci_low')}"
            f"{_numeric_cell(row.final_hard_loss_ci_high, SUMMARY_FILE, prefix + '/final_hard_loss_ci_high')}"
            f"{_numeric_cell(row.success_rate, SUMMARY_FILE, prefix + '/success_rate')}"
            f"{_numeric_cell(row.held_out_loss_mean, SUMMARY_FILE, prefix + '/held_out_loss_mean')}"
            "</tr>"
        )
    return _table(
        "Multi-seed optimization summaries",
        (
            "Scenario",
            "Method",
            "Final hard loss",
            "CI low",
            "CI high",
            "Success rate",
            "Held-out loss",
        ),
        rows,
    )


def _performance_table(model: ReportModel) -> str:
    rows = []
    raw_by_id = {row.row_id: row for row in model.performance_rows}
    selected = sorted(
        model.summary.performance_summaries, key=lambda row: (row.scenario, row.method)
    )
    for row in selected:
        prefix = f"/performance_summaries/{row.index}"
        raw = raw_by_id[row.source_row_ids[0]]
        rows.append(
            "<tr>"
            f"{_provenance_cell(row.scenario, SUMMARY_FILE, prefix + '/scenario')}"
            f"{_provenance_cell(model.summary.method_labels[row.method], SUMMARY_FILE, f'/method_labels/{_json_pointer_part(row.method)}')}"
            f"{_provenance_cell(raw.device, raw.source_file, f'/rows/{raw.source_index}/device')}"
            f"{_numeric_cell(row.cold_compile_time, SUMMARY_FILE, prefix + '/cold_compile_time')}"
            f"{_numeric_cell(row.warm_median, SUMMARY_FILE, prefix + '/warm_median')}"
            f"{_numeric_cell(row.warm_iqr, SUMMARY_FILE, prefix + '/warm_iqr')}"
            f"{_numeric_cell(row.forward_executions, SUMMARY_FILE, prefix + '/forward_executions')}"
            f"{_numeric_cell(row.backward_executions, SUMMARY_FILE, prefix + '/backward_executions')}"
            f"{_optional_numeric_cell(row.tracemalloc_peak, SUMMARY_FILE, prefix + '/tracemalloc_peak')}"
            f"{_optional_numeric_cell(row.rss_delta, SUMMARY_FILE, prefix + '/rss_delta')}"
            f"{_optional_numeric_cell(row.warp_allocation_peak, SUMMARY_FILE, prefix + '/warp_allocation_peak')}"
            f"{_optional_numeric_cell(row.device_free_memory_delta, SUMMARY_FILE, prefix + '/device_free_memory_delta')}"
            "</tr>"
        )
    return _table(
        "Cold and warmed method cost",
        (
            "Scenario",
            "Method",
            "Device",
            "Cold compile",
            "Warm median",
            "Warm IQR",
            "Forwards",
            "Backwards",
            "Tracemalloc peak",
            "RSS delta",
            "Warp allocation peak",
            "Device free-memory delta",
        ),
        rows,
    )


def _validity_table(model: ReportModel) -> str:
    rows = []
    for row in sorted(model.summary.scenario_validity, key=lambda item: item.scenario):
        for metric, value in sorted(row.metrics.items()):
            rows.append(
                "<tr>"
                f"{_provenance_cell(row.scenario, SUMMARY_FILE, f'/scenario_validity/{row.index}/scenario')}"
                f"{_value_cell(row.accepted, SUMMARY_FILE, f'/scenario_validity/{row.index}/accepted')}"
                f"{_text_cell(metric)}"
                f"{_value_cell(value, SUMMARY_FILE, f'/scenario_validity/{row.index}/metrics/{_json_pointer_part(metric)}')}"
                "</tr>"
            )
    return _table(
        "Scenario validity gates", ("Scenario", "Status", "Diagnostic", "Value"), rows
    )


def _reference_table(model: ReportModel) -> str:
    rows = []
    for row in model.reference_rows:
        prefix = f"/rows/{row.source_index}/counts"
        samples = _integer(
            row.counts.get("samples"), name=f"{row.cell_id}.counts.samples"
        )
        replicates = _integer(
            row.counts.get("replicates"), name=f"{row.cell_id}.counts.replicates"
        )
        rows.append(
            "<tr>"
            f"{_provenance_cell(row.cell_id, row.source_file, f'/rows/{row.source_index}/cell_id')}"
            f"{_numeric_cell(samples, row.source_file, prefix + '/samples')}"
            f"{_numeric_cell(replicates, row.source_file, prefix + '/replicates')}"
            f"{_vector_cell(row.reference_gradient, row.source_file, f'/rows/{row.source_index}/reference_gradient')}"
            "</tr>"
        )
    return _table(
        "Accepted reference gradients",
        ("Cell", "Samples", "Replicates", "Reference gradient"),
        rows,
    )


def _residual_counts_table(model: ReportModel) -> str:
    rows = []
    selected = [
        row
        for row in model.gradient_rows
        if row.method == "residual_control_variate"
        and row.hard_forward_executions is not None
        and row.soft_forward_executions is not None
    ]
    if not selected:
        raise BuildError(
            "residual control-variate rows must report hard and soft execution counts"
        )
    for row in selected:
        prefix = f"/rows/{row.source_index}"
        rows.append(
            "<tr>"
            f"{_provenance_cell(row.scenario, row.source_file, f'/rows/{row.source_index}/scenario')}"
            f"{_numeric_cell(row.samples, row.source_file, prefix + '/samples')}"
            f"{_numeric_cell(row.hard_forward_executions, row.source_file, prefix + '/hard_forward_executions')}"
            f"{_numeric_cell(row.soft_forward_executions, row.source_file, prefix + '/soft_forward_executions')}"
            f"{_numeric_cell(row.backward_executions, row.source_file, prefix + '/backward_executions')}"
            "</tr>"
        )
    return _table(
        "Residual control-variate execution accounting",
        ("Scenario", "Samples", "Hard forwards", "Soft forwards", "Backwards"),
        rows,
    )


def _figure_markup(model: ReportModel, token: str) -> str:
    figures = []
    for asset, source_file, title, caption in FIGURE_SPECS[token]:
        records = _select_figure_records(model, asset, source_file)
        source_keys = [
            _source_fragment(f"/rows/{record.source_index}") for record in records
        ]
        plot_ids = [record.plot_id for record in records]
        provenance = (
            f' data-asset="{_escape(asset)}"'
            + _source_attributes(source_file, f"/rows/{records[0].source_index}")
            + f' data-plot-id="{_escape(records[0].plot_id)}"'
            + f' data-source-keys="{_escape(json.dumps(source_keys, separators=(",", ":")))}"'
            + f' data-plot-ids="{_escape(json.dumps(plot_ids, separators=(",", ":")))}"'
        )
        image_field = IMAGE_FIELDS.get(asset)
        if image_field is not None:
            provenance += f' data-source-value-key="{_source_fragment(f"/rows/{records[0].source_index}/{image_field}")}"'
        figures.append(
            f'<figure class="evidence-figure"{provenance}>'
            f'<img src="{_escape(asset)}" alt="{_escape(title)}" loading="lazy">'
            f"<figcaption><strong>{_escape(title)}.</strong> {_escape(caption)}</figcaption>"
            "</figure>"
        )
    return '<div class="figure-grid">' + "".join(figures) + "</div>"


def _hero_summary(model: ReportModel) -> str:
    source = model.manifest.source
    device = _string(source.get("device"), name="manifest.source.device")
    families = ", ".join(model.summary.scenario_families)
    accepted = all(value is True for value in model.manifest.accepted.values())
    status = (
        "passed its declared acceptance gates"
        if accepted
        else "contains rejected acceptance gates"
    )
    text = (
        "This experimental report is rendered only from the validated report-tier bundle. "
        f"The measured scenario families are {_escape(families)}; execution metadata identifies {_escape(device)}. "
        f"The imported evidence {_escape(status)}."
    )
    source_files = [SUMMARY_FILE, "data/manifest.json", "data/manifest.json"]
    source_keys = [
        _source_fragment("/scenario_families"),
        _source_fragment("/source/device"),
        _source_fragment("/accepted"),
    ]
    return (
        '<p class="lede"'
        f' data-source-files="{_escape(json.dumps(source_files, separators=(",", ":")))}"'
        f' data-source-keys="{_escape(json.dumps(source_keys, separators=(",", ":")))}">'
        f"{text}</p>"
    )


def _protocol_tables(model: ReportModel) -> str:
    config_rows = []
    for key, value in sorted(model.manifest.config.items()):
        config_rows.append(
            "<tr>"
            f"{_text_cell(key.replace('_', ' '))}"
            f"{_value_cell(value, 'data/manifest.json', f'/config/{_json_pointer_part(key)}')}"
            "</tr>"
        )
    source = model.manifest.source
    source_rows = [
        "<tr>"
        f"{_text_cell('device')}"
        f"{_value_cell(source.get('device'), 'data/manifest.json', '/source/device')}"
        "</tr>",
        "<tr>"
        f"{_text_cell('CPU threads')}"
        f"{_value_cell(source.get('cpu_threads'), 'data/manifest.json', '/source/cpu_threads')}"
        "</tr>",
    ]
    seeds = _mapping(source.get("seeds"), name="manifest.source.seeds")
    for key, value in sorted(seeds.items()):
        source_rows.append(
            "<tr>"
            f"{_text_cell('seed ' + key.replace('_', ' '))}"
            f"{_value_cell(value, 'data/manifest.json', f'/source/seeds/{_json_pointer_part(key)}')}"
            "</tr>"
        )
    return _table(
        "Manifest budgets and schedules",
        ("Configuration", "Declared value"),
        config_rows,
    ) + _table(
        "Execution device and seed domains",
        ("Source field", "Declared value"),
        source_rows,
    )


def _reproducibility(model: ReportModel) -> str:
    source = model.manifest.source
    commit = _string(source.get("commit"), name="manifest.source.commit")
    command = [
        _string(part, name="manifest.source.command item")
        for part in _sequence(source.get("command"), name="manifest.source.command")
    ]
    fields = (
        ("Manifest schema", model.manifest.schema_version, "/schema_version"),
        ("Summary schema", model.summary.schema_version, None),
        ("Report tier", model.manifest.tier, "/tier"),
        ("Warp commit", commit, "/source/commit"),
        ("Producer command", command, "/source/command"),
        (
            "Python",
            _string(source.get("python"), name="manifest.source.python"),
            "/source/python",
        ),
        (
            "Warp",
            _string(source.get("warp"), name="manifest.source.warp"),
            "/source/warp",
        ),
        (
            "Platform",
            _string(source.get("platform"), name="manifest.source.platform"),
            "/source/platform",
        ),
        (
            "Compiler",
            _string(source.get("compiler"), name="manifest.source.compiler"),
            "/source/compiler",
        ),
        (
            "CPU model",
            _string(source.get("cpu_model"), name="manifest.source.cpu_model"),
            "/source/cpu_model",
        ),
        (
            "CPU threads",
            _integer(source.get("cpu_threads"), name="manifest.source.cpu_threads"),
            "/source/cpu_threads",
        ),
        (
            "Device",
            _string(source.get("device"), name="manifest.source.device"),
            "/source/device",
        ),
    )
    metadata_rows = []
    for label, value, pointer in fields:
        source_file = (
            SUMMARY_FILE if label == "Summary schema" else "data/manifest.json"
        )
        source_key = "/schema_version" if pointer is None else pointer
        metadata_rows.append(
            f"<tr>{_text_cell(label)}{_value_cell(value, source_file, source_key)}</tr>"
        )
    digest_rows = []
    for relative, descriptor_value in sorted(model.manifest.files.items()):
        descriptor = _mapping(descriptor_value, name=f"manifest.files.{relative}")
        prefix = f"/files/{_json_pointer_part(relative)}"
        digest_rows.append(
            "<tr>"
            f"{_text_cell(relative)}"
            f"{_numeric_cell(_integer(descriptor.get('bytes'), name=f'{relative}.bytes'), 'data/manifest.json', prefix + '/bytes')}"
            f"{_provenance_cell(_string(descriptor.get('sha256'), name=f'{relative}.sha256'), 'data/manifest.json', prefix + '/sha256')}"
            "</tr>"
        )
    return (
        _table("Build and source metadata", ("Field", "Value"), metadata_rows)
        + _table("Artifact digests", ("Artifact", "Bytes", "SHA-256"), digest_rows)
        + "<p>The manifest records exact byte counts and SHA-256 digests for every JSON and image artifact. "
        "Reproduction starts from the producer command above, then validates the imported bundle before rendering. "
        '<a href="https://github.com/NVIDIA/warp">Warp source repository</a>.</p>'
    )


def _token_values(model: ReportModel) -> dict[str, str]:
    commit = _string(model.manifest.source.get("commit"), name="manifest.source.commit")
    values = {
        "SOURCE_COMMIT_ATTR": _escape(commit),
        "HERO_SUMMARY": _hero_summary(model),
        "HEADLINE_TABLE": _headline_table(model),
        "METHOD_TABLE": _method_table(model),
        "RESIDUAL_COUNTS_TABLE": _residual_counts_table(model),
        "REFERENCE_TABLE": _reference_table(model),
        "METHOD_SUMMARY_TABLE": _method_summary_table(model),
        "OPTIMIZATION_TABLE": _optimization_table(model),
        "PERFORMANCE_TABLE": _performance_table(model),
        "VALIDITY_TABLE": _validity_table(model),
        "PROTOCOL_TABLES": _protocol_tables(model),
        "REPRODUCIBILITY": _reproducibility(model),
    }
    for token in FIGURE_SPECS:
        values[token] = _figure_markup(model, token)
    return values


def _render_template(template: str, values: Mapping[str, str]) -> str:
    invalid_tokens = sorted(
        token
        for token in ANY_TOKEN_RE.findall(template)
        if TOKEN_RE.fullmatch(token) is None
    )
    tokens = TOKEN_RE.findall(template)
    token_set = set(tokens)
    value_set = set(values)
    unknown = sorted(token_set - value_set)
    missing = sorted(value_set - token_set)
    duplicates = sorted(token for token in token_set if tokens.count(token) != 1)
    if invalid_tokens or unknown or missing or duplicates:
        raise BuildError(
            "template token contract failed: "
            f"invalid={invalid_tokens}, unknown={unknown}, missing={missing}, "
            f"duplicate={duplicates}"
        )
    rendered = TOKEN_RE.sub(lambda match: values[match.group(1)], template)
    remaining = ANY_TOKEN_RE.findall(rendered)
    if remaining:
        raise BuildError(f"template has unreplaced tokens: {sorted(set(remaining))}")
    return rendered


def _validated_evidence_attributes(
    tag: str, attrs: list[tuple[str, str | None]]
) -> dict[str, str]:
    validated: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name, value in attrs:
        normalized = name.casefold()
        if normalized in seen:
            raise BuildError(f"duplicate HTML attribute {name!r} on <{tag}>")
        seen.add(normalized)
        validated.append((normalized, value or ""))
    return dict(validated)


class _EvidenceHTMLParser(HTMLParser):
    _VOID_ELEMENTS = frozenset(
        {
            "area",
            "base",
            "br",
            "col",
            "embed",
            "hr",
            "img",
            "input",
            "link",
            "meta",
            "param",
            "source",
            "track",
            "wbr",
        }
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.cells: list[tuple[dict[str, str], str]] = []
        self.figures: list[dict[str, str]] = []
        self.images: list[tuple[dict[str, str], int | None]] = []
        self.narratives: list[dict[str, str]] = []
        self._cell_attributes: dict[str, str] | None = None
        self._cell_text: list[str] = []
        self._figure_stack: list[int] = []
        self._open_tags: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = _validated_evidence_attributes(tag, attrs)
        if tag == "td":
            if self._cell_attributes is not None:
                raise BuildError("rendered report nests evidence table cells")
            self._cell_attributes = attributes
            self._cell_text = []
        elif tag == "figure":
            self.figures.append(attributes)
            self._figure_stack.append(len(self.figures) - 1)
        elif tag == "img":
            direct_figure = (
                self._figure_stack[-1]
                if self._figure_stack and self._open_tags[-1:] == ["figure"]
                else None
            )
            self.images.append((attributes, direct_figure))
        elif tag == "p" and "lede" in attributes.get("class", "").split():
            self.narratives.append(attributes)
        if tag not in self._VOID_ELEMENTS:
            self._open_tags.append(tag)

    def handle_data(self, data: str) -> None:
        if self._cell_attributes is not None:
            self._cell_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self._cell_attributes is not None:
            self.cells.append((self._cell_attributes, "".join(self._cell_text).strip()))
            self._cell_attributes = None
            self._cell_text = []
        if tag == "figure":
            if not self._figure_stack:
                raise BuildError("rendered report closes an unopened evidence figure")
            self._figure_stack.pop()
        if tag in self._open_tags:
            reverse_index = self._open_tags[::-1].index(tag)
            del self._open_tags[len(self._open_tags) - reverse_index - 1 :]


def _decode_source_key(value: Any, *, name: str) -> str:
    raw = _string(value, name=name)
    if not raw.startswith("#"):
        raise BuildError(f"{name} must be a fragment JSON pointer")
    pointer = unquote(raw[1:])
    if not pointer.startswith("/"):
        raise BuildError(f"{name} must resolve to an absolute JSON pointer")
    return pointer


def _resolve_pointer(value: Any, pointer: str, *, name: str) -> Any:
    current = value
    for raw_part in pointer[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            if not part.isdigit():
                raise BuildError(f"{name} has a non-index array pointer component")
            index = int(part)
            if index >= len(current):
                raise BuildError(f"{name} points outside an evidence array")
            current = current[index]
        elif isinstance(current, dict):
            if part not in current:
                raise BuildError(f"{name} points to a missing evidence key {part!r}")
            current = current[part]
        else:
            raise BuildError(f"{name} traverses a scalar evidence value")
    return current


def _resolve_evidence_source(
    model: ReportModel,
    source_file: Any,
    source_key: Any,
    cache: dict[str, Any],
    *,
    name: str,
) -> Any:
    relative = _string(source_file, name=f"{name}.source_file")
    allowed = {"data/manifest.json", *model.manifest.files}
    if relative not in allowed or not relative.endswith(".json"):
        raise BuildError(f"{name} cites undeclared JSON evidence {relative!r}")
    if relative not in cache:
        try:
            cache[relative] = json.loads(
                (model.root / relative).read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise BuildError(f"cannot resolve {name} evidence: {error}") from error
    pointer = _decode_source_key(source_key, name=f"{name}.source_key")
    return _resolve_pointer(cache[relative], pointer, name=name)


def _looks_like_measured_value(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    try:
        converted = float(stripped)
    except ValueError:
        converted = None
    if converted is not None and math.isfinite(converted):
        return True
    if stripped[0] not in "[{":
        return False
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return False
    return isinstance(value, (list, dict))


def _validate_evidence_provenance(page: str, model: ReportModel) -> None:
    parser = _EvidenceHTMLParser()
    try:
        parser.feed(page)
        parser.close()
    except BuildError:
        raise
    except Exception as error:
        raise BuildError(f"cannot parse evidence provenance: {error}") from error
    cache: dict[str, Any] = {}
    for index, (attributes, text) in enumerate(parser.cells):
        source_file = attributes.get("data-source-file")
        source_key = attributes.get("data-source-key")
        static_key = attributes.get("data-static-key")
        sourced = source_file is not None or source_key is not None
        static = static_key is not None
        if sourced == static or (
            sourced and (source_file is None or source_key is None)
        ):
            raise BuildError(
                f"evidence cell {index} requires exactly one source pointer or static label key"
            )
        if sourced:
            source_value = _resolve_evidence_source(
                model,
                source_file,
                source_key,
                cache,
                name=f"evidence cell {index}",
            )
            expected_text = _canonical_evidence_display(source_value)
            if text != expected_text:
                raise BuildError(
                    f"evidence cell {index} display does not match its source value"
                )
        elif not static_key or _looks_like_measured_value(text):
            raise BuildError(
                f"evidence cell {index} has an invalid static provenance label"
            )

    expected_sources: dict[str, str] = {}
    for specs in FIGURE_SPECS.values():
        for asset, source_file, _title, _caption in specs:
            _insert_unique_semantic_record(
                expected_sources,
                asset,
                source_file,
                description="fixed figure asset identity",
            )
    observed_assets = [attributes.get("data-asset") for attributes in parser.figures]
    if len(observed_assets) != len(set(observed_assets)) or set(observed_assets) != set(
        expected_sources
    ):
        raise BuildError(
            "rendered figures do not exactly cover the fixed asset mapping"
        )
    for attributes in parser.figures:
        asset = attributes["data-asset"]
        source_file = attributes.get("data-source-file")
        if source_file != expected_sources[asset]:
            raise BuildError(f"figure {asset!r} cites the wrong plot dataset")
        primary = _resolve_evidence_source(
            model,
            source_file,
            attributes.get("data-source-key"),
            cache,
            name=f"figure {asset}",
        )
        if not isinstance(primary, dict) or primary.get("plot_id") != attributes.get(
            "data-plot-id"
        ):
            raise BuildError(f"figure {asset!r} primary plot provenance mismatch")
        try:
            source_keys = json.loads(
                _string(
                    attributes.get("data-source-keys"),
                    name=f"figure {asset}.data-source-keys",
                )
            )
            plot_ids = json.loads(
                _string(
                    attributes.get("data-plot-ids"),
                    name=f"figure {asset}.data-plot-ids",
                )
            )
        except json.JSONDecodeError as error:
            raise BuildError(
                f"figure {asset!r} has invalid provenance arrays"
            ) from error
        if (
            not isinstance(source_keys, list)
            or not isinstance(plot_ids, list)
            or not source_keys
            or len(source_keys) != len(plot_ids)
        ):
            raise BuildError(f"figure {asset!r} has incomplete provenance arrays")
        for record_index, (source_key, plot_id) in enumerate(
            zip(source_keys, plot_ids, strict=True)
        ):
            record = _resolve_evidence_source(
                model,
                source_file,
                source_key,
                cache,
                name=f"figure {asset} record {record_index}",
            )
            if not isinstance(record, dict) or record.get("plot_id") != plot_id:
                raise BuildError(f"figure {asset!r} plot provenance mismatch")
        image_field = IMAGE_FIELDS.get(asset)
        value_key = attributes.get("data-source-value-key")
        if image_field is not None:
            image = _resolve_evidence_source(
                model,
                source_file,
                value_key,
                cache,
                name=f"figure {asset} image",
            )
            if not isinstance(image, list) or not image:
                raise BuildError(f"figure {asset!r} image provenance is empty")
        elif value_key is not None:
            raise BuildError(f"non-image figure {asset!r} has image provenance")

    image_sources = [attributes.get("src") for attributes, _parent in parser.images]
    if (
        len(image_sources) != len(expected_sources)
        or len(image_sources) != len(set(image_sources))
        or set(image_sources) != set(expected_sources)
    ):
        raise BuildError(
            "rendered evidence images do not exactly cover the fixed asset mapping"
        )
    figure_image_counts = [0] * len(parser.figures)
    for index, (attributes, parent) in enumerate(parser.images):
        if parent is None:
            raise BuildError(
                f"evidence image {index} is not a direct child of its bound figure"
            )
        figure_image_counts[parent] += 1
        source = attributes.get("src")
        if source != parser.figures[parent].get("data-asset"):
            raise BuildError(
                f"evidence image {source!r} does not match its bound figure asset"
            )
    if any(count != 1 for count in figure_image_counts):
        raise BuildError("each bound evidence figure must contain exactly one image")

    if len(parser.narratives) != 1:
        raise BuildError(
            "rendered report requires exactly one data-derived hero narrative"
        )
    narrative = parser.narratives[0]
    try:
        source_files = json.loads(
            _string(
                narrative.get("data-source-files"),
                name="hero narrative data-source-files",
            )
        )
        source_keys = json.loads(
            _string(
                narrative.get("data-source-keys"),
                name="hero narrative data-source-keys",
            )
        )
    except json.JSONDecodeError as error:
        raise BuildError("hero narrative has invalid provenance arrays") from error
    if (
        not isinstance(source_files, list)
        or not isinstance(source_keys, list)
        or not source_files
        or len(source_files) != len(source_keys)
    ):
        raise BuildError("hero narrative has incomplete provenance arrays")
    for index, (source_file, source_key) in enumerate(
        zip(source_files, source_keys, strict=True)
    ):
        _resolve_evidence_source(
            model,
            source_file,
            source_key,
            cache,
            name=f"hero narrative source {index}",
        )


def _validate_rendered_contract(page: str, model: ReportModel) -> None:
    _validate_evidence_provenance(page, model)
    parser = _ReportHTMLParser()
    try:
        parser.feed(page)
        parser.close()
    except Exception as error:
        raise BuildError(f"rendered report is invalid HTML: {error}") from error
    try:
        for index, text in enumerate(parser.text_fragments):
            _assert_no_local_path(text, location=f"rendered text[{index}]")
        for index, value in enumerate(parser.attribute_values):
            _assert_no_local_path(value, location=f"rendered attribute[{index}]")
    except ValidationError as error:
        raise BuildError(f"rendered report is unsafe: {error}") from error
    if parser.duplicate_ids:
        raise BuildError(
            f"rendered report has duplicate IDs: {sorted(parser.duplicate_ids)}"
        )
    if parser.active_content_violations:
        raise BuildError(
            "rendered report has unsafe active content: "
            f"{parser.active_content_violations}"
        )
    for section_id in REQUIRED_SECTION_IDS:
        if parser.ids.get(section_id) != ("section", False):
            raise BuildError(f"rendered template is missing section {section_id!r}")
    for method in REQUIRED_METHOD_IDS:
        if not " ".join(parser.method_text.get(method, ())).strip():
            raise BuildError(f"rendered template is missing method label {method!r}")
    commit = _string(model.manifest.source.get("commit"), name="manifest.source.commit")
    if parser.source_commit != commit:
        raise BuildError("rendered report source commit does not match manifest")
    declared_assets = set(model.manifest.files)
    for tag, attribute, raw_reference in parser.references:
        split = urlsplit(raw_reference)
        if split.scheme:
            if split.scheme in {"http", "https"} and tag == "a" and attribute == "href":
                continue
            raise BuildError(f"rendered report has unsafe reference {raw_reference!r}")
        if split.netloc:
            raise BuildError(
                f"rendered report has protocol-relative reference {raw_reference!r}"
            )
        if not split.path:
            continue
        pure = PurePosixPath(split.path)
        if pure.is_absolute() or ".." in pure.parts or "\\" in split.path:
            raise BuildError(
                f"rendered report has unsafe local reference {raw_reference!r}"
            )
        if pure.as_posix() not in declared_assets:
            raise BuildError(
                f"rendered report references undeclared asset {raw_reference!r}"
            )
        if tag not in {"a", "img", "source"}:
            raise BuildError(
                f"rendered report uses an unsafe local reference element {tag!r}"
            )


def render_report(*, root: Path, template_path: Path | None = None) -> bytes:
    root = Path(root)
    template_path = (
        root / "report_template.html" if template_path is None else Path(template_path)
    )
    model = load_report(Path(root))
    try:
        template = Path(template_path).read_text(encoding="utf-8")
    except (OSError, UnicodeError, ValueError) as error:
        raise BuildError(
            f"cannot read UTF-8 report template {template_path}: {error}"
        ) from error
    rendered = _render_template(template, _token_values(model))
    _validate_rendered_contract(rendered, model)
    return rendered.encode("utf-8")


def _resolve_filesystem_path(path: Path, *, name: str, strict: bool) -> Path:
    try:
        return Path(path).resolve(strict=strict)
    except (OSError, RuntimeError, ValueError) as error:
        raise BuildError(
            f"cannot resolve {name} filesystem path {path}: {error}"
        ) from error


def _guard_output_location(
    *, resolved_root: Path, resolved_output: Path, resolved_template: Path
) -> None:
    if resolved_output == resolved_template:
        raise BuildError("report output cannot overwrite its template")
    try:
        relative_output = resolved_output.relative_to(resolved_root).as_posix()
    except ValueError:
        relative_output = None
    if relative_output == "data/manifest.json" or (
        relative_output is not None
        and (
            relative_output.startswith("data/") or relative_output.startswith("assets/")
        )
    ):
        raise BuildError("report output cannot overwrite a publication bundle artifact")


@dataclass(slots=True)
class _OutputDestination:
    output: Path
    resolved_output: Path
    parent: Path
    name: str
    parent_fd: int
    parent_device: int
    parent_inode: int
    closed: bool = False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        descriptor = self.parent_fd
        self.parent_fd = -1
        try:
            os.close(descriptor)
        except OSError:
            pass

    def __enter__(self) -> _OutputDestination:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def _open_output_destination(
    *,
    root: Path,
    output: Path,
    template_path: Path,
    create_parent: bool = True,
) -> _OutputDestination:
    root = Path(root)
    output = Path(output)
    template_path = Path(template_path)
    if not output.name or output.name in {".", ".."}:
        raise BuildError(f"report output requires a regular filename: {output}")

    resolved_root = _resolve_filesystem_path(root, name="report root", strict=True)
    resolved_template = _resolve_filesystem_path(
        template_path, name="report template", strict=True
    )
    resolved_output = _resolve_filesystem_path(
        output, name="report output", strict=False
    )
    _guard_output_location(
        resolved_root=resolved_root,
        resolved_output=resolved_output,
        resolved_template=resolved_template,
    )

    parent = output.parent
    if create_parent:
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise BuildError(
                f"cannot create report output directory {parent}: {error}"
            ) from error

    required_flags = ("O_DIRECTORY", "O_NOFOLLOW")
    missing_flags = [name for name in required_flags if not hasattr(os, name)]
    if missing_flags:
        raise BuildError(
            "secure report output requires no-follow directory descriptors: "
            f"missing {', '.join(missing_flags)}"
        )
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor: int | None = None
    try:
        descriptor = os.open(parent, flags)
        descriptor_stat = os.fstat(descriptor)
        path_stat = os.stat(parent, follow_symlinks=False)
        if not stat.S_ISDIR(path_stat.st_mode) or (
            path_stat.st_dev,
            path_stat.st_ino,
        ) != (descriptor_stat.st_dev, descriptor_stat.st_ino):
            raise BuildError(
                f"report output directory changed while it was being secured: {parent}"
            )
        current_output = _resolve_filesystem_path(
            output, name="report output", strict=False
        )
        _guard_output_location(
            resolved_root=resolved_root,
            resolved_output=current_output,
            resolved_template=resolved_template,
        )
        if current_output != resolved_output:
            raise BuildError(
                f"report output changed while it was being secured: {output}"
            )
        destination = _OutputDestination(
            output=output,
            resolved_output=resolved_output,
            parent=parent,
            name=output.name,
            parent_fd=descriptor,
            parent_device=descriptor_stat.st_dev,
            parent_inode=descriptor_stat.st_ino,
        )
        descriptor = None
        return destination
    except BuildError:
        raise
    except OSError as error:
        raise BuildError(
            f"cannot secure report output directory {parent}: {error}"
        ) from error
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _validate_output_destination(destination: _OutputDestination) -> None:
    if destination.closed or destination.parent_fd < 0:
        raise BuildError("report output directory descriptor is closed")
    try:
        descriptor_stat = os.fstat(destination.parent_fd)
        path_stat = os.stat(destination.parent, follow_symlinks=False)
    except OSError as error:
        raise BuildError(
            f"report output directory changed before write: {destination.parent}: {error}"
        ) from error
    expected_identity = (destination.parent_device, destination.parent_inode)
    if (
        not stat.S_ISDIR(path_stat.st_mode)
        or (descriptor_stat.st_dev, descriptor_stat.st_ino) != expected_identity
        or (path_stat.st_dev, path_stat.st_ino) != expected_identity
    ):
        raise BuildError(
            f"report output directory changed before write: {destination.parent}"
        )
    current_output = _resolve_filesystem_path(
        destination.output, name="report output", strict=False
    )
    if current_output != destination.resolved_output:
        raise BuildError(f"report output changed before write: {destination.output}")


def _read_output(destination: _OutputDestination) -> bytes:
    _validate_output_destination(destination)
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor: int | None = None
    try:
        descriptor = os.open(destination.name, flags, dir_fd=destination.parent_fd)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    except OSError as error:
        raise BuildError(
            f"generated page is missing or unreadable: {destination.output}: {error}"
        ) from error
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _atomic_write(destination: _OutputDestination, data: bytes) -> None:
    temporary_name: str | None = None
    descriptor: int | None = None
    try:
        _validate_output_destination(destination)
        temporary_name = f".{destination.name}.{secrets.token_hex(16)}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        descriptor = os.open(
            temporary_name,
            flags,
            0o600,
            dir_fd=destination.parent_fd,
        )
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("atomic report write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        _validate_output_destination(destination)
        os.replace(
            temporary_name,
            destination.name,
            src_dir_fd=destination.parent_fd,
            dst_dir_fd=destination.parent_fd,
        )
        os.fsync(destination.parent_fd)
    except BuildError:
        raise
    except OSError as error:
        raise BuildError(
            f"cannot write report output {destination.output}: {error}"
        ) from error
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=destination.parent_fd)
            except OSError:
                pass


def build_report(
    *,
    root: Path,
    output: Path,
    template_path: Path | None = None,
    check: bool = False,
) -> bytes:
    root = Path(root)
    output = Path(output)
    template_path = (
        root / "report_template.html" if template_path is None else Path(template_path)
    )
    destination = _open_output_destination(
        root=root,
        output=output,
        template_path=template_path,
        create_parent=not check,
    )
    try:
        rendered = render_report(root=root, template_path=template_path)
        if check:
            current = _read_output(destination)
            if current != rendered:
                raise BuildError(f"generated page differs from {output}")
        else:
            _atomic_write(destination, rendered)
        return rendered
    finally:
        destination.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the deterministic Warp branch-smoothing report"
    )
    parser.add_argument("--root", type=Path, default=REPORT_ROOT)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    template = (
        arguments.template
        if arguments.template is not None
        else arguments.root / "report_template.html"
    )
    output = (
        arguments.output
        if arguments.output is not None
        else arguments.root / "index.html"
    )
    try:
        rendered = build_report(
            root=arguments.root,
            output=output,
            template_path=template,
            check=arguments.check,
        )
    except BuildError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if arguments.check:
        print(f"PASS: report is current ({len(rendered)} bytes)")
    else:
        print(f"WROTE: {output} ({len(rendered)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
