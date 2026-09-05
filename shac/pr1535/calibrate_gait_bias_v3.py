#!/usr/bin/env python3
"""Apply an explicit actor-output bias to a v3 gait checkpoint.

This is a small, auditable calibration step for a deterministic policy.  The
bias is added to the final linear layer *before* tanh, so the saved result is
still an ordinary PPOActor and needs no special inference wrapper.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


SUPPORTED_FORMATS = {
    "mjwarp-pr1535-full-gait-v3",
    "mjwarp-pr1535-shac-gait-v3",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def parse_bias(value: str) -> tuple[int, float]:
    try:
        index_text, delta_text = value.split("=", maxsplit=1)
        return int(index_text), float(delta_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("bias must be INDEX=DELTA") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy", choices=("best", "final"), default="best")
    parser.add_argument("--bias", type=parse_bias, action="append", required=True)
    parser.add_argument("--previous-action-alpha", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix != ".pt":
        parser.error("--output must end in .pt")
    if not 0.0 < args.previous_action_alpha <= 1.0:
        parser.error("--previous-action-alpha must be in (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    source_path = args.checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    if source.get("format") not in SUPPORTED_FORMATS:
        raise ValueError("unsupported gait checkpoint format")

    actor_key = f"{args.policy}_actor"
    critic_key = f"{args.policy}_critic"
    normalizer_key = f"{args.policy}_normalizer"
    actor = copy.deepcopy(source[actor_key])
    output_bias = actor["policy.output.bias"]
    applied: dict[str, float] = {}
    for index, delta in args.bias:
        if index < 0 or index >= output_bias.numel():
            raise IndexError(f"actor output index {index} is out of range")
        output_bias[index] += delta
        applied[str(index)] = applied.get(str(index), 0.0) + delta

    result = copy.deepcopy(source)
    for prefix in ("initial", "best", "final"):
        result[f"{prefix}_actor"] = copy.deepcopy(actor)
        result[f"{prefix}_critic"] = copy.deepcopy(source[critic_key])
        result[f"{prefix}_normalizer"] = copy.deepcopy(source[normalizer_key])
    result["best_update"] = 0
    result["calibration"] = {
        "method": "pre-tanh final-layer bias",
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": sha256_file(source_path),
        "source_policy": args.policy,
        "bias_by_action_index": applied,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    result["control_conditioning"] = {
        "previous_action_low_pass_alpha": args.previous_action_alpha,
        "method": "causal blend of desired and previously applied control",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    manifest_path = output_path.with_suffix(".json")
    write_json(
        manifest_path,
        {
            "schema": "mjwarp-pr1535-gait-bias-calibration-v3",
            "task": source["task"],
            "output_checkpoint": str(output_path),
            "output_checkpoint_sha256": sha256_file(output_path),
            "control_conditioning": result["control_conditioning"],
            **result["calibration"],
        },
    )
    print(f"Wrote {output_path} and {manifest_path}")


if __name__ == "__main__":
    main()
