#!/usr/bin/env python3
"""Microbenchmarks for Newton mimic-joint Model parameterization options.

This is intentionally dependency-light. It measures the metadata traversal cost of
several representational choices, not full solver dynamics. The existing report
asset_benchmark_*.json files cover real Newton FK and solver timings.
"""
from __future__ import annotations

import json
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    name: str
    joints: int
    dofs: int
    mimics: int
    repeats: int = 9
    loops: int = 2000


def median_us(samples: list[float]) -> float:
    return statistics.median(samples) * 1e6


def bench_case(case: Case) -> dict:
    # Deterministic synthetic data. Leaders occupy the first dofs; followers are
    # distributed across the joint array to make branch prediction realistic.
    joints, dofs, mimics = case.joints, max(case.dofs, 1), case.mimics
    q = [0.001 * ((i % 97) - 48) for i in range(dofs)]
    qd = [0.002 * ((i % 89) - 44) for i in range(dofs)]
    joint_q_start = [min(i, dofs - 1) for i in range(joints)]
    joint_type = [1] * joints
    mimic_joint_indices = []
    for k in range(mimics):
        j = (joints - mimics) + k
        if 0 <= j < joints:
            joint_type[j] = 99
            mimic_joint_indices.append(j)
    # Current branch: full-frequency arrays on joint_count.
    dense_leader = [-1] * joints
    dense_offset = [0.0] * joints
    dense_mult = [1.0] * joints
    for n, j in enumerate(mimic_joint_indices):
        dense_leader[j] = n % dofs
        dense_offset[j] = 0.01 * (n % 7)
        dense_mult[j] = -1.0 if n % 2 else 1.0
    # Compact joint table: mimic-only tuples, with follower joint index retained.
    compact = [(j, n % dofs, 0.01 * (n % 7), -1.0 if n % 2 else 1.0) for n, j in enumerate(mimic_joint_indices)]
    # Hypothetical no-new-array overload: joint_parent stores leader joint index,
    # joint_target_ke stores offset, and joint_target_kd stores multiplier at joint
    # frequency. This is fast to look up but semantically invalid for current Newton:
    # joint_parent is a body index, and target arrays are per DOF, not per joint.
    overloaded_parent = dense_leader
    overloaded_target_ke = dense_offset
    overloaded_target_kd = dense_mult
    # Legacy separate concept: same scalar math, plus enabled flag and follower lookup.
    constraints = [(j, n % dofs, 0.01 * (n % 7), -1.0 if n % 2 else 1.0, True) for n, j in enumerate(mimic_joint_indices)]

    def fk_baseline() -> float:
        acc = 0.0
        for _ in range(case.loops):
            for j in range(joints):
                s = joint_q_start[j]
                acc += q[s] * 1.000001 + qd[s] * 0.5
        return acc

    def dense_joint_arrays() -> float:
        acc = 0.0
        for _ in range(case.loops):
            for j in range(joints):
                if joint_type[j] == 99:
                    leader = dense_leader[j]
                    acc += dense_offset[j] + dense_mult[j] * q[leader] + dense_mult[j] * qd[leader]
                else:
                    s = joint_q_start[j]
                    acc += q[s] * 1.000001 + qd[s] * 0.5
        return acc

    def compact_mimic_table() -> float:
        acc = 0.0
        for _ in range(case.loops):
            # normal FK remains a straight joint loop; derived followers are a compact side pass
            for j in range(joints):
                s = joint_q_start[j]
                acc += q[s] * 1.000001 + qd[s] * 0.5
            for _j, leader, offset, mult in compact:
                acc += offset + mult * q[leader] + mult * qd[leader]
        return acc

    def overloaded_existing_attributes() -> float:
        acc = 0.0
        for _ in range(case.loops):
            for j in range(joints):
                if joint_type[j] == 99:
                    leader = overloaded_parent[j]
                    acc += overloaded_target_ke[j] + overloaded_target_kd[j] * q[leader] + overloaded_target_kd[j] * qd[leader]
                else:
                    s = joint_q_start[j]
                    acc += q[s] * 1.000001 + qd[s] * 0.5
        return acc

    def legacy_constraint_pass() -> float:
        acc = 0.0
        for _ in range(case.loops):
            for j in range(joints):
                s = joint_q_start[j]
                acc += q[s] * 1.000001 + qd[s] * 0.5
            for _follower, leader, offset, mult, enabled in constraints:
                if enabled:
                    acc += offset + mult * q[leader] + mult * qd[leader]
        return acc

    benches = {
        "baseline_no_mimic_us": fk_baseline,
        "joint_dense_branch_arrays_us": dense_joint_arrays,
        "overloaded_existing_attributes_us": overloaded_existing_attributes,
        "joint_compact_relation_table_us": compact_mimic_table,
        "separate_constraint_pass_us": legacy_constraint_pass,
    }
    results = {}
    for name, fn in benches.items():
        samples = []
        guard = None
        for _ in range(case.repeats):
            t0 = time.perf_counter()
            guard = fn()
            samples.append(time.perf_counter() - t0)
        results[name] = {
            "median_us": median_us(samples),
            "mean_us": statistics.mean(samples) * 1e6,
            "min_us": min(samples) * 1e6,
            "max_us": max(samples) * 1e6,
            "guard": guard,
        }
    # Approximate fixed metadata storage. Does not include labels/list overhead.
    # dense: leader int32 + coef[2] float32 + type int32 + axis vec3 float32 = 28 bytes/joint
    # compact relation: follower int32 + leader int32 + offset/mult float32 + axis vec3 + type int32 = 28 bytes/mimic
    # legacy: follower int32 + leader int32 + coef0/coef1 float32 + enabled bool + world int32 ~= 21 bytes/mimic, padded to 24
    # overloaded existing attrs: zero incremental arrays only if joint_parent and
    # joint_target_ke/kd are redefined/overloaded. If current semantics must be
    # preserved, this hidden cost becomes a replacement parent-body array and
    # joint-frequency target metadata, so it is not actually free.
    results["memory_bytes"] = {
        "joint_dense_branch_arrays": joints * 28,
        "overloaded_existing_attributes_incremental": 0,
        "joint_compact_relation_table": mimics * 28,
        "separate_constraint_pass": mimics * 24,
        "dense_minus_compact_bytes": joints * 28 - mimics * 28,
    }
    return results


def main() -> None:
    cases = [
        Case("synthetic_robotiq_like", joints=7, dofs=1, mimics=5, loops=20000),
        Case("robotiq_imported_like", joints=15, dofs=6, mimics=8, loops=10000),
        Case("hand_24dof_sparse_mimic", joints=32, dofs=24, mimics=4, loops=8000),
        Case("humanoid_43dof_sparse_mimic", joints=64, dofs=43, mimics=6, loops=6000),
        Case("large_batched_10pct_mimic", joints=1024, dofs=768, mimics=102, loops=600),
    ]
    out = {
        "benchmark": "mimic parameterization metadata traversal microbenchmark",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cases": {case.name: bench_case(case) for case in cases},
    }
    path = Path(__file__).with_name("parameterization_benchmark.json")
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
