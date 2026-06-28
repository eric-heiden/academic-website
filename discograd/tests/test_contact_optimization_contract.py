from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from discograd.tests.test_publication import (
    SCHEMA_VERSION,
    _refresh_manifest,
    make_valid_fixture,
    write_fixture_json,
)
from discograd.validate_report import ValidationError, validate_publication


CONTACT_METHODS = (
    "soft_ad",
    "straight_through_ad",
    "residual_control_variate",
)
CONTACT_EVENT_LABELS = (
    "pair_01",
    "pair_02",
    "pair_12",
    "floor_0",
    "floor_1",
    "floor_2",
    "ramp_0",
    "ramp_1",
    "ramp_2",
)
CONTACT_STEPS = 180
CONTACT_SWEEPS = 4
OPTIMIZATION_STEPS = 64
OPTIMIZATION_SCHEDULES = 16
CONTACT_IMPULSE_THRESHOLD = 1.0e-8


def _physical_validity() -> dict[str, object]:
    positive = [[[] for _ in range(CONTACT_SWEEPS)] for _ in range(CONTACT_STEPS)]
    positive[0][0] = ["pair_01", "floor_0"]
    positive[1][0] = ["pair_12", "ramp_0"]
    corrections = copy.deepcopy(positive)
    checks = {
        "pair_01_contact": True,
        "pair_12_contact": True,
        "ordered_pair_01_then_pair_12": True,
        "floor_contact": True,
        "ramp_contact": True,
        "stick_mode": True,
        "slide_mode": True,
        "no_zero_limit_slide": True,
        "penetration_bounded": True,
        "contact_energy_bounded": True,
        "pair_momentum_conserved": True,
        "pair_angular_momentum_conserved": True,
        "meaningful_positive_impulse": True,
    }
    return {
        "valid": True,
        "checks": checks,
        "pair_contact_counts": {"0-1": 1, "0-2": 0, "1-2": 1},
        "static_contact_counts": {"floor": 1, "ramp": 1},
        "pair_correction_counts": {"0-1": 1, "0-2": 0, "1-2": 1},
        "static_correction_counts": {"floor": 1, "ramp": 1},
        "positive_impulse_event_types_by_step_and_sweep": positive,
        "correction_event_types_by_step_and_sweep": corrections,
        "canonical_solver_event_order": list(CONTACT_EVENT_LABELS),
        "event_sequence_semantics": (
            "ordered_per_step_per_sweep_solver_call_events_with_multiplicity"
        ),
        "stick_contacts": 1,
        "slide_contacts": 3,
        "zero_limit_slide_contacts": 0,
        "body_steps": 3 * CONTACT_STEPS,
        "contact_sweeps": CONTACT_SWEEPS * CONTACT_STEPS,
        "pair_solver_calls": 3 * CONTACT_SWEEPS * CONTACT_STEPS,
        "static_solver_calls": 6 * CONTACT_SWEEPS * CONTACT_STEPS,
        "minimum_positive_normal_impulse": 1.0e-6,
        "positive_impulse_threshold": CONTACT_IMPULSE_THRESHOLD,
        "max_penetration": 0.02,
        "max_contact_energy_gain": 0.01,
        "max_pair_momentum_error": 5.0e-6,
        "max_pair_angular_momentum_error": 5.0e-6,
        "thresholds": {
            "max_penetration": 0.03,
            "max_contact_energy_gain": 0.02,
            "max_pair_momentum_error": 1.0e-5,
            "max_pair_angular_momentum_error": 1.0e-5,
        },
    }


def _gradient_work(method: str, schedule: int) -> list[dict[str, object]]:
    base_seed = 6000 + schedule * OPTIMIZATION_STEPS
    if method == "residual_control_variate":
        method_work = {
            "samples": 8,
            "forward_executions": 16,
            "backward_executions": 8,
            "independent_contributions": 4,
            "parameter_perturbations": 8,
            "hard_forward_executions": 8,
            "soft_forward_executions": 8,
        }
    else:
        method_work = {
            "samples": 1,
            "forward_executions": 1,
            "backward_executions": 1,
            "independent_contributions": 1,
            "parameter_perturbations": 1,
            "hard_forward_executions": None,
            "soft_forward_executions": None,
        }
    return [
        {
            "step": step,
            "outer_seed": base_seed + step,
            "inner_seed": None,
            **method_work,
        }
        for step in range(OPTIMIZATION_STEPS)
    ]


def _hard_evaluation_work() -> dict[str, int]:
    return {
        "initial_forward_executions": 1,
        "line_search_batches": OPTIMIZATION_STEPS,
        "line_search_candidates_per_batch": 6,
        "line_search_forward_executions": 6 * OPTIMIZATION_STEPS,
        "final_forward_executions": 1,
        "recheck_forward_executions": 1,
        "total_forward_executions": 6 * OPTIMIZATION_STEPS + 3,
    }


def _contact_rows(*, source_commit: str, device: str) -> list[dict[str, object]]:
    rows = []
    losses = [
        1.0 - 0.5 * step / OPTIMIZATION_STEPS for step in range(OPTIMIZATION_STEPS + 1)
    ]
    parameters = [[0.0, 0.0, 0.0] for _ in range(OPTIMIZATION_STEPS + 1)]
    for schedule in range(OPTIMIZATION_SCHEDULES):
        seed_base = 6000 + schedule * OPTIMIZATION_STEPS
        realized_seeds = list(range(seed_base, seed_base + OPTIMIZATION_STEPS))
        for method in CONTACT_METHODS:
            rows.append(
                {
                    "row_id": f"contact:optimization:{method}:{schedule}",
                    "scenario_family": "contact_3d",
                    "scenario": "three_sphere_floor_ramp",
                    "method": method,
                    "schedule_id": schedule,
                    "initial_hard_loss": 1.0,
                    "final_hard_loss": 0.5,
                    "held_out_loss": 0.5,
                    "success": True,
                    "accepted": True,
                    "source_commit": source_commit,
                    "device": device,
                    "losses": list(losses),
                    "parameters": copy.deepcopy(parameters),
                    "final_target_position_error": 0.5,
                    "final_physical_validity": _physical_validity(),
                    "gradient_work": _gradient_work(method, schedule),
                    "hard_evaluation_work": _hard_evaluation_work(),
                    "realized_outer_seeds": realized_seeds,
                }
            )
    return rows


def _contact_optimization_summaries(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            "scenario": "contact_3d",
            "method": method,
            "final_hard_loss_mean": 0.5,
            "final_hard_loss_ci_low": 0.5,
            "final_hard_loss_ci_high": 0.5,
            "success_rate": 1.0,
            "held_out_loss_mean": 0.5,
            "source_row_ids": [
                row["row_id"] for row in rows if row["method"] == method
            ],
        }
        for method in sorted(CONTACT_METHODS)
    ]


def _contact_optimization_validity(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    successful_methods = [row["method"] for row in rows if row["success"]]
    accepted_count = sum(row["accepted"] is True for row in rows)
    return {
        "scenario": "contact_3d_optimization",
        "accepted": bool(rows)
        and accepted_count == len(rows)
        and bool(successful_methods),
        "metrics": {
            "row_count": len(rows),
            "accepted_count": accepted_count,
            "success_count": len(successful_methods),
            "successful_methods": successful_methods,
        },
        "source_row_ids": [row["row_id"] for row in rows],
    }


def make_valid_contact_optimization_fixture(test: unittest.TestCase) -> Path:
    """Upgrade the shared publication fixture with the canonical report contact contract."""

    root = make_valid_fixture(test)
    manifest_path = root / "data/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = _contact_rows(
        source_commit=manifest["source"]["commit"],
        device=manifest["source"]["device"],
    )
    write_fixture_json(
        root / "data/raw/contact_3d_optimization.json",
        {
            "schema_version": SCHEMA_VERSION,
            "dataset": "contact_3d_optimization",
            "rows": rows,
        },
    )

    summary_path = root / "data/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["optimization_summaries"] = [
        row
        for row in summary["optimization_summaries"]
        if row.get("scenario") != "contact_3d"
    ]
    summary["optimization_summaries"].extend(_contact_optimization_summaries(rows))
    summary["optimization_summaries"].sort(
        key=lambda row: (str(row["scenario"]), str(row["method"]))
    )
    summary["scenario_validity"] = [
        row
        for row in summary["scenario_validity"]
        if row.get("scenario") != "contact_3d_optimization"
    ]
    summary["scenario_validity"].append(_contact_optimization_validity(rows))
    write_fixture_json(summary_path, summary)

    manifest["config"]["optimization_steps"] = OPTIMIZATION_STEPS
    manifest["config"]["optimization_schedules"] = OPTIMIZATION_SCHEDULES
    write_fixture_json(manifest_path, manifest)
    _refresh_manifest(root)
    return root


def _load_contact_rows(
    root: Path,
) -> tuple[Path, dict[str, object], list[dict[str, object]]]:
    path = root / "data/raw/contact_3d_optimization.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return path, payload, payload["rows"]


def _write_contact_rows(root: Path, path: Path, payload: dict[str, object]) -> None:
    write_fixture_json(path, payload)
    _refresh_manifest(root)


class ContactOptimizationContractTests(unittest.TestCase):
    def assert_contact_contract_rejected(self, root: Path, pattern: str) -> None:
        with self.assertRaisesRegex(ValidationError, pattern):
            validate_publication(root)

    def test_valid_contact_optimization_contract_is_accepted(self):
        root = make_valid_contact_optimization_fixture(self)
        _, _, rows = _load_contact_rows(root)
        self.assertEqual(len(rows), len(CONTACT_METHODS) * OPTIMIZATION_SCHEDULES)
        self.assertEqual(
            {(row["method"], row["schedule_id"]) for row in rows},
            {
                (method, schedule)
                for method in CONTACT_METHODS
                for schedule in range(OPTIMIZATION_SCHEDULES)
            },
        )
        for row in rows:
            schedule = row["schedule_id"]
            expected_seeds = list(
                range(
                    6000 + schedule * OPTIMIZATION_STEPS,
                    6000 + (schedule + 1) * OPTIMIZATION_STEPS,
                )
            )
            self.assertEqual(row["realized_outer_seeds"], expected_seeds)
            self.assertEqual(len(row["gradient_work"]), OPTIMIZATION_STEPS)
            validity = row["final_physical_validity"]
            self.assertEqual(
                len(validity["positive_impulse_event_types_by_step_and_sweep"]),
                CONTACT_STEPS,
            )
            self.assertTrue(
                all(
                    len(step) == CONTACT_SWEEPS
                    for step in validity[
                        "positive_impulse_event_types_by_step_and_sweep"
                    ]
                )
            )
            self.assertTrue(row["accepted"])
            self.assertTrue(row["success"])

        summary = json.loads((root / "data/summary.json").read_text(encoding="utf-8"))
        contact_summaries = [
            row
            for row in summary["optimization_summaries"]
            if row["scenario"] == "contact_3d"
        ]
        self.assertEqual(
            {row["method"] for row in contact_summaries}, set(CONTACT_METHODS)
        )
        validity_records = [
            row
            for row in summary["scenario_validity"]
            if row["scenario"] == "contact_3d_optimization"
        ]
        self.assertEqual(validity_records, [_contact_optimization_validity(rows)])

        result = validate_publication(root)
        self.assertEqual(result["files"], 31)

    def test_contact_optimization_requires_exact_method_schedule_identities(self):
        root = make_valid_contact_optimization_fixture(self)
        path, payload, rows = _load_contact_rows(root)
        rows[0]["schedule_id"] = 1
        _write_contact_rows(root, path, payload)

        self.assert_contact_contract_rejected(
            root,
            r"contact.*(identity|schedule|method)|optimization.*(identity|schedule|method)",
        )

    def test_contact_optimization_seed_domains_are_canonical_and_disjoint(self):
        root = make_valid_contact_optimization_fixture(self)
        path, payload, rows = _load_contact_rows(root)
        row = next(
            row
            for row in rows
            if row["method"] == "soft_ad" and row["schedule_id"] == 1
        )
        row["realized_outer_seeds"] = list(range(6000, 6000 + OPTIMIZATION_STEPS))
        for step, work in enumerate(row["gradient_work"]):
            work["outer_seed"] = 6000 + step
        _write_contact_rows(root, path, payload)

        self.assert_contact_contract_rejected(
            root, r"contact.*(seed|domain)|optimization.*(seed|domain)"
        )

    def test_contact_optimization_work_accounting_is_exact(self):
        def coherent_hard_work_change(row: dict[str, object]) -> None:
            work = row["hard_evaluation_work"]
            work["line_search_batches"] = OPTIMIZATION_STEPS - 1
            work["line_search_forward_executions"] = 6 * (OPTIMIZATION_STEPS - 1)
            work["total_forward_executions"] = 6 * (OPTIMIZATION_STEPS - 1) + 3

        mutations = (
            (
                "soft-gradient",
                "soft_ad",
                lambda row: row["gradient_work"][0].__setitem__(
                    "forward_executions", 2
                ),
            ),
            (
                "straight-through-seed-linkage",
                "straight_through_ad",
                lambda row: row["gradient_work"][0].__setitem__(
                    "outer_seed", row["gradient_work"][0]["outer_seed"] + 1
                ),
            ),
            (
                "residual-gradient",
                "residual_control_variate",
                lambda row: row["gradient_work"][0].__setitem__(
                    "hard_forward_executions", 7
                ),
            ),
            ("coherent-hard-evaluation", "soft_ad", coherent_hard_work_change),
        )
        for label, method, mutate in mutations:
            with self.subTest(work=label):
                root = make_valid_contact_optimization_fixture(self)
                path, payload, rows = _load_contact_rows(root)
                row = next(
                    row
                    for row in rows
                    if row["method"] == method and row["schedule_id"] == 0
                )
                mutate(row)
                _write_contact_rows(root, path, payload)
                self.assert_contact_contract_rejected(
                    root, r"contact.*work|gradient work|hard evaluation"
                )

    def test_contact_optimization_recomputes_acceptance_and_success_from_losses(self):
        def forge_success(row: dict[str, object], summary: dict[str, object]) -> None:
            del summary
            row["initial_hard_loss"] = 0.5
            row["losses"] = [0.5] * (OPTIMIZATION_STEPS + 1)

        def forge_acceptance(
            row: dict[str, object], summary: dict[str, object]
        ) -> None:
            row["held_out_loss"] = 0.75
            method_summary = next(
                item
                for item in summary["optimization_summaries"]
                if item["scenario"] == "contact_3d" and item["method"] == row["method"]
            )
            method_summary["held_out_loss_mean"] = (15 * 0.5 + 0.75) / 16

        for label, mutate in (
            ("success", forge_success),
            ("accepted", forge_acceptance),
        ):
            with self.subTest(boolean=label):
                root = make_valid_contact_optimization_fixture(self)
                path, payload, rows = _load_contact_rows(root)
                summary_path = root / "data/summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                mutate(rows[0], summary)
                write_fixture_json(path, payload)
                write_fixture_json(summary_path, summary)
                _refresh_manifest(root)
                self.assert_contact_contract_rejected(
                    root,
                    r"contact.*(accepted|success|loss)|recomputed.*(accepted|success)",
                )

    def test_contact_optimization_physical_thresholds_are_strict(self):
        root = make_valid_contact_optimization_fixture(self)
        path, payload, rows = _load_contact_rows(root)
        rows[0]["final_physical_validity"]["max_penetration"] = 0.03
        _write_contact_rows(root, path, payload)

        self.assert_contact_contract_rejected(
            root, r"contact.*(physical|threshold|penetration)|physical.*valid"
        )

    def test_contact_optimization_rejects_reordered_or_duplicate_events(self):
        def reorder(validity: dict[str, object]) -> None:
            for field in (
                "positive_impulse_event_types_by_step_and_sweep",
                "correction_event_types_by_step_and_sweep",
            ):
                sequence = validity[field]
                sequence[0][0], sequence[1][0] = sequence[1][0], sequence[0][0]

        def duplicate(validity: dict[str, object]) -> None:
            for field in (
                "positive_impulse_event_types_by_step_and_sweep",
                "correction_event_types_by_step_and_sweep",
            ):
                validity[field][0][0] = ["pair_01", "pair_01", "floor_0"]

        def positive_without_correction(validity: dict[str, object]) -> None:
            validity["positive_impulse_event_types_by_step_and_sweep"][0][0] = [
                "pair_01",
                "pair_02",
                "floor_0",
            ]
            validity["pair_contact_counts"]["0-2"] = 1
            validity["stick_contacts"] = 2

        def malformed_correction_shape(validity: dict[str, object]) -> None:
            validity["correction_event_types_by_step_and_sweep"].pop()

        for label, mutate in (
            ("order", reorder),
            ("multiplicity", duplicate),
            ("positive-subset", positive_without_correction),
            ("correction-shape", malformed_correction_shape),
        ):
            with self.subTest(events=label):
                root = make_valid_contact_optimization_fixture(self)
                path, payload, rows = _load_contact_rows(root)
                mutate(rows[0]["final_physical_validity"])
                _write_contact_rows(root, path, payload)
                self.assert_contact_contract_rejected(
                    root, r"contact.*(event|order|multiplicity)|canonical solver order"
                )

    def test_contact_optimization_requires_meaningful_positive_impulse(self):
        root = make_valid_contact_optimization_fixture(self)
        path, payload, rows = _load_contact_rows(root)
        rows[0]["final_physical_validity"]["minimum_positive_normal_impulse"] = 0.0
        _write_contact_rows(root, path, payload)

        self.assert_contact_contract_rejected(
            root, r"contact.*(impulse|physical)|meaningful positive impulse"
        )

    def test_contact_optimization_requires_at_least_one_successful_method(self):
        root = make_valid_contact_optimization_fixture(self)
        path, payload, rows = _load_contact_rows(root)
        for row in rows:
            row["initial_hard_loss"] = 0.5
            row["losses"] = [0.5] * (OPTIMIZATION_STEPS + 1)
            row["success"] = False
        write_fixture_json(path, payload)

        summary_path = root / "data/summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in summary["optimization_summaries"]:
            if row["scenario"] == "contact_3d":
                row["success_rate"] = 0.0
        validity = next(
            row
            for row in summary["scenario_validity"]
            if row["scenario"] == "contact_3d_optimization"
        )
        validity["accepted"] = False
        validity["metrics"]["success_count"] = 0
        validity["metrics"]["successful_methods"] = []
        write_fixture_json(summary_path, summary)
        _refresh_manifest(root)

        self.assert_contact_contract_rejected(
            root, r"contact.*(successful|improvement|accepted)|optimization.*success"
        )

    def test_contact_optimization_summaries_and_validity_are_recomputed(self):
        mutations = (
            (
                "optimization-summary",
                lambda summary: next(
                    row
                    for row in summary["optimization_summaries"]
                    if row["scenario"] == "contact_3d"
                ).__setitem__("success_rate", 0.0),
            ),
            (
                "scenario-validity",
                lambda summary: next(
                    row
                    for row in summary["scenario_validity"]
                    if row["scenario"] == "contact_3d_optimization"
                )["metrics"].__setitem__("accepted_count", 47),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(summary=label):
                root = make_valid_contact_optimization_fixture(self)
                summary_path = root / "data/summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                mutate(summary)
                write_fixture_json(summary_path, summary)
                _refresh_manifest(root)
                self.assert_contact_contract_rejected(
                    root, r"contact.*(summary|validity|aggregate)|optimization.*summary"
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
