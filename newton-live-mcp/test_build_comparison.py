"""Check comparison accounting and redaction using fixtures and development copies."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from build_comparison import (
    audit_candidates,
    audit_events,
    build,
    export_evidence,
    paired_statistics,
    sanitize,
    sha256,
)


class TestComparisonBuilder(unittest.TestCase):
    def setUp(self):
        """Create an isolated evidence tree."""
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.registration = {
            "created_utc": "2026-09-20T00:00:00Z",
            "source_commit": "fixture-commit",
            "model": "gpt-6-astra",
            "reasoning_effort": "xhigh",
            "trials": [],
        }

    def tearDown(self):
        """Remove only test-owned copies."""
        self.temporary.cleanup()

    def write(self, path, value):
        """Write a local JSON fixture."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n")

    def trial(
        self,
        condition,
        variant=0,
        *,
        elapsed=20.0,
        success=True,
        cohort="existing_primary",
    ):
        """Create a complete trial using the recorded development schema."""
        name = f"panda-{condition}-{variant}-{cohort}"
        workspace = self.root / name
        workspace.mkdir()
        sources = {"physics.py": "a" * 64}
        self.write(workspace / "integrity-manifest.json", {"source_hashes": sources})
        task = {
            "scenario": "panda",
            "variant": variant,
            "condition": condition,
            "phase": "confirmation",
            "budget_seconds": 600,
            "candidate_budget": 12,
            "thresholds": {"tracking_rmse_rad": 0.05},
            "integrity_manifest_sha256": sha256(workspace / "integrity-manifest.json"),
        }
        self.write(workspace / "task.json", task)
        (workspace / "TASK.md").write_text(
            "Recorded schema fixture; no agent launched.\n"
        )
        quality = {
            "scenario": "panda",
            "variant": variant,
            "success": success,
            "finite": True,
            "frames": 1500,
            "sample_count": 1500,
            "expected_frames": 1500,
            "thresholds": task["thresholds"],
            "tracking_rmse_rad": 0.01 if success else 0.2,
            "config": {"kp": 100.0},
        }
        self.write(workspace / "verification/metrics.json", quality)
        failed = quality | {
            "success": False,
            "tracking_rmse_rad": 0.3,
            "config": {"kp": 10.0},
        }
        (workspace / "live_rollouts.jsonl").write_text(
            json.dumps(failed) + "\n" + json.dumps(quality) + "\n"
        )
        process = {
            "event": "simulation_process_start",
            "pid": 123,
            "wall_time_unix": 1.0,
            "live": condition != "restart",
        }
        (workspace / "process_events.jsonl").write_text(json.dumps(process) + "\n")
        (workspace / "verification/process_events.jsonl").write_text(
            json.dumps(process | {"pid": 124}) + "\n"
        )
        usage = {
            "input_tokens": 1000,
            "cached_input_tokens": 800,
            "cache_write_input_tokens": 0,
            "output_tokens": 100,
            "reasoning_output_tokens": 40,
        }
        events = [
            {
                "type": "item.completed",
                "item": {
                    "id": "command-1",
                    "type": "command_execution",
                    "exit_code": 0,
                },
            },
            {"type": "item.completed", "item": {"id": "edit-1", "type": "file_change"}},
            {"type": "turn.completed", "usage": usage},
        ]
        (workspace / "agent.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in events)
        )
        summary = {
            "scenario": "panda",
            "variant": variant,
            "condition": condition,
            "phase": "confirmation",
            "model": "gpt-6-astra",
            "reasoning_effort": "xhigh",
            "usage": usage,
            "tool_items": 1,
            "mcp_tool_items": 0,
            "failed_command_ids": [],
            "mcp_error_indication_ids": [],
            "candidate_rollouts": 2,
            "simulation_process_starts_during_trial": 1,
            "verification_process_starts": 1,
            "shared_sources_unchanged": True,
            "task_unchanged": True,
            "external_sources_unchanged": True,
            "integrity_manifest_unchanged": True,
            "references_unchanged": True,
            "within_candidate_budget": True,
            "agent_elapsed_seconds": elapsed - 2,
            "application_startup_seconds": 2,
            "startup_inclusive_seconds": elapsed,
            "timed_out": False,
            "exit_code": 0,
            "task_source_hashes": sources,
            "study_success": success,
            "quality": quality,
        }
        self.write(workspace / "summary.json", summary)
        entry = {
            "id": name,
            "scenario": "panda",
            "variant": variant,
            "condition": condition,
            "cohort": cohort,
            "workspace": str(workspace),
            "task_sha256": sha256(workspace / "task.json"),
            "prompt_sha256": sha256(workspace / "TASK.md"),
            "integrity_manifest_sha256": sha256(workspace / "integrity-manifest.json"),
        }
        self.registration["trials"].append(entry)
        return workspace

    def run_builder(self, **kwargs):
        """Build test outputs without publishing."""
        self.write(self.root / "registration.json", self.registration)
        return build(self.root / "registration.json", self.root / "out", **kwargs)

    def test_failure_retained_and_never_fastest(self):
        """A fast failed run stays in denominators/costs without winning a metric."""
        self.trial("live", elapsed=30)
        self.trial("restart", elapsed=50)
        self.trial("ipython", elapsed=3, success=False)
        result = self.run_builder(require_complete=True, strict_audit=True)
        rows = result["trials"]
        self.assertEqual(result["successful_eligible_trials"], 2)
        self.assertEqual(rows[2]["best_eligible_metrics"], [])
        self.assertEqual(rows[0]["input_output_tokens"], 1100)
        self.assertEqual(rows[0]["uncached_input_output_tokens"], 300)
        self.assertEqual(rows[0]["failed_candidates"], 1)
        group = next(
            item
            for item in result["groups"]
            if item["condition"] == "ipython" and item["scope"] == "all"
        )
        self.assertEqual(group["success_rate_all_registered"], 0)
        self.assertEqual(group["all_completed_costs"]["startup_inclusive_seconds"], 3)
        pair = next(
            item
            for item in result["pairwise"]
            if item["scope"] == "all" and item["numerator_condition"] == "restart"
        )
        self.assertAlmostEqual(
            pair["metrics"]["startup_inclusive_seconds"]["geometric_ratio"], 50 / 30
        )

    def test_raw_accounting_mismatches_reject_eligibility(self):
        """Do not accept edited summary totals that disagree with raw evidence."""
        workspace = self.trial("live")
        summary = json.loads((workspace / "summary.json").read_text())
        summary["usage"]["input_tokens"] += 1
        summary["candidate_rollouts"] += 1
        self.write(workspace / "summary.json", summary)
        row = self.run_builder()["trials"][0]
        self.assertFalse(row["eligible_success"])
        self.assertTrue(any("Raw token" in issue for issue in row["issues"]))
        self.assertTrue(any("candidate" in issue for issue in row["issues"]))
        with self.assertRaisesRegex(ValueError, "accounting"):
            self.run_builder(strict_audit=True)

    def test_immutable_hash_change_is_visible(self):
        """Prompt mutations invalidate a row without removing its cost."""
        workspace = self.trial("live")
        (workspace / "TASK.md").write_text("changed")
        result = self.run_builder()
        self.assertFalse(result["trials"][0]["eligible"])
        self.assertEqual(result["groups"][0]["completed_trials"], 1)

    def test_missing_rows_and_duplicate_workspaces(self):
        """Retain missing registered rows and reject workspace reuse."""
        workspace = self.trial("live")
        (workspace / "summary.json").unlink()
        result = self.run_builder()
        self.assertEqual(result["registered_trials"], 1)
        self.assertEqual(result["completed_trials"], 0)
        with self.assertRaisesRegex(ValueError, "missing"):
            self.run_builder(require_complete=True)
        duplicate = dict(self.registration["trials"][0])
        duplicate.update(id="different-id", condition="restart")
        self.registration["trials"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "reused"):
            self.run_builder()

    def test_exact_pairing_and_sensitivity(self):
        """Pair the same scenario/variant across explicitly labeled sensitivity cohorts."""
        self.trial("ipython", variant=0, elapsed=40)
        self.trial("ipython", variant=1, elapsed=90)
        self.trial("ipython_fixed", variant=0, elapsed=20, cohort="sensitivity")
        result = self.run_builder()
        pair = next(
            item
            for item in result["pairwise"]
            if item["cohort"] == "sensitivity"
            and item["scope"] == "all"
            and item["denominator_condition"] == "ipython"
        )
        self.assertEqual(pair["metrics"]["startup_inclusive_seconds"]["n"], 1)
        self.assertEqual(
            pair["metrics"]["startup_inclusive_seconds"]["geometric_ratio"], 0.5
        )
        self.assertEqual(pair["registered_pairs"], 1)

    def test_exact_tests_and_bootstrap(self):
        """Recover exact small-n probabilities and deterministic paired intervals."""
        result = paired_statistics([2, 4, 6], [1, 2, 3], bootstrap_samples=1000)
        self.assertAlmostEqual(result["geometric_ratio"], 2)
        self.assertEqual(result["exact_two_sided_sign_p"], 0.25)
        self.assertEqual(result["exact_two_sided_log_ratio_permutation_p"], 0.25)
        self.assertEqual(result["bootstrap_95_interval"], [2.0, 2.0])
        tied = paired_statistics([1, 2], [1, 2], bootstrap_samples=100)
        self.assertEqual(tied["ties"], 2)
        self.assertEqual(tied["exact_two_sided_sign_p"], 1)
        with self.assertRaises(ValueError):
            paired_statistics([0], [1])

    def test_missing_usage_has_incomplete_coverage(self):
        """Timeouts lacking a final usage event do not imply zero token expenditure."""
        workspace = self.trial("live")
        (workspace / "agent.jsonl").write_text(
            json.dumps({"type": "turn.failed", "error": "timeout"}) + "\n"
        )
        summary = json.loads((workspace / "summary.json").read_text())
        summary.update(
            usage=dict.fromkeys(summary["usage"], 0),
            tool_items=0,
            timed_out=True,
            exit_code=-15,
            study_success=False,
        )
        self.write(workspace / "summary.json", summary)
        result = self.run_builder()
        self.assertFalse(result["trials"][0]["usage_observed"])
        self.assertEqual(
            result["groups"][0]["cost_coverage_completed_trials"][
                "input_output_tokens"
            ],
            0,
        )
        self.assertFalse(result["trials"][0]["eligible_success"])

    def test_export_redacts_key_prefixes_and_omits_connections(self):
        """Sanitize upstream connect replies and retain separate raw/export hashes."""
        workspace = self.trial("ipython")
        prefix = "credential-test-prefix"
        response = {
            "type": "item.completed",
            "item": {
                "type": "mcp_tool_call",
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"✅ Connected\n🔑 Using key: {prefix}...",
                        },
                        {
                            "type": "text",
                            "text": json.dumps(
                                {"key": prefix, "nested": {"api_key": prefix}}
                            ),
                        },
                    ]
                },
            },
        }
        with (workspace / "agent.jsonl").open("a") as stream:
            stream.write(json.dumps(response) + "\n")
        blocked = (
            "connection.json",
            "kernel.json",
            "server.log",
            "kernel-secret.cfg",
            "connection.txt",
            "server.out",
        )
        for name in blocked:
            (workspace / name).write_text(prefix)
        target = self.root / "exports"
        manifest = export_evidence(workspace, target)
        text = "".join(path.read_text() for path in target.rglob("*") if path.is_file())
        self.assertNotIn(prefix, text)
        for name in blocked:
            self.assertFalse((target / name).exists())
            self.assertNotIn(
                "raw_sha256", next(row for row in manifest if row["path"] == name)
            )
        exported = next(row for row in manifest if row["path"] == "agent.jsonl")
        self.assertNotEqual(exported["raw_sha256"], exported["export_sha256"])
        self.assertEqual(exported["raw_sha256"], sha256(workspace / "agent.jsonl"))
        self.assertNotIn(prefix, str(sanitize({"message": f"key=b'{prefix}'"})))
        helper = workspace / "helpers/fit.py"
        helper.parent.mkdir()
        helper.write_text("import numpy as np\nresult = np.ones(7)\n")
        self.write(workspace / "fitted.json", {"mass": [1] * 7})
        export_evidence(workspace, self.root / "second-export")
        self.assertEqual(
            (self.root / "second-export/helpers/fit.py").read_text(), helper.read_text()
        )
        self.assertTrue((self.root / "second-export/fitted.json").is_file())

    @unittest.skipUnless(
        Path(
            "/home/horde/artifacts/newton-live-mcp-v2/development/panda-real-live-0/summary.json"
        ).exists(),
        "retained development evidence is optional",
    )
    def test_development_copy_raw_recount(self):
        """Recount copied development evidence without promoting it to confirmation."""
        original = Path(
            "/home/horde/artifacts/newton-live-mcp-v2/development/panda-real-live-0"
        )
        target = self.root / "development-copy"
        target.mkdir()
        for name in ("agent.jsonl", "live_rollouts.jsonl", "process_events.jsonl"):
            shutil.copyfile(original / name, target / name)
        summary = json.loads((original / "summary.json").read_text())
        events, candidates = audit_events(target), audit_candidates(target)
        for key in ("input_tokens", "output_tokens", "cached_input_tokens"):
            self.assertEqual(events["usage"][key], summary["usage"][key])
        self.assertEqual(
            candidates["candidate_rollouts"], summary["candidate_rollouts"]
        )
        self.assertEqual(events["tool_items"], summary["tool_items"])
        self.assertEqual(
            events["mcp_error_indication_ids"], summary["mcp_error_indication_ids"]
        )


if __name__ == "__main__":
    unittest.main()
