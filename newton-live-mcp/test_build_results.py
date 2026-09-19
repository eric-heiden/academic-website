"""Verify report eligibility, incomplete-run handling, and evidence redaction."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from build_results import (
    build,
    export_file,
    pairs_for,
    plan_entries,
    safe_files,
    sanitize,
    trial_record,
)


class TestBuildResults(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.hashes = {"simulation.py": "a" * 64}
        self.registration = {
            "trials": [],
            "trial_count": 2,
            "agent_budget_seconds": 600,
            "completed_candidate_ceiling": 12,
            "task_source_sha256": self.hashes,
        }
        self.summary = {
            "scenario": "panda",
            "variant": 0,
            "condition": "live",
            "phase": "confirmation",
            "model": "gpt-6-astra",
            "reasoning_effort": "xhigh",
            "agent_elapsed_seconds": 100,
            "live_startup_seconds": 3,
            "startup_inclusive_seconds": 103,
            "exit_code": 0,
            "timed_out": False,
            "within_candidate_budget": True,
            "candidate_rollouts": 9,
            "shared_sources_unchanged": True,
            "task_source_hashes": self.hashes,
            "quality": {"success": True},
            "usage": {
                "input_tokens": 1000,
                "cached_input_tokens": 800,
                "output_tokens": 100,
                "reasoning_output_tokens": 50,
                "cache_write_input_tokens": 0,
            },
        }
        self.task = {
            "scenario": "panda",
            "variant": 0,
            "condition": "live",
            "phase": "confirmation",
            "source_hashes": self.hashes,
            "budget_seconds": 600,
        }

    def fixture(self, condition="live", changes=None):
        run = f"panda-{condition}-0"
        folder = self.root / run
        folder.mkdir(exist_ok=True)
        summary = (
            copy.deepcopy(self.summary) | {"condition": condition} | (changes or {})
        )
        (folder / "summary.json").write_text(json.dumps(summary))
        (folder / "task.json").write_text(
            json.dumps(self.task | {"condition": condition})
        )
        return {
            "run_id": run,
            "scenario": "panda",
            "variant": 0,
            "condition": condition,
            "pair_id": "panda-0",
            "workspace": folder,
        }

    def test_quality_alone_does_not_establish_success(self):
        for change, expected in [
            ({"candidate_rollouts": 13}, "candidate_budget"),
            (
                {"agent_elapsed_seconds": 601, "startup_inclusive_seconds": 604},
                "time_budget",
            ),
            ({"timed_out": True}, "no_timeout"),
            ({"shared_sources_unchanged": False}, "shared_sources_unchanged"),
            ({"exit_code": 1}, "clean_exit"),
        ]:
            with self.subTest(expected=expected):
                row, _, _ = trial_record(
                    self.fixture(changes=change), self.registration
                )
                self.assertTrue(row["quality"]["success"])
                self.assertFalse(row["eligible_success"])
                self.assertIn(expected, row["failure_reasons"])

    def test_calibration_requires_training_and_references(self):
        entry = self.fixture()
        entry["scenario"] = "panda_calibration"
        summary = self.summary | {
            "scenario": "panda_calibration",
            "quality": {
                "success": True,
                "held_out_success": True,
                "training_success": False,
            },
            "references_unchanged": True,
        }
        (entry["workspace"] / "summary.json").write_text(json.dumps(summary))
        (entry["workspace"] / "task.json").write_text(
            json.dumps(self.task | {"scenario": "panda_calibration"})
        )
        row, _, _ = trial_record(entry, self.registration)
        self.assertFalse(row["eligible_success"])
        self.assertIn("training_quality", row["failure_reasons"])

    def test_per_trial_versions_preserve_original_and_retry_identity(self):
        old_entry = self.fixture()
        new_entry = self.fixture("restart")
        new_hashes = {"simulation.py": "b" * 64}
        for name, field in (
            ("summary.json", "task_source_hashes"),
            ("task.json", "source_hashes"),
        ):
            path = new_entry["workspace"] / name
            data = json.loads(path.read_text())
            data[field] = new_hashes
            path.write_text(json.dumps(data))
        self.registration["source_commit"] = "old-commit"
        self.registration["trials"] = [
            {key: value for key, value in old_entry.items() if key != "workspace"},
            {
                **{
                    key: value for key, value in new_entry.items() if key != "workspace"
                },
                "workspace": str(new_entry["workspace"]),
                "trial_id": new_entry["run_id"],
                "run_id": new_entry["run_id"] + "-attempt2",
                "source_commit": "new-commit",
                "task_source_sha256": new_hashes,
            },
        ]
        plan = plan_entries(self.registration, self.root)
        old_row, _, _ = trial_record(plan[0], self.registration)
        new_row, _, _ = trial_record(plan[1], self.registration)
        self.assertTrue(old_row["eligible_success"])
        self.assertTrue(new_row["eligible_success"])
        self.assertEqual(old_row["source_commit"], "old-commit")
        self.assertEqual(new_row["source_commit"], "new-commit")
        self.assertEqual(new_row["trial_id"], "panda-restart-0")
        self.assertEqual(new_row["run_id"], "panda-restart-0-attempt2")
        pair = pairs_for([old_row, new_row])[0]
        self.assertFalse(pair["matched_sources_and_references"])
        self.assertTrue(
            all(value is None for value in pair["ratios_restart_over_live"].values())
        )

    def test_infrastructure_attempt_and_registration_history_stay_separate(self):
        self.registration["trial_count"] = 1
        self.registration["trials"] = [
            {"scenario": "panda", "variant": 0, "condition": "live"}
        ]
        folder = self.root / "rejected-startup"
        folder.mkdir()
        (folder / "startup.log").write_text("argparse rejected variant\n")
        self.registration["infrastructure_attempts"] = [
            {
                "run_id": "rejected-startup",
                "workspace": str(folder),
                "agent_launched": False,
                "elapsed_seconds": 2.68,
                "logs": ["startup.log"],
                "logs_base": "workspace",
            }
        ]
        path = self.root / "registration.json"
        path.write_text(json.dumps(self.registration))
        history = self.root / "registration-v1.json"
        history.write_text('{"source_commit":"original"}\n')
        original = history.read_bytes()
        output = self.root / "output"
        result = build(
            path,
            self.root,
            output,
            export_evidence=True,
            registration_history=[history],
        )
        self.assertEqual(result["registered_trials"], 1)
        self.assertEqual(result["completed_trials"], 0)
        self.assertEqual(result["pre_agent_startup_failures"], 1)
        self.assertIsNone(
            result["condition_totals"][0]["sums"]["startup_inclusive_seconds"]
        )
        self.assertTrue(
            (output / "infrastructure/rejected-startup/startup.log").is_file()
        )
        self.assertEqual(history.read_bytes(), original)
        self.assertEqual(
            json.loads(
                (output / "registration-history/registration-v1.json").read_text()
            )["source_commit"],
            "original",
        )
        self.registration["infrastructure_attempts"][0]["agent_launched"] = True
        path.write_text(json.dumps(self.registration))
        with self.assertRaisesRegex(ValueError, "agent_launched"):
            build(path, self.root, self.root / "invalid")

    def test_task_cannot_extend_registered_budget(self):
        entry = self.fixture()
        path = entry["workspace"] / "task.json"
        task = json.loads(path.read_text()) | {"budget_seconds": 700}
        path.write_text(json.dumps(task))
        row, _, _ = trial_record(entry, self.registration)
        self.assertFalse(row["eligible_success"])
        self.assertIn("registered_time_budget", row["failure_reasons"])

    def test_ratios_suppress_failed_pairs_and_keep_token_subsets(self):
        live, _, _ = trial_record(self.fixture(), self.registration)
        restart, _, _ = trial_record(
            self.fixture(
                "restart",
                {
                    "agent_elapsed_seconds": 206,
                    "live_startup_seconds": 0,
                    "startup_inclusive_seconds": 206,
                },
            ),
            self.registration,
        )
        self.assertEqual(live["uncached_input_tokens"], 200)
        self.assertEqual(live["input_output_tokens"], 1100)
        self.assertEqual(
            pairs_for([live, restart])[0]["ratios_restart_over_live"][
                "startup_inclusive_seconds"
            ],
            2,
        )
        restart["eligible_success"] = False
        self.assertTrue(
            all(
                value is None
                for value in pairs_for([live, restart])[0][
                    "ratios_restart_over_live"
                ].values()
            )
        )

    def test_incomplete_trials_remain_pending_and_block_final_build(self):
        self.fixture()
        self.registration["trials"] = [
            {"scenario": "panda", "variant": 0, "condition": condition}
            for condition in ("live", "restart")
        ]
        path = self.root / "registration.json"
        path.write_text(json.dumps(self.registration))
        result = build(path, self.root, self.root / "partial")
        self.assertEqual(result["completed_trials"], 1)
        self.assertEqual(result["pending_trials"], ["panda-restart-0"])
        self.assertFalse(result["pairs"][0]["complete"])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            build(path, self.root, self.root / "final", require_complete=True)
        self.assertFalse((self.root / "final").exists())

    def test_partial_summary_is_not_exported_as_completed(self):
        entry = self.fixture()
        (entry["workspace"] / "summary.json").write_text('{"scenario":')
        row, summary, _ = trial_record(entry, self.registration)
        self.assertEqual(row["status"], "incomplete_evidence")
        self.assertIsNone(summary)

    def test_nested_credentials_images_and_private_references_are_removed(self):
        secret = "sensitive-value-never-public"
        value = {
            "payload": json.dumps({"token": secret, "source_sha256": "a" * 64}),
            "image": {"type": "image", "data": "A" * 9000},
            "command": "read /private/calibration-private/truth.json",
        }
        rendered = json.dumps(sanitize(value))
        self.assertNotIn(secret, rendered)
        self.assertNotIn("A" * 100, rendered)
        self.assertNotIn("truth.json", rendered)
        self.assertIn("a" * 64, rendered)

    def test_allowlist_never_exports_connection_truth_or_symlinks(self):
        folder = self.fixture()["workspace"]
        for name in (
            "connection.json",
            "truth.json",
            "reference.npz",
            "TASK.md",
            "config.py",
        ):
            (folder / name).write_text("test")
        (folder / "candidate").mkdir()
        (folder / "candidate" / "rollouts.jsonl").symlink_to(folder / "connection.json")
        names = {path.name for path in safe_files(folder)}
        self.assertIn("config.py", names)
        self.assertFalse(
            names & {"connection.json", "truth.json", "reference.npz", "rollouts.jsonl"}
        )

    def test_sanitized_jsonl_retains_malformed_line_and_provenance(self):
        source = self.root / "agent.jsonl"
        source.write_text('{"token":"unsafe-secret-value"}\nnot json\n')
        output = self.root / "export" / "agent.jsonl"
        info = export_file(source, output)
        self.assertEqual(info["records"], 2)
        self.assertEqual(info["malformed_lines"], 1)
        self.assertNotIn("unsafe-secret-value", output.read_text())
        self.assertNotEqual(info["original_sha256"], info["export_sha256"])


if __name__ == "__main__":
    unittest.main()
