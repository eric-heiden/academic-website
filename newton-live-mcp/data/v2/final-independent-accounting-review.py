"""Independent final JSON audit; primary recount extended to completed sensitivity.

The separate primary review files are preserved unchanged.
"""

import collections
import datetime
import hashlib
import itertools
import json
import math
import os
from pathlib import Path

ROOT = Path('/home/horde/artifacts/newton-live-mcp-v2')
REGISTRATION = ROOT / 'CONFIRMATION_REGISTRATION.json'
TOKEN_FIELDS = ('input_tokens', 'cached_input_tokens', 'cache_write_input_tokens',
                'output_tokens', 'reasoning_output_tokens')
TEST_EPISODES = [21, 22, 23, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines(path):
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            try:
                yield json.loads(line)
            except ValueError as error:
                raise ValueError(f'{path}:{number}: malformed JSON') from error


def gates(quality, thresholds, frames, episodes=None, real=False):
    failures = []
    for key in ('frames', 'sample_count', 'expected_frames'):
        if quality.get(key) != frames:
            failures.append({'gate': key, 'value': quality.get(key), 'expected': frames})
    if quality.get('finite') is not True:
        failures.append({'gate': 'finite', 'value': quality.get('finite')})
    if quality.get('thresholds') != thresholds:
        failures.append({'gate': 'threshold_definition'})
    if episodes is not None and quality.get('episodes') != episodes:
        failures.append({'gate': 'episode_membership'})
    per_episode = quality.get('per_episode', [])
    if episodes is not None and [x.get('episode') for x in per_episode] != episodes:
        failures.append({'gate': 'per_episode_membership'})
    for item in [quality, *per_episode]:
        episode = item.get('episode', 'pooled')
        if real and episode != 'pooled' and item.get('sample_count') != 600:
            failures.append({'gate': 'per_episode_sample_count', 'episode': episode})
        for key, limit in thresholds.items():
            if not real and episode != 'pooled' and key not in item:
                continue
            value = item.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0 or value > limit:
                detail = {'gate': key, 'episode': episode, 'value': value, 'limit': limit}
                if isinstance(value, (int, float)) and math.isfinite(value):
                    detail['limit_ratio'] = value / limit
                if key == 'max_joint_torque_normalized_rmse':
                    values = item.get('torque_normalized_rmse_per_joint', [])
                    detail['failed_joints_one_based'] = [j + 1 for j, v in enumerate(values) if v > limit]
                failures.append(detail)
    if bool(quality.get('success')) != (not failures):
        failures.append({'gate': 'reported_success_disagrees_with_recomputed_gates'})
    return failures


def review(entry, registration):
    assert entry['cohort'] in ('existing_primary', 'real_primary', 'sensitivity')
    assert entry['condition'] in ('live', 'restart', 'ipython', 'ipython_fixed')
    workspace = Path(entry['workspace'])
    summary = read_json(workspace / 'summary.json')
    task = read_json(workspace / 'task.json')
    manifest = read_json(workspace / 'integrity-manifest.json')
    row = {k: entry[k] for k in ('id', 'scenario', 'variant', 'condition', 'cohort')}
    row['issues'] = issues = []
    for name, commitment in [('task.json', 'task_sha256'), ('TASK.md', 'prompt_sha256'), ('integrity-manifest.json', 'integrity_manifest_sha256')]:
        if digest(workspace / name) != entry[commitment]:
            issues.append(f'Registration hash mismatch: {name}')
    for key in ('scenario', 'variant', 'condition'):
        if summary[key] != entry[key] or task[key] != entry[key]:
            issues.append(f'Identity mismatch: {key}')
    for key in ('model', 'reasoning_effort'):
        if summary[key] != registration[key]:
            issues.append(f'Model mismatch: {key}')
    if summary['phase'] != 'confirmation' or task['phase'] != 'confirmation':
        issues.append('Wrong phase')
    if summary['task_source_hashes'] != manifest['source_hashes']:
        issues.append('Frozen source identity differs from summary')
    if task['integrity_manifest_sha256'] != entry['integrity_manifest_sha256']:
        issues.append('Task manifest commitment mismatch')
    for key in ('shared_sources_unchanged', 'task_unchanged', 'external_sources_unchanged', 'references_unchanged'):
        if summary.get(key) is not True:
            issues.append(f'Integrity flag: {key}')
    for key in ('integrity_manifest_unchanged', 'geometry_unchanged'):
        if key in summary and summary[key] is not True:
            issues.append(f'Integrity flag: {key}')
    row['source_identity'] = hashlib.sha256(json.dumps(manifest['source_hashes'], sort_keys=True).encode()).hexdigest()
    row['reference_identity'] = {k: task[k] for k in ('reference_sha256', 'verification_reference_sha256', 'input_hashes', 'geometry_hashes') if k in task}
    row['summary_sha256'] = digest(workspace / 'summary.json')
    tokens = dict.fromkeys(TOKEN_FIELDS, 0)
    item_counts = collections.Counter()
    failed_commands, mcp_errors = [], []
    completed_turns = turn_errors = 0
    for event in lines(workspace / 'agent.jsonl'):
        kind = event.get('type')
        if kind == 'turn.completed':
            completed_turns += 1
            for key in TOKEN_FIELDS:
                value = event.get('usage', {}).get(key, 0)
                if type(value) is not int or value < 0:
                    issues.append(f'Invalid token count: {key}')
                else:
                    tokens[key] += value
        if kind in ('turn.failed', 'error'):
            turn_errors += 1
        if kind != 'item.completed':
            continue
        item = event.get('item', {})
        item_type = item.get('type')
        item_counts[item_type] += 1
        if item_type == 'command_execution' and item.get('exit_code') not in (None, 0):
            failed_commands.append(item.get('id'))
        if item_type == 'mcp_tool_call':
            result = item.get('result') or {}
            indicated = item.get('error') or item.get('status') == 'failed'
            if isinstance(result, dict):
                indicated = indicated or result.get('isError') or any(p.get('type') == 'text' and p.get('text', '').lstrip().startswith('❌') for p in result.get('content', []))
            if indicated:
                mcp_errors.append(item.get('id'))
    for key in TOKEN_FIELDS:
        if summary['usage'].get(key, 0) != tokens[key]:
            issues.append(f'Raw token mismatch: {key}')
    if tokens['cached_input_tokens'] > tokens['input_tokens'] or tokens['reasoning_output_tokens'] > tokens['output_tokens']:
        issues.append('Token subsets exceed totals')
    counts = {'tool_items': sum(item_counts[k] for k in ('command_execution', 'mcp_tool_call', 'tool_call')), 'mcp_tool_items': item_counts['mcp_tool_call']}
    for key, value in [*counts.items(), ('failed_command_ids', failed_commands), ('mcp_error_indication_ids', mcp_errors)]:
        if summary[key] != value:
            issues.append(f'Raw event mismatch: {key}')
    candidates, process_count, verifier_count = [], 0, 0
    seen = set()
    for path in workspace.rglob('*.jsonl'):
        if path.name not in ('process_events.jsonl', 'rollouts.jsonl', 'live_rollouts.jsonl'):
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(workspace.resolve()):
            issues.append(f'Unexpected log symlink: {path.relative_to(workspace)}')
            continue
        if path.resolve() in seen:
            continue
        seen.add(path.resolve())
        verifier = path.relative_to(workspace).parts[0] == 'verification'
        for record in lines(path):
            if path.name == 'process_events.jsonl':
                if record.get('event') != 'simulation_process_start':
                    issues.append('Unexpected process record')
                if verifier:
                    verifier_count += 1
                else:
                    process_count += 1
            elif not verifier:
                candidates.append(record)
    for key, value in [('candidate_rollouts', len(candidates)), ('simulation_process_starts_during_trial', process_count), ('verification_process_starts', verifier_count)]:
        if summary[key] != value:
            issues.append(f'Raw candidate/process mismatch: {key}')
    startup = summary['application_startup_seconds']
    elapsed = summary['agent_elapsed_seconds']
    inclusive = summary['startup_inclusive_seconds']
    if not all(math.isfinite(x) and x >= 0 for x in (startup, elapsed, inclusive)) or not math.isclose(inclusive, startup + elapsed, abs_tol=1e-8):
        issues.append('Startup arithmetic')
    if summary['timed_out'] or summary['exit_code'] or elapsed > task['budget_seconds'] or not summary['within_candidate_budget'] or len(candidates) > task.get('candidate_budget', 12):
        issues.append('Execution/budget eligibility')
    if completed_turns == 0 or turn_errors:
        issues.append('Incomplete or failed raw agent turn')
    quality = read_json(workspace / 'verification/metrics.json')
    real = entry['scenario'] == 'panda_real'
    calibration = entry['scenario'] == 'panda_calibration'
    frames = 9600 if real else 1500
    episodes = TEST_EPISODES if real else [2] if calibration else None
    failures = gates(quality, task['thresholds'], frames, episodes, real)
    if quality.get('config') != summary['quality'].get('config'):
        issues.append('Verifier/summary configuration mismatch')
    row['training_success'] = None
    if real or calibration:
        training_episodes = [2, 3, 4] if real else [0, 1]
        training_frames = 1800 if real else 3000
        matching = [c for c in candidates if c.get('config') == quality['config'] and not gates(c, task['thresholds'], training_frames, training_episodes, real)]
        row['training_success'] = bool(matching)
        if not matching:
            failures.append({'gate': 'matching_successful_training_candidate'})
        if real:
            fresh = read_json(workspace / 'verification/training/metrics.json')
            fresh_failures = gates(fresh, task['thresholds'], 1800, training_episodes, True)
            if fresh.get('config') != quality['config'] or read_json(workspace / 'config.json') != quality['config']:
                issues.append('Submitted/fresh-training/heldout configuration mismatch')
            if fresh_failures:
                failures.append({'gate': 'fresh_training', 'failures': fresh_failures})
            row['fresh_training_success'] = not fresh_failures
            row['training_reference_sha256'] = fresh.get('reference_sha256')
            if fresh.get('reference_sha256') != task['input_hashes']['training.npz']:
                issues.append('Training reference identity mismatch')
        elif quality.get('reference_sha256') != task['verification_reference_sha256']:
            issues.append('Calibration heldout reference identity mismatch')
    if bool(summary['study_success']) != (not failures and not issues):
        issues.append('Summary study_success differs from independent review')
    row.update(tokens)
    row.update(counts)
    row.update(startup_inclusive_seconds=inclusive, agent_elapsed_seconds=elapsed,
               application_startup_seconds=startup,
               input_output_tokens=tokens['input_tokens'] + tokens['output_tokens'],
               uncached_input_output_tokens=tokens['input_tokens'] - tokens['cached_input_tokens'] + tokens['output_tokens'],
               candidate_rollouts=len(candidates), failed_candidates=sum(not c.get('success', False) for c in candidates),
               simulation_process_starts=process_count, verification_process_starts=verifier_count,
               failed_commands=len(failed_commands), mcp_errors=len(mcp_errors), turn_error_events=turn_errors,
               physics_success=not failures, eligible=not issues, failures=failures,
               heldout_reference_sha256=quality.get('reference_sha256'))
    return row


def close(a, b):
    return math.isclose(a, b, rel_tol=2e-12, abs_tol=2e-12)


def independently_calculate_statistics(numerators, denominators):
    import numpy as np

    ratios = [a / b for a, b in zip(numerators, denominators, strict=True)]
    logs = [math.log(x) for x in ratios]
    n = len(logs)
    mean = math.fsum(logs) / n
    positive = sum(x > 1e-12 for x in logs)
    negative = sum(x < -1e-12 for x in logs)
    non_ties = positive + negative
    sign_probability = min(1.0, 2 * math.fsum(math.comb(non_ties, k) for k in range(min(positive, negative) + 1)) / (2 ** non_ties)) if non_ties else 1.0
    extreme = 0
    for mask in range(2 ** n):
        permuted_mean = math.fsum(value * (1 if mask & (1 << j) else -1) for j, value in enumerate(logs)) / n
        extreme += abs(permuted_mean) >= abs(mean) - 1e-12
    samples, seed = 20000, 20260920
    indices = np.random.default_rng(seed).integers(n, size=(samples, n))
    draws = np.asarray(logs)[indices].sum(axis=1) / n
    ordered = np.sort(np.exp(draws))
    percentiles = []
    for fraction in (0.025, 0.975):
        position = (samples - 1) * fraction
        low = int(position)
        percentiles.append(float(ordered[low] + (position - low) * (ordered[low + 1] - ordered[low])))
    return {'n': n, 'geometric_ratio': math.exp(mean), 'aggregate_ratio': math.fsum(numerators) / math.fsum(denominators), 'paired_ratios': ratios, 'numerator_wins': negative, 'denominator_wins': positive, 'ties': n - non_ties, 'exact_two_sided_sign_p': sign_probability, 'exact_two_sided_log_ratio_permutation_p': extreme / (2 ** n), 'bootstrap_95_interval': percentiles, 'bootstrap_samples': samples, 'bootstrap_seed': seed}


def main():
    registration = read_json(REGISTRATION)
    report_root = Path('/home/horde/repos/academic-website-reports/newton-live-mcp')
    comparison_path = report_root / 'data/v2/comparison.json'
    comparison = read_json(comparison_path)
    original_path = ROOT / 'primary-independent-accounting-review.json'
    original = read_json(original_path)
    assert len(registration['trials']) == 61
    rows = [review(entry, registration) for entry in registration['trials']]
    by_id = {row['id']: row for row in rows}
    issues = [f"{r['id']}: {issue}" for r in rows for issue in r['issues']]
    audited = {row['id']: row for row in comparison['trials']}
    if set(by_id) != set(audited):
        issues.append('Registered/recount/builder membership mismatch')
    if comparison['registered_trials'] != 61 or comparison['completed_trials'] != 61:
        issues.append('Final comparison is incomplete')
    if comparison['registration_raw_sha256'] != digest(REGISTRATION):
        issues.append('Comparison registration digest mismatch')
    for prior in original['trials']:
        if prior != by_id[prior['id']]:
            issues.append(f"Primary recount changed: {prior['id']}")
    ignored = {'issues', 'failures', 'training_success', 'fresh_training_success', 'training_reference_sha256', 'heldout_reference_sha256'}
    renamed = {'source_identity': 'source_identity_sha256'}
    for row in rows:
        target = audited[row['id']]
        for key, value in row.items():
            if key in ignored:
                continue
            if target.get(renamed.get(key, key)) != value:
                issues.append(f"Builder row mismatch: {row['id']} {key}")
        expected_success = row['physics_success'] and row['eligible']
        if target['eligible_success'] != expected_success or not target['usage_complete'] or target['status'] != 'completed' or target['issues']:
            issues.append(f"Builder eligibility/usage/status mismatch: {row['id']}")
        if not expected_success and target.get('best_eligible_metrics'):
            issues.append(f"Failed context marked best: {row['id']}")
    if comparison['successful_eligible_trials'] != sum(r['physics_success'] and r['eligible'] for r in rows):
        issues.append('Overall success count mismatch')
    for group in comparison['groups']:
        subset = [r for r in rows if r['cohort'] == group['cohort'] and r['condition'] == group['condition'] and (group['scope'] == 'all' or r['scenario'] == group['scope'])]
        for key in ('registered_trials', 'completed_trials', 'eligible_trials'):
            if group[key] != len(subset):
                issues.append(f'Group count mismatch: {group["cohort"]}/{group["condition"]}/{group["scope"]} {key}')
        success = sum(r['physics_success'] and r['eligible'] for r in subset)
        if group['successful_eligible_trials'] != success or group['success_rate_all_registered'] != success / len(subset):
            issues.append('Group success mismatch')
        for metric, recorded in group['all_completed_costs'].items():
            expected = sum(r[metric] for r in subset)
            if not close(recorded, expected) or group['cost_coverage_completed_trials'][metric] != len(subset):
                issues.append(f'Group cost/coverage mismatch: {metric}')
    statistical_checks, headline = [], []
    for comparison_pair in comparison['pairwise']:
        cohort, scope = comparison_pair['cohort'], comparison_pair['scope']
        denominator, numerator = comparison_pair['denominator_condition'], comparison_pair['numerator_condition']
        targets = sorted([r for r in rows if r['cohort'] == cohort and r['condition'] == numerator and (scope == 'all' or r['scenario'] == scope)], key=lambda r: (r['scenario'], r['variant']))
        expected_pairs, excluded = [], []
        for target in targets:
            counterpart = next(r for r in rows if r['scenario'] == target['scenario'] and r['variant'] == target['variant'] and r['condition'] == denominator and (r['cohort'] == cohort if cohort != 'sensitivity' else r['cohort'] in ('existing_primary', 'real_primary')))
            if not all(r['physics_success'] and r['eligible'] for r in (target, counterpart)):
                excluded.append({'scenario': target['scenario'], 'variant': target['variant'], 'reason': 'one or both trials failed quality or eligibility'})
            else:
                if target['source_identity'] != counterpart['source_identity'] or target['reference_identity'] != counterpart['reference_identity'] or target['heldout_reference_sha256'] != counterpart['heldout_reference_sha256']:
                    issues.append('Pair source/reference mismatch')
                expected_pairs.append([counterpart['id'], target['id']])
        if comparison_pair['registered_pairs'] != len(targets) or comparison_pair['successful_eligible_pairs'] != len(expected_pairs) or comparison_pair['excluded_pairs'] != excluded:
            issues.append(f'Pair selection/exclusion mismatch: {cohort}/{scope}/{numerator}/{denominator}')
        checked = {'cohort': cohort, 'scope': scope, 'numerator_condition': numerator, 'denominator_condition': denominator, 'included_pairs': expected_pairs, 'excluded_pairs': excluded, 'metrics': {}}
        for metric, stats in comparison_pair['metrics'].items():
            if stats['pair_ids'] != expected_pairs:
                issues.append('Metric pairing mismatch')
            values = independently_calculate_statistics([by_id[b][metric] for a, b in expected_pairs], [by_id[a][metric] for a, b in expected_pairs])
            for key, expected in values.items():
                recorded = stats[key]
                matches = all(close(a, b) for a, b in zip(expected, recorded, strict=True)) if isinstance(expected, list) else close(expected, recorded)
                if not matches:
                    issues.append(f'Statistic mismatch: {cohort}/{scope}/{numerator}/{denominator}/{metric}/{key}')
            display = values.copy()
            if cohort != 'sensitivity':
                display['geometric_ratio'] = 1 / values['geometric_ratio']
                display['bootstrap_95_interval'] = [1 / values['bootstrap_95_interval'][1], 1 / values['bootstrap_95_interval'][0]]
            checked['metrics'][metric] = {'builder_orientation': values, 'displayed_geometric_ratio': display['geometric_ratio'], 'displayed_bootstrap_95_interval': display['bootstrap_95_interval']}
        statistical_checks.append(checked)
        if scope == 'all' or cohort == 'sensitivity' and scope == 'panda_real':
            headline.append(checked)
    figures_path = report_root / 'assets/v2/comparison-figure-data.json'
    figures = read_json(figures_path)
    for item in figures['performance']['rows']:
        live = next(r for r in rows if r['cohort'] == item['cohort'] and r['scenario'] == item['scenario'] and r['variant'] == item['variant'] and r['condition'] == 'live')
        other = next(r for r in rows if r['cohort'] == item['cohort'] and r['scenario'] == item['scenario'] and r['variant'] == item['variant'] and r['condition'] == item['comparator'])
        included = all(r['physics_success'] and r['eligible'] for r in (live, other))
        if item['included'] != included or included and not close(item['newton_over_comparator'], live[item['metric']] / other[item['metric']]):
            issues.append('Figure performance ratio/exclusion mismatch')
    expected_quality_ids = {r['id'] for r in rows if r['cohort'] == 'real_primary'}
    if {r['id'] for r in figures['quality']['rows']} != expected_quality_ids:
        issues.append('Quality figure membership mismatch')
    for item in figures['quality']['rows']:
        measured = audited[item['id']]['quality']
        normalized = [max(part[key] for part in [measured, *measured['per_episode']]) / measured['thresholds'][key] for key in figures['quality']['metrics']]
        if not all(close(a, b) for a, b in zip(normalized, item['normalized_worst_pooled_or_recording_values'], strict=True)) or item['eligible_success'] != audited[item['id']]['eligible_success']:
            issues.append('Quality figure normalization/status mismatch')
    sensitivity_quality = []
    for entry in registration['trials']:
        if entry['cohort'] != 'sensitivity':
            continue
        measured = read_json(Path(entry['workspace']) / 'verification/metrics.json')
        worst = {key: max(part[key] for part in [measured, *measured.get('per_episode', [])] if key in part) for key in measured['thresholds']}
        normalized = {key: value / measured['thresholds'][key] for key, value in worst.items()}
        sensitivity_quality.append({'id': entry['id'], 'frames': measured['frames'], 'sample_count': measured['sample_count'], 'finite': measured['finite'], 'thresholds': measured['thresholds'], 'worst_pooled_or_recording': worst, 'normalized_worst_pooled_or_recording': normalized, 'largest_normalized_gate': max(normalized, key=normalized.get), 'largest_limit_ratio': max(normalized.values())})
    output = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'source_commit': registration['source_commit'], 'registration_sha256': digest(REGISTRATION), 'comparison_sha256': digest(comparison_path), 'primary_review_sha256': digest(original_path), 'figure_data_sha256': digest(figures_path), 'recount_script_sha256': digest(Path(__file__)), 'scope': 'All 61 completed contexts; primary review preserved. Raw event/candidate/process recount independent of the report builder. Recorded scalar verification gates checked, not rerun physics.', 'issue_count': len(issues), 'issues': issues, 'primary_rows_reconciled': len(original['trials']), 'total_rows_reconciled': len(rows), 'groups_reconciled': len(comparison['groups']), 'pairwise_blocks_reconciled': len(statistical_checks), 'metric_statistics_recomputed': sum(len(p['metrics']) for p in statistical_checks), 'quality_figure_rows_reconciled': len(figures['quality']['rows']), 'eligible': sum(r['eligible'] for r in rows), 'physics_successful': sum(r['physics_success'] for r in rows), 'sensitivity_rows': [r for r in rows if r['cohort'] == 'sensitivity'], 'sensitivity_quality': sensitivity_quality, 'sensitivity_groups': [g for g in comparison['groups'] if g['cohort'] == 'sensitivity'], 'headline_ratios': headline, 'all_statistics': statistical_checks, 'interpretation': ['Primary ratios are rendered as denominator/numerator relative to builder storage; reciprocal interval endpoints must be reversed. Sensitivity remains corrected/original comparator, as recorded.', 'Two-sided exact sign tests and sign-flip log-ratio tests recomputed by enumeration. Bootstrap uses 20,000 whole-pair resamples with fixed seed 20260920 and linear-interpolated 2.5/97.5 percentiles; results agree within floating-point tolerance.', 'Different exact tests and percentile-bootstrap intervals need not agree about crossing 1 versus a p-value threshold. These small-sample unadjusted descriptive comparisons should not be presented as causal or universal superiority.', 'Real-data paired ratios exclude failed variants 3 and 5 from Newton comparisons, leaving n=7; restart/IPython excludes variant 3, leaving n=8. All failures remain in all-completed costs and success denominators.', 'All corrected-IPython runs followed the primary block and cover seven selected cases, including three real-data repeats. Differences can reflect run order and agent search variation; the request-correlation and wait changes are not isolated causally.']}
    target = ROOT / 'final-independent-accounting-review.json'
    target.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    compact = {'artifact': str(target), 'issues': issues, 'rows': len(rows), 'groups': len(comparison['groups']), 'statistic_blocks': len(statistical_checks), 'sensitivity_totals': next(g for g in output['sensitivity_groups'] if g['scope'] == 'all'), 'headlines': [{**{k: p[k] for k in ('cohort', 'scope', 'numerator_condition', 'denominator_condition')}, 'n': len(p['included_pairs']), 'metrics': {key: {'ratio': value['displayed_geometric_ratio'], 'interval': value['displayed_bootstrap_95_interval'], 'sign_p': value['builder_orientation']['exact_two_sided_sign_p'], 'permutation_p': value['builder_orientation']['exact_two_sided_log_ratio_permutation_p']} for key, value in p['metrics'].items()}} for p in headline]}
    print(json.dumps(compact, indent=2))


if __name__ == '__main__':
    main()
