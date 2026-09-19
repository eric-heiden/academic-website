"""Read-only stdlib accounting audit; never opens private calibration inputs."""

import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

BASE = Path('/home/horde/artifacts/newton-live-mcp')
ROOT = Path('/home/horde/apps/newton-live-mcp')
registration = json.loads((BASE / 'confirmation-registration.json').read_text())
problems = []
results = []
tasks = {}
previous_summary_time = None


def check(condition, label):
    if not condition:
        problems.append(label)


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def score(row, task, frames):
    valid = row['finite'] and row['frames'] == frames and row['sample_count'] == frames
    valid = valid and row['expected_frames'] == frames
    valid = valid and row['thresholds'] == task['thresholds']
    valid = valid and all(math.isfinite(row[k]) and row[k] <= bound for k, bound in task['thresholds'].items())
    for episode in row.get('per_episode', []):
        valid = valid and all(math.isfinite(episode[k]) and episode[k] <= task['thresholds'][k]
                              for k in ('trajectory_rmse_rad', 'trajectory_p95_rad'))
    return bool(valid)


for name, expected in registration['protocol_sha256'].items():
    check(hashlib.sha256((BASE / name).read_bytes()).hexdigest() == expected, f'protocol hash: {name}')

for trial in registration['trials']:
    workspace = Path(trial['workspace'])
    if not (workspace / 'summary.json').exists():
        continue
    label = trial['run_id']
    summary = json.loads((workspace / 'summary.json').read_text())
    task = json.loads((workspace / 'task.json').read_text())
    tasks[(task['scenario'], task['variant'], task['condition'])] = {k: v for k, v in task.items() if k != 'condition'}
    check(summary['task_source_hashes'] == task['source_hashes'] == trial['task_source_sha256'], f'{label}: per-trial hashes')
    check(summary['model'] == registration['model'] and summary['reasoning_effort'] == registration['reasoning_effort'], f'{label}: model/settings')
    check(summary['phase'] == task['phase'] == 'confirmation' and task['budget_seconds'] == 600, f'{label}: phase/budget')
    check(summary['shared_sources_unchanged'], f'{label}: source mutation')
    check(summary['exit_code'] == 0 and not summary['timed_out'], f'{label}: final exit/timeout')
    events = records(workspace / 'agent.jsonl')
    items = [event['item'] for event in events if event.get('type') == 'item.completed']
    kinds = Counter(item['type'] for item in items)
    for item in items:
        if item['type'] == 'mcp_tool_call':
            check(item['server'] == 'newton' and item['status'] == 'completed' and not item.get('error') and not item.get('result', {}).get('isError'), f'{label}: MCP success/identity')
        if item['type'] == 'file_change':
            check(all(Path(change['path']) == workspace / 'config.py' for change in item['changes']), f'{label}: recorded file-change scope')
    check(summary['tool_items'] == sum(kinds[k] for k in ('command_execution', 'mcp_tool_call', 'tool_call')), f'{label}: command+MCP count')
    check(summary['mcp_tool_items'] == kinds['mcp_tool_call'], f'{label}: MCP count')
    check(summary['malformed_event_count'] == 0, f'{label}: malformed events')
    turns = [event for event in events if event.get('type') == 'turn.completed']
    usage = {k: sum(event.get('usage', {}).get(k, 0) for event in turns) for k in summary['usage'] if k != 'note'}
    check(usage == {k: v for k, v in summary['usage'].items() if k != 'note'}, f'{label}: usage')
    check(summary['startup_inclusive_seconds'] == summary['agent_elapsed_seconds'] + summary['live_startup_seconds'], f'{label}: inclusive time arithmetic')
    check(summary['agent_elapsed_seconds'] <= task['budget_seconds'], f'{label}: time budget')
    check((summary['live_startup_seconds'] > 0) == (task['condition'] == 'live'), f'{label}: startup condition')
    logs = {'candidate_rollout_logs': [], 'simulation_process_logs': []}
    candidates, processes, seen = [], [], set()
    for path in sorted(workspace.rglob('*.jsonl')):
        if path.name not in ('rollouts.jsonl', 'live_rollouts.jsonl', 'process_events.jsonl'):
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(workspace) or resolved in seen or resolved.relative_to(workspace).parts[0] == 'verification':
            continue
        seen.add(resolved)
        rows = records(path)
        key = 'simulation_process_logs' if path.name == 'process_events.jsonl' else 'candidate_rollout_logs'
        logs[key].append({'path': str(path.relative_to(workspace)), 'records': len(rows)})
        (processes if key == 'simulation_process_logs' else candidates).extend(rows)
    check(all(summary[k] == value for k, value in logs.items()), f'{label}: recursive log manifests')
    check(len(candidates) == summary['candidate_rollouts'], f'{label}: candidate count')
    check(len(processes) == summary['simulation_process_starts_during_trial'], f'{label}: process count')
    check(len({p['pid'] for p in processes}) == len(processes), f'{label}: unique process count')
    check(all(p['event'] == 'simulation_process_start' for p in processes), f'{label}: process event types')
    if previous_summary_time is not None:
        check(min(p['wall_time_unix'] for p in processes) > previous_summary_time, f'{label}: recorded serial process ordering')
    previous_summary_time = (workspace / 'summary.json').stat().st_mtime
    check(len(candidates) <= 12 and summary['within_candidate_budget'], f'{label}: candidate budget')
    check(len(processes) == (1 if task['condition'] == 'live' else len(candidates)), f'{label}: process lifecycle')
    check((kinds['mcp_tool_call'] > 0) == (task['condition'] == 'live'), f'{label}: actual MCP condition')
    frames = task.get('expected_frames', 1500)
    for n, row in enumerate(candidates):
        check(row['scenario'] == task['scenario'] and row['variant'] == task['variant'], f'{label}: candidate {n} identity')
        check(set(row['config']) == set(task['bounds']) and all(low <= row['config'][k] <= high for k, (low, high) in task['bounds'].items()), f'{label}: candidate {n} config bounds')
        check(row['success'] == score(row, task, frames), f'{label}: candidate {n} score')
        if task['scenario'] == 'panda_calibration':
            check(row['episodes'] == [0, 1] and row['reference_sha256'] == task['reference_sha256'], f'{label}: training episodes/reference')
    config_tree = ast.parse((workspace / 'config.py').read_text())
    config = next(ast.literal_eval(node.value) for node in config_tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'CONFIG' for target in node.targets))
    matches = [row for row in candidates if row['config'] == config and row['success']]
    check(bool(matches), f'{label}: submitted measured passing config')
    verifier = json.loads((workspace / summary['verification_metrics']).read_text())
    vprocesses = records(workspace / 'verification/process_events.jsonl')
    check(len(vprocesses) == summary['verification_process_starts'] == 1, f'{label}: verifier count')
    check(verifier['pid'] == vprocesses[0]['pid'] and verifier['pid'] not in {p['pid'] for p in processes}, f'{label}: fresh verifier PID')
    check(vprocesses[0]['wall_time_unix'] > max(p['wall_time_unix'] for p in processes), f'{label}: verifier ordering')
    check(verifier['config'] == config and verifier['success'] == score(verifier, task, 1500), f'{label}: verifier config/score')
    check(summary['quality']['success'], f'{label}: final quality')
    if task['scenario'] == 'panda_calibration':
        check(summary['training_quality'] in matches, f'{label}: exact training-quality record')
        check(summary['references_unchanged'] and verifier['episodes'] == [2], f'{label}: heldout/reference check')
        check(verifier['reference_sha256'] == task['verification_reference_sha256'], f'{label}: heldout digest commitment')
        check(hashlib.sha256((workspace / 'reference.npz').read_bytes()).hexdigest() == task['reference_sha256'], f'{label}: training bytes')
        expected_quality = dict(verifier, held_out_success=verifier['success'], training_success=summary['training_quality']['success'])
        expected_quality['success'] = expected_quality['held_out_success'] and expected_quality['training_success'] and summary['references_unchanged']
    else:
        expected_quality = verifier
        check(all(matches[-1][k] == verifier[k] for k in [*task['thresholds'], 'finite', 'frames', 'sample_count', 'config']), f'{label}: exact fresh scored-metric parity')
    check(summary['quality'] == expected_quality, f'{label}: summary versus verification metrics')
    results.append(dict(index=trial['index'], run_id=label, scenario=task['scenario'], variant=task['variant'], condition=task['condition'], source_revision=trial['source_revision'],
                        candidates=len(candidates), failed_candidates=sum(not row['success'] for row in candidates), processes=len(processes),
                        process_pids=[p['pid'] for p in processes], verifier_pid=verifier['pid'],
                        command_items=kinds['command_execution'], mcp_items=kinds['mcp_tool_call'], file_change_items=kinds['file_change'],
                        command_failures=[{'id': item['id'], 'exit_code': item['exit_code']} for item in items if item['type'] == 'command_execution' and item['exit_code'] != 0],
                        mcp_failures=[item['id'] for item in items if item['type'] == 'mcp_tool_call' and item.get('error')],
                        agent_seconds=summary['agent_elapsed_seconds'], startup_seconds=summary['live_startup_seconds'], inclusive_seconds=summary['startup_inclusive_seconds'],
                        usage=usage, uncached_input_tokens=usage['input_tokens'] - usage['cached_input_tokens'], quality={k: verifier[k] for k in [*task['thresholds'], 'success']},
                        summary_path=str((workspace / 'summary.json').relative_to(BASE)), raw_events_path=str((workspace / 'agent.jsonl').relative_to(BASE)), logs=logs))

for (scenario, variant, condition), task in tasks.items():
    opposite = (scenario, variant, 'restart' if condition == 'live' else 'live')
    if opposite in tasks:
        check(task == tasks[opposite], f'{scenario} {variant}: identical paired task info except condition')
latest_hashes = registration['trials'][-1]['task_source_sha256']
for name, checksum in latest_hashes.items():
    check(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == checksum, f'current frozen source: {name}')
infra = registration['infrastructure_attempts'][0]
check(not (Path(infra['workspace']) / 'agent.jsonl').exists(), 'startup failure has no agent transcript')
check(hashlib.sha256(Path(infra['server_log']).read_bytes()).hexdigest() == infra['server_log_sha256'], 'retained infrastructure server log digest')
output = dict(audit_utc=datetime.now(timezone.utc).isoformat(), completed=len(results), problems=problems, trials=results,
              scope='Stdlib artifact checks only; no private calibration input opened and no simulation rerun.')
(BASE / 'EVALUATION_AUDIT_RECOMPUTED.json').write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps(output, indent=2))
