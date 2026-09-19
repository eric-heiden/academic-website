"""Export measured confirmation comparisons as standalone scientific figures."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
OUTPUT = ROOT / 'assets'
RESULTS = json.loads((ROOT / 'data/confirmation/results.json').read_text())
HISTORY = json.loads((ROOT / 'data/confirmation/candidate-history.json').read_text())
COLORS = ('#2563a6', '#b66b27', '#555555')
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none', 'savefig.facecolor': 'white'})

def save(figure, stem):
    for extension in ('svg', 'png'):
        figure.savefig(OUTPUT / f'{stem}.{extension}', dpi=180, bbox_inches='tight')
    plt.close(figure)

pairs = [pair for pair in RESULTS['pairs'] if pair['both_eligible_successes'] and pair['matched_sources_and_references']]
labels = [pair['pair_id'].replace('panda_calibration-', 'Calibration ').replace('panda-', 'Panda ').replace('allegro-', 'Allegro ').replace('hug-', 'HUG ') for pair in pairs]
figure, ax = plt.subplots(figsize=(7.5, 4.6), layout='constrained')
for offset, key, label, color, marker in [(-0.15, 'startup_inclusive_seconds', 'Elapsed time', COLORS[0], 'o'), (0.15, 'input_output_tokens', 'Input + output', COLORS[1], 's'), (0, 'uncached_input_tokens', 'Uncached input', COLORS[2], 'D')]:
    ax.scatter([pair['ratios_restart_over_live'][key] for pair in pairs], np.arange(len(pairs)) + offset, label=label, color=color, marker=marker, s=28, facecolors='none' if marker == 'D' else color)
ax.set_yticks(range(len(pairs)), labels)
ax.invert_yaxis()
ax.axvline(1, color='#777777', linestyle=':', linewidth=1)
ax.set_xlabel('Restart / live (dimensionless ratio; >1 favors live)')
ax.set_xlim(left=0)
ax.grid(axis='x', alpha=0.16)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncols=3, frameon=False, fontsize=9)
save(figure, 'confirmation-ratios')

figure, axes = plt.subplots(3, 1, figsize=(7.5, 8.4), layout='constrained', sharex=True, sharey=True)
for instance, ax in enumerate(axes):
    pair = next(pair for pair in pairs if pair['pair_id'] == f'panda_calibration-{instance}')
    for condition, color, marker, linestyle in [('live', COLORS[0], 'o', '-'), ('restart', COLORS[1], 's', '--')]:
        records = [record for record in HISTORY[pair[condition]]['records'] if record['complete']]
        ax.plot([record['candidate_index'] for record in records], [record['metrics']['trajectory_rmse_rad'] * 1000 for record in records], color=color, marker=marker, linestyle=linestyle, label=condition.title(), markersize=4)
    ax.axhline(0.35, color='#777777', linestyle=':', linewidth=1, label='Pooled RMSE limit')
    ax.set_yscale('log')
    ax.set_ylim(0.15, 50)
    ax.set_ylabel('Training RMSE (mrad)')
    ax.set_title(f'Synthetic instance {instance}', loc='left', fontsize=11)
    ax.grid(alpha=0.16)
    ax.set_xticks(range(1, 10))
axes[0].legend(frameon=False, fontsize=9, loc='upper right')
axes[-1].set_xlabel('Completed candidate (saved order)')
save(figure, 'calibration-convergence')
print('Exported paired ratios and all three calibration histories as SVG and PNG.')

systems = json.loads((ROOT / 'data/systems-timing.json').read_text())
figure, axes = plt.subplots(3, 1, figsize=(7.5, 8.4), layout='constrained', sharex=True, sharey=True)
for ax, case in zip(axes, systems['cases'], strict=True):
    live = np.r_[case['startup_seconds'], case['startup_seconds'] + np.cumsum(case['live_seconds'])]
    restart = np.r_[0, np.cumsum(case['restart_seconds'])]
    ax.plot(range(6), live, color=COLORS[0], marker='o', label='Live, including startup')
    ax.plot(range(6), restart, color=COLORS[1], marker='s', linestyle='--', label='Restart')
    ax.set_title(case['scenario'].capitalize() if case['scenario'] != 'hug' else 'HUG', loc='left', fontsize=11)
    ax.set_ylabel('Cumulative elapsed time (s)')
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.16)
axes[0].legend(frameon=False, fontsize=9)
axes[-1].set_xlabel('Completed scripted evaluations')
axes[-1].set_xticks(range(6))
save(figure, 'systems-cumulative')
