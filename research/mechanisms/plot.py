"""Visualize controlled decision probes and one-change mask ablations."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
out = ROOT / 'build/mechanism-audit'
p = np.load(out / 'plateau.npz')
r = json.loads((out / 'results.json').read_text())
variants = {v['name']: v for v in r['variants']}
fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), layout='constrained')
for ax, ripple in zip(axes[0], ('0.0', '0.05')):
    x = np.arange(290)
    ax.plot(x, p['input_' + ripple][:290], color='#263b57', label='Magnitude supplied to first pass')
    ax.plot(x, p['floor_' + ripple][:290], color='#bf4d25', label='Occupancy threshold')
    records = json.loads((out / 'plateau.json').read_text())['records']
    threshold = next(v['reference_threshold'] for v in records if str(v['ripple']) == ripple)
    ax.axhline(threshold, color='#24806d', ls='--', label='Cleanup mean row threshold')
    ax.set(title='Steady plateau' if ripple == '0.0' else 'Same plateau + small fixed ripple', xlabel='Frequency cell', ylabel='Magnitude / threshold', ylim=(-.08, 2.5))
axes[0, 0].legend(fontsize=8, loc='upper right')
ax = axes[1, 0]
for key, label, color in [('reference_0.05', 'Cleanup', '#24806d'), ('occupancy_0.05', 'Occupancy', '#bf4d25')]:
    # Display separate rows to make the binary decisions readable.
    row = 1 if label == 'Cleanup' else 0
    accepted = p[key][60:180] > .5
    ax.scatter(np.arange(60, 180)[accepted], np.full(accepted.sum(), row), marker='|', s=130, color=color, label=label)
ax.axvspan(80, 159, color='#263b57', alpha=.06)
ax.set(yticks=[0, 1], yticklabels=['Occupancy', 'Cleanup'], ylim=(-.6, 1.6), xlabel='Frequency cell', title='First-pass admission, rippled plateau')
ax = axes[1, 1]
names = ['registered-reference', 'binary-second-pass']
labels = ['Cleanup reference', 'Only second pass made binary']
for key, label, color in zip(names, labels, ['#24806d', '#bf4d25']):
    values = np.array(variants[key]['radio_stage_means'])[:, 0]
    ax.plot(np.arange(4), values * 100, '-o', label=label, color=color)
ax.set(xticks=np.arange(4), xticklabels=['First pass', 'Combined\nsecond pass', 'Recursive\nsmoothing', 'Final gain'], ylabel='Mean in-band gain × 100', title='Actual radio: identical first selection')
ax.legend(fontsize=8)
fig.suptitle('Cleanup mechanism audit — decision behavior, not a perceptual ranking', fontsize=14)
fig.savefig(out / 'mechanisms.png', dpi=170)
