#!/usr/bin/env python3
"""Generate performance graphs for GEMM implementations."""

import matplotlib.pyplot as plt
import numpy as np

# Benchmark results from M2 MacBook
sizes = [256, 512, 1024]
naive = [1.87, 1.88, 1.73]
blocked = [4.88, 4.73, 3.15]
simd = [18.8, 18.0, 16.5]

x = np.arange(len(sizes))
width = 0.25

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart
bars1 = ax1.bar(x - width, naive, width, label='Naive', color='#e74c3c')
bars2 = ax1.bar(x, blocked, width, label='Blocked', color='#f39c12')
bars3 = ax1.bar(x + width, simd, width, label='SIMD', color='#27ae60')

ax1.set_xlabel('Matrix Size (N×N)')
ax1.set_ylabel('GFLOPS')
ax1.set_title('GEMM Performance by Implementation')
ax1.set_xticks(x)
ax1.set_xticklabels([f'{s}×{s}' for s in sizes])
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8)

# Speedup chart
speedup_blocked = [b/n for b, n in zip(blocked, naive)]
speedup_simd = [s/n for s, n in zip(simd, naive)]

ax2.plot(sizes, [1]*3, 'o--', label='Naive (baseline)', color='#e74c3c', markersize=8)
ax2.plot(sizes, speedup_blocked, 'o-', label='Blocked', color='#f39c12', markersize=8)
ax2.plot(sizes, speedup_simd, 'o-', label='SIMD', color='#27ae60', markersize=8)

ax2.set_xlabel('Matrix Size (N×N)')
ax2.set_ylabel('Speedup vs Naive')
ax2.set_title('Speedup Over Naive Implementation')
ax2.set_xticks(sizes)
ax2.legend()
ax2.grid(alpha=0.3)

# Add value labels
for i, (sb, ss) in enumerate(zip(speedup_blocked, speedup_simd)):
    ax2.annotate(f'{sb:.1f}×', (sizes[i], sb), textcoords="offset points",
                 xytext=(5, 5), fontsize=9)
    ax2.annotate(f'{ss:.1f}×', (sizes[i], ss), textcoords="offset points",
                 xytext=(5, 5), fontsize=9)

plt.tight_layout()
plt.savefig('performance.png', dpi=150)
plt.savefig('performance.svg')
print("Saved performance.png and performance.svg")
