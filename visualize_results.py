import json
import matplotlib.pyplot as plt
import numpy as np

with open('evaluation_results.json', 'r') as f:
    results = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
iterations = list(range(1, len(results['rd_losses']) + 1))
ax1.plot(iterations, results['rd_losses'], 'b-o', label='RD-MCTS', linewidth=2)
ax1.plot(iterations, results['baseline_losses'], 'r-s', label='Baseline MCTS', linewidth=2)
ax1.set_xlabel('Training Iteration', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
match_results = results['match_results']
categories = ['RD-MCTS\nWins', 'Baseline\nWins', 'Draws']
values = [match_results['rd_wins'], match_results['baseline_wins'], match_results['draws']]
colors = ['#2ecc71', '#e74c3c', '#95a5a6']
bars = ax2.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

for bar, value in zip(bars, values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(value)}',
             ha='center', va='bottom', fontsize=14, fontweight='bold')

ax2.set_ylabel('Number of Games', fontsize=12)
ax2.set_title('Head-to-Head Match Results (20 games)', fontsize=14, fontweight='bold')
ax2.set_ylim(0, max(values) * 1.2)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
print("Visualization saved to evaluation_results.png")

win_rate = match_results['rd_wins'] / (match_results['rd_wins'] + match_results['baseline_wins'] + match_results['draws'])
print(f"\nRD-MCTS Win Rate: {win_rate*100:.1f}%")
print(f"Performance Advantage: {match_results['rd_wins']}W - {match_results['baseline_wins']}L - {match_results['draws']}D")
