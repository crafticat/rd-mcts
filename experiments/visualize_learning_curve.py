"""
Visualization script for learning curve results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_learning_curves(results_path="learning_curve_results.json", output_path="learning_curve_analysis.png"):
    """Generate learning curve visualization."""
    
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    checkpoints = results["checkpoints"]
    rd_win_rates = results["rd_mcts"]["win_rates"]
    std_win_rates = results["standard_mcts"]["win_rates"]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Learning Curve Analysis: Sample Efficiency Comparison", fontsize=16, fontweight='bold')
    
    # Plot 1: Win rates over training
    ax1 = axes[0, 0]
    ax1.plot(checkpoints, rd_win_rates, 'o-', linewidth=2, markersize=8, label='RD-MCTS (λ=1)', color='#2E86AB')
    ax1.plot(checkpoints, std_win_rates, 's-', linewidth=2, markersize=8, label='Standard MCTS', color='#A23B72')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
    ax1.set_xlabel('Training Games', fontsize=12)
    ax1.set_ylabel('Win Rate vs Benchmark', fontsize=12)
    ax1.set_title('Win Rate Over Training', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Plot 2: Win rate difference
    ax2 = axes[0, 1]
    win_rate_diff = [rd - std for rd, std in zip(rd_win_rates, std_win_rates)]
    colors = ['green' if d > 0 else 'red' for d in win_rate_diff]
    ax2.bar(checkpoints, win_rate_diff, width=8, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Training Games', fontsize=12)
    ax2.set_ylabel('Win Rate Difference\n(RD-MCTS - Standard)', fontsize=12)
    ax2.set_title('Advantage of RD-MCTS Over Training', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Cumulative wins
    ax3 = axes[1, 0]
    rd_wins = results["rd_mcts"]["wins"]
    std_wins = results["standard_mcts"]["wins"]
    
    x = np.arange(len(checkpoints))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, rd_wins, width, label='RD-MCTS', color='#2E86AB', alpha=0.8)
    bars2 = ax3.bar(x + width/2, std_wins, width, label='Standard MCTS', color='#A23B72', alpha=0.8)
    
    ax3.set_xlabel('Training Games', fontsize=12)
    ax3.set_ylabel('Wins (out of 20 games)', fontsize=12)
    ax3.set_title('Wins vs Benchmark at Each Checkpoint', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(checkpoints)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate statistics
    rd_mean = np.mean(rd_win_rates)
    std_mean = np.mean(std_win_rates)
    rd_final = rd_win_rates[-1] if rd_win_rates else 0
    std_final = std_win_rates[-1] if std_win_rates else 0
    
    # Find when each agent reaches 50% win rate
    rd_50_games = next((cp for cp, wr in zip(checkpoints, rd_win_rates) if wr >= 0.5), "Never")
    std_50_games = next((cp for cp, wr in zip(checkpoints, std_win_rates) if wr >= 0.5), "Never")
    
    summary_text = f"""
LEARNING CURVE SUMMARY

Sample Efficiency:
  RD-MCTS reaches 50% at:     {rd_50_games} games
  Standard MCTS reaches 50% at: {std_50_games} games

Average Performance (across all checkpoints):
  RD-MCTS:      {rd_mean:.1%}
  Standard MCTS: {std_mean:.1%}
  Difference:    {(rd_mean - std_mean):.1%}

Final Performance (100 games):
  RD-MCTS:      {rd_final:.1%}
  Standard MCTS: {std_final:.1%}
  Difference:    {(rd_final - std_final):.1%}

Key Finding:
  {"RD-MCTS learns faster" if rd_50_games != "Never" and (std_50_games == "Never" or rd_50_games < std_50_games) else "Similar learning speed"}
  {"and achieves higher final performance" if rd_final > std_final else ""}

Benchmark: {results["config"]["benchmark_type"]}
Games per checkpoint: {results["config"]["num_games"]}
"""
    
    ax4.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_path}")
    
    return fig


def create_summary_report(results_path="learning_curve_results.json", output_path="learning_curve_summary.txt"):
    """Create text summary of learning curve results."""
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    checkpoints = results["checkpoints"]
    rd_win_rates = results["rd_mcts"]["win_rates"]
    std_win_rates = results["standard_mcts"]["win_rates"]
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("LEARNING CURVE EXPERIMENT SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write("## Experiment Configuration\n\n")
        f.write(f"Benchmark: {results['config']['benchmark_type']}\n")
        f.write(f"Games per checkpoint: {results['config']['num_games']}\n")
        f.write(f"Simulations per move: {results['config']['simulations']}\n")
        f.write(f"Checkpoints: {checkpoints}\n\n")
        
        f.write("## Results\n\n")
        f.write("Training Games | RD-MCTS | Standard MCTS | Difference\n")
        f.write("-" * 60 + "\n")
        for cp, rd_wr, std_wr in zip(checkpoints, rd_win_rates, std_win_rates):
            diff = rd_wr - std_wr
            f.write(f"{cp:14d} | {rd_wr:7.1%} | {std_wr:13.1%} | {diff:+10.1%}\n")
        
        f.write("\n## Key Findings\n\n")
        
        # Sample efficiency
        rd_50 = next((cp for cp, wr in zip(checkpoints, rd_win_rates) if wr >= 0.5), None)
        std_50 = next((cp for cp, wr in zip(checkpoints, std_win_rates) if wr >= 0.5), None)
        
        f.write("1. Sample Efficiency:\n")
        if rd_50 and std_50:
            f.write(f"   - RD-MCTS reaches 50% win rate at {rd_50} games\n")
            f.write(f"   - Standard MCTS reaches 50% win rate at {std_50} games\n")
            if rd_50 < std_50:
                speedup = std_50 / rd_50
                f.write(f"   - RD-MCTS is {speedup:.1f}x faster to reach competence\n")
        elif rd_50:
            f.write(f"   - RD-MCTS reaches 50% win rate at {rd_50} games\n")
            f.write(f"   - Standard MCTS never reaches 50% win rate\n")
        else:
            f.write(f"   - Neither agent reaches 50% win rate in 100 games\n")
        
        f.write("\n2. Average Performance:\n")
        rd_mean = np.mean(rd_win_rates)
        std_mean = np.mean(std_win_rates)
        f.write(f"   - RD-MCTS: {rd_mean:.1%}\n")
        f.write(f"   - Standard MCTS: {std_mean:.1%}\n")
        f.write(f"   - Difference: {(rd_mean - std_mean):+.1%}\n")
        
        f.write("\n3. Final Performance (100 games):\n")
        rd_final = rd_win_rates[-1]
        std_final = std_win_rates[-1]
        f.write(f"   - RD-MCTS: {rd_final:.1%}\n")
        f.write(f"   - Standard MCTS: {std_final:.1%}\n")
        f.write(f"   - Difference: {(rd_final - std_final):+.1%}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ Saved {output_path}")


def main():
    """Generate all learning curve visualizations."""
    print("\n" + "="*80)
    print("LEARNING CURVE VISUALIZATION")
    print("="*80)
    
    results_path = "learning_curve_results.json"
    
    if not Path(results_path).exists():
        print(f"\nError: {results_path} not found!")
        print("Please run eval_learning_curve.py first.")
        return
    
    print("\nGenerating visualizations...")
    plot_learning_curves()
    create_summary_report()
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - learning_curve_analysis.png")
    print("  - learning_curve_summary.txt")


if __name__ == "__main__":
    main()
