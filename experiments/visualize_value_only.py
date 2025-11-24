"""
Visualize value-only learning curve results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import matplotlib.pyplot as plt
import numpy as np


def main():
    print("="*80)
    print("VALUE-ONLY LEARNING CURVE VISUALIZATION")
    print("="*80)
    
    # Load results
    with open("../learning_curve_value_only_results.json", "r") as f:
        results = json.load(f)
    
    # Extract data
    rd_checkpoints = sorted([int(k) for k in results["rd_mcts"].keys()])
    rd_win_rates = [results["rd_mcts"][str(c)]["win_rate"] for c in rd_checkpoints]
    
    scalar_checkpoints = sorted([int(k) for k in results["scalar_mcts"].keys()])
    scalar_win_rates = [results["scalar_mcts"][str(c)]["win_rate"] for c in scalar_checkpoints]
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(rd_checkpoints, rd_win_rates, 'o-', label='RD-MCTS (Value-Only)', 
            color='#2E86AB', linewidth=2, markersize=8)
    ax.plot(scalar_checkpoints, scalar_win_rates, 's-', label='Scalar MCTS (Value-Only)', 
            color='#A23B72', linewidth=2, markersize=8)
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Play')
    
    ax.set_xlabel('Training Games', fontsize=12, fontweight='bold')
    ax.set_ylabel('Win Rate vs Minimax (depth=4)', fontsize=12, fontweight='bold')
    ax.set_title('Learning Curve: Value-Only Networks\n(Pure Thompson Sampling, No Policy Network)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('../learning_curve_value_only.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: learning_curve_value_only.png")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    rd_avg = np.mean(rd_win_rates)
    scalar_avg = np.mean(scalar_win_rates)
    
    print(f"\nRD-MCTS (Value-Only):")
    print(f"  Average win rate: {rd_avg:.1%}")
    print(f"  Best win rate: {max(rd_win_rates):.1%} (at {rd_checkpoints[rd_win_rates.index(max(rd_win_rates))]} games)")
    
    print(f"\nScalar MCTS (Value-Only):")
    print(f"  Average win rate: {scalar_avg:.1%}")
    print(f"  Best win rate: {max(scalar_win_rates):.1%} (at {scalar_checkpoints[scalar_win_rates.index(max(scalar_win_rates))]} games)")
    
    print(f"\nRD-MCTS advantage: {(rd_avg - scalar_avg) * 100:.1f} percentage points")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
