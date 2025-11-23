"""
Visualize results from extended experiments (noise sweep + MCTS comparison).
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def plot_noise_sweep_results():
    """Plot how penalty effectiveness changes with noise level."""
    with open("noise_sweep_results.json", "r") as f:
        results = json.load(f)
    
    noise_levels = [r["noise_std"] for r in results]
    win_rates_lambda_1 = [r["win_rate_lambda_1"] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Win rate vs noise
    ax1.plot(noise_levels, win_rates_lambda_1, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% (neutral)')
    ax1.set_xlabel('Noise Level (σ)', fontsize=12)
    ax1.set_ylabel('Win Rate for RD-MCTS (λ=1)', fontsize=12)
    ax1.set_title('Penalty Effectiveness vs Noise Level', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0, 1])
    
    # Add annotations
    for i, (noise, wr) in enumerate(zip(noise_levels, win_rates_lambda_1)):
        ax1.annotate(f'{wr:.0%}', (noise, wr), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    # Plot 2: Win/Loss/Draw breakdown
    width = 0.06
    x = np.array(noise_levels)
    
    for i, r in enumerate(results):
        wins_l0 = r["wins_lambda_0"]
        wins_l1 = r["wins_lambda_1"]
        draws = r["draws"]
        total = wins_l0 + wins_l1 + draws
        
        ax2.bar(x[i] - width, wins_l0/total, width, label='λ=0 wins' if i == 0 else '', color='#A23B72')
        ax2.bar(x[i], wins_l1/total, width, label='λ=1 wins' if i == 0 else '', color='#2E86AB')
        ax2.bar(x[i] + width, draws/total, width, label='Draws' if i == 0 else '', color='#F18F01')
    
    ax2.set_xlabel('Noise Level (σ)', fontsize=12)
    ax2.set_ylabel('Proportion of Games', fontsize=12)
    ax2.set_title('Game Outcomes by Noise Level', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('noise_sweep_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved noise_sweep_analysis.png")
    plt.close()

def plot_mcts_comparison_results():
    """Plot round-robin tournament results."""
    with open("mcts_comparison_results.json", "r") as f:
        results = json.load(f)
    
    # Calculate overall win rates
    algorithms = [
        "Standard PUCT", "Thompson Scalar", "UCB-V", "RAVE",
        "RD-MCTS (λ=0)", "RD-MCTS (λ=1)", "RD-MCTS (λ=2)"
    ]
    
    win_counts = {name: 0 for name in algorithms}
    total_games = {name: 0 for name in algorithms}
    
    for matchup_key, result in results.items():
        name1 = result["player1"]
        name2 = result["player2"]
        
        win_counts[name1] += result["wins1"]
        win_counts[name2] += result["wins2"]
        total_games[name1] += 20
        total_games[name2] += 20
    
    win_rates = {name: win_counts[name] / total_games[name] for name in algorithms}
    
    # Sort by win rate
    sorted_algos = sorted(algorithms, key=lambda x: win_rates[x], reverse=True)
    sorted_win_rates = [win_rates[name] for name in sorted_algos]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Overall win rates
    colors = ['#2E86AB' if 'RD-MCTS' in name else '#A23B72' if 'Thompson' in name 
              else '#F18F01' if name in ['UCB-V', 'RAVE'] else '#C73E1D' 
              for name in sorted_algos]
    
    bars = ax1.barh(range(len(sorted_algos)), sorted_win_rates, color=colors)
    ax1.set_yticks(range(len(sorted_algos)))
    ax1.set_yticklabels(sorted_algos)
    ax1.set_xlabel('Win Rate', fontsize=12)
    ax1.set_title('MCTS Algorithms: Overall Win Rates', fontsize=14, fontweight='bold')
    ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add win rate labels
    for i, (bar, wr) in enumerate(zip(bars, sorted_win_rates)):
        ax1.text(wr + 0.01, i, f'{wr:.1%}', va='center', fontsize=10)
    
    # Plot 2: Head-to-head heatmap (RD-MCTS variants vs others)
    rd_variants = ["RD-MCTS (λ=0)", "RD-MCTS (λ=1)", "RD-MCTS (λ=2)"]
    baselines = ["Standard PUCT", "Thompson Scalar", "UCB-V", "RAVE"]
    
    heatmap_data = np.zeros((len(rd_variants), len(baselines)))
    
    for i, rd_name in enumerate(rd_variants):
        for j, baseline_name in enumerate(baselines):
            # Find the matchup
            for matchup_key, result in results.items():
                if (result["player1"] == rd_name and result["player2"] == baseline_name):
                    heatmap_data[i, j] = result["wins1"] / 20
                    break
                elif (result["player1"] == baseline_name and result["player2"] == rd_name):
                    heatmap_data[i, j] = result["wins2"] / 20
                    break
    
    im = ax2.imshow(heatmap_data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    
    ax2.set_xticks(range(len(baselines)))
    ax2.set_yticks(range(len(rd_variants)))
    ax2.set_xticklabels(baselines, rotation=45, ha='right')
    ax2.set_yticklabels(rd_variants)
    ax2.set_title('RD-MCTS Variants vs Baselines\n(Win Rate)', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(rd_variants)):
        for j in range(len(baselines)):
            text = ax2.text(j, i, f'{heatmap_data[i, j]:.0%}',
                           ha="center", va="center", color="black", fontsize=11)
    
    plt.colorbar(im, ax=ax2, label='Win Rate')
    
    plt.tight_layout()
    plt.savefig('mcts_comparison_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved mcts_comparison_analysis.png")
    plt.close()

def create_summary_report():
    """Create a summary report of extended experiments."""
    with open("noise_sweep_results.json", "r") as f:
        noise_results = json.load(f)
    
    with open("mcts_comparison_results.json", "r") as f:
        comparison_results = json.load(f)
    
    report = []
    report.append("="*80)
    report.append("EXTENDED EXPERIMENTS SUMMARY REPORT")
    report.append("="*80)
    
    report.append("\n## EXPERIMENT 1: Noise Sweep (When Does Penalty Help?)")
    report.append("\nTesting RD-MCTS (λ=0) vs RD-MCTS (λ=1) at different noise levels:\n")
    
    for r in noise_results:
        noise = r["noise_std"]
        wr_l1 = r["win_rate_lambda_1"]
        wins_l0 = r["wins_lambda_0"]
        wins_l1 = r["wins_lambda_1"]
        draws = r["draws"]
        
        report.append(f"Noise σ={noise}:")
        report.append(f"  λ=0: {wins_l0} wins, λ=1: {wins_l1} wins, draws: {draws}")
        report.append(f"  Win rate for λ=1: {wr_l1:.1%}")
        
        if wr_l1 > 0.55:
            report.append(f"  → Penalty HELPS at this noise level")
        elif wr_l1 < 0.45:
            report.append(f"  → Penalty HURTS at this noise level")
        else:
            report.append(f"  → Penalty has NEUTRAL effect")
        report.append("")
    
    report.append("\n## EXPERIMENT 2: RD-MCTS vs Other MCTS Optimizations")
    report.append("\nRound-robin tournament with fixed network:\n")
    
    # Calculate overall win rates
    algorithms = [
        "Standard PUCT", "Thompson Scalar", "UCB-V", "RAVE",
        "RD-MCTS (λ=0)", "RD-MCTS (λ=1)", "RD-MCTS (λ=2)"
    ]
    
    win_counts = {name: 0 for name in algorithms}
    total_games = {name: 0 for name in algorithms}
    
    for matchup_key, result in comparison_results.items():
        name1 = result["player1"]
        name2 = result["player2"]
        
        win_counts[name1] += result["wins1"]
        win_counts[name2] += result["wins2"]
        total_games[name1] += 20
        total_games[name2] += 20
    
    win_rates = {name: win_counts[name] / total_games[name] for name in algorithms}
    
    # Sort by win rate
    sorted_algos = sorted(algorithms, key=lambda x: win_rates[x], reverse=True)
    
    report.append("Overall Rankings:")
    for i, name in enumerate(sorted_algos, 1):
        wr = win_rates[name]
        wins = win_counts[name]
        total = total_games[name]
        report.append(f"  {i}. {name}: {wr:.1%} ({wins}/{total} wins)")
    
    report.append("\n## KEY FINDINGS")
    
    # Find best RD-MCTS variant
    rd_variants = ["RD-MCTS (λ=0)", "RD-MCTS (λ=1)", "RD-MCTS (λ=2)"]
    best_rd = max(rd_variants, key=lambda x: win_rates[x])
    
    report.append(f"\n1. Best RD-MCTS variant: {best_rd} ({win_rates[best_rd]:.1%} win rate)")
    
    # Compare RD-MCTS to baselines
    baselines = ["Standard PUCT", "UCB-V", "RAVE"]
    avg_baseline_wr = np.mean([win_rates[name] for name in baselines])
    avg_rd_wr = np.mean([win_rates[name] for name in rd_variants])
    
    report.append(f"\n2. RD-MCTS variants average: {avg_rd_wr:.1%}")
    report.append(f"   Baselines average: {avg_baseline_wr:.1%}")
    
    if avg_rd_wr > avg_baseline_wr + 0.05:
        report.append("   → RD-MCTS significantly outperforms baselines")
    elif avg_rd_wr > avg_baseline_wr:
        report.append("   → RD-MCTS slightly outperforms baselines")
    else:
        report.append("   → RD-MCTS comparable to baselines")
    
    # Noise sweep conclusion
    high_noise_wr = noise_results[-1]["win_rate_lambda_1"]  # Highest noise level
    low_noise_wr = noise_results[0]["win_rate_lambda_1"]   # No noise
    
    report.append(f"\n3. Penalty effectiveness:")
    report.append(f"   At σ=0.0: λ=1 wins {low_noise_wr:.0%}")
    report.append(f"   At σ={noise_results[-1]['noise_std']}: λ=1 wins {high_noise_wr:.0%}")
    
    if high_noise_wr > low_noise_wr + 0.1:
        report.append("   → Penalty becomes MORE helpful with increased noise")
    elif high_noise_wr < low_noise_wr - 0.1:
        report.append("   → Penalty becomes LESS helpful with increased noise")
    else:
        report.append("   → Penalty effect is relatively stable across noise levels")
    
    report.append("\n" + "="*80)
    
    report_text = "\n".join(report)
    
    with open("extended_experiments_summary.txt", "w") as f:
        f.write(report_text)
    
    print("\n" + report_text)
    print("\n✓ Saved extended_experiments_summary.txt")

if __name__ == "__main__":
    print("Generating visualizations for extended experiments...")
    
    try:
        plot_noise_sweep_results()
        plot_mcts_comparison_results()
        create_summary_report()
        
        print("\n" + "="*80)
        print("ALL VISUALIZATIONS COMPLETE!")
        print("="*80)
        print("\nGenerated files:")
        print("  - noise_sweep_analysis.png")
        print("  - mcts_comparison_analysis.png")
        print("  - extended_experiments_summary.txt")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Make sure extended experiments have completed and generated result files.")
