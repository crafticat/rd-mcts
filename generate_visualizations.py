"""
Generate comprehensive visualizations from experiment results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(filename):
    """Load results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def plot_training_curves(results, output_file='training_curves.png'):
    """Plot training loss curves for all experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Loss Curves Across Experiments', fontsize=16, fontweight='bold')
    
    # Suite 1: Value Representations
    ax = axes[0, 0]
    for name in ['Categorical + RD-MCTS', 'Gaussian + RD-MCTS', 'Scalar + RD-MCTS']:
        if name in results and 'training_losses' in results[name]:
            losses = results[name]['training_losses']
            if losses:
                ax.plot(range(1, len(losses) + 1), losses, marker='o', label=name, linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Suite 1: Value Representation Comparison', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Suite 2: Search Algorithms
    ax = axes[0, 1]
    suite2_names = ['Categorical + RD-MCTS', 'Categorical + Standard MCTS', 
                    'Categorical + RD-MCTS (no penalty)', 'Scalar + Thompson MCTS']
    for name in suite2_names:
        if name in results and 'training_losses' in results[name]:
            losses = results[name]['training_losses']
            if losses:
                ax.plot(range(1, len(losses) + 1), losses, marker='s', label=name, linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Suite 2: Search Algorithm Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Combined view
    ax = axes[1, 0]
    all_experiments = {}
    for suite_results in [results]:
        for name, data in suite_results.items():
            if isinstance(data, dict) and 'training_losses' in data:
                losses = data['training_losses']
                if losses:
                    all_experiments[name] = losses
    
    for name, losses in all_experiments.items():
        ax.plot(range(1, len(losses) + 1), losses, marker='o', label=name[:30], linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('All Experiments Combined', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Loss convergence rate
    ax = axes[1, 1]
    for name, losses in all_experiments.items():
        if len(losses) >= 2:
            improvement = [(losses[0] - losses[i]) / losses[0] * 100 for i in range(len(losses))]
            ax.plot(range(1, len(improvement) + 1), improvement, marker='o', label=name[:30], linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss Improvement (%)', fontsize=12)
    ax.set_title('Training Convergence Rate', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved training curves to {output_file}")
    plt.close()

def plot_head_to_head_results(results, output_file='head_to_head_results.png'):
    """Plot head-to-head match results."""
    matches = {}
    for key, value in results.items():
        if '_vs_' in key and isinstance(value, dict):
            if 'agent1_wins' in value:
                matches[key] = value
    
    if not matches:
        print("No head-to-head matches found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Head-to-Head Match Results', fontsize=16, fontweight='bold')
    
    # Bar chart
    ax = axes[0]
    match_names = list(matches.keys())
    agent1_wins = [matches[m]['agent1_wins'] for m in match_names]
    agent2_wins = [matches[m]['agent2_wins'] for m in match_names]
    draws = [matches[m]['draws'] for m in match_names]
    
    x = np.arange(len(match_names))
    width = 0.25
    
    ax.bar(x - width, agent1_wins, width, label='Agent 1 Wins', color='#2ecc71')
    ax.bar(x, agent2_wins, width, label='Agent 2 Wins', color='#e74c3c')
    ax.bar(x + width, draws, width, label='Draws', color='#95a5a6')
    
    ax.set_xlabel('Match', fontsize=12)
    ax.set_ylabel('Number of Games', fontsize=12)
    ax.set_title('Win/Loss/Draw Distribution', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' vs ', '\nvs\n')[:50] for m in match_names], rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Win rate percentages
    ax = axes[1]
    win_rates = []
    labels = []
    for match_name in match_names:
        m = matches[match_name]
        total = m['agent1_wins'] + m['agent2_wins'] + m['draws']
        if total > 0:
            agent1_rate = m['agent1_wins'] / total * 100
            win_rates.append(agent1_rate)
            labels.append(match_name.replace(' vs ', '\nvs\n')[:50])
    
    colors = ['#2ecc71' if wr > 50 else '#e74c3c' if wr < 50 else '#95a5a6' for wr in win_rates]
    ax.barh(range(len(win_rates)), win_rates, color=colors, alpha=0.7)
    ax.set_xlabel('Agent 1 Win Rate (%)', fontsize=12)
    ax.set_title('Agent 1 Win Rate by Match', fontsize=13, fontweight='bold')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(x=50, color='k', linestyle='--', alpha=0.3, label='50% (Even)')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved head-to-head results to {output_file}")
    plt.close()

def plot_trap_avoidance(results, output_file='trap_avoidance.png'):
    """Plot trap avoidance analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Trap Avoidance Analysis', fontsize=16, fontweight='bold')
    
    experiments = []
    for name, data in results.items():
        if isinstance(data, dict) and 'trap_initial' in data and 'trap_final' in data:
            experiments.append((name, data))
    
    if not experiments:
        print("No trap avoidance data found")
        return
    
    # Initial vs Final trap avoidance
    ax = axes[0, 0]
    for name, data in experiments[:6]:
        trap_initial = data['trap_initial']
        trap_final = data['trap_final']
        
        if trap_initial and trap_final:
            moves_initial = sorted(trap_initial.keys())
            moves_final = sorted(trap_final.keys())
            
            if moves_initial:
                scores_initial = [trap_initial[str(m)].get('score', trap_initial[str(m)].get('value', 0)) 
                                 for m in moves_initial]
                ax.plot(moves_initial, scores_initial, marker='o', linestyle='--', 
                       label=f"{name[:25]} (initial)", alpha=0.5)
            
            if moves_final:
                scores_final = [trap_final[str(m)].get('score', trap_final[str(m)].get('value', 0)) 
                               for m in moves_final]
                ax.plot(moves_final, scores_final, marker='s', linestyle='-', 
                       label=f"{name[:25]} (final)", linewidth=2)
    
    ax.set_xlabel('Move', fontsize=12)
    ax.set_ylabel('Score (Mean - Std)', fontsize=12)
    ax.set_title('Trap Position Evaluation: Initial vs Final', fontsize=13, fontweight='bold')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Variance/uncertainty analysis
    ax = axes[0, 1]
    for name, data in experiments[:6]:
        trap_final = data['trap_final']
        if trap_final:
            moves = sorted(trap_final.keys())
            stds = [trap_final[str(m)].get('std', 0.1) for m in moves]
            ax.plot(moves, stds, marker='o', label=name[:25], linewidth=2)
    
    ax.set_xlabel('Move', fontsize=12)
    ax.set_ylabel('Uncertainty (Std Dev)', fontsize=12)
    ax.set_title('Value Uncertainty by Move', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Visit distribution
    ax = axes[1, 0]
    for name, data in experiments[:6]:
        trap_final = data['trap_final']
        if trap_final:
            moves = sorted(trap_final.keys())
            visits = [trap_final[str(m)].get('visits', 0) for m in moves]
            ax.bar([f"{name[:15]}\nMove {m}" for m in moves], visits, alpha=0.7, label=name[:25])
    
    ax.set_xlabel('Experiment & Move', fontsize=12)
    ax.set_ylabel('Visits', fontsize=12)
    ax.set_title('MCTS Visit Distribution in Trap Position', fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Score improvement
    ax = axes[1, 1]
    improvements = []
    labels = []
    for name, data in experiments:
        trap_initial = data['trap_initial']
        trap_final = data['trap_final']
        
        if trap_initial and trap_final:
            moves = [m for m in trap_initial.keys() if m in trap_final]
            if moves:
                initial_scores = [trap_initial[m].get('score', trap_initial[m].get('value', 0)) for m in moves]
                final_scores = [trap_final[m].get('score', trap_final[m].get('value', 0)) for m in moves]
                
                avg_improvement = np.mean([f - i for i, f in zip(initial_scores, final_scores)])
                improvements.append(avg_improvement)
                labels.append(name[:30])
    
    if improvements:
        colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
        ax.barh(range(len(improvements)), improvements, color=colors, alpha=0.7)
        ax.set_xlabel('Average Score Improvement', fontsize=12)
        ax.set_title('Trap Avoidance Learning', fontsize=13, fontweight='bold')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved trap avoidance analysis to {output_file}")
    plt.close()

def plot_value_representation_comparison(results, output_file='value_representation_comparison.png'):
    """Compare different value representations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Value Representation Comparison', fontsize=16, fontweight='bold')
    
    # Extract categorical, gaussian, and scalar experiments
    categorical_exp = None
    gaussian_exp = None
    scalar_exp = None
    
    for name, data in results.items():
        if 'Categorical' in name and 'RD-MCTS' in name and isinstance(data, dict):
            if 'training_losses' in data:
                categorical_exp = (name, data)
        elif 'Gaussian' in name and 'RD-MCTS' in name and isinstance(data, dict):
            if 'training_losses' in data:
                gaussian_exp = (name, data)
        elif 'Scalar' in name and 'RD-MCTS' in name and isinstance(data, dict):
            if 'training_losses' in data:
                scalar_exp = (name, data)
    
    # Training efficiency
    ax = axes[0, 0]
    for exp in [categorical_exp, gaussian_exp, scalar_exp]:
        if exp:
            name, data = exp
            losses = data['training_losses']
            ax.plot(range(1, len(losses) + 1), losses, marker='o', label=name.split('+')[0].strip(), linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Efficiency by Value Representation', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Final loss comparison
    ax = axes[0, 1]
    final_losses = []
    labels = []
    for exp in [categorical_exp, gaussian_exp, scalar_exp]:
        if exp:
            name, data = exp
            losses = data['training_losses']
            if losses:
                final_losses.append(losses[-1])
                labels.append(name.split('+')[0].strip())
    
    if final_losses:
        colors = ['#3498db', '#e67e22', '#9b59b6']
        ax.bar(labels, final_losses, color=colors, alpha=0.7)
        ax.set_ylabel('Final Loss', fontsize=12)
        ax.set_title('Final Training Loss', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Trap avoidance effectiveness
    ax = axes[1, 0]
    for exp in [categorical_exp, gaussian_exp, scalar_exp]:
        if exp:
            name, data = exp
            trap_final = data.get('trap_final', {})
            if trap_final:
                moves = sorted(trap_final.keys())
                scores = [trap_final[str(m)].get('score', trap_final[str(m)].get('value', 0)) for m in moves]
                ax.plot(moves, scores, marker='o', label=name.split('+')[0].strip(), linewidth=2)
    ax.set_xlabel('Move', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Trap Position Evaluation', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Win rates
    ax = axes[1, 1]
    match_results = {}
    for key, value in results.items():
        if '_vs_' in key and isinstance(value, dict) and 'agent1_wins' in value:
            if any(rep in key for rep in ['Categorical', 'Gaussian', 'Scalar']):
                match_results[key] = value
    
    if match_results:
        win_rates = {}
        for match_name, match_data in match_results.items():
            agents = match_name.split(' vs ')
            if len(agents) == 2:
                agent1 = agents[0].split('+')[0].strip()
                agent2 = agents[1].split('+')[0].strip()
                
                total = match_data['agent1_wins'] + match_data['agent2_wins'] + match_data['draws']
                if total > 0:
                    if agent1 not in win_rates:
                        win_rates[agent1] = []
                    if agent2 not in win_rates:
                        win_rates[agent2] = []
                    
                    win_rates[agent1].append(match_data['agent1_wins'] / total * 100)
                    win_rates[agent2].append(match_data['agent2_wins'] / total * 100)
        
        avg_win_rates = {k: np.mean(v) for k, v in win_rates.items()}
        if avg_win_rates:
            labels = list(avg_win_rates.keys())
            values = list(avg_win_rates.values())
            colors = ['#3498db', '#e67e22', '#9b59b6'][:len(labels)]
            ax.bar(labels, values, color=colors, alpha=0.7)
            ax.set_ylabel('Average Win Rate (%)', fontsize=12)
            ax.set_title('Average Win Rate by Value Representation', fontsize=13, fontweight='bold')
            ax.axhline(y=50, color='k', linestyle='--', alpha=0.3, label='50% baseline')
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved value representation comparison to {output_file}")
    plt.close()

def generate_all_visualizations():
    """Generate all visualizations from experiment results."""
    print("="*80)
    print("GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("="*80)
    
    # Load all results
    all_results = {}
    
    for suite_file in ['suite1_value_representations.json', 'suite2_search_algorithms.json', 'suite3_fixed_network.json']:
        if Path(suite_file).exists():
            print(f"\nLoading {suite_file}...")
            results = load_results(suite_file)
            all_results.update(results)
    
    if not all_results:
        print("No results found! Make sure experiments have completed.")
        return
    
    print(f"\nLoaded {len(all_results)} experiment results")
    
    # Generate all plots
    print("\n" + "="*80)
    print("Generating plots...")
    print("="*80)
    
    plot_training_curves(all_results, 'training_curves.png')
    plot_head_to_head_results(all_results, 'head_to_head_results.png')
    plot_trap_avoidance(all_results, 'trap_avoidance.png')
    plot_value_representation_comparison(all_results, 'value_representation_comparison.png')
    
    print("\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - training_curves.png")
    print("  - head_to_head_results.png")
    print("  - trap_avoidance.png")
    print("  - value_representation_comparison.png")

if __name__ == "__main__":
    generate_all_visualizations()
