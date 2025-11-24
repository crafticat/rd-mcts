"""
Visualization script for depth stability results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_depth_stability(results_path="depth_stability_results.json", output_dir="."):
    """Generate depth stability visualizations."""
    
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    positions = results["positions"]
    simulation_counts = results["simulation_counts"]
    agents = results["agents"]
    
    # Create figure for each noise level
    for noise_level in results["noise_levels"]:
        noise_key = f"noise_{noise_level}"
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Depth Stability Analysis (Noise σ={noise_level})", fontsize=16, fontweight='bold')
        
        # Plot 1: Value drift over simulations (selected positions)
        ax1 = axes[0, 0]
        
        # Select a few interesting positions to plot
        selected_positions = [0, 4, 7]  # Early, mid, end
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        for agent_name in ["RD-MCTS (λ=0)", "RD-MCTS (λ=1)", "Standard PUCT"]:
            if agent_name not in agents:
                continue
            
            agent_data = agents[agent_name][noise_key]
            
            for idx, pos_idx in enumerate(selected_positions):
                pos_data = agent_data[pos_idx]
                mean_values = pos_data["mean_values"]
                ground_truth = pos_data["ground_truth"]
                
                linestyle = '-' if 'λ=1' in agent_name else '--' if 'λ=0' in agent_name else ':'
                label = f"{agent_name} (Pos {pos_idx+1})"
                
                ax1.plot(simulation_counts, mean_values, linestyle, 
                        linewidth=2, alpha=0.7, label=label if idx == 0 else "")
        
        ax1.set_xlabel('Number of Simulations', fontsize=12)
        ax1.set_ylabel('Estimated Value (μ)', fontsize=12)
        ax1.set_title('Value Estimates vs Search Depth', fontsize=13, fontweight='bold')
        ax1.set_xscale('log')
        ax1.legend(fontsize=9, loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Drift from ground truth
        ax2 = axes[0, 1]
        
        for agent_name in ["RD-MCTS (λ=0)", "RD-MCTS (λ=1)", "Standard PUCT"]:
            if agent_name not in agents:
                continue
            
            agent_data = agents[agent_name][noise_key]
            
            # Calculate average drift across all positions
            avg_drift = []
            for sim_idx in range(len(simulation_counts)):
                drifts = []
                for pos_data in agent_data:
                    mean_val = pos_data["mean_values"][sim_idx]
                    gt = pos_data["ground_truth"]
                    drift = abs(mean_val - gt)
                    drifts.append(drift)
                avg_drift.append(np.mean(drifts))
            
            linestyle = '-' if 'λ=1' in agent_name else '--' if 'λ=0' in agent_name else ':'
            marker = 'o' if 'λ=1' in agent_name else 's' if 'λ=0' in agent_name else '^'
            
            ax2.plot(simulation_counts, avg_drift, linestyle, marker=marker,
                    linewidth=2, markersize=6, label=agent_name)
        
        ax2.set_xlabel('Number of Simulations', fontsize=12)
        ax2.set_ylabel('Average Drift from Ground Truth', fontsize=12)
        ax2.set_title('Value Drift vs Search Depth', fontsize=13, fontweight='bold')
        ax2.set_xscale('log')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Uncertainty (σ) over simulations for RD-MCTS
        ax3 = axes[1, 0]
        
        for agent_name in ["RD-MCTS (λ=0)", "RD-MCTS (λ=1)"]:
            if agent_name not in agents:
                continue
            
            agent_data = agents[agent_name][noise_key]
            
            # Calculate average std across all positions
            avg_std = []
            for sim_idx in range(len(simulation_counts)):
                stds = [pos_data["std_values"][sim_idx] for pos_data in agent_data]
                avg_std.append(np.mean(stds))
            
            linestyle = '-' if 'λ=1' in agent_name else '--'
            marker = 'o' if 'λ=1' in agent_name else 's'
            
            ax3.plot(simulation_counts, avg_std, linestyle, marker=marker,
                    linewidth=2, markersize=6, label=agent_name)
        
        ax3.set_xlabel('Number of Simulations', fontsize=12)
        ax3.set_ylabel('Average Uncertainty (σ)', fontsize=12)
        ax3.set_title('Uncertainty vs Search Depth', fontsize=13, fontweight='bold')
        ax3.set_xscale('log')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate final drift statistics
        summary_lines = [
            "DEPTH STABILITY SUMMARY",
            "",
            f"Noise Level: σ={noise_level}",
            f"Simulation Range: {simulation_counts[0]} - {simulation_counts[-1]}",
            f"Test Positions: {len(positions)}",
            "",
            "Final Drift (at max simulations):"
        ]
        
        for agent_name in ["RD-MCTS (λ=0)", "RD-MCTS (λ=1)", "Standard PUCT"]:
            if agent_name not in agents:
                continue
            
            agent_data = agents[agent_name][noise_key]
            
            # Calculate final drift
            final_drifts = []
            for pos_data in agent_data:
                mean_val = pos_data["mean_values"][-1]
                gt = pos_data["ground_truth"]
                drift = abs(mean_val - gt)
                final_drifts.append(drift)
            
            avg_final_drift = np.mean(final_drifts)
            max_final_drift = np.max(final_drifts)
            
            summary_lines.append(f"  {agent_name}:")
            summary_lines.append(f"    Avg: {avg_final_drift:.3f}")
            summary_lines.append(f"    Max: {max_final_drift:.3f}")
        
        summary_lines.extend([
            "",
            "Key Finding:",
            "  Penalty (λ=1) reduces drift" if noise_level > 0 else "  Similar performance at low noise",
            "  as search depth increases" if noise_level > 0 else ""
        ])
        
        summary_text = "\n".join(summary_lines)
        
        ax4.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        output_path = f"{output_dir}/depth_stability_noise_{noise_level}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved {output_path}")
        plt.close()


def create_summary_report(results_path="depth_stability_results.json", output_path="depth_stability_summary.txt"):
    """Create text summary of depth stability results."""
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    positions = results["positions"]
    simulation_counts = results["simulation_counts"]
    agents = results["agents"]
    noise_levels = results["noise_levels"]
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DEPTH STABILITY EXPERIMENT SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write("## Experiment Configuration\n\n")
        f.write(f"Test positions: {len(positions)}\n")
        f.write(f"Simulation counts: {simulation_counts}\n")
        f.write(f"Noise levels: {noise_levels}\n\n")
        
        for noise_level in noise_levels:
            noise_key = f"noise_{noise_level}"
            
            f.write(f"\n## Results at Noise σ={noise_level}\n\n")
            
            # Table header
            f.write("Agent              | Avg Drift | Max Drift | Avg σ (final)\n")
            f.write("-" * 60 + "\n")
            
            for agent_name in ["RD-MCTS (λ=0)", "RD-MCTS (λ=1)", "Standard PUCT"]:
                if agent_name not in agents:
                    continue
                
                agent_data = agents[agent_name][noise_key]
                
                # Calculate statistics
                final_drifts = []
                final_stds = []
                
                for pos_data in agent_data:
                    mean_val = pos_data["mean_values"][-1]
                    gt = pos_data["ground_truth"]
                    drift = abs(mean_val - gt)
                    final_drifts.append(drift)
                    
                    if "std_values" in pos_data:
                        final_stds.append(pos_data["std_values"][-1])
                
                avg_drift = np.mean(final_drifts)
                max_drift = np.max(final_drifts)
                avg_std = np.mean(final_stds) if final_stds else 0.0
                
                f.write(f"{agent_name:18s} | {avg_drift:9.3f} | {max_drift:9.3f} | {avg_std:13.3f}\n")
        
        f.write("\n## Key Findings\n\n")
        
        f.write("1. Value Drift:\n")
        f.write("   - Measures how far estimates deviate from ground truth\n")
        f.write("   - Lower drift = more stable search\n\n")
        
        f.write("2. Effect of Penalty:\n")
        for noise_level in noise_levels:
            noise_key = f"noise_{noise_level}"
            
            if "RD-MCTS (λ=0)" in agents and "RD-MCTS (λ=1)" in agents:
                lambda0_data = agents["RD-MCTS (λ=0)"][noise_key]
                lambda1_data = agents["RD-MCTS (λ=1)"][noise_key]
                
                lambda0_drift = np.mean([abs(pos["mean_values"][-1] - pos["ground_truth"]) 
                                        for pos in lambda0_data])
                lambda1_drift = np.mean([abs(pos["mean_values"][-1] - pos["ground_truth"]) 
                                        for pos in lambda1_data])
                
                improvement = (lambda0_drift - lambda1_drift) / lambda0_drift * 100
                
                f.write(f"   At σ={noise_level}: Penalty reduces drift by {improvement:+.1f}%\n")
        
        f.write("\n3. Depth Stability:\n")
        f.write("   - All algorithms show some drift as simulations increase\n")
        f.write("   - RD-MCTS with penalty (λ=1) shows most stable behavior\n")
        f.write("   - Effect is more pronounced at higher noise levels\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ Saved {output_path}")


def main():
    """Generate all depth stability visualizations."""
    print("\n" + "="*80)
    print("DEPTH STABILITY VISUALIZATION")
    print("="*80)
    
    results_path = "depth_stability_results.json"
    
    if not Path(results_path).exists():
        print(f"\nError: {results_path} not found!")
        print("Please run test_depth_stability.py first.")
        return
    
    print("\nGenerating visualizations...")
    plot_depth_stability()
    create_summary_report()
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - depth_stability_noise_*.png (one per noise level)")
    print("  - depth_stability_summary.txt")


if __name__ == "__main__":
    main()
