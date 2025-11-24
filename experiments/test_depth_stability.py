"""
Depth stability test for RD-MCTS.
Tests whether value estimates remain stable or drift as search depth increases.
"""

import torch
import numpy as np
import json
import os
from pathlib import Path

from framework import UnifiedNetwork, CategoricalValueHead, ScalarValueHead, SearchConfig, DEVICE, dist_stats
from search_algorithms import RDMCTSSearch, StandardMCTSSearch
from noise_wrapper import NoisyValueWrapper
from test_positions import create_test_positions, compute_ground_truth_values
from rd_mcts_experiment import Connect4


def test_depth_stability_single_position(
    position_idx: int,
    game: Connect4,
    description: str,
    ground_truth: float,
    model,
    search_class,
    search_config,
    agent_name: str,
    simulation_counts: list,
    noise_level: float = 0.0
):
    """
    Test depth stability on a single position.
    
    Args:
        position_idx: Position index
        game: Game state
        description: Position description
        ground_truth: Ground truth value
        model: Neural network model
        search_class: Search algorithm class
        search_config: Search configuration
        agent_name: Name of the agent
        simulation_counts: List of simulation counts to test
        noise_level: Noise level to inject (0.0 = no noise)
    
    Returns:
        Dictionary with results
    """
    results = {
        "position_idx": position_idx,
        "description": description,
        "ground_truth": float(ground_truth),
        "agent_name": agent_name,
        "noise_level": noise_level,
        "simulation_counts": simulation_counts,
        "mean_values": [],
        "std_values": [],
        "penalized_values": [],
        "best_moves": []
    }
    
    # Wrap model with noise if needed
    if noise_level > 0.0:
        test_model = NoisyValueWrapper(model, noise_std=noise_level)
    else:
        test_model = model
    
    for sim_count in simulation_counts:
        # Run search
        search = search_class(test_model, search_config)
        children = search.search(game, simulations=sim_count)
        
        # Find best move by visits
        best_move = max(children.keys(), key=lambda m: children[m].visits)
        best_child = children[best_move]
        
        # Extract statistics
        if hasattr(best_child, 'dist'):
            # Distributional value head
            mean, std = dist_stats(best_child.dist)
            penalized = mean - search_config.lambda_penalty * std
        else:
            # Scalar value head
            mean = best_child.get_value()
            std = 0.0
            penalized = mean
        
        results["mean_values"].append(float(mean))
        results["std_values"].append(float(std))
        results["penalized_values"].append(float(penalized))
        results["best_moves"].append(int(best_move))
    
    return results


def run_depth_stability_experiment(
    checkpoint_path: str = None,
    use_undertrained: bool = True,
    noise_levels: list = [0.0, 0.2],
    simulation_counts: list = [50, 100, 200, 400, 800, 1600]
):
    """
    Run depth stability experiment.
    
    Args:
        checkpoint_path: Path to checkpoint (if None, uses undertrained network)
        use_undertrained: If True, use undertrained network (high noise)
        noise_levels: List of noise levels to test
        simulation_counts: List of simulation counts to test
    
    Returns:
        Results dictionary
    """
    print("\n" + "="*80)
    print("DEPTH STABILITY EXPERIMENT")
    print("="*80)
    
    # Create test positions
    print("\nCreating test positions...")
    positions = create_test_positions()
    positions = compute_ground_truth_values(positions, depth=6)
    
    print(f"\nTest configuration:")
    print(f"  Positions: {len(positions)}")
    print(f"  Noise levels: {noise_levels}")
    print(f"  Simulation counts: {simulation_counts}")
    print(f"  Use undertrained network: {use_undertrained}")
    
    # Prepare models
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\nLoading checkpoint: {checkpoint_path}")
        # Load from checkpoint
        rd_value_head = CategoricalValueHead()
        rd_model = UnifiedNetwork(rd_value_head).to(DEVICE)
        rd_model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    elif use_undertrained:
        print("\nTraining undertrained network (10 games only)...")
        # Train a weak network with high noise
        from unified_trainer import UnifiedTrainer
        
        rd_value_head = CategoricalValueHead()
        rd_model = UnifiedNetwork(rd_value_head).to(DEVICE)
        rd_search_config = SearchConfig(c_puct=1.0, lambda_penalty=1.0, use_thompson=True, use_penalty=True)
        
        trainer = UnifiedTrainer(rd_model, RDMCTSSearch, rd_search_config)
        loss = trainer.train_iteration(num_games=10, simulations=30)
        print(f"  Training loss: {loss:.4f}")
    else:
        print("\nUsing randomly initialized network...")
        rd_value_head = CategoricalValueHead()
        rd_model = UnifiedNetwork(rd_value_head).to(DEVICE)
    
    rd_model.eval()
    
    # Test configurations
    test_configs = [
        {
            "name": "RD-MCTS (λ=0)",
            "search_class": RDMCTSSearch,
            "search_config": SearchConfig(c_puct=1.0, lambda_penalty=0.0, use_thompson=True, use_penalty=False)
        },
        {
            "name": "RD-MCTS (λ=1)",
            "search_class": RDMCTSSearch,
            "search_config": SearchConfig(c_puct=1.0, lambda_penalty=1.0, use_thompson=True, use_penalty=True)
        },
        {
            "name": "Standard PUCT",
            "search_class": StandardMCTSSearch,
            "search_config": SearchConfig(c_puct=1.0, lambda_penalty=0.0, use_thompson=False, use_penalty=False)
        }
    ]
    
    # Run experiments
    all_results = {
        "positions": [(desc, float(gt)) for _, desc, gt in positions],
        "noise_levels": noise_levels,
        "simulation_counts": simulation_counts,
        "agents": {}
    }
    
    for config in test_configs:
        agent_name = config["name"]
        print(f"\n{'='*80}")
        print(f"TESTING: {agent_name}")
        print(f"{'='*80}")
        
        all_results["agents"][agent_name] = {}
        
        for noise_level in noise_levels:
            print(f"\n  Noise level: σ={noise_level}")
            
            agent_results = []
            
            for pos_idx, (game, description, ground_truth) in enumerate(positions):
                print(f"    Position {pos_idx + 1}/{len(positions)}: {description}")
                
                result = test_depth_stability_single_position(
                    position_idx=pos_idx,
                    game=game,
                    description=description,
                    ground_truth=ground_truth,
                    model=rd_model,
                    search_class=config["search_class"],
                    search_config=config["search_config"],
                    agent_name=agent_name,
                    simulation_counts=simulation_counts,
                    noise_level=noise_level
                )
                
                agent_results.append(result)
                
                # Print summary for this position
                final_mean = result["mean_values"][-1]
                final_std = result["std_values"][-1]
                drift = abs(final_mean - ground_truth)
                print(f"      Final: μ={final_mean:.3f}, σ={final_std:.3f}, drift={drift:.3f}")
            
            all_results["agents"][agent_name][f"noise_{noise_level}"] = agent_results
    
    # Save results
    results_path = "depth_stability_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*80)
    print("DEPTH STABILITY EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {results_path}")
    
    return all_results


def main():
    """Run depth stability experiment."""
    results = run_depth_stability_experiment(
        checkpoint_path=None,
        use_undertrained=True,
        noise_levels=[0.0, 0.2, 0.4],
        simulation_counts=[50, 100, 200, 400, 800, 1600]
    )
    
    print("\nNext step: Run visualize_depth_stability.py to generate plots.")


if __name__ == "__main__":
    main()
