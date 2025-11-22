"""
Quick validation experiment to test the framework end-to-end.
Runs a minimal version of the comprehensive experiments.
"""

import torch
import numpy as np

from unified_trainer import (
    ExperimentConfig, run_single_experiment, play_match, save_results
)

def run_quick_validation():
    """Run a quick validation with minimal training."""
    print("="*80)
    print("QUICK VALIDATION EXPERIMENT")
    print("="*80)
    print("Testing framework with minimal training (2 iterations, 5 games each)")
    print()
    
    configs = [
        ExperimentConfig(
            name="Categorical + RD-MCTS",
            value_head_type="categorical",
            search_type="rd_mcts",
            training_mode="separate",
            num_iterations=2,
            games_per_iteration=5,
            simulations_train=20,
            simulations_eval=30
        ),
        ExperimentConfig(
            name="Scalar + Standard MCTS",
            value_head_type="scalar",
            search_type="standard",
            training_mode="separate",
            num_iterations=2,
            games_per_iteration=5,
            simulations_train=20,
            simulations_eval=30
        ),
    ]
    
    agents = []
    all_results = {}
    
    for config in configs:
        model, search_class, search_config, results = run_single_experiment(config)
        agents.append((config.name, model, search_class, search_config))
        all_results[config.name] = results
    
    print("\n" + "="*80)
    print("HEAD-TO-HEAD MATCH")
    print("="*80)
    
    name1, model1, search1, config1 = agents[0]
    name2, model2, search2, config2 = agents[1]
    
    print(f"\n{name1} vs {name2}")
    wins = play_match(model1, search1, config1, model2, search2, config2, num_games=6, simulations=30)
    print(f"  {name1}: {wins[0]} wins")
    print(f"  {name2}: {wins[1]} wins")
    print(f"  Draws: {wins[2]}")
    
    all_results[f"{name1}_vs_{name2}"] = {
        'agent1_wins': wins[0],
        'agent2_wins': wins[1],
        'draws': wins[2]
    }
    
    save_results(all_results, 'quick_validation_results.json')
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE!")
    print("="*80)
    print("Results saved to: quick_validation_results.json")
    print("\nFramework is working correctly!")
    
    return all_results

if __name__ == "__main__":
    run_quick_validation()
