"""
Comprehensive experiment runner for testing multiple MCTS variants.
"""

import torch
import numpy as np
from typing import List, Dict
import time
from dataclasses import asdict

from framework import SearchConfig, DEVICE
from search_algorithms import RDMCTSSearch, StandardMCTSSearch
from unified_trainer import (
    ExperimentConfig, evaluate_trap_avoidance,
    play_match, save_results, run_single_experiment
)

def run_experiment_suite_1():
    """
    Experiment Suite 1: Different value representations with RD-MCTS
    Tests: Categorical vs Gaussian vs Scalar, all using RD-MCTS search
    """
    print("\n" + "="*80)
    print("EXPERIMENT SUITE 1: Value Representation Comparison")
    print("="*80)
    
    configs = [
        ExperimentConfig(
            name="Categorical + RD-MCTS",
            value_head_type="categorical",
            search_type="rd_mcts",
            training_mode="separate",
            num_iterations=5,
            games_per_iteration=10
        ),
        ExperimentConfig(
            name="Gaussian + RD-MCTS",
            value_head_type="gaussian",
            search_type="rd_mcts",
            training_mode="separate",
            num_iterations=5,
            games_per_iteration=10
        ),
        ExperimentConfig(
            name="Scalar + RD-MCTS",
            value_head_type="scalar",
            search_type="rd_mcts",
            training_mode="separate",
            num_iterations=5,
            games_per_iteration=10
        ),
    ]
    
    agents = []
    all_results = {}
    
    for config in configs:
        model, search_class, search_config, results = run_single_experiment(config)
        agents.append((config.name, model, search_class, search_config))
        all_results[config.name] = results
    
    print("\n" + "="*80)
    print("HEAD-TO-HEAD MATCHES")
    print("="*80)
    
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            name1, model1, search1, config1 = agents[i]
            name2, model2, search2, config2 = agents[j]
            
            print(f"\n{name1} vs {name2}")
            wins = play_match(model1, search1, config1, model2, search2, config2, num_games=20, simulations=50)
            print(f"  {name1}: {wins[0]} wins")
            print(f"  {name2}: {wins[1]} wins")
            print(f"  Draws: {wins[2]}")
            
            all_results[f"{name1}_vs_{name2}"] = {
                'agent1_wins': wins[0],
                'agent2_wins': wins[1],
                'draws': wins[2]
            }
    
    save_results(all_results, 'suite1_value_representations.json')
    return all_results

def run_experiment_suite_2():
    """
    Experiment Suite 2: Different search algorithms with same value head
    Tests: RD-MCTS vs Standard vs RD-no-penalty vs Thompson-Scalar
    """
    print("\n" + "="*80)
    print("EXPERIMENT SUITE 2: Search Algorithm Comparison")
    print("="*80)
    
    configs = [
        ExperimentConfig(
            name="Categorical + RD-MCTS",
            value_head_type="categorical",
            search_type="rd_mcts",
            training_mode="separate",
            num_iterations=5,
            games_per_iteration=10
        ),
        ExperimentConfig(
            name="Categorical + Standard MCTS",
            value_head_type="categorical",
            search_type="standard",
            training_mode="separate",
            num_iterations=5,
            games_per_iteration=10
        ),
        ExperimentConfig(
            name="Categorical + RD-MCTS (no penalty)",
            value_head_type="categorical",
            search_type="rd_no_penalty",
            training_mode="separate",
            num_iterations=5,
            games_per_iteration=10
        ),
        ExperimentConfig(
            name="Scalar + Thompson MCTS",
            value_head_type="scalar",
            search_type="thompson_scalar",
            training_mode="separate",
            num_iterations=5,
            games_per_iteration=10
        ),
    ]
    
    agents = []
    all_results = {}
    
    for config in configs:
        model, search_class, search_config, results = run_single_experiment(config)
        agents.append((config.name, model, search_class, search_config))
        all_results[config.name] = results
    
    print("\n" + "="*80)
    print("HEAD-TO-HEAD MATCHES")
    print("="*80)
    
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            name1, model1, search1, config1 = agents[i]
            name2, model2, search2, config2 = agents[j]
            
            print(f"\n{name1} vs {name2}")
            wins = play_match(model1, search1, config1, model2, search2, config2, num_games=20, simulations=50)
            print(f"  {name1}: {wins[0]} wins")
            print(f"  {name2}: {wins[1]} wins")
            print(f"  Draws: {wins[2]}")
            
            all_results[f"{name1}_vs_{name2}"] = {
                'agent1_wins': wins[0],
                'agent2_wins': wins[1],
                'draws': wins[2]
            }
    
    save_results(all_results, 'suite2_search_algorithms.json')
    return all_results

def run_experiment_suite_3():
    """
    Experiment Suite 3: Fixed network ablation
    Train one network, test with multiple search algorithms
    """
    print("\n" + "="*80)
    print("EXPERIMENT SUITE 3: Fixed Network Ablation")
    print("="*80)
    
    print("\nTraining base network with Categorical + Standard MCTS...")
    base_config = ExperimentConfig(
        name="Base Network (Categorical + Standard)",
        value_head_type="categorical",
        search_type="standard",
        training_mode="separate",
        num_iterations=5,
        games_per_iteration=10
    )
    
    base_model, _, _, base_results = run_single_experiment(base_config)
    
    print("\n" + "="*80)
    print("TESTING SAME NETWORK WITH DIFFERENT SEARCH ALGORITHMS")
    print("="*80)
    
    search_variants = [
        ("Standard MCTS", StandardMCTSSearch, SearchConfig(c_puct=1.0)),
        ("RD-MCTS", RDMCTSSearch, SearchConfig(c_puct=1.0, lambda_penalty=1.0, use_thompson=True, use_penalty=True)),
        ("RD-MCTS (no penalty)", RDMCTSSearch, SearchConfig(c_puct=1.0, lambda_penalty=0.0, use_thompson=True, use_penalty=False)),
        ("RD-MCTS (high penalty)", RDMCTSSearch, SearchConfig(c_puct=1.0, lambda_penalty=2.0, use_thompson=True, use_penalty=True)),
    ]
    
    all_results = {'base_training': base_results}
    agents = []
    
    for name, search_class, search_config in search_variants:
        print(f"\nEvaluating with {name}...")
        trap_results = evaluate_trap_avoidance(base_model, search_class, search_config, simulations=100)
        all_results[f"fixed_network_{name}"] = {
            'trap_analysis': trap_results
        }
        agents.append((name, base_model, search_class, search_config))
    
    print("\n" + "="*80)
    print("HEAD-TO-HEAD MATCHES (Same Network, Different Search)")
    print("="*80)
    
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            name1, model1, search1, config1 = agents[i]
            name2, model2, search2, config2 = agents[j]
            
            print(f"\n{name1} vs {name2}")
            wins = play_match(model1, search1, config1, model2, search2, config2, num_games=20, simulations=50)
            print(f"  {name1}: {wins[0]} wins")
            print(f"  {name2}: {wins[1]} wins")
            print(f"  Draws: {wins[2]}")
            
            all_results[f"{name1}_vs_{name2}"] = {
                'agent1_wins': wins[0],
                'agent2_wins': wins[1],
                'draws': wins[2]
            }
    
    save_results(all_results, 'suite3_fixed_network.json')
    return all_results

if __name__ == "__main__":
    print("="*80)
    print("COMPREHENSIVE RD-MCTS EXPERIMENTAL EVALUATION")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Starting comprehensive experiments...")
    
    start_time = time.time()
    
    suite1_results = run_experiment_suite_1()
    
    suite2_results = run_experiment_suite_2()
    
    suite3_results = run_experiment_suite_3()
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*80)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print("\nResults saved to:")
    print("  - suite1_value_representations.json")
    print("  - suite2_search_algorithms.json")
    print("  - suite3_fixed_network.json")
