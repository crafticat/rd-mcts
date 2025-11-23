"""
Extended experiments comparing RD-MCTS against other MCTS optimizations,
including noise injection tests to understand when penalty helps.
"""

import json
import numpy as np
import torch
from rd_mcts_experiment import Connect4
from framework import UnifiedNetwork, CategoricalValueHead, ScalarValueHead, SearchConfig
from search_algorithms import (
    RDMCTSSearch, StandardMCTSSearch, ThompsonScalarMCTSSearch,
    UCBVMCTSSearch, RAVEMCTSSearch
)
from unified_trainer import create_model_and_search, play_match, UnifiedTrainer, ExperimentConfig
from noise_wrapper import create_noisy_model

def run_noise_sweep_experiment():
    """
    Test how noise affects RD-MCTS with vs without penalty.
    This directly tests the paper's theoretical claims about trap avoidance.
    """
    print("="*80)
    print("EXPERIMENT: Noise Sweep (When Does Penalty Help?)")
    print("="*80)
    
    # Train a base model with categorical value head
    print("\n1. Training base model (5 iterations)...")
    config = ExperimentConfig(
        name="noise_sweep_base",
        value_head_type="categorical",
        search_type="rd_mcts",
        training_mode="separate",
        lambda_penalty=1.0,
        use_penalty=True,
        use_thompson=True,
        num_iterations=5,
        games_per_iteration=10,
        simulations_train=30
    )
    
    model, search_class, search_config = create_model_and_search(config)
    trainer = UnifiedTrainer(model, search_class, search_config)
    
    for i in range(config.num_iterations):
        loss = trainer.train_iteration(config.games_per_iteration, config.simulations_train)
        print(f"  Iteration {i+1}/5: loss={loss:.4f}")
    
    # Save the trained model
    base_model = model
    
    # Test with different noise levels
    noise_levels = [0.0, 0.1, 0.2, 0.4]
    results = []
    
    print("\n2. Testing RD-MCTS (λ=0) vs RD-MCTS (λ=1) at different noise levels...")
    
    for noise_std in noise_levels:
        print(f"\n  Noise level σ={noise_std}:")
        
        # Create noisy versions of the model
        noisy_model_l0 = create_noisy_model(base_model, noise_std)
        noisy_model_l1 = create_noisy_model(base_model, noise_std)
        
        # Create search algorithms
        config_l0 = SearchConfig(c_puct=1.0, lambda_penalty=0.0, use_penalty=False, use_thompson=True)
        config_l1 = SearchConfig(c_puct=1.0, lambda_penalty=1.0, use_penalty=True, use_thompson=True)
        
        search_l0 = RDMCTSSearch(noisy_model_l0, config_l0)
        search_l1 = RDMCTSSearch(noisy_model_l1, config_l1)
        
        # Play matches
        num_games = 20
        match_results = play_match(
            noisy_model_l0, RDMCTSSearch, config_l0,
            noisy_model_l1, RDMCTSSearch, config_l1,
            num_games=num_games, simulations=30
        )
        
        wins_l0 = match_results[0]
        wins_l1 = match_results[1]
        draws = match_results[2]
        
        print(f"    Games {num_games}/{num_games}: λ=0: {wins_l0}, λ=1: {wins_l1}, draws: {draws}")
        
        win_rate_l1 = wins_l1 / num_games
        
        results.append({
            "noise_std": noise_std,
            "wins_lambda_0": wins_l0,
            "wins_lambda_1": wins_l1,
            "draws": draws,
            "win_rate_lambda_1": win_rate_l1
        })
        
        print(f"    Final: λ=0: {wins_l0}, λ=1: {wins_l1}, draws: {draws}")
        print(f"    Win rate for λ=1: {win_rate_l1:.1%}")
    
    # Save results
    with open("noise_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("NOISE SWEEP COMPLETE")
    print("="*80)
    print("\nKey findings:")
    for r in results:
        print(f"  σ={r['noise_std']}: λ=1 win rate = {r['win_rate_lambda_1']:.1%}")
    
    return results

def run_mcts_comparison_experiment():
    """
    Compare RD-MCTS against other known MCTS optimizations.
    Uses fixed network ablation to isolate search algorithm effects.
    """
    print("\n" + "="*80)
    print("EXPERIMENT: RD-MCTS vs Other MCTS Optimizations")
    print("="*80)
    
    # Train a base model with scalar value head (works with all search algorithms)
    print("\n1. Training base model with Standard MCTS (5 iterations)...")
    config = ExperimentConfig(
        name="mcts_comparison_base",
        value_head_type="scalar",
        search_type="standard",
        training_mode="separate",
        num_iterations=5,
        games_per_iteration=10,
        simulations_train=30
    )
    
    model, search_class, search_config = create_model_and_search(config)
    trainer = UnifiedTrainer(model, search_class, search_config)
    
    for i in range(config.num_iterations):
        loss = trainer.train_iteration(config.games_per_iteration, config.simulations_train)
        print(f"  Iteration {i+1}/5: loss={loss:.4f}")
    
    # Create all search algorithms with the same model
    base_config = SearchConfig(c_puct=1.0)
    
    algorithms = {
        "Standard PUCT": StandardMCTSSearch(model, base_config),
        "Thompson Scalar": ThompsonScalarMCTSSearch(model, base_config),
        "UCB-V": UCBVMCTSSearch(model, base_config),
        "RAVE": RAVEMCTSSearch(model, base_config),
        "RD-MCTS (λ=0)": RDMCTSSearch(model, SearchConfig(c_puct=1.0, lambda_penalty=0.0, use_penalty=False, use_thompson=True)),
        "RD-MCTS (λ=1)": RDMCTSSearch(model, SearchConfig(c_puct=1.0, lambda_penalty=1.0, use_penalty=True, use_thompson=True)),
        "RD-MCTS (λ=2)": RDMCTSSearch(model, SearchConfig(c_puct=1.0, lambda_penalty=2.0, use_penalty=True, use_thompson=True)),
    }
    
    # Run round-robin tournament
    print("\n2. Running round-robin tournament (20 games per matchup)...")
    
    results = {}
    algo_names = list(algorithms.keys())
    
    for i, name1 in enumerate(algo_names):
        for name2 in algo_names[i+1:]:
            print(f"\n  {name1} vs {name2}:")
            
            # Use the play_match function with correct signature
            match_results = play_match(
                model, type(algorithms[name1]), algorithms[name1].config,
                model, type(algorithms[name2]), algorithms[name2].config,
                num_games=20, simulations=30
            )
            
            wins1 = match_results[0]
            wins2 = match_results[1]
            draws = match_results[2]
            
            matchup_key = f"{name1}_vs_{name2}"
            results[matchup_key] = {
                "player1": name1,
                "player2": name2,
                "wins1": wins1,
                "wins2": wins2,
                "draws": draws,
                "win_rate1": wins1 / 20
            }
            
            print(f"    Result: {wins1}-{wins2}-{draws}")
    
    # Save results
    with open("mcts_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("MCTS COMPARISON COMPLETE")
    print("="*80)
    
    # Calculate win rates for each algorithm
    win_counts = {name: 0 for name in algo_names}
    total_games = {name: 0 for name in algo_names}
    
    for matchup_key, result in results.items():
        name1 = result["player1"]
        name2 = result["player2"]
        
        win_counts[name1] += result["wins1"]
        win_counts[name2] += result["wins2"]
        total_games[name1] += 20
        total_games[name2] += 20
    
    print("\nOverall win rates:")
    for name in algo_names:
        win_rate = win_counts[name] / total_games[name] if total_games[name] > 0 else 0
        print(f"  {name}: {win_rate:.1%} ({win_counts[name]}/{total_games[name]})")
    
    return results

if __name__ == "__main__":
    print("\n" + "="*80)
    print("EXTENDED EXPERIMENTS: RD-MCTS vs MCTS Optimizations + Noise Analysis")
    print("="*80)
    
    # Run noise sweep to understand when penalty helps
    print("\n[1/2] Running noise sweep experiment...")
    noise_results = run_noise_sweep_experiment()
    
    # Run MCTS comparison
    print("\n[2/2] Running MCTS comparison experiment...")
    comparison_results = run_mcts_comparison_experiment()
    
    print("\n" + "="*80)
    print("ALL EXTENDED EXPERIMENTS COMPLETE!")
    print("="*80)
    print("\nResults saved to:")
    print("  - noise_sweep_results.json")
    print("  - mcts_comparison_results.json")
