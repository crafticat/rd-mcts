"""
Evaluation script for learning curve experiments.
Evaluates each checkpoint against a fixed benchmark opponent.
"""

import torch
import numpy as np
import os
from pathlib import Path
import json

from framework import UnifiedNetwork, CategoricalValueHead, ScalarValueHead, SearchConfig, DEVICE
from search_algorithms import RDMCTSSearch, StandardMCTSSearch
from rd_mcts_experiment import Connect4
from minimax_agent import MinimaxAgent


def play_game(agent1_model, agent1_search_class, agent1_config, 
              agent2_model, agent2_search_class, agent2_config,
              simulations=50):
    """
    Play one game between two agents.
    
    Returns:
        1 if agent1 wins, -1 if agent2 wins, 0 if draw
    """
    game = Connect4()
    player_idx = 0
    
    agents = [
        (agent1_model, agent1_search_class, agent1_config),
        (agent2_model, agent2_search_class, agent2_config)
    ]
    
    while not game.is_terminal():
        model, search_class, config = agents[player_idx]
        
        if model is None:  # Minimax agent
            minimax = MinimaxAgent(depth=4)
            move = minimax.get_best_move(game)
        else:  # Neural network agent
            search = search_class(model, config)
            children = search.search(game, simulations=simulations)
            move = max(children.keys(), key=lambda m: children[m].visits)
        
        game = game.make_move(move)
        player_idx = 1 - player_idx
    
    result = game.check_win()
    return result


def evaluate_checkpoint(
    checkpoint_path: str,
    value_head,
    search_class,
    search_config,
    benchmark_type: str = "minimax",
    benchmark_model=None,
    benchmark_search_class=None,
    benchmark_config=None,
    num_games: int = 20,
    simulations: int = 50
):
    """
    Evaluate a checkpoint against a fixed benchmark.
    
    Args:
        checkpoint_path: Path to checkpoint file
        value_head: Value head instance for the checkpoint
        search_class: Search algorithm class
        search_config: Search configuration
        benchmark_type: "minimax" or "network"
        benchmark_model: Benchmark model (if benchmark_type="network")
        benchmark_search_class: Benchmark search class (if benchmark_type="network")
        benchmark_config: Benchmark search config (if benchmark_type="network")
        num_games: Number of games to play
        simulations: Simulations per move
    
    Returns:
        Win rate (0.0 to 1.0)
    """
    # Load checkpoint
    model = UnifiedNetwork(value_head).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    
    wins = 0
    losses = 0
    draws = 0
    
    for game_num in range(num_games):
        # Alternate who plays first
        if game_num % 2 == 0:
            # Checkpoint plays first
            if benchmark_type == "minimax":
                result = play_game(
                    model, search_class, search_config,
                    None, None, None,  # Minimax
                    simulations=simulations
                )
            else:
                result = play_game(
                    model, search_class, search_config,
                    benchmark_model, benchmark_search_class, benchmark_config,
                    simulations=simulations
                )
            
            if result == 1:
                wins += 1
            elif result == -1:
                losses += 1
            else:
                draws += 1
        else:
            # Benchmark plays first
            if benchmark_type == "minimax":
                result = play_game(
                    None, None, None,  # Minimax
                    model, search_class, search_config,
                    simulations=simulations
                )
            else:
                result = play_game(
                    benchmark_model, benchmark_search_class, benchmark_config,
                    model, search_class, search_config,
                    simulations=simulations
                )
            
            if result == 1:
                losses += 1
            elif result == -1:
                wins += 1
            else:
                draws += 1
    
    win_rate = wins / num_games
    return win_rate, wins, losses, draws


def main():
    """Evaluate learning curves for both agents."""
    print("\n" + "="*80)
    print("LEARNING CURVE EVALUATION")
    print("="*80)
    
    checkpoint_dir = "checkpoints"
    checkpoints = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    num_games = 20
    simulations = 50
    benchmark_type = "minimax"  # or "network"
    
    print(f"\nBenchmark: {benchmark_type}")
    print(f"Games per checkpoint: {num_games}")
    print(f"Simulations per move: {simulations}")
    
    results = {
        "checkpoints": checkpoints,
        "rd_mcts": {
            "win_rates": [],
            "wins": [],
            "losses": [],
            "draws": []
        },
        "standard_mcts": {
            "win_rates": [],
            "wins": [],
            "losses": [],
            "draws": []
        },
        "config": {
            "benchmark_type": benchmark_type,
            "num_games": num_games,
            "simulations": simulations
        }
    }
    
    # Evaluate RD-MCTS checkpoints
    print("\n" + "="*80)
    print("EVALUATING RD-MCTS CHECKPOINTS")
    print("="*80)
    
    rd_value_head = CategoricalValueHead()
    rd_search_config = SearchConfig(
        c_puct=1.0,
        lambda_penalty=1.0,
        use_thompson=True,
        use_penalty=True
    )
    
    for cp in checkpoints:
        checkpoint_path = os.path.join(checkpoint_dir, f"rd_mcts_games_{cp}.pt")
        
        if not os.path.exists(checkpoint_path):
            print(f"  Checkpoint not found: {checkpoint_path}")
            continue
        
        print(f"\n  Evaluating checkpoint: games_{cp}")
        win_rate, wins, losses, draws = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            value_head=rd_value_head,
            search_class=RDMCTSSearch,
            search_config=rd_search_config,
            benchmark_type=benchmark_type,
            num_games=num_games,
            simulations=simulations
        )
        
        results["rd_mcts"]["win_rates"].append(win_rate)
        results["rd_mcts"]["wins"].append(wins)
        results["rd_mcts"]["losses"].append(losses)
        results["rd_mcts"]["draws"].append(draws)
        
        print(f"    Win rate: {win_rate:.1%} ({wins}-{losses}-{draws})")
    
    # Evaluate Standard MCTS checkpoints
    print("\n" + "="*80)
    print("EVALUATING STANDARD MCTS CHECKPOINTS")
    print("="*80)
    
    std_value_head = ScalarValueHead()
    std_search_config = SearchConfig(
        c_puct=1.0,
        lambda_penalty=0.0,
        use_thompson=False,
        use_penalty=False
    )
    
    for cp in checkpoints:
        checkpoint_path = os.path.join(checkpoint_dir, f"standard_mcts_games_{cp}.pt")
        
        if not os.path.exists(checkpoint_path):
            print(f"  Checkpoint not found: {checkpoint_path}")
            continue
        
        print(f"\n  Evaluating checkpoint: games_{cp}")
        win_rate, wins, losses, draws = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            value_head=std_value_head,
            search_class=StandardMCTSSearch,
            search_config=std_search_config,
            benchmark_type=benchmark_type,
            num_games=num_games,
            simulations=simulations
        )
        
        results["standard_mcts"]["win_rates"].append(win_rate)
        results["standard_mcts"]["wins"].append(wins)
        results["standard_mcts"]["losses"].append(losses)
        results["standard_mcts"]["draws"].append(draws)
        
        print(f"    Win rate: {win_rate:.1%} ({wins}-{losses}-{draws})")
    
    # Save results
    results_path = "learning_curve_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {results_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\nRD-MCTS Learning Curve:")
    for i, cp in enumerate(checkpoints):
        if i < len(results["rd_mcts"]["win_rates"]):
            wr = results["rd_mcts"]["win_rates"][i]
            print(f"  Games {cp:3d}: {wr:.1%}")
    
    print("\nStandard MCTS Learning Curve:")
    for i, cp in enumerate(checkpoints):
        if i < len(results["standard_mcts"]["win_rates"]):
            wr = results["standard_mcts"]["win_rates"][i]
            print(f"  Games {cp:3d}: {wr:.1%}")
    
    print("\nNext step: Run visualize_learning_curve.py to generate plots.")


if __name__ == "__main__":
    main()
