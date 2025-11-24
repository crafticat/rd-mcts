"""
Evaluate value-only checkpoints against Minimax benchmark.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
from game import Connect4
from game.minimax_agent import MinimaxAgent
from networks import ValueOnlyNetwork
from mcts import ValueOnlyRDMCTSSearch, ValueOnlyScalarMCTSSearch
from utils import CategoricalValueHead, ScalarValueHead, SearchConfig, DEVICE


def play_game(agent1, agent2, simulations=50):
    """Play one game between two agents. Returns 1 if agent1 wins, -1 if agent2 wins, 0 for draw."""
    game = Connect4()
    
    while not game.is_terminal():
        if game.player == 1:
            # Agent 1's turn
            if isinstance(agent1, MinimaxAgent):
                move = agent1.get_best_move(game)
            else:
                children = agent1.search(game, simulations=simulations)
                move = max(children.items(), key=lambda x: x[1].visits)[0]
        else:
            # Agent 2's turn
            if isinstance(agent2, MinimaxAgent):
                move = agent2.get_best_move(game)
            else:
                children = agent2.search(game, simulations=simulations)
                move = max(children.items(), key=lambda x: x[1].visits)[0]
        
        game = game.make_move(move)
    
    result = game.check_win()
    return result


def evaluate_checkpoint(model_path, search_class, config, value_head_class, num_games=20):
    """Evaluate a checkpoint against Minimax."""
    # Load model
    model = ValueOnlyNetwork(value_head_class()).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # Create agents
    mcts_agent = search_class(model, config)
    minimax_agent = MinimaxAgent(depth=4)
    
    wins = 0
    losses = 0
    draws = 0
    
    for game_idx in range(num_games):
        # Alternate who goes first
        if game_idx % 2 == 0:
            result = play_game(mcts_agent, minimax_agent, simulations=50)
        else:
            result = -play_game(minimax_agent, mcts_agent, simulations=50)
        
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1
        
        if (game_idx + 1) % 5 == 0:
            print(f"    Game {game_idx + 1}/{num_games}: W={wins}, L={losses}, D={draws}")
    
    win_rate = wins / num_games
    return wins, losses, draws, win_rate


def main():
    print("="*80)
    print("VALUE-ONLY CHECKPOINT EVALUATION")
    print("="*80)
    
    results = {
        "rd_mcts": {},
        "scalar_mcts": {}
    }
    
    # Evaluate RD-MCTS checkpoints
    print("\n" + "="*80)
    print("Evaluating RD-MCTS (Value-Only) Checkpoints")
    print("="*80)
    
    rd_config = SearchConfig(c_puct=1.0, lambda_penalty=1.0, use_thompson=True, use_penalty=True)
    
    for checkpoint in range(10, 101, 10):
        model_path = f"../checkpoints_value_only/rd_mcts_checkpoint_{checkpoint}.pt"
        
        if not os.path.exists(model_path):
            print(f"\nCheckpoint {checkpoint}: NOT FOUND")
            continue
        
        print(f"\nCheckpoint {checkpoint} (after {checkpoint} games):")
        wins, losses, draws, win_rate = evaluate_checkpoint(
            model_path, ValueOnlyRDMCTSSearch, rd_config, CategoricalValueHead, num_games=20
        )
        
        results["rd_mcts"][checkpoint] = {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": win_rate
        }
        
        print(f"  Results: W={wins}, L={losses}, D={draws}, Win Rate={win_rate:.1%}")
    
    # Evaluate Scalar MCTS checkpoints
    print("\n" + "="*80)
    print("Evaluating Scalar MCTS (Value-Only) Checkpoints")
    print("="*80)
    
    scalar_config = SearchConfig(c_puct=1.0, lambda_penalty=0.0, use_thompson=True, use_penalty=False)
    
    for checkpoint in range(10, 101, 10):
        model_path = f"../checkpoints_value_only/scalar_mcts_checkpoint_{checkpoint}.pt"
        
        if not os.path.exists(model_path):
            print(f"\nCheckpoint {checkpoint}: NOT FOUND")
            continue
        
        print(f"\nCheckpoint {checkpoint} (after {checkpoint} games):")
        wins, losses, draws, win_rate = evaluate_checkpoint(
            model_path, ValueOnlyScalarMCTSSearch, scalar_config, ScalarValueHead, num_games=20
        )
        
        results["scalar_mcts"][checkpoint] = {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": win_rate
        }
        
        print(f"  Results: W={wins}, L={losses}, D={draws}, Win Rate={win_rate:.1%}")
    
    # Save results
    with open("../learning_curve_value_only_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: learning_curve_value_only_results.json")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\nRD-MCTS (Value-Only):")
    for checkpoint, data in sorted(results["rd_mcts"].items()):
        print(f"  {checkpoint} games: {data['win_rate']:.1%} win rate")
    
    print("\nScalar MCTS (Value-Only):")
    for checkpoint, data in sorted(results["scalar_mcts"].items()):
        print(f"  {checkpoint} games: {data['win_rate']:.1%} win rate")


if __name__ == "__main__":
    main()
