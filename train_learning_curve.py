"""
Training script for learning curve experiments.
Trains RD-MCTS and Standard MCTS from scratch, saving checkpoints every 10 games.
"""

import torch
import numpy as np
import os
from pathlib import Path

from framework import UnifiedNetwork, CategoricalValueHead, ScalarValueHead, SearchConfig, DEVICE
from search_algorithms import RDMCTSSearch, StandardMCTSSearch
from unified_trainer import UnifiedTrainer
from rd_mcts_experiment import Connect4


def train_with_checkpoints(
    agent_name: str,
    value_head,
    search_class,
    search_config,
    total_games: int = 100,
    checkpoint_interval: int = 10,
    simulations: int = 30,
    checkpoint_dir: str = "checkpoints"
):
    """
    Train an agent from scratch, saving checkpoints at regular intervals.
    
    Args:
        agent_name: Name for this agent (e.g., "rd_mcts", "standard")
        value_head: Value head instance
        search_class: Search algorithm class
        search_config: Search configuration
        total_games: Total number of self-play games to train
        checkpoint_interval: Save checkpoint every N games
        simulations: Simulations per move during training
        checkpoint_dir: Directory to save checkpoints
    
    Returns:
        Final model
    """
    print(f"\n{'='*80}")
    print(f"TRAINING: {agent_name}")
    print(f"{'='*80}")
    print(f"Total games: {total_games}")
    print(f"Checkpoint interval: {checkpoint_interval}")
    print(f"Simulations per move: {simulations}")
    
    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(exist_ok=True)
    
    # Initialize model and trainer
    model = UnifiedNetwork(value_head).to(DEVICE)
    trainer = UnifiedTrainer(model, search_class, search_config)
    
    # Track training progress
    games_played = 0
    checkpoint_losses = []
    
    # Training loop
    while games_played < total_games:
        # Determine how many games to play in this batch
        games_this_batch = min(checkpoint_interval, total_games - games_played)
        
        print(f"\nGames {games_played + 1}-{games_played + games_this_batch}:")
        
        # Train for this batch
        loss = trainer.train_iteration(num_games=games_this_batch, simulations=simulations)
        games_played += games_this_batch
        
        print(f"  Loss: {loss:.4f}")
        checkpoint_losses.append(loss)
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"{agent_name}_games_{games_played}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE: {agent_name}")
    print(f"{'='*80}")
    print(f"Total games played: {games_played}")
    print(f"Checkpoints saved: {len(checkpoint_losses)}")
    print(f"Final loss: {checkpoint_losses[-1]:.4f}")
    
    # Save training history
    history_path = os.path.join(checkpoint_dir, f"{agent_name}_training_history.npy")
    np.save(history_path, np.array(checkpoint_losses))
    print(f"Training history saved: {history_path}")
    
    return model


def main():
    """Run learning curve training for both RD-MCTS and Standard MCTS."""
    print("\n" + "="*80)
    print("LEARNING CURVE TRAINING EXPERIMENT")
    print("="*80)
    print("\nThis experiment trains two agents from scratch:")
    print("  1. RD-MCTS (categorical value head, λ=1)")
    print("  2. Standard MCTS (scalar value head)")
    print("\nCheckpoints are saved every 10 games (up to 100 games total).")
    print("These checkpoints will be evaluated against a fixed benchmark.")
    
    # Configuration
    total_games = 100
    checkpoint_interval = 10
    simulations = 30
    checkpoint_dir = "checkpoints"
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Train RD-MCTS agent
    print("\n" + "="*80)
    print("AGENT 1: RD-MCTS")
    print("="*80)
    
    rd_value_head = CategoricalValueHead()
    rd_search_config = SearchConfig(
        c_puct=1.0,
        lambda_penalty=1.0,
        use_thompson=True,
        use_penalty=True
    )
    
    rd_model = train_with_checkpoints(
        agent_name="rd_mcts",
        value_head=rd_value_head,
        search_class=RDMCTSSearch,
        search_config=rd_search_config,
        total_games=total_games,
        checkpoint_interval=checkpoint_interval,
        simulations=simulations,
        checkpoint_dir=checkpoint_dir
    )
    
    # Train Standard MCTS agent
    print("\n" + "="*80)
    print("AGENT 2: Standard MCTS")
    print("="*80)
    
    std_value_head = ScalarValueHead()
    std_search_config = SearchConfig(
        c_puct=1.0,
        lambda_penalty=0.0,  # Not used for scalar
        use_thompson=False,
        use_penalty=False
    )
    
    std_model = train_with_checkpoints(
        agent_name="standard_mcts",
        value_head=std_value_head,
        search_class=StandardMCTSSearch,
        search_config=std_search_config,
        total_games=total_games,
        checkpoint_interval=checkpoint_interval,
        simulations=simulations,
        checkpoint_dir=checkpoint_dir
    )
    
    print("\n" + "="*80)
    print("ALL TRAINING COMPLETE!")
    print("="*80)
    print(f"\nCheckpoints saved in: {checkpoint_dir}/")
    print(f"  - rd_mcts_games_10.pt, rd_mcts_games_20.pt, ..., rd_mcts_games_100.pt")
    print(f"  - standard_mcts_games_10.pt, standard_mcts_games_20.pt, ..., standard_mcts_games_100.pt")
    print(f"\nNext step: Run eval_learning_curve.py to evaluate these checkpoints.")


if __name__ == "__main__":
    main()
