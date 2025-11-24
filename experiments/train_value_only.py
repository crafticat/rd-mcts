"""
Train value-only RD-MCTS and Scalar MCTS from scratch.
Save checkpoints every 10 games for learning curve analysis.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
import numpy as np
import copy
from game import Connect4
from networks import ValueOnlyNetwork
from mcts import ValueOnlyRDMCTSSearch, ValueOnlyScalarMCTSSearch
from utils import CategoricalValueHead, ScalarValueHead, SearchConfig, DEVICE, project_gaussian


def self_play_game(search_agent, game, simulations=50):
    """Play one self-play game and collect training data."""
    states = []
    policies = []
    
    while not game.is_terminal():
        # Run MCTS search
        children = search_agent.search(game, simulations=simulations)
        
        # Get visit counts as policy target
        moves = list(children.keys())
        visits = np.array([children[m].visits for m in moves])
        
        # Create policy distribution
        policy = np.zeros(7)
        for move, visit_count in zip(moves, visits):
            policy[move] = visit_count
        policy = policy / np.sum(policy)
        
        # Store state and policy
        states.append(game.get_canonical_state().copy())
        policies.append(policy)
        
        # Make move (sample from policy)
        move = np.random.choice(7, p=policy)
        game = game.make_move(move)
    
    # Get final result
    result = game.check_win()
    if result == 0:
        value = 0.0
    else:
        value = result
    
    # Assign values (alternating sign for each player)
    values = []
    for i in range(len(states)):
        if i % 2 == 0:
            values.append(value)
        else:
            values.append(-value)
    
    return states, policies, values


def train_iteration(model, search_class, config, num_games=10, simulations=30):
    """Train for one iteration."""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    all_states = []
    all_policies = []
    all_values = []
    
    # Collect self-play data
    for game_idx in range(num_games):
        game = Connect4()
        search_agent = search_class(model, config)
        
        states, policies, values = self_play_game(search_agent, game, simulations=simulations)
        
        all_states.extend(states)
        all_policies.extend(policies)
        all_values.extend(values)
        
        if (game_idx + 1) % 5 == 0:
            print(f"  Game {game_idx + 1}/{num_games} complete")
    
    # Train on collected data
    total_loss = 0.0
    num_batches = 0
    
    for epoch in range(3):
        indices = np.random.permutation(len(all_states))
        
        for i in range(0, len(indices), 32):
            batch_indices = indices[i:i+32]
            
            batch_states = torch.FloatTensor([all_states[j] for j in batch_indices]).to(DEVICE)
            batch_values = torch.FloatTensor([all_values[j] for j in batch_indices]).to(DEVICE)
            
            optimizer.zero_grad()
            
            value_reps = model(batch_states)
            loss = model.value_head.loss(value_reps, batch_values)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


def main():
    print("="*80)
    print("VALUE-ONLY LEARNING CURVE TRAINING")
    print("="*80)
    
    # Create checkpoint directory
    os.makedirs("../checkpoints_value_only", exist_ok=True)
    
    # Train RD-MCTS (value-only)
    print("\n" + "="*80)
    print("Training RD-MCTS (Value-Only)")
    print("="*80)
    
    rd_model = ValueOnlyNetwork(CategoricalValueHead()).to(DEVICE)
    rd_config = SearchConfig(c_puct=1.0, lambda_penalty=1.0, use_thompson=True, use_penalty=True)
    
    for checkpoint in range(1, 11):
        print(f"\nCheckpoint {checkpoint}/10 (Games {(checkpoint-1)*10 + 1}-{checkpoint*10})")
        loss = train_iteration(rd_model, ValueOnlyRDMCTSSearch, rd_config, num_games=10, simulations=30)
        print(f"  Average loss: {loss:.4f}")
        
        # Save checkpoint
        torch.save(rd_model.state_dict(), f"../checkpoints_value_only/rd_mcts_checkpoint_{checkpoint*10}.pt")
        print(f"  Saved checkpoint: rd_mcts_checkpoint_{checkpoint*10}.pt")
    
    # Train Scalar MCTS (value-only)
    print("\n" + "="*80)
    print("Training Scalar MCTS (Value-Only)")
    print("="*80)
    
    scalar_model = ValueOnlyNetwork(ScalarValueHead()).to(DEVICE)
    scalar_config = SearchConfig(c_puct=1.0, lambda_penalty=0.0, use_thompson=True, use_penalty=False)
    
    for checkpoint in range(1, 11):
        print(f"\nCheckpoint {checkpoint}/10 (Games {(checkpoint-1)*10 + 1}-{checkpoint*10})")
        loss = train_iteration(scalar_model, ValueOnlyScalarMCTSSearch, scalar_config, num_games=10, simulations=30)
        print(f"  Average loss: {loss:.4f}")
        
        # Save checkpoint
        torch.save(scalar_model.state_dict(), f"../checkpoints_value_only/scalar_mcts_checkpoint_{checkpoint*10}.pt")
        print(f"  Saved checkpoint: scalar_mcts_checkpoint_{checkpoint*10}.pt")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nCheckpoints saved to: checkpoints_value_only/")
    print("Ready for evaluation against Minimax benchmark.")


if __name__ == "__main__":
    main()
