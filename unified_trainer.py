"""
Unified trainer that works with any combination of value head and search algorithm.
"""

import torch
import torch.optim as optim
import numpy as np
import copy
from typing import List, Tuple
from dataclasses import dataclass, asdict
import json

from framework import (
    UnifiedNetwork, CategoricalValueHead, GaussianValueHead, ScalarValueHead,
    SearchConfig, Search, DEVICE
)
from search_algorithms import RDMCTSSearch, StandardMCTSSearch, ThompsonScalarMCTSSearch
from rd_mcts_experiment import Connect4

@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str
    value_head_type: str  # 'categorical', 'gaussian', 'scalar'
    search_type: str  # 'rd_mcts', 'standard', 'thompson_scalar', 'rd_no_penalty'
    training_mode: str  # 'separate', 'fixed_network', 'shared_backbone'
    c_puct: float = 1.0
    lambda_penalty: float = 1.0
    use_thompson: bool = True
    use_penalty: bool = True
    num_iterations: int = 5
    games_per_iteration: int = 10
    simulations_train: int = 30
    simulations_eval: int = 50
    seed: int = 42

class UnifiedTrainer:
    """Trainer that works with any value head and search algorithm."""
    
    def __init__(self, model: UnifiedNetwork, search_class, search_config):
        self.model = model
        self.search_class = search_class
        self.search_config = search_config
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.training_history = []
    
    def self_play_game(self, simulations=50):
        """Play one self-play game."""
        game = Connect4()
        states = []
        policies = []
        
        while not game.is_terminal():
            search = self.search_class(self.model, self.search_config)
            children = search.search(game, simulations=simulations)
            
            state = game.get_canonical_state()
            states.append(state)
            
            policy = np.zeros(7)
            total_visits = sum(child.visits for child in children.values())
            for move, child in children.items():
                policy[move] = child.visits / total_visits
            policies.append(policy)
            
            move = np.random.choice(7, p=policy)
            game = game.make_move(move)
        
        result = game.check_win()
        if result == 0:
            final_value = 0.0
        else:
            final_value = result
        
        values = []
        for i in range(len(states)):
            if i % 2 == 0:
                values.append(final_value)
            else:
                values.append(-final_value)
        
        return states, policies, values
    
    def train_on_batch(self, states, policies, values):
        """Train on a batch of data."""
        self.model.train()
        self.optimizer.zero_grad()
        
        states_tensor = torch.FloatTensor(np.array(states)).to(DEVICE)
        policies_tensor = torch.FloatTensor(np.array(policies)).to(DEVICE)
        values_tensor = torch.FloatTensor(np.array(values)).to(DEVICE)
        
        pi_pred, value_rep = self.model(states_tensor)
        
        policy_loss = -torch.mean(torch.sum(policies_tensor * pi_pred, dim=1))
        value_loss = self.model.value_head.loss(value_rep, values_tensor)
        
        total_loss = policy_loss + value_loss
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), policy_loss.item(), value_loss.item()
    
    def train_iteration(self, num_games=10, simulations=50):
        """Run one training iteration."""
        all_states = []
        all_policies = []
        all_values = []
        
        for i in range(num_games):
            states, policies, values = self.self_play_game(simulations=simulations)
            all_states.extend(states)
            all_policies.extend(policies)
            all_values.extend(values)
        
        batch_size = 32
        total_loss = 0
        num_batches = 0
        
        for i in range(0, len(all_states), batch_size):
            batch_states = all_states[i:i+batch_size]
            batch_policies = all_policies[i:i+batch_size]
            batch_values = all_values[i:i+batch_size]
            
            loss, pol_loss, val_loss = self.train_on_batch(batch_states, batch_policies, batch_values)
            total_loss += loss
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        self.training_history.append(avg_loss)
        
        return avg_loss

def create_trap_position():
    """Create a trap position for testing."""
    game = Connect4()
    moves = [3, 3, 2, 4, 2, 4, 1, 5]
    for move in moves:
        if move in game.get_valid_moves():
            game = game.make_move(move)
    return game

def evaluate_trap_avoidance(model, search_class, search_config, simulations=100):
    """Evaluate trap avoidance on a constructed position."""
    from framework import dist_stats
    
    trap_game = create_trap_position()
    search = search_class(model, search_config)
    children = search.search(trap_game, simulations=simulations)
    
    results = {}
    for move, child in children.items():
        if hasattr(child, 'dist') and child.visits > 0:
            mean, std = dist_stats(child.dist)
            results[move] = {
                'visits': child.visits,
                'mean': float(mean),
                'std': float(std),
                'score': float(mean - std)
            }
        else:
            value = child.get_value()
            results[move] = {
                'visits': child.visits,
                'value': float(value),
                'score': float(value)
            }
    
    return results

def play_match(model1, search_class1, config1, model2, search_class2, config2, num_games=10, simulations=50):
    """Play a match between two agents."""
    wins = [0, 0, 0]
    
    for game_num in range(num_games):
        game = Connect4()
        
        if game_num % 2 == 0:
            models = [model1, model2]
            search_classes = [search_class1, search_class2]
            configs = [config1, config2]
        else:
            models = [model2, model1]
            search_classes = [search_class2, search_class1]
            configs = [config2, config1]
        
        player_idx = 0
        
        while not game.is_terminal():
            search = search_classes[player_idx](models[player_idx], configs[player_idx])
            children = search.search(game, simulations=simulations)
            
            best_move = max(children.keys(), key=lambda m: children[m].visits)
            game = game.make_move(best_move)
            
            player_idx = 1 - player_idx
        
        result = game.check_win()
        
        if game_num % 2 == 0:
            if result == 1:
                wins[0] += 1
            elif result == -1:
                wins[1] += 1
            else:
                wins[2] += 1
        else:
            if result == 1:
                wins[1] += 1
            elif result == -1:
                wins[0] += 1
            else:
                wins[2] += 1
    
    return wins

def save_results(results: dict, filename: str):
    """Save results to JSON file."""
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)

def create_model_and_search(config: ExperimentConfig):
    """Create model and search class based on config."""
    if config.value_head_type == 'categorical':
        value_head = CategoricalValueHead()
    elif config.value_head_type == 'gaussian':
        value_head = GaussianValueHead()
    elif config.value_head_type == 'scalar':
        value_head = ScalarValueHead()
    else:
        raise ValueError(f"Unknown value head type: {config.value_head_type}")
    
    model = UnifiedNetwork(value_head).to(DEVICE)
    
    search_config = SearchConfig(
        c_puct=config.c_puct,
        lambda_penalty=config.lambda_penalty,
        use_thompson=config.use_thompson,
        use_penalty=config.use_penalty
    )
    
    if config.search_type == 'rd_mcts':
        search_class = RDMCTSSearch
    elif config.search_type == 'standard':
        search_class = StandardMCTSSearch
    elif config.search_type == 'thompson_scalar':
        search_class = ThompsonScalarMCTSSearch
    elif config.search_type == 'rd_no_penalty':
        search_class = RDMCTSSearch
        search_config.use_penalty = False
    else:
        raise ValueError(f"Unknown search type: {config.search_type}")
    
    return model, search_class, search_config

def run_single_experiment(config: ExperimentConfig):
    """Run a single experiment with the given configuration."""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {config.name}")
    print(f"{'='*80}")
    print(f"Value Head: {config.value_head_type}")
    print(f"Search: {config.search_type}")
    print(f"Training Mode: {config.training_mode}")
    print(f"Lambda: {config.lambda_penalty}, C_PUCT: {config.c_puct}")
    
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    model, search_class, search_config = create_model_and_search(config)
    
    results = {
        'config': asdict(config),
        'training_losses': [],
        'trap_initial': {},
        'trap_final': {},
    }
    
    print("\n[1/3] Testing initial trap avoidance...")
    results['trap_initial'] = evaluate_trap_avoidance(
        model, search_class, search_config, simulations=100
    )
    
    if config.training_mode == 'separate':
        print("\n[2/3] Training with self-play...")
        trainer = UnifiedTrainer(model, search_class, search_config)
        
        for iteration in range(config.num_iterations):
            print(f"  Iteration {iteration + 1}/{config.num_iterations}")
            loss = trainer.train_iteration(
                num_games=config.games_per_iteration,
                simulations=config.simulations_train
            )
            results['training_losses'].append(float(loss))
            print(f"    Loss: {loss:.4f}")
    
    print("\n[3/3] Testing final trap avoidance...")
    results['trap_final'] = evaluate_trap_avoidance(
        model, search_class, search_config, simulations=100
    )
    
    return model, search_class, search_config, results
