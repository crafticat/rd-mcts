import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import time
import json
from collections import defaultdict
import matplotlib.pyplot as plt

from rd_mcts_experiment import (
    Connect4, DistributionalResNet, RDMCTS, dist_stats, 
    DEVICE, SUPPORT_SIZE, ATOMS, project_gaussian
)
from baseline_mcts import ScalarResNet, StandardMCTS

class Trainer:
    def __init__(self, model, mcts_class, is_distributional=True):
        self.model = model
        self.mcts_class = mcts_class
        self.is_distributional = is_distributional
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.training_history = []
        
    def self_play_game(self, simulations=50):
        game = Connect4()
        states = []
        policies = []
        
        while not game.is_terminal():
            mcts = self.mcts_class(self.model)
            children = mcts.search(game, simulations=simulations)
            
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
        self.model.train()
        self.optimizer.zero_grad()
        
        states_tensor = torch.FloatTensor(np.array(states)).to(DEVICE)
        policies_tensor = torch.FloatTensor(np.array(policies)).to(DEVICE)
        values_tensor = torch.FloatTensor(np.array(values)).to(DEVICE)
        
        if self.is_distributional:
            pi_pred, v_dist_pred = self.model(states_tensor)
            
            policy_loss = -torch.mean(torch.sum(policies_tensor * pi_pred, dim=1))
            
            target_dists = []
            for v in values_tensor.cpu().numpy():
                target_dist = project_gaussian(v, 0.1)
                target_dists.append(target_dist)
            target_dists_tensor = torch.FloatTensor(np.array(target_dists)).to(DEVICE)
            
            value_loss = -torch.mean(torch.sum(target_dists_tensor * torch.log(v_dist_pred + 1e-8), dim=1))
        else:
            pi_pred, v_pred = self.model(states_tensor)
            
            policy_loss = -torch.mean(torch.sum(policies_tensor * pi_pred, dim=1))
            value_loss = torch.mean((v_pred.squeeze() - values_tensor) ** 2)
        
        total_loss = policy_loss + value_loss
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), policy_loss.item(), value_loss.item()
    
    def train_iteration(self, num_games=10, simulations=50):
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
    game = Connect4()
    
    moves = [3, 3, 2, 4, 2, 4, 1, 5]
    for move in moves:
        if move in game.get_valid_moves():
            game = game.make_move(move)
    
    return game

def evaluate_trap_avoidance(model, mcts_class, is_distributional=True, simulations=100):
    trap_game = create_trap_position()
    
    mcts = mcts_class(model)
    children = mcts.search(trap_game, simulations=simulations)
    
    results = {}
    for move, child in children.items():
        if is_distributional:
            mean, std = dist_stats(child.dist)
            results[move] = {
                'visits': child.visits,
                'mean': mean,
                'std': std,
                'score': mean - std
            }
        else:
            results[move] = {
                'visits': child.visits,
                'value': child.get_value(),
                'score': child.get_value()
            }
    
    return results

def play_match(model1, mcts_class1, model2, mcts_class2, num_games=10, simulations=50):
    wins = [0, 0, 0]
    
    for game_num in range(num_games):
        game = Connect4()
        
        if game_num % 2 == 0:
            current_model = model1
            current_mcts = mcts_class1
            models = [model1, model2]
            mcts_classes = [mcts_class1, mcts_class2]
        else:
            current_model = model2
            current_mcts = mcts_class2
            models = [model2, model1]
            mcts_classes = [mcts_class2, mcts_class1]
        
        player_idx = 0
        
        while not game.is_terminal():
            mcts = mcts_classes[player_idx](models[player_idx])
            children = mcts.search(game, simulations=simulations)
            
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

def run_comprehensive_evaluation(num_training_iterations=5, games_per_iteration=10):
    print("="*80)
    print("COMPREHENSIVE RD-MCTS EVALUATION")
    print("="*80)
    
    print("\n[1/5] Initializing Models...")
    rd_model = DistributionalResNet().to(DEVICE)
    baseline_model = ScalarResNet().to(DEVICE)
    
    rd_trainer = Trainer(rd_model, RDMCTS, is_distributional=True)
    baseline_trainer = Trainer(baseline_model, StandardMCTS, is_distributional=False)
    
    print(f"Device: {DEVICE}")
    print(f"RD-MCTS Parameters: {sum(p.numel() for p in rd_model.parameters()):,}")
    print(f"Baseline Parameters: {sum(p.numel() for p in baseline_model.parameters()):,}")
    
    print("\n[2/5] Testing Initial Trap Avoidance...")
    print("\nRD-MCTS (Untrained) Trap Analysis:")
    rd_trap_initial = evaluate_trap_avoidance(rd_model, RDMCTS, is_distributional=True)
    for move, stats in sorted(rd_trap_initial.items()):
        print(f"  Move {move}: visits={stats['visits']}, mean={stats['mean']:.3f}, std={stats['std']:.3f}, score={stats['score']:.3f}")
    
    print("\nBaseline MCTS (Untrained) Trap Analysis:")
    baseline_trap_initial = evaluate_trap_avoidance(baseline_model, StandardMCTS, is_distributional=False)
    for move, stats in sorted(baseline_trap_initial.items()):
        print(f"  Move {move}: visits={stats['visits']}, value={stats['value']:.3f}, score={stats['score']:.3f}")
    
    print("\n[3/5] Training Both Models...")
    rd_losses = []
    baseline_losses = []
    
    for iteration in range(num_training_iterations):
        print(f"\nIteration {iteration + 1}/{num_training_iterations}")
        
        print("  Training RD-MCTS...")
        rd_loss = rd_trainer.train_iteration(num_games=games_per_iteration, simulations=30)
        rd_losses.append(rd_loss)
        print(f"    Loss: {rd_loss:.4f}")
        
        print("  Training Baseline MCTS...")
        baseline_loss = baseline_trainer.train_iteration(num_games=games_per_iteration, simulations=30)
        baseline_losses.append(baseline_loss)
        print(f"    Loss: {baseline_loss:.4f}")
    
    print("\n[4/5] Testing Post-Training Trap Avoidance...")
    print("\nRD-MCTS (Trained) Trap Analysis:")
    rd_trap_final = evaluate_trap_avoidance(rd_model, RDMCTS, is_distributional=True)
    for move, stats in sorted(rd_trap_final.items()):
        print(f"  Move {move}: visits={stats['visits']}, mean={stats['mean']:.3f}, std={stats['std']:.3f}, score={stats['score']:.3f}")
    
    print("\nBaseline MCTS (Trained) Trap Analysis:")
    baseline_trap_final = evaluate_trap_avoidance(baseline_model, StandardMCTS, is_distributional=False)
    for move, stats in sorted(baseline_trap_final.items()):
        print(f"  Move {move}: visits={stats['visits']}, value={stats['value']:.3f}, score={stats['score']:.3f}")
    
    print("\n[5/5] Head-to-Head Match...")
    wins = play_match(rd_model, RDMCTS, baseline_model, StandardMCTS, num_games=20, simulations=50)
    print(f"\nResults (RD-MCTS vs Baseline):")
    print(f"  RD-MCTS Wins: {wins[0]}")
    print(f"  Baseline Wins: {wins[1]}")
    print(f"  Draws: {wins[2]}")
    
    results = {
        'rd_losses': rd_losses,
        'baseline_losses': baseline_losses,
        'rd_trap_initial': rd_trap_initial,
        'rd_trap_final': rd_trap_final,
        'baseline_trap_initial': baseline_trap_initial,
        'baseline_trap_final': baseline_trap_final,
        'match_results': {
            'rd_wins': wins[0],
            'baseline_wins': wins[1],
            'draws': wins[2]
        }
    }
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    
    return results

if __name__ == "__main__":
    results = run_comprehensive_evaluation(num_training_iterations=5, games_per_iteration=10)
    
    with open('evaluation_results.json', 'w') as f:
        json.dump({
            'rd_losses': results['rd_losses'],
            'baseline_losses': results['baseline_losses'],
            'match_results': results['match_results']
        }, f, indent=2)
    
    print("\nResults saved to evaluation_results.json")
