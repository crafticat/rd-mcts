"""
Value-only MCTS search algorithms (pure Thompson Sampling, no policy priors).
This is the correct implementation for testing the paper's claims.
"""

import numpy as np
import torch
import copy
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.framework import (
    SearchNode, SearchConfig, convolve_max_and_penalize,
    ATOMS, SUPPORT_SIZE, DEVICE, project_gaussian
)


class ValueOnlyRDMCTSSearch:
    """
    RD-MCTS with pure Thompson Sampling (no policy priors, no UCB).
    
    Selection: Sample from value distribution, pick argmax
    Backpropagation: Penalized max-convolution
    """
    
    def __init__(self, model, config: SearchConfig):
        self.model = model
        self.config = config
    
    def search(self, root_state, simulations=50):
        root = SearchNode(prior=1.0)  # Uniform prior (not used)
        
        # Evaluate root
        board_tensor = torch.FloatTensor(root_state.get_canonical_state()).to(DEVICE)
        with torch.no_grad():
            value_rep = self.model(board_tensor)
        
        root.dist = self.model.value_head.to_distribution(value_rep)
        valid_moves = root_state.get_valid_moves()
        
        # Initialize children with uniform priors (not used in selection)
        uniform_prior = 1.0 / len(valid_moves)
        for move in valid_moves:
            root.children[move] = SearchNode(prior=uniform_prior)
        
        for _ in range(simulations):
            node = root
            game = copy.deepcopy(root_state)
            path = [node]
            
            # Selection: Pure Thompson Sampling (no UCB term)
            while node.children:
                best_score = -float('inf')
                best_move = -1
                best_child = None
                
                for move, child in node.children.items():
                    # Ensure unvisited nodes get explored first
                    if child.visits == 0:
                        score = float('inf')
                    else:
                        # Pure Thompson Sampling: sample from distribution
                        if self.config.use_thompson:
                            sample_v = np.random.choice(ATOMS, p=child.dist)
                        else:
                            # Fallback to mean if Thompson disabled
                            sample_v = np.sum(child.dist * ATOMS)
                        
                        # No UCB term, no policy prior - just the sampled value
                        score = sample_v
                    
                    if score > best_score:
                        best_score = score
                        best_move = move
                        best_child = child
                
                game = game.make_move(best_move)
                node = best_child
                path.append(node)
            
            # Expansion and evaluation
            if not game.is_terminal():
                board_tensor = torch.FloatTensor(game.get_canonical_state()).to(DEVICE)
                with torch.no_grad():
                    value_rep = self.model(board_tensor)
                leaf_dist = self.model.value_head.to_distribution(value_rep)
                
                # Expand with uniform priors
                valid = game.get_valid_moves()
                uniform_prior = 1.0 / len(valid) if valid else 1.0
                for m in valid:
                    node.children[m] = SearchNode(prior=uniform_prior)
            else:
                result = game.check_win()
                if result == 0:
                    result = 0.0
                else:
                    result = -result * game.player
                leaf_dist = project_gaussian(result, 0.01)
            
            # Backpropagation: Penalized max-convolution
            node.visits += 1
            node.dist = leaf_dist
            
            for i in range(len(path) - 2, -1, -1):
                parent = path[i]
                parent.visits += 1
                
                # Collect child distributions (inverted for zero-sum)
                child_dists_inverted = []
                for child in parent.children.values():
                    if child.visits > 0:
                        d_inv = np.flip(child.dist)
                        child_dists_inverted.append(d_inv)
                
                if child_dists_inverted:
                    lambda_param = self.config.lambda_penalty if self.config.use_penalty else 0.0
                    new_dist = convolve_max_and_penalize(child_dists_inverted, lambda_param)
                    parent.dist = new_dist
        
        return root.children


class ValueOnlyScalarMCTSSearch:
    """
    Scalar MCTS with pure Thompson Sampling (no policy priors, no UCB).
    
    Selection: Sample from scalar value with uncertainty, pick argmax
    Backpropagation: Standard averaging
    """
    
    def __init__(self, model, config: SearchConfig):
        self.model = model
        self.config = config
    
    def search(self, root_state, simulations=50):
        root = SearchNode(prior=1.0)  # Uniform prior (not used)
        
        # Evaluate root
        board_tensor = torch.FloatTensor(root_state.get_canonical_state()).to(DEVICE)
        with torch.no_grad():
            value_rep = self.model(board_tensor)
        
        root.value_sum = self.model.value_head.to_scalar(value_rep)
        root.visits = 1
        root.uncertainty = 0.5  # Initial uncertainty
        valid_moves = root_state.get_valid_moves()
        
        # Initialize children with uniform priors (not used in selection)
        uniform_prior = 1.0 / len(valid_moves)
        for move in valid_moves:
            child = SearchNode(prior=uniform_prior)
            child.uncertainty = 0.5
            root.children[move] = child
        
        for _ in range(simulations):
            node = root
            game = copy.deepcopy(root_state)
            path = [node]
            
            # Selection: Pure Thompson Sampling (no UCB term)
            while node.children:
                best_score = -float('inf')
                best_move = -1
                best_child = None
                
                for move, child in node.children.items():
                    # Ensure unvisited nodes get explored first
                    if child.visits == 0:
                        score = float('inf')
                    else:
                        q_value = child.get_value()
                        uncertainty = child.uncertainty if hasattr(child, 'uncertainty') else 0.5
                        
                        # Pure Thompson Sampling: sample from Gaussian
                        if self.config.use_thompson:
                            sample_v = np.random.normal(q_value, uncertainty)
                        else:
                            sample_v = q_value
                        
                        # No UCB term, no policy prior - just the sampled value
                        score = sample_v
                    
                    if score > best_score:
                        best_score = score
                        best_move = move
                        best_child = child
                
                game = game.make_move(best_move)
                node = best_child
                path.append(node)
            
            # Expansion and evaluation
            if not game.is_terminal():
                board_tensor = torch.FloatTensor(game.get_canonical_state()).to(DEVICE)
                with torch.no_grad():
                    value_rep = self.model(board_tensor)
                leaf_value = self.model.value_head.to_scalar(value_rep)
                
                # Expand with uniform priors
                valid = game.get_valid_moves()
                uniform_prior = 1.0 / len(valid) if valid else 1.0
                for m in valid:
                    child = SearchNode(prior=uniform_prior)
                    child.uncertainty = 0.5
                    node.children[m] = child
            else:
                result = game.check_win()
                if result == 0:
                    leaf_value = 0.0
                else:
                    leaf_value = -result * game.player
            
            # Backpropagation: Standard averaging with uncertainty tracking
            for n in path:
                n.visits += 1
                old_mean = n.value_sum / max(1, n.visits - 1) if n.visits > 1 else 0
                n.value_sum += leaf_value
                new_mean = n.value_sum / n.visits
                
                # Track variance for uncertainty estimation
                if not hasattr(n, 'M2'):
                    n.M2 = 0.0
                delta = leaf_value - old_mean
                delta2 = leaf_value - new_mean
                n.M2 += delta * delta2
                
                n.uncertainty = np.sqrt(n.M2 / n.visits) if n.visits > 1 else 0.5
                
                leaf_value = -leaf_value
        
        return root.children
