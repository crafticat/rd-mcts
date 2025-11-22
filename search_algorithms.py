"""
Different MCTS search algorithm implementations.
"""

import numpy as np
import torch
import copy
from framework import (
    Search, SearchNode, SearchConfig, convolve_max_and_penalize,
    ATOMS, SUPPORT_SIZE, DEVICE, project_gaussian
)

class RDMCTSSearch(Search):
    """RD-MCTS with Thompson Sampling and penalized max-convolution."""
    
    def search(self, root_state, simulations=50):
        root = SearchNode(0)
        
        board_tensor = torch.FloatTensor(root_state.get_canonical_state()).to(DEVICE)
        with torch.no_grad():
            pi, value_rep = self.model(board_tensor)
        
        root.dist = self.model.value_head.to_distribution(value_rep)
        valid_moves = root_state.get_valid_moves()
        
        noise = np.random.dirichlet([0.3] * len(valid_moves))
        
        for idx, move in enumerate(valid_moves):
            root.children[move] = SearchNode(prior=np.exp(pi.cpu().numpy()[0][move]))
            root.children[move].prior = 0.75 * root.children[move].prior + 0.25 * noise[idx]
        
        for _ in range(simulations):
            node = root
            game = copy.deepcopy(root_state)
            path = [node]
            
            while node.children:
                best_score = -float('inf')
                best_move = -1
                best_child = None
                
                for move, child in node.children.items():
                    if self.config.use_thompson:
                        sample_v = np.random.choice(ATOMS, p=child.dist)
                    else:
                        sample_v = np.sum(child.dist * ATOMS)
                    
                    u = self.config.c_puct * child.prior * np.sqrt(node.visits) / (1 + child.visits)
                    score = sample_v + u
                    
                    if score > best_score:
                        best_score = score
                        best_move = move
                        best_child = child
                
                game = game.make_move(best_move)
                node = best_child
                path.append(node)
            
            if not game.is_terminal():
                board_tensor = torch.FloatTensor(game.get_canonical_state()).to(DEVICE)
                with torch.no_grad():
                    pi, value_rep = self.model(board_tensor)
                leaf_dist = self.model.value_head.to_distribution(value_rep)
                
                valid = game.get_valid_moves()
                probs = np.exp(pi.cpu().numpy()[0])
                for m in valid:
                    node.children[m] = SearchNode(probs[m])
            else:
                result = game.check_win()
                if result == 0:
                    result = 0.0
                else:
                    result = -result * game.player
                leaf_dist = project_gaussian(result, 0.01)
            
            node.visits += 1
            node.dist = leaf_dist
            
            for i in range(len(path) - 2, -1, -1):
                parent = path[i]
                parent.visits += 1
                
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

class StandardMCTSSearch(Search):
    """Standard MCTS with UCB/PUCT selection."""
    
    def search(self, root_state, simulations=50):
        root = SearchNode(0)
        
        board_tensor = torch.FloatTensor(root_state.get_canonical_state()).to(DEVICE)
        with torch.no_grad():
            pi, value_rep = self.model(board_tensor)
        
        root.value_sum = self.model.value_head.to_scalar(value_rep)
        root.visits = 1
        valid_moves = root_state.get_valid_moves()
        
        noise = np.random.dirichlet([0.3] * len(valid_moves))
        
        for idx, move in enumerate(valid_moves):
            root.children[move] = SearchNode(prior=np.exp(pi.cpu().numpy()[0][move]))
            root.children[move].prior = 0.75 * root.children[move].prior + 0.25 * noise[idx]
        
        for _ in range(simulations):
            node = root
            game = copy.deepcopy(root_state)
            path = [node]
            
            while node.children:
                best_score = -float('inf')
                best_move = -1
                best_child = None
                
                for move, child in node.children.items():
                    q_value = child.get_value()
                    u = self.config.c_puct * child.prior * np.sqrt(node.visits) / (1 + child.visits)
                    score = q_value + u
                    
                    if score > best_score:
                        best_score = score
                        best_move = move
                        best_child = child
                
                game = game.make_move(best_move)
                node = best_child
                path.append(node)
            
            if not game.is_terminal():
                board_tensor = torch.FloatTensor(game.get_canonical_state()).to(DEVICE)
                with torch.no_grad():
                    pi, value_rep = self.model(board_tensor)
                leaf_value = self.model.value_head.to_scalar(value_rep)
                
                valid = game.get_valid_moves()
                probs = np.exp(pi.cpu().numpy()[0])
                for m in valid:
                    node.children[m] = SearchNode(probs[m])
            else:
                result = game.check_win()
                if result == 0:
                    leaf_value = 0.0
                else:
                    leaf_value = -result * game.player
            
            for n in path:
                n.visits += 1
                n.value_sum += leaf_value
                leaf_value = -leaf_value
        
        return root.children

class ThompsonScalarMCTSSearch(Search):
    """MCTS with Thompson Sampling but scalar values (with uncertainty estimate)."""
    
    def search(self, root_state, simulations=50):
        root = SearchNode(0)
        
        board_tensor = torch.FloatTensor(root_state.get_canonical_state()).to(DEVICE)
        with torch.no_grad():
            pi, value_rep = self.model(board_tensor)
        
        root.value_sum = self.model.value_head.to_scalar(value_rep)
        root.visits = 1
        root.uncertainty = 0.5
        valid_moves = root_state.get_valid_moves()
        
        noise = np.random.dirichlet([0.3] * len(valid_moves))
        
        for idx, move in enumerate(valid_moves):
            root.children[move] = SearchNode(prior=np.exp(pi.cpu().numpy()[0][move]))
            root.children[move].prior = 0.75 * root.children[move].prior + 0.25 * noise[idx]
            root.children[move].uncertainty = 0.5
        
        for _ in range(simulations):
            node = root
            game = copy.deepcopy(root_state)
            path = [node]
            
            while node.children:
                best_score = -float('inf')
                best_move = -1
                best_child = None
                
                for move, child in node.children.items():
                    q_value = child.get_value()
                    uncertainty = child.uncertainty if hasattr(child, 'uncertainty') else 0.5
                    
                    sample_v = np.random.normal(q_value, uncertainty)
                    
                    u = self.config.c_puct * child.prior * np.sqrt(node.visits) / (1 + child.visits)
                    score = sample_v + u
                    
                    if score > best_score:
                        best_score = score
                        best_move = move
                        best_child = child
                
                game = game.make_move(best_move)
                node = best_child
                path.append(node)
            
            if not game.is_terminal():
                board_tensor = torch.FloatTensor(game.get_canonical_state()).to(DEVICE)
                with torch.no_grad():
                    pi, value_rep = self.model(board_tensor)
                leaf_value = self.model.value_head.to_scalar(value_rep)
                
                valid = game.get_valid_moves()
                probs = np.exp(pi.cpu().numpy()[0])
                for m in valid:
                    node.children[m] = SearchNode(probs[m])
                    node.children[m].uncertainty = 0.5
            else:
                result = game.check_win()
                if result == 0:
                    leaf_value = 0.0
                else:
                    leaf_value = -result * game.player
            
            for n in path:
                n.visits += 1
                old_mean = n.value_sum / max(1, n.visits - 1) if n.visits > 1 else 0
                n.value_sum += leaf_value
                new_mean = n.value_sum / n.visits
                
                if not hasattr(n, 'M2'):
                    n.M2 = 0.0
                delta = leaf_value - old_mean
                delta2 = leaf_value - new_mean
                n.M2 += delta * delta2
                
                n.uncertainty = np.sqrt(n.M2 / n.visits) if n.visits > 1 else 0.5
                
                leaf_value = -leaf_value
        
        return root.children
