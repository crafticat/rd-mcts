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

class UCBVMCTSSearch(Search):
    """UCB1-Tuned / UCB-V: Variance-aware UCB for risk-sensitive search."""
    
    def search(self, root_state, simulations=50):
        root = SearchNode(0)
        
        board_tensor = torch.FloatTensor(root_state.get_canonical_state()).to(DEVICE)
        with torch.no_grad():
            pi, value_rep = self.model(board_tensor)
        
        root.value_sum = self.model.value_head.to_scalar(value_rep)
        root.visits = 1
        root.M2 = 0.0
        valid_moves = root_state.get_valid_moves()
        
        noise = np.random.dirichlet([0.3] * len(valid_moves))
        
        for idx, move in enumerate(valid_moves):
            root.children[move] = SearchNode(prior=np.exp(pi.cpu().numpy()[0][move]))
            root.children[move].prior = 0.75 * root.children[move].prior + 0.25 * noise[idx]
            root.children[move].M2 = 0.0
        
        for _ in range(simulations):
            node = root
            game = copy.deepcopy(root_state)
            path = [node]
            
            while node.children:
                best_score = -float('inf')
                best_move = -1
                best_child = None
                
                for move, child in node.children.items():
                    if child.visits == 0:
                        score = float('inf')
                    else:
                        q_value = child.get_value()
                        
                        # UCB-V: variance-aware exploration
                        variance = child.M2 / child.visits if child.visits > 0 else 0.25
                        
                        # UCB1-Tuned formula: includes variance in exploration term
                        V_bound = variance + np.sqrt(2 * np.log(node.visits) / child.visits)
                        V_bound = min(V_bound, 0.25)  # Clamp to [0, 0.25] for values in [-1, 1]
                        
                        exploration = np.sqrt(np.log(node.visits) / child.visits * V_bound)
                        
                        # Add policy prior like PUCT
                        u = self.config.c_puct * child.prior * np.sqrt(node.visits) / (1 + child.visits)
                        
                        score = q_value + exploration + u
                    
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
                    node.children[m].M2 = 0.0
            else:
                result = game.check_win()
                if result == 0:
                    leaf_value = 0.0
                else:
                    leaf_value = -result * game.player
            
            # Backpropagate with variance tracking
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
                
                leaf_value = -leaf_value
        
        return root.children

class RAVEMCTSSearch(Search):
    """RAVE (Rapid Action Value Estimation): Uses AMAF statistics for faster learning."""
    
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
            # AMAF (All-Moves-As-First) statistics
            root.children[move].amaf_sum = 0.0
            root.children[move].amaf_visits = 0
        
        for _ in range(simulations):
            node = root
            game = copy.deepcopy(root_state)
            path = [node]
            moves_played = []
            
            while node.children:
                best_score = -float('inf')
                best_move = -1
                best_child = None
                
                for move, child in node.children.items():
                    if child.visits == 0:
                        score = float('inf')
                    else:
                        q_value = child.get_value()
                        
                        # RAVE: Mix regular Q with AMAF Q
                        amaf_q = child.amaf_sum / child.amaf_visits if child.amaf_visits > 0 else 0
                        
                        # Beta schedule: weight AMAF more early, regular Q more later
                        # Common formula: β = sqrt(k / (3*n + k)) where k is a constant
                        k = 100  # Tunable parameter
                        beta = np.sqrt(k / (3 * child.visits + k))
                        
                        mixed_q = (1 - beta) * q_value + beta * amaf_q
                        
                        # UCB exploration
                        u = self.config.c_puct * child.prior * np.sqrt(node.visits) / (1 + child.visits)
                        
                        score = mixed_q + u
                    
                    if score > best_score:
                        best_score = score
                        best_move = move
                        best_child = child
                
                moves_played.append(best_move)
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
                    node.children[m].amaf_sum = 0.0
                    node.children[m].amaf_visits = 0
            else:
                result = game.check_win()
                if result == 0:
                    leaf_value = 0.0
                else:
                    leaf_value = -result * game.player
            
            # Backpropagate regular statistics
            for n in path:
                n.visits += 1
                n.value_sum += leaf_value
                leaf_value = -leaf_value
            
            # Update AMAF statistics for all moves played
            # For each node in path, update AMAF for all moves that were played later
            for i, node in enumerate(path[:-1]):  # Exclude leaf
                amaf_value = leaf_value if i % 2 == 0 else -leaf_value
                for move in moves_played[i:]:
                    if move in node.children:
                        if not hasattr(node.children[move], 'amaf_sum'):
                            node.children[move].amaf_sum = 0.0
                            node.children[move].amaf_visits = 0
                        node.children[move].amaf_sum += amaf_value
                        node.children[move].amaf_visits += 1
        
        return root.children
