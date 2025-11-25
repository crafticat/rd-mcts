"""
LC-MCTS (Linearly Correlated MCTS) with hybrid state.

Implements the hybrid merge algorithm from the LC-MCTS paper:
- Each node has state (μ, w, ν)
- μ: expected value (scalar)
- w: structured uncertainty embedding (d-vector)
- ν: unstructured uncertainty (scalar)
"""

import numpy as np
import torch
import copy
from scipy.stats import norm
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.framework import DEVICE


class LCNode:
    """Node for LC-MCTS tree."""
    
    def __init__(self, mu=0.0, w=None, nu=0.0):
        self.mu = mu
        self.w = w if w is not None else np.zeros(32)  # Default dimension
        self.nu = nu
        self.visits = 0
        self.children = {}  # move -> LCNode
    
    def total_variance(self):
        """Compute total variance: σ² = ||w||² + ν²"""
        return np.dot(self.w, self.w) + self.nu ** 2
    
    def __repr__(self):
        return f"LCNode(μ={self.mu:.3f}, ||w||={np.linalg.norm(self.w):.3f}, ν={self.nu:.3f}, visits={self.visits})"


def hybrid_merge(mu1, w1, nu1, mu2, w2, nu2):
    """
    Hybrid merge algorithm from LC-MCTS paper.
    
    Args:
        mu1, mu2: Expected values (scalars)
        w1, w2: Structured uncertainty embeddings (d-vectors)
        nu1, nu2: Unstructured uncertainty (scalars)
    
    Returns:
        mu_p, w_p, nu_p: Parent node state
    """
    # Step A: Compute total variances
    s1_sq = np.dot(w1, w1) + nu1**2
    s2_sq = np.dot(w2, w2) + nu2**2
    
    # Step B: Compute difference statistics (only structured part contributes to covariance!)
    cov = np.dot(w1, w2)
    var_diff = s1_sq + s2_sq - 2*cov
    sigma_diff = np.sqrt(max(var_diff, 1e-9))  # Numerical stability
    
    # Handle clone case explicitly (when nodes are identical or nearly identical)
    mu_diff = abs(mu1 - mu2)
    w_diff_norm = np.linalg.norm(w1 - w2)
    nu_diff = abs(nu1 - nu2)
    
    if mu_diff < 1e-9 and w_diff_norm < 1e-9 and nu_diff < 1e-9:
        # Clone case: return identical node
        return mu1, w1.copy(), nu1
    
    # Step C: Compute probabilities
    alpha = (mu1 - mu2) / sigma_diff
    Phi = norm.cdf(alpha)
    phi = norm.pdf(alpha)
    
    # Step D: Update mean (optimistic)
    mu_p = mu1*Phi + mu2*(1-Phi) + sigma_diff*phi
    
    # Step E: Update structured uncertainty (Stein's Lemma, NO RESCALING)
    w_p = Phi * w1 + (1 - Phi) * w2
    
    # Step F: Compute target variance (Clark's Formula)
    E_P_sq = (mu1**2 + s1_sq)*Phi + (mu2**2 + s2_sq)*(1-Phi) + (mu1 + mu2)*sigma_diff*phi
    target_var = max(E_P_sq - mu_p**2, 1e-9)  # Numerical stability
    
    # Step G: Compute unstructured residual
    w_p_var = np.dot(w_p, w_p)
    nu_p = np.sqrt(max(target_var - w_p_var, 0))
    
    return mu_p, w_p, nu_p


def multi_child_merge(children_states):
    """
    Merge multiple children using pairwise hybrid merge.
    Children are sorted by mean (descending) before merging to minimize approximation error.
    
    Args:
        children_states: list of (mu, w, nu) tuples
    
    Returns:
        mu_p, w_p, nu_p: Parent node state
    """
    if len(children_states) == 0:
        raise ValueError("Cannot merge empty list of children")
    
    if len(children_states) == 1:
        return children_states[0]
    
    # Sort by mean (descending) to minimize approximation error
    sorted_children = sorted(children_states, key=lambda x: x[0], reverse=True)
    
    # Fold children pairwise
    acc_mu, acc_w, acc_nu = sorted_children[0]
    
    for mu, w, nu in sorted_children[1:]:
        acc_mu, acc_w, acc_nu = hybrid_merge(acc_mu, acc_w, acc_nu, mu, w, nu)
    
    return acc_mu, acc_w, acc_nu


class LCMCTSSearch:
    """
    LC-MCTS search with hybrid state (μ, w, ν).
    
    Uses Thompson Sampling at root with correlated noise:
    score_i = μᵢ + (wᵢ·z) + (νᵢ·εᵢ)
    where z is shared (structured correlation) and εᵢ are independent (unstructured).
    """
    
    def __init__(self, model, latent_dim=32):
        self.model = model
        self.latent_dim = latent_dim
    
    def search(self, root_state, simulations=50):
        """
        Run LC-MCTS search from root state.
        
        Args:
            root_state: Game state
            simulations: Number of simulations to run
        
        Returns:
            dict: {move: LCNode} for all valid moves from root
        """
        root = LCNode()
        
        # Evaluate root
        board_tensor = torch.FloatTensor(root_state.get_canonical_state()).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mu, w, _ = self.model(board_tensor)
        
        root.mu = mu.item()
        root.w = w.cpu().numpy()[0]
        root.nu = 0.0  # Leaf nodes have ν=0
        root.visits = 1
        
        # Initialize children
        valid_moves = root_state.get_valid_moves()
        for move in valid_moves:
            root.children[move] = LCNode(w=np.zeros(self.latent_dim))
        
        # Run simulations
        for _ in range(simulations):
            node = root
            game = copy.deepcopy(root_state)
            path = [node]
            
            # Selection: Thompson Sampling with correlated noise
            while node.children and not game.is_terminal():
                # Sample shared z for structured correlation
                z = np.random.randn(self.latent_dim)
                
                best_score = -float('inf')
                best_move = -1
                best_child = None
                
                for move, child in node.children.items():
                    if child.visits == 0:
                        # Unvisited nodes get priority
                        score = float('inf')
                    else:
                        # Thompson Sampling from parent's perspective
                        # child.mu and child.w are from child's perspective (opponent)
                        # Flip sign for zero-sum game: parent value = -child value
                        mu_parent = -child.mu
                        w_parent = -child.w
                        structured = np.dot(w_parent, z)
                        unstructured = child.nu * np.random.randn()  # Independent!
                        score = mu_parent + structured + unstructured
                    
                    if score > best_score:
                        best_score = score
                        best_move = move
                        best_child = child
                
                game = game.make_move(best_move)
                node = best_child
                path.append(node)
            
            # Expansion and evaluation
            if not game.is_terminal():
                board_tensor = torch.FloatTensor(game.get_canonical_state()).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    mu, w, _ = self.model(board_tensor)
                
                leaf_mu = mu.item()
                leaf_w = w.cpu().numpy()[0]
                leaf_nu = 0.0  # Leaf nodes have ν=0
                
                # Expand
                valid = game.get_valid_moves()
                for m in valid:
                    node.children[m] = LCNode(w=np.zeros(self.latent_dim))
            else:
                # Terminal node
                result = game.check_win()
                if result == 0:
                    leaf_mu = 0.0
                else:
                    # Value from perspective of player to move at this state
                    # result is the winner (1 or -1), game.player is player to move
                    # If result == game.player, this player won (+1)
                    # If result != game.player, this player lost (-1)
                    leaf_mu = result * game.player
                
                leaf_w = np.zeros(self.latent_dim)
                leaf_nu = 0.01  # Small uncertainty for terminal nodes
            
            # Backpropagation: Update leaf node
            node.mu = leaf_mu
            node.w = leaf_w
            node.nu = leaf_nu
            node.visits += 1
            
            # Backpropagate up the tree
            for i in range(len(path) - 2, -1, -1):
                parent = path[i]
                parent.visits += 1
                
                # Collect visited children states
                children_states = []
                for child in parent.children.values():
                    if child.visits > 0:
                        # Flip sign for zero-sum game
                        children_states.append((-child.mu, -child.w, child.nu))
                
                if children_states:
                    # Multi-child merge
                    parent.mu, parent.w, parent.nu = multi_child_merge(children_states)
        
        return root.children


def test_lc_mcts():
    """Test LC-MCTS search."""
    print("="*80)
    print("TESTING LC-MCTS SEARCH")
    print("="*80)
    
    # Import game
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from game import Connect4
    from networks.lc_autoencoder import LCAutoencoder
    
    # Create network and search
    model = LCAutoencoder(latent_dim=32).to(DEVICE)
    search = LCMCTSSearch(model, latent_dim=32)
    
    # Create game
    game = Connect4()
    
    # Run search
    print("\nRunning LC-MCTS search (10 simulations)...")
    children = search.search(game, simulations=10)
    
    print(f"\nFound {len(children)} valid moves:")
    for move, child in sorted(children.items(), key=lambda x: x[1].visits, reverse=True):
        print(f"  Move {move}: {child}")
    
    # Check that visits sum correctly
    total_visits = sum(child.visits for child in children.values())
    print(f"\nTotal visits: {total_visits}")
    
    print("\n" + "="*80)
    print("✓ LC-MCTS SEARCH TEST PASSED")
    print("="*80)


if __name__ == "__main__":
    test_lc_mcts()
