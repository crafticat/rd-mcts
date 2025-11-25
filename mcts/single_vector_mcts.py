"""
Single-vector correlation MCTS variants from the LC-MCTS paper.

These are the "old approaches" that have issues:
- Original: Uses Stein's Lemma merge but suffers from variance collapse
- Rescaled: Fixes variance but breaks correlations
"""

import numpy as np
import torch
import copy
from scipy.stats import norm
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.framework import DEVICE


class SVNode:
    """Node for single-vector correlation MCTS."""
    
    def __init__(self, mu=0.0, w=None):
        self.mu = mu
        self.w = w if w is not None else np.zeros(32)
        self.visits = 0
        self.children = {}
    
    def variance(self):
        """Compute variance: σ² = ||w||²"""
        return np.dot(self.w, self.w)
    
    def __repr__(self):
        return f"SVNode(μ={self.mu:.3f}, ||w||={np.linalg.norm(self.w):.3f}, visits={self.visits})"


def single_vector_merge_original(mu1, w1, mu2, w2):
    """
    Original single-vector merge (Stein's Lemma only).
    
    Issues:
    - Variance collapse: ||w_p|| < max(||w1||, ||w2||)
    - Underestimates uncertainty
    
    Args:
        mu1, mu2: Expected values
        w1, w2: Uncertainty embeddings
    
    Returns:
        mu_p, w_p: Parent state
    """
    # Compute difference statistics
    var1 = np.dot(w1, w1)
    var2 = np.dot(w2, w2)
    cov = np.dot(w1, w2)
    var_diff = var1 + var2 - 2*cov
    sigma_diff = np.sqrt(max(var_diff, 1e-9))
    
    # Handle clone case
    if abs(mu1 - mu2) < 1e-9 and np.linalg.norm(w1 - w2) < 1e-9:
        return mu1, w1.copy()
    
    # Probabilities
    alpha = (mu1 - mu2) / sigma_diff
    Phi = norm.cdf(alpha)
    phi = norm.pdf(alpha)
    
    # Update mean (optimistic)
    mu_p = mu1*Phi + mu2*(1-Phi) + sigma_diff*phi
    
    # Update embedding (Stein's Lemma, NO RESCALING)
    w_p = Phi * w1 + (1 - Phi) * w2
    
    # NOTE: This suffers from variance collapse!
    # ||w_p||² < ||w1||² and ||w_p||² < ||w2||² in general
    
    return mu_p, w_p


def single_vector_merge_rescaled(mu1, w1, mu2, w2):
    """
    Rescaled single-vector merge (fixes variance but breaks correlation).
    
    Issues:
    - Breaks correlations with external nodes
    - Rescaling changes all dot products
    
    Args:
        mu1, mu2: Expected values
        w1, w2: Uncertainty embeddings
    
    Returns:
        mu_p, w_p: Parent state
    """
    # First do original merge
    mu_p, w_p_unscaled = single_vector_merge_original(mu1, w1, mu2, w2)
    
    # Compute target variance using Clark's formula
    var1 = np.dot(w1, w1)
    var2 = np.dot(w2, w2)
    cov = np.dot(w1, w2)
    var_diff = var1 + var2 - 2*cov
    sigma_diff = np.sqrt(max(var_diff, 1e-9))
    
    alpha = (mu1 - mu2) / sigma_diff
    Phi = norm.cdf(alpha)
    phi = norm.pdf(alpha)
    
    E_P_sq = (mu1**2 + var1)*Phi + (mu2**2 + var2)*(1-Phi) + (mu1 + mu2)*sigma_diff*phi
    target_var = max(E_P_sq - mu_p**2, 1e-9)
    
    # Rescale w_p to match target variance
    current_var = np.dot(w_p_unscaled, w_p_unscaled)
    
    if current_var > 1e-9:
        scale = np.sqrt(target_var / current_var)
        w_p = w_p_unscaled * scale
    else:
        # If w_p is zero, can't rescale - just return it
        w_p = w_p_unscaled
    
    # NOTE: This breaks correlations!
    # Cov(P, C) = w_p · w_c = scale * (w_p_unscaled · w_c) ≠ expected covariance
    
    return mu_p, w_p


def multi_child_merge_sv(children_states, use_rescaling=False):
    """
    Merge multiple children using single-vector merge.
    
    Args:
        children_states: list of (mu, w) tuples
        use_rescaling: if True, use rescaled merge; otherwise use original
    
    Returns:
        mu_p, w_p: Parent state
    """
    if len(children_states) == 0:
        raise ValueError("Cannot merge empty list")
    
    if len(children_states) == 1:
        return children_states[0]
    
    # Sort by mean (descending)
    sorted_children = sorted(children_states, key=lambda x: x[0], reverse=True)
    
    # Fold children pairwise
    acc_mu, acc_w = sorted_children[0]
    
    merge_fn = single_vector_merge_rescaled if use_rescaling else single_vector_merge_original
    
    for mu, w in sorted_children[1:]:
        acc_mu, acc_w = merge_fn(acc_mu, acc_w, mu, w)
    
    return acc_mu, acc_w


class SingleVectorMCTSSearch:
    """
    Single-vector correlation MCTS.
    
    Can use either:
    - Original merge (variance collapse)
    - Rescaled merge (breaks correlations)
    """
    
    def __init__(self, model, latent_dim=32, use_rescaling=False):
        self.model = model
        self.latent_dim = latent_dim
        self.use_rescaling = use_rescaling
    
    def search(self, root_state, simulations=50):
        """Run single-vector MCTS search."""
        root = SVNode()
        
        # Evaluate root
        board_tensor = torch.FloatTensor(root_state.get_canonical_state()).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mu, w, _ = self.model(board_tensor)
        
        root.mu = mu.item()
        root.w = w.cpu().numpy()[0]
        root.visits = 1
        
        # Initialize children
        valid_moves = root_state.get_valid_moves()
        for move in valid_moves:
            root.children[move] = SVNode(w=np.zeros(self.latent_dim))
        
        # Run simulations
        for _ in range(simulations):
            node = root
            game = copy.deepcopy(root_state)
            path = [node]
            
            # Selection: Thompson Sampling
            while node.children and not game.is_terminal():
                z = np.random.randn(self.latent_dim)
                
                best_score = -float('inf')
                best_move = -1
                best_child = None
                
                for move, child in node.children.items():
                    if child.visits == 0:
                        score = float('inf')
                    else:
                        # Thompson Sampling from parent's perspective
                        # Flip sign for zero-sum game
                        mu_parent = -child.mu
                        w_parent = -child.w
                        score = mu_parent + np.dot(w_parent, z)
                    
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
                
                # Expand
                valid = game.get_valid_moves()
                for m in valid:
                    node.children[m] = SVNode(w=np.zeros(self.latent_dim))
            else:
                # Terminal node
                result = game.check_win()
                if result == 0:
                    leaf_mu = 0.0
                else:
                    # Value from perspective of player to move at this state
                    leaf_mu = result * game.player
                
                leaf_w = np.zeros(self.latent_dim)
            
            # Backpropagation
            node.mu = leaf_mu
            node.w = leaf_w
            node.visits += 1
            
            for i in range(len(path) - 2, -1, -1):
                parent = path[i]
                parent.visits += 1
                
                # Collect visited children
                children_states = []
                for child in parent.children.values():
                    if child.visits > 0:
                        children_states.append((-child.mu, -child.w))
                
                if children_states:
                    parent.mu, parent.w = multi_child_merge_sv(children_states, self.use_rescaling)
        
        return root.children


def test_single_vector_mcts():
    """Test single-vector MCTS variants."""
    print("="*80)
    print("TESTING SINGLE-VECTOR MCTS")
    print("="*80)
    
    # Import dependencies
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from game import Connect4
    from networks.lc_autoencoder import LCAutoencoder
    
    # Create network
    model = LCAutoencoder(latent_dim=32).to(DEVICE)
    
    # Test Original variant
    print("\n--- Testing Original (no rescaling) ---")
    search_orig = SingleVectorMCTSSearch(model, latent_dim=32, use_rescaling=False)
    game = Connect4()
    children_orig = search_orig.search(game, simulations=10)
    
    print(f"Found {len(children_orig)} moves:")
    for move, child in sorted(children_orig.items(), key=lambda x: x[1].visits, reverse=True):
        print(f"  Move {move}: {child}")
    
    # Test Rescaled variant
    print("\n--- Testing Rescaled ---")
    search_resc = SingleVectorMCTSSearch(model, latent_dim=32, use_rescaling=True)
    game = Connect4()
    children_resc = search_resc.search(game, simulations=10)
    
    print(f"Found {len(children_resc)} moves:")
    for move, child in sorted(children_resc.items(), key=lambda x: x[1].visits, reverse=True):
        print(f"  Move {move}: {child}")
    
    print("\n" + "="*80)
    print("✓ SINGLE-VECTOR MCTS TEST PASSED")
    print("="*80)


if __name__ == "__main__":
    test_single_vector_mcts()
