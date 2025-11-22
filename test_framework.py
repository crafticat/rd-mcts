"""
Quick test to verify the framework works before running full experiments.
"""

import torch
import numpy as np

from framework import (
    UnifiedNetwork, CategoricalValueHead, GaussianValueHead, ScalarValueHead,
    SearchConfig, DEVICE
)
from search_algorithms import RDMCTSSearch, StandardMCTSSearch
from rd_mcts_experiment import Connect4

def test_value_heads():
    """Test all value head types."""
    print("Testing value heads...")
    
    game = Connect4()
    state = torch.FloatTensor(game.get_canonical_state()).to(DEVICE)
    
    for name, head_class in [("Categorical", CategoricalValueHead), 
                              ("Gaussian", GaussianValueHead), 
                              ("Scalar", ScalarValueHead)]:
        print(f"  Testing {name} head...")
        head = head_class()
        model = UnifiedNetwork(head).to(DEVICE)
        
        pi, value_rep = model(state)
        assert pi.shape == (1, 7), f"Policy shape wrong: {pi.shape}"
        
        scalar = model.value_head.to_scalar(value_rep)
        assert -1 <= scalar <= 1, f"Scalar value out of range: {scalar}"
        
        dist = model.value_head.to_distribution(value_rep)
        assert dist.shape == (51,), f"Distribution shape wrong: {dist.shape}"
        assert abs(dist.sum() - 1.0) < 1e-4, f"Distribution doesn't sum to 1: {dist.sum()}"
        
        print(f"    ✓ {name} head works (scalar={scalar:.3f})")
    
    print("✓ All value heads work!\n")

def test_search_algorithms():
    """Test all search algorithms."""
    print("Testing search algorithms...")
    
    game = Connect4()
    config = SearchConfig()
    
    for name, search_class in [("RD-MCTS", RDMCTSSearch), 
                                ("Standard", StandardMCTSSearch)]:
        print(f"  Testing {name}...")
        head = CategoricalValueHead()
        model = UnifiedNetwork(head).to(DEVICE)
        
        search = search_class(model, config)
        children = search.search(game, simulations=10)
        
        assert len(children) > 0, "No children returned"
        total_visits = sum(c.visits for c in children.values())
        assert total_visits >= 10, f"Not enough visits: {total_visits}"
        
        print(f"    ✓ {name} works ({len(children)} children, {total_visits} visits)")
    
    print("✓ All search algorithms work!\n")

def test_gaussian_head_training():
    """Test that Gaussian head can compute gradients."""
    print("Testing Gaussian head gradients...")
    
    head = GaussianValueHead()
    model = UnifiedNetwork(head).to(DEVICE)
    
    game = Connect4()
    state = torch.FloatTensor(game.get_canonical_state()).to(DEVICE)
    target = torch.FloatTensor([0.5]).to(DEVICE)
    
    pi, value_rep = model(state)
    loss = model.value_head.loss(value_rep, target)
    loss.backward()
    
    has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_gradients, "No gradients computed!"
    
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradients computed: {has_gradients}")
    print("✓ Gaussian head gradients work!\n")

if __name__ == "__main__":
    print("="*60)
    print("FRAMEWORK VALIDATION TESTS")
    print("="*60)
    print()
    
    test_value_heads()
    test_search_algorithms()
    test_gaussian_head_training()
    
    print("="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
