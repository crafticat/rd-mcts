#!/usr/bin/env python3
"""
Comprehensive test suite to validate RD-MCTS implementation
"""

import numpy as np
import torch
from rd_mcts_experiment import Connect4, DistributionalResNet, RDMCTS, dist_stats, convolve_max_and_penalize, project_gaussian, SUPPORT_SIZE, ATOMS
from baseline_mcts import ScalarResNet, StandardMCTS

def test_connect4_basic():
    """Test basic Connect4 functionality"""
    print("Testing Connect4 basic functionality...")
    game = Connect4()
    assert len(game.get_valid_moves()) == 7, "Empty board should have 7 valid moves"
    assert not game.is_terminal(), "Empty board should not be terminal"
    assert game.check_win() == 0, "Empty board should have no winner"
    print("  ✓ Basic functionality works")

def test_connect4_win_detection():
    """Test Connect4 win detection"""
    print("Testing Connect4 win detection...")
    
    game = Connect4()
    for move in [3, 4, 3, 4, 3, 4, 3]:
        game = game.make_move(move)
    
    winner = game.check_win()
    assert winner != 0, "Vertical 4-in-a-row should be detected"
    print(f"  ✓ Win detection works (winner: {winner})")

def test_distributional_network():
    """Test distributional network output"""
    print("Testing distributional network...")
    model = DistributionalResNet()
    game = Connect4()
    
    state = torch.FloatTensor(game.get_canonical_state())
    pi, v_dist = model(state)
    
    assert pi.shape == (1, 7), f"Policy shape should be (1, 7), got {pi.shape}"
    assert v_dist.shape == (1, SUPPORT_SIZE), f"Value dist shape should be (1, {SUPPORT_SIZE}), got {v_dist.shape}"
    assert torch.allclose(v_dist.sum(dim=1), torch.ones(1), atol=1e-5), "Value distribution should sum to 1"
    print("  ✓ Distributional network works")

def test_scalar_network():
    """Test scalar network output"""
    print("Testing scalar network...")
    model = ScalarResNet()
    game = Connect4()
    
    state = torch.FloatTensor(game.get_canonical_state())
    pi, v = model(state)
    
    assert pi.shape == (1, 7), f"Policy shape should be (1, 7), got {pi.shape}"
    assert v.shape == (1, 1), f"Value shape should be (1, 1), got {v.shape}"
    assert -1 <= v.item() <= 1, f"Value should be in [-1, 1], got {v.item()}"
    print("  ✓ Scalar network works")

def test_dist_stats():
    """Test distribution statistics calculation"""
    print("Testing distribution statistics...")
    
    dist = project_gaussian(0.5, 0.2)
    mean, std = dist_stats(dist)
    
    assert abs(mean - 0.5) < 0.1, f"Mean should be close to 0.5, got {mean}"
    assert abs(std - 0.2) < 0.1, f"Std should be close to 0.2, got {std}"
    print(f"  ✓ Distribution stats work (mean={mean:.3f}, std={std:.3f})")

def test_max_convolution():
    """Test penalized max-convolution"""
    print("Testing penalized max-convolution...")
    
    dist1 = project_gaussian(0.3, 0.1)
    dist2 = project_gaussian(0.5, 0.1)
    dist3 = project_gaussian(0.4, 0.3)
    
    result = convolve_max_and_penalize([dist1, dist2, dist3], lambda_param=1.0)
    
    assert result.shape == (SUPPORT_SIZE,), f"Result shape should be ({SUPPORT_SIZE},), got {result.shape}"
    assert abs(result.sum() - 1.0) < 1e-5, f"Result should sum to 1, got {result.sum()}"
    
    mean, std = dist_stats(result)
    print(f"  ✓ Max-convolution works (result mean={mean:.3f}, std={std:.3f})")

def test_rdmcts_search():
    """Test RD-MCTS search"""
    print("Testing RD-MCTS search...")
    
    model = DistributionalResNet()
    mcts = RDMCTS(model)
    game = Connect4()
    
    children = mcts.search(game, simulations=10)
    
    assert len(children) > 0, "Search should return children"
    total_visits = sum(child.visits for child in children.values())
    assert total_visits >= 10, f"Total visits should be at least 10, got {total_visits}"
    
    visited_children = 0
    for move, child in children.items():
        assert 0 <= move <= 6, f"Move should be in [0, 6], got {move}"
        if child.visits > 0:
            visited_children += 1
            mean, std = dist_stats(child.dist)
            assert -1 <= mean <= 1, f"Mean should be in [-1, 1], got {mean}"
    
    assert visited_children > 0, "At least one child should have visits"
    print(f"  ✓ RD-MCTS search works ({len(children)} children, {visited_children} visited, {total_visits} total visits)")

def test_baseline_mcts_search():
    """Test baseline MCTS search"""
    print("Testing baseline MCTS search...")
    
    model = ScalarResNet()
    mcts = StandardMCTS(model)
    game = Connect4()
    
    children = mcts.search(game, simulations=10)
    
    assert len(children) > 0, "Search should return children"
    total_visits = sum(child.visits for child in children.values())
    assert total_visits >= 10, f"Total visits should be at least 10, got {total_visits}"
    
    visited_children = 0
    for move, child in children.items():
        assert 0 <= move <= 6, f"Move should be in [0, 6], got {move}"
        if child.visits > 0:
            visited_children += 1
            value = child.get_value()
            assert -1 <= value <= 1, f"Value should be in [-1, 1], got {value}"
    
    assert visited_children > 0, "At least one child should have visits"
    print(f"  ✓ Baseline MCTS search works ({len(children)} children, {visited_children} visited, {total_visits} total visits)")

def test_thompson_sampling():
    """Test that Thompson Sampling explores uncertain moves"""
    print("Testing Thompson Sampling exploration...")
    
    model = DistributionalResNet()
    mcts = RDMCTS(model)
    game = Connect4()
    
    children = mcts.search(game, simulations=50)
    
    visit_counts = [child.visits for child in children.values()]
    assert max(visit_counts) > min(visit_counts), "Visit distribution should be non-uniform"
    
    low_visit_children = [child for child in children.values() if child.visits < 5]
    if low_visit_children:
        for child in low_visit_children:
            mean, std = dist_stats(child.dist)
            assert std > 0.1, f"Low-visit nodes should have high uncertainty, got std={std}"
    
    print(f"  ✓ Thompson Sampling explores appropriately (visit range: {min(visit_counts)}-{max(visit_counts)})")

def run_all_tests():
    """Run all tests"""
    print("="*80)
    print("RUNNING COMPREHENSIVE TESTS")
    print("="*80)
    print()
    
    tests = [
        test_connect4_basic,
        test_connect4_win_detection,
        test_distributional_network,
        test_scalar_network,
        test_dist_stats,
        test_max_convolution,
        test_rdmcts_search,
        test_baseline_mcts_search,
        test_thompson_sampling,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        print()
    
    print("="*80)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*80)
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
