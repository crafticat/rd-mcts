"""
Test that reorganized code structure works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from game import Connect4
from networks import ValueOnlyNetwork
from mcts import ValueOnlyRDMCTSSearch, ValueOnlyScalarMCTSSearch
from utils import CategoricalValueHead, ScalarValueHead, SearchConfig, DEVICE


def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    print("  ✓ Game module imported")
    print("  ✓ Networks module imported")
    print("  ✓ MCTS module imported")
    print("  ✓ Utils module imported")


def test_game():
    """Test that game works."""
    print("\nTesting game...")
    game = Connect4()
    assert len(game.get_valid_moves()) == 7
    game = game.make_move(3)
    assert game.player == -1
    print("  ✓ Connect4 game works")


def test_network():
    """Test that network works."""
    print("\nTesting network...")
    
    cat_head = CategoricalValueHead()
    cat_net = ValueOnlyNetwork(cat_head).to(DEVICE)
    
    game = Connect4()
    board_tensor = torch.FloatTensor(game.get_canonical_state()).to(DEVICE)
    
    value_rep = cat_net(board_tensor)
    assert value_rep.rep_type == "categorical"
    print("  ✓ ValueOnlyNetwork works")


def test_search():
    """Test that search works."""
    print("\nTesting search...")
    
    cat_head = CategoricalValueHead()
    cat_net = ValueOnlyNetwork(cat_head).to(DEVICE)
    
    config = SearchConfig(c_puct=1.0, lambda_penalty=1.0, use_thompson=True, use_penalty=True)
    search = ValueOnlyRDMCTSSearch(cat_net, config)
    
    game = Connect4()
    children = search.search(game, simulations=10)
    
    assert len(children) > 0
    print("  ✓ ValueOnlyRDMCTSSearch works")


if __name__ == "__main__":
    print("="*80)
    print("REORGANIZATION TEST")
    print("="*80)
    
    test_imports()
    test_game()
    test_network()
    test_search()
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80)
    print("\nReorganized code structure is working correctly.")
    print("Ready to re-run experiments with value-only implementation.")
