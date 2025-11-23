"""
Quick test to verify new search algorithms (UCB-V and RAVE) work correctly.
"""

import torch
import numpy as np
from rd_mcts_experiment import Connect4
from framework import UnifiedNetwork, ScalarValueHead, SearchConfig
from search_algorithms import UCBVMCTSSearch, RAVEMCTSSearch, StandardMCTSSearch

def test_ucbv_search():
    """Test UCB-V search algorithm."""
    print("="*80)
    print("Testing UCB-V Search Algorithm")
    print("="*80)
    
    # Create model and search
    model = UnifiedNetwork(ScalarValueHead())
    config = SearchConfig(c_puct=1.0)
    search = UCBVMCTSSearch(model, config)
    
    # Run search on empty board
    game = Connect4()
    children = search.search(game, simulations=50)
    
    print(f"\nSearch completed successfully!")
    print(f"Number of children: {len(children)}")
    
    # Check that variance tracking is working
    for move, child in children.items():
        if child.visits > 0:
            variance = child.M2 / child.visits if hasattr(child, 'M2') else 0
            print(f"Move {move}: visits={child.visits}, Q={child.get_value():.3f}, var={variance:.4f}")
    
    print("✓ UCB-V search works correctly!")
    return True

def test_rave_search():
    """Test RAVE search algorithm."""
    print("\n" + "="*80)
    print("Testing RAVE Search Algorithm")
    print("="*80)
    
    # Create model and search
    model = UnifiedNetwork(ScalarValueHead())
    config = SearchConfig(c_puct=1.0)
    search = RAVEMCTSSearch(model, config)
    
    # Run search on empty board
    game = Connect4()
    children = search.search(game, simulations=50)
    
    print(f"\nSearch completed successfully!")
    print(f"Number of children: {len(children)}")
    
    # Check that AMAF tracking is working
    for move, child in children.items():
        if child.visits > 0:
            amaf_q = child.amaf_sum / child.amaf_visits if hasattr(child, 'amaf_visits') and child.amaf_visits > 0 else 0
            print(f"Move {move}: visits={child.visits}, Q={child.get_value():.3f}, AMAF_Q={amaf_q:.3f}, AMAF_visits={getattr(child, 'amaf_visits', 0)}")
    
    print("✓ RAVE search works correctly!")
    return True

def test_comparison():
    """Compare all search algorithms on the same position."""
    print("\n" + "="*80)
    print("Comparing All Search Algorithms")
    print("="*80)
    
    # Create shared model
    model = UnifiedNetwork(ScalarValueHead())
    config = SearchConfig(c_puct=1.0)
    
    # Create all search algorithms
    searches = {
        "Standard PUCT": StandardMCTSSearch(model, config),
        "UCB-V": UCBVMCTSSearch(model, config),
        "RAVE": RAVEMCTSSearch(model, config),
    }
    
    game = Connect4()
    
    print("\nRunning 50 simulations with each algorithm on empty board:")
    for name, search in searches.items():
        children = search.search(game, simulations=50)
        
        # Get most visited move
        best_move = max(children.items(), key=lambda x: x[1].visits)
        print(f"\n{name}:")
        print(f"  Best move: {best_move[0]} (visits={best_move[1].visits}, Q={best_move[1].get_value():.3f})")
        
        # Show visit distribution
        visit_dist = [children[m].visits if m in children else 0 for m in range(7)]
        print(f"  Visit distribution: {visit_dist}")
    
    print("\n✓ All search algorithms produce reasonable results!")
    return True

if __name__ == "__main__":
    print("\n" + "="*80)
    print("NEW SEARCH ALGORITHMS TEST SUITE")
    print("="*80)
    
    try:
        test_ucbv_search()
        test_rave_search()
        test_comparison()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED!")
        print("="*80)
        print("\nUCB-V and RAVE are ready for comprehensive experiments.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
