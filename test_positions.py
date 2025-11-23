"""
Test positions for depth stability experiments.
Contains 10 diverse positions: 3 early game, 4 mid-game, 3 tactical endgame.
"""

from rd_mcts_experiment import Connect4
from minimax_agent import MinimaxAgent


def create_test_positions():
    """
    Create 10 test positions with ground truth values.
    
    Returns:
        List of (position, description, ground_truth_value) tuples
    """
    positions = []
    
    # ========== EARLY GAME POSITIONS (3) ==========
    
    # Early 1: Empty board
    game1 = Connect4()
    positions.append((game1, "Early: Empty board", 0.0))
    
    # Early 2: Opening moves (center control)
    game2 = Connect4()
    for move in [3, 3]:  # Both players take center
        game2 = game2.make_move(move)
    positions.append((game2, "Early: Center control", 0.0))
    
    # Early 3: Asymmetric opening
    game3 = Connect4()
    for move in [3, 2, 4, 5]:  # Asymmetric development
        game3 = game3.make_move(move)
    positions.append((game3, "Early: Asymmetric opening", 0.0))
    
    # ========== MID-GAME POSITIONS (4) ==========
    
    # Mid 1: Balanced position with threats
    game4 = Connect4()
    for move in [3, 3, 2, 4, 2, 4, 1, 5]:
        game4 = game4.make_move(move)
    positions.append((game4, "Mid: Balanced with threats", 0.0))
    
    # Mid 2: Trap position (risky vs safe)
    game5 = Connect4()
    for move in [3, 2, 3, 4, 3, 5]:  # Column 3 has 3 pieces
        game5 = game5.make_move(move)
    positions.append((game5, "Mid: Trap position", None))  # Will compute
    
    # Mid 3: Complex tactical position
    game6 = Connect4()
    for move in [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]:
        game6 = game6.make_move(move)
    positions.append((game6, "Mid: Complex tactics", None))  # Will compute
    
    # Mid 4: Positional advantage
    game7 = Connect4()
    for move in [3, 6, 3, 6, 2, 0, 4, 0]:
        game7 = game7.make_move(move)
    positions.append((game7, "Mid: Positional advantage", None))  # Will compute
    
    # ========== ENDGAME POSITIONS (3) ==========
    
    # End 1: Forced win in 1 move
    game8 = Connect4()
    for move in [0, 1, 0, 1, 0, 1]:  # Column 0 has 3 in a row
        game8 = game8.make_move(move)
    positions.append((game8, "End: Forced win in 1", 1.0))
    
    # End 2: Must block or lose
    game9 = Connect4()
    for move in [0, 1, 0, 1, 0, 2]:  # Opponent threatens column 0
        game9 = game9.make_move(move)
    positions.append((game9, "End: Must block", None))  # Will compute
    
    # End 3: Complex endgame
    game10 = Connect4()
    for move in [3, 3, 2, 4, 2, 4, 1, 5, 1, 5, 0, 6]:
        game10 = game10.make_move(move)
    positions.append((game10, "End: Complex endgame", None))  # Will compute
    
    return positions


def compute_ground_truth_values(positions, depth=6):
    """
    Compute ground truth values for positions using Minimax.
    
    Args:
        positions: List of (game, description, ground_truth) tuples
        depth: Minimax search depth
    
    Returns:
        Updated positions with ground truth values
    """
    print(f"Computing ground truth values with Minimax (depth={depth})...")
    minimax = MinimaxAgent(depth=depth)
    
    updated_positions = []
    
    for i, (game, description, ground_truth) in enumerate(positions):
        if ground_truth is None:
            print(f"  Position {i+1}: {description}")
            ground_truth = minimax.get_position_value(game)
            print(f"    Ground truth: {ground_truth:.3f} (nodes: {minimax.nodes_evaluated})")
        else:
            print(f"  Position {i+1}: {description} (predefined: {ground_truth:.3f})")
        
        updated_positions.append((game, description, ground_truth))
    
    return updated_positions


def visualize_position(game: Connect4):
    """Print a visual representation of the board."""
    symbols = {0: '.', 1: 'X', -1: 'O'}
    print("\n  " + " ".join(str(i) for i in range(7)))
    for row in range(6):
        print("  " + " ".join(symbols[game.board[row][col]] for col in range(7)))
    print()


def main():
    """Test the test positions."""
    print("\n" + "="*80)
    print("TEST POSITIONS FOR DEPTH STABILITY")
    print("="*80)
    
    # Create positions
    positions = create_test_positions()
    
    print(f"\nCreated {len(positions)} test positions:")
    for i, (game, description, gt) in enumerate(positions):
        gt_str = f"{gt:.3f}" if gt is not None else "TBD"
        print(f"  {i+1}. {description} (GT: {gt_str})")
    
    # Compute ground truth for positions without predefined values
    positions = compute_ground_truth_values(positions, depth=6)
    
    # Visualize a few positions
    print("\n" + "="*80)
    print("SAMPLE POSITIONS")
    print("="*80)
    
    for i in [0, 4, 7]:  # Show early, mid, end examples
        game, description, gt = positions[i]
        print(f"\nPosition {i+1}: {description}")
        print(f"Ground truth: {gt:.3f}")
        print(f"Current player: {'X' if game.player == 1 else 'O'}")
        visualize_position(game)
    
    print("\n" + "="*80)
    print("TEST POSITIONS READY")
    print("="*80)
    print(f"\nAll {len(positions)} positions have ground truth values.")
    print("These will be used in test_depth_stability.py")


if __name__ == "__main__":
    main()
