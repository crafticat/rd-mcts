"""
Minimax agent with alpha-beta pruning for Connect4.
Used as a fixed benchmark opponent for learning curve experiments.
"""

import numpy as np
import copy
from rd_mcts_experiment import Connect4


class MinimaxAgent:
    """Minimax agent with alpha-beta pruning for Connect4."""
    
    def __init__(self, depth=4):
        """
        Initialize Minimax agent.
        
        Args:
            depth: Maximum search depth (4 is reasonable for Connect4)
        """
        self.depth = depth
        self.nodes_evaluated = 0
    
    def evaluate_position(self, game: Connect4) -> float:
        """
        Static evaluation function for non-terminal positions.
        Returns value in [-1, 1] from current player's perspective.
        """
        # Check for immediate win/loss
        result = game.check_win()
        if result != 0:
            # Return from current player's perspective
            return -result * game.player
        
        # Check if board is full (draw)
        if len(game.get_valid_moves()) == 0:
            return 0.0
        
        # Heuristic evaluation based on threats and opportunities
        score = 0.0
        
        # Evaluate all possible 4-in-a-row windows
        # Horizontal
        for row in range(game.rows):
            for col in range(game.cols - 3):
                window = [game.board[row][col + i] for i in range(4)]
                score += self._evaluate_window(window, game.player)
        
        # Vertical
        for row in range(game.rows - 3):
            for col in range(game.cols):
                window = [game.board[row + i][col] for i in range(4)]
                score += self._evaluate_window(window, game.player)
        
        # Diagonal (positive slope)
        for row in range(game.rows - 3):
            for col in range(game.cols - 3):
                window = [game.board[row + i][col + i] for i in range(4)]
                score += self._evaluate_window(window, game.player)
        
        # Diagonal (negative slope)
        for row in range(3, game.rows):
            for col in range(game.cols - 3):
                window = [game.board[row - i][col + i] for i in range(4)]
                score += self._evaluate_window(window, game.player)
        
        # Center column preference (Connect4 strategy)
        center_col = game.cols // 2
        center_count = sum(1 for row in range(game.rows) if game.board[row][center_col] == game.player)
        score += center_count * 0.03
        
        # Normalize to [-1, 1]
        return np.tanh(score)
    
    def _evaluate_window(self, window, player):
        """Evaluate a 4-cell window."""
        score = 0.0
        opponent = -player
        
        player_count = window.count(player)
        opponent_count = window.count(opponent)
        empty_count = window.count(0)
        
        # Scoring heuristic
        if player_count == 4:
            score += 1.0  # Win
        elif player_count == 3 and empty_count == 1:
            score += 0.1  # Strong threat
        elif player_count == 2 and empty_count == 2:
            score += 0.02  # Weak threat
        
        if opponent_count == 3 and empty_count == 1:
            score -= 0.08  # Block opponent threat
        elif opponent_count == 2 and empty_count == 2:
            score -= 0.01  # Block weak opponent threat
        
        return score
    
    def minimax(self, game: Connect4, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        """
        Minimax with alpha-beta pruning.
        
        Args:
            game: Current game state
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            maximizing: True if maximizing player, False if minimizing
        
        Returns:
            Best value found
        """
        self.nodes_evaluated += 1
        
        # Terminal conditions
        result = game.check_win()
        if result != 0:
            # Return from original caller's perspective
            # If we're at depth d, we've made d moves, so flip d times
            return -result * game.player * (1.0 + depth * 0.01)  # Prefer faster wins
        
        if len(game.get_valid_moves()) == 0:
            return 0.0  # Draw
        
        if depth == 0:
            return self.evaluate_position(game)
        
        valid_moves = game.get_valid_moves()
        
        # Move ordering: try center columns first (better for alpha-beta pruning)
        center = game.cols // 2
        valid_moves = sorted(valid_moves, key=lambda m: abs(m - center))
        
        if maximizing:
            max_eval = -float('inf')
            for move in valid_moves:
                child = game.make_move(move)
                eval_score = self.minimax(child, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval
        else:
            min_eval = float('inf')
            for move in valid_moves:
                child = game.make_move(move)
                eval_score = self.minimax(child, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval
    
    def get_best_move(self, game: Connect4) -> int:
        """
        Get the best move using minimax search.
        
        Args:
            game: Current game state
        
        Returns:
            Best move (column index)
        """
        self.nodes_evaluated = 0
        valid_moves = game.get_valid_moves()
        
        if not valid_moves:
            return -1
        
        # Move ordering: try center columns first
        center = game.cols // 2
        valid_moves = sorted(valid_moves, key=lambda m: abs(m - center))
        
        best_move = valid_moves[0]
        best_value = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        
        for move in valid_moves:
            child = game.make_move(move)
            # After making a move, we're minimizing from opponent's perspective
            value = self.minimax(child, self.depth - 1, alpha, beta, False)
            
            if value > best_value:
                best_value = value
                best_move = move
            
            alpha = max(alpha, value)
        
        return best_move
    
    def get_position_value(self, game: Connect4) -> float:
        """
        Get the value of a position using minimax search.
        Used for ground truth in depth stability tests.
        
        Args:
            game: Current game state
        
        Returns:
            Position value in [-1, 1] from current player's perspective
        """
        self.nodes_evaluated = 0
        
        # Check terminal
        result = game.check_win()
        if result != 0:
            return -result * game.player
        
        if len(game.get_valid_moves()) == 0:
            return 0.0
        
        # Run minimax
        return self.minimax(game, self.depth, -float('inf'), float('inf'), True)


def test_minimax():
    """Test the minimax agent."""
    print("Testing Minimax Agent...")
    
    # Test 1: Empty board
    game = Connect4()
    agent = MinimaxAgent(depth=4)
    move = agent.get_best_move(game)
    value = agent.get_position_value(game)
    print(f"Empty board: Best move = {move}, Value = {value:.3f}, Nodes = {agent.nodes_evaluated}")
    
    # Test 2: Near-win position
    game = Connect4()
    for m in [3, 3, 3]:  # Stack in center
        game = game.make_move(m)
    move = agent.get_best_move(game)
    value = agent.get_position_value(game)
    print(f"Near-win: Best move = {move}, Value = {value:.3f}, Nodes = {agent.nodes_evaluated}")
    
    # Test 3: Blocking position
    game = Connect4()
    for m in [0, 1, 0, 1, 0]:  # Opponent threatens column 0
        game = game.make_move(m)
    move = agent.get_best_move(game)
    value = agent.get_position_value(game)
    print(f"Blocking: Best move = {move}, Value = {value:.3f}, Nodes = {agent.nodes_evaluated}")
    
    print("Minimax tests complete!")


if __name__ == "__main__":
    test_minimax()
