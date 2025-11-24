"""
Connect4 game environment.
"""

import numpy as np
import copy


class Connect4:
    """Connect4 game implementation."""
    
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.player = 1  # 1 or -1
        self.last_move = None

    def get_valid_moves(self):
        """Return list of valid column indices."""
        return [c for c in range(self.cols) if self.board[0][c] == 0]

    def make_move(self, col):
        """Make a move and return new game state."""
        cp = copy.deepcopy(self)
        row = max([r for r in range(self.rows) if cp.board[r][col] == 0])
        cp.board[row][col] = cp.player
        cp.player *= -1
        cp.last_move = col
        return cp

    def check_win(self):
        """Check if game is won. Returns 1, -1, or 0."""
        if self.last_move is None:
            return 0
        
        last_player = -self.player
        col = self.last_move
        row = None
        for r in range(self.rows):
            if self.board[r][col] == last_player:
                row = r
                break
        
        if row is None:
            return 0
        
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1
            for direction in [1, -1]:
                r, c = row + dr * direction, col + dc * direction
                while 0 <= r < self.rows and 0 <= c < self.cols and self.board[r][c] == last_player:
                    count += 1
                    r += dr * direction
                    c += dc * direction
            
            if count >= 4:
                return last_player
        
        return 0
    
    def is_terminal(self):
        """Check if game is over."""
        return len(self.get_valid_moves()) == 0 or self.check_win() != 0

    def get_canonical_state(self):
        """Return board from perspective of current player."""
        return self.board * self.player
