import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
import matplotlib.pyplot as plt

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUPPORT_SIZE = 51
V_MIN = -1.0
V_MAX = 1.0
ATOMS = np.linspace(V_MIN, V_MAX, SUPPORT_SIZE)
DELTA_Z = (V_MAX - V_MIN) / (SUPPORT_SIZE - 1)
LAMBDA_PENALTY = 1.0  # The "Risk Aversion" parameter

# --- 1. The Environment (Connect 4) ---
class Connect4:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.player = 1 # 1 or -1
        self.last_move = None

    def get_valid_moves(self):
        return [c for c in range(self.cols) if self.board[0][c] == 0]

    def make_move(self, col):
        cp = copy.deepcopy(self)
        row = max([r for r in range(self.rows) if cp.board[r][col] == 0])
        cp.board[row][col] = cp.player
        cp.player *= -1
        cp.last_move = col
        return cp

    def check_win(self):
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
        return len(self.get_valid_moves()) == 0 or self.check_win() != 0

    def get_canonical_state(self):
        # Return board from perspective of current player
        return self.board * self.player

# --- 2. The Distributional Network (C51 Style) ---
class DistributionalResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Small body
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64)
            ) for _ in range(4)
        ])
        
        # Policy Head
        self.pol_conv = nn.Conv2d(64, 32, 1)
        self.pol_fc = nn.Linear(32 * 6 * 7, 7)
        
        # Value Head (Distributional)
        self.val_conv = nn.Conv2d(64, 32, 1)
        self.val_fc = nn.Linear(32 * 6 * 7, 256)
        self.val_dist = nn.Linear(256, SUPPORT_SIZE)

    def forward(self, x):
        x = x.view(-1, 1, 6, 7)
        x = F.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            residual = x
            x = F.relu(block(x) + residual)
            
        p = F.relu(self.pol_conv(x))
        p = p.view(p.size(0), -1)
        pi = self.pol_fc(p)
        
        v = F.relu(self.val_conv(x))
        v = v.view(v.size(0), -1)
        v = F.relu(self.val_fc(v))
        v_logits = self.val_dist(v)
        
        # Use log_softmax for policy, softmax for distribution
        return F.log_softmax(pi, dim=1), F.softmax(v_logits, dim=1)

# --- 3. The Core Logic: Math Helpers ---

def dist_stats(dist):
    """Returns mean and std_dev of a categorical distribution."""
    mean = np.sum(dist * ATOMS)
    var = np.sum(dist * (ATOMS - mean)**2)
    return mean, np.sqrt(var)

def project_gaussian(mean, std):
    """Creates a discrete distribution from analytical Gaussian parameters."""
    # This is used to reconstruct the 'Penalized' distribution
    std = max(std, 1e-4) # Prevent div by zero
    dist = np.exp(-0.5 * ((ATOMS - mean)/std)**2)
    return dist / np.sum(dist)

def convolve_max_and_penalize(child_dists, lambda_param):
    """
    1. Computes Distribution of Max(Children).
    2. Applies Variance Penalty.
    """
    if not child_dists:
        return np.ones(SUPPORT_SIZE) / SUPPORT_SIZE # Uniform uncertainty

    # --- Optimization: The CDF Trick for Max ---
    # CDF_max(x) = Product(CDF_i(x))
    # This is O(N) instead of O(N^2)
    
    cdfs = [np.cumsum(d) for d in child_dists]
    max_cdf = np.ones(SUPPORT_SIZE)
    for cdf in cdfs:
        max_cdf *= cdf
        
    # Convert CDF back to PDF
    max_pdf = np.diff(max_cdf, prepend=0)
    max_pdf = max_pdf / np.sum(max_pdf) # Renormalize for numerical errors
    
    # --- The Correction Step ---
    mu, sigma = dist_stats(max_pdf)
    
    # Apply Penalty: Shift mean to the left
    penalized_mean = mu - (lambda_param * sigma)
    
    # Re-project to support
    # We maintain the shape (sigma) but shift the center
    final_dist = project_gaussian(penalized_mean, sigma)
    
    return final_dist

# --- 4. The RD-MCTS Engine ---

class Node:
    def __init__(self, prior):
        self.visits = 0
        self.prior = prior
        self.children = {}
        # The crucial change: Storing a distribution, not a scalar Q
        self.dist = np.ones(SUPPORT_SIZE) / SUPPORT_SIZE 

class RDMCTS:
    def __init__(self, model, c_puct=1.0):
        self.model = model
        self.c_puct = c_puct
        
    def search(self, root_state, simulations=50):
        root = Node(0)
        
        # Evaluate Root first
        board_tensor = torch.FloatTensor(root_state.get_canonical_state()).to(DEVICE)
        with torch.no_grad():
            pi, v_dist = self.model(board_tensor)
        
        root.dist = v_dist.cpu().numpy()[0]
        valid_moves = root_state.get_valid_moves()
        
        # Add Dirichlet noise to root (Standard AlphaZero practice)
        noise = np.random.dirichlet([0.3] * len(valid_moves))
        
        for idx, move in enumerate(valid_moves):
            root.children[move] = Node(prior=np.exp(pi.cpu().numpy()[0][move]))
            # Mix noise
            root.children[move].prior = 0.75 * root.children[move].prior + 0.25 * noise[idx]

        for _ in range(simulations):
            node = root
            game = copy.deepcopy(root_state)
            path = [node]
            
            # 1. Selection (Thompson Sampling ish)
            # In pure Thompson Sampling, we sample from dist. 
            # In Tree search, we often combine Prior (Policy) with Value.
            # Here, let's use a hybrid: UCB score based on SAMPLE from dist.
            
            while node.children:
                best_score = -float('inf')
                best_move = -1
                best_child = None
                
                for move, child in node.children.items():
                    # THOMPSON SAMPLING STEP:
                    # Sample a value 'v' from the child's distribution
                    sample_v = np.random.choice(ATOMS, p=child.dist)
                    
                    # Add Exploration term (PUCT)
                    u = self.c_puct * child.prior * np.sqrt(node.visits) / (1 + child.visits)
                    
                    # Score = Sampled Value + Exploration
                    score = sample_v + u
                    
                    if score > best_score:
                        best_score = score
                        best_move = move
                        best_child = child
                
                game = game.make_move(best_move)
                node = best_child
                path.append(node)
            
            # 2. Evaluation
            if not game.is_terminal():
                board_tensor = torch.FloatTensor(game.get_canonical_state()).to(DEVICE)
                with torch.no_grad():
                    pi, v_dist_tensor = self.model(board_tensor)
                leaf_dist = v_dist_tensor.cpu().numpy()[0]
                
                # Expand
                valid = game.get_valid_moves()
                probs = np.exp(pi.cpu().numpy()[0])
                for m in valid:
                    node.children[m] = Node(probs[m])
            else:
                result = game.check_win()
                if result == 0:
                    result = 0.0
                else:
                    result = -result * game.player
                leaf_dist = project_gaussian(result, 0.01)

            # 3. Backpropagation (Penalized Max-Conv)
            # Reverse path (Leaf -> Root)
            
            # Update Leaf first
            node.visits += 1
            node.dist = leaf_dist # Leaf value is ground truth from Net/Env
            
            # Propagate up
            for i in range(len(path) - 2, -1, -1):
                parent = path[i]
                parent.visits += 1
                
                # In standard MCTS: Parent Q = Average(Children Q)
                # In RD-MCTS: Parent Dist = PenalizedMaxConv(Children Dists)
                
                # Important: AlphaZero alternates players. 
                # Parent (Player A) wants to Maximize value.
                # Children (Player B) want to Minimize value (from A's perspective).
                # To simply reuse Max logic, we can invert the distribution of the child
                # (since V_parent = -V_child in zero-sum).
                
                child_dists_inverted = []
                for child in parent.children.values():
                    if child.visits > 0:
                        # Flip distribution indices to simulate negation (V -> -V)
                        d_inv = np.flip(child.dist)
                        child_dists_inverted.append(d_inv)
                
                if child_dists_inverted:
                    # Apply The Algorithm
                    new_dist = convolve_max_and_penalize(child_dists_inverted, LAMBDA_PENALTY)
                    
                    # Soft update (Running Average) to stabilize learning
                    # parent.dist = (parent.dist * (parent.visits-1) + new_dist) / parent.visits
                    # OR: Direct Replacement (since Max-Conv represents best-play knowledge)
                    # Direct replacement is more theoretically aligned with Bellman
                    parent.dist = new_dist 

        return root.children

# --- 5. Main Execution ---

if __name__ == "__main__":
    print("Initializing RD-MCTS System...")
    net = DistributionalResNet().to(DEVICE)
    mcts = RDMCTS(net)
    game = Connect4()
    
    print("Running Search on Empty Board...")
    children = mcts.search(game, simulations=100)
    
    print("\nSearch Complete. Root Children Estimates:")
    for move, child in children.items():
        mu, sigma = dist_stats(child.dist)
        print(f"Move {move}: Visits={child.visits}, Est. Value={mu:.3f}, Uncertainty={sigma:.3f}")
        
    print("\nVerification:")
    print("If 'Uncertainty' is high for low-visit nodes, Thompson Sampling is working.")
    print("If 'Est. Value' avoids high-sigma traps, Penalty is working.")
