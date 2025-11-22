import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ScalarResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64)
            ) for _ in range(4)
        ])
        
        self.pol_conv = nn.Conv2d(64, 32, 1)
        self.pol_fc = nn.Linear(32 * 6 * 7, 7)
        
        self.val_conv = nn.Conv2d(64, 32, 1)
        self.val_fc = nn.Linear(32 * 6 * 7, 256)
        self.val_out = nn.Linear(256, 1)

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
        v_scalar = torch.tanh(self.val_out(v))
        
        return F.log_softmax(pi, dim=1), v_scalar

class ScalarNode:
    def __init__(self, prior):
        self.visits = 0
        self.prior = prior
        self.children = {}
        self.value_sum = 0.0
        
    def get_value(self):
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

class StandardMCTS:
    def __init__(self, model, c_puct=1.0):
        self.model = model
        self.c_puct = c_puct
        
    def search(self, root_state, simulations=50):
        root = ScalarNode(0)
        
        board_tensor = torch.FloatTensor(root_state.get_canonical_state()).to(DEVICE)
        with torch.no_grad():
            pi, v = self.model(board_tensor)
        
        root.value_sum = v.item()
        root.visits = 1
        valid_moves = root_state.get_valid_moves()
        
        noise = np.random.dirichlet([0.3] * len(valid_moves))
        
        for idx, move in enumerate(valid_moves):
            root.children[move] = ScalarNode(prior=np.exp(pi.cpu().numpy()[0][move]))
            root.children[move].prior = 0.75 * root.children[move].prior + 0.25 * noise[idx]

        for _ in range(simulations):
            node = root
            game = copy.deepcopy(root_state)
            path = [node]
            
            while node.children:
                best_score = -float('inf')
                best_move = -1
                best_child = None
                
                for move, child in node.children.items():
                    q_value = child.get_value()
                    u = self.c_puct * child.prior * np.sqrt(node.visits) / (1 + child.visits)
                    score = q_value + u
                    
                    if score > best_score:
                        best_score = score
                        best_move = move
                        best_child = child
                
                game = game.make_move(best_move)
                node = best_child
                path.append(node)
            
            if not game.is_terminal():
                board_tensor = torch.FloatTensor(game.get_canonical_state()).to(DEVICE)
                with torch.no_grad():
                    pi, v_tensor = self.model(board_tensor)
                leaf_value = v_tensor.item()
                
                valid = game.get_valid_moves()
                probs = np.exp(pi.cpu().numpy()[0])
                for m in valid:
                    node.children[m] = ScalarNode(probs[m])
            else:
                result = game.check_win()
                if result == 0:
                    leaf_value = 0.0
                else:
                    leaf_value = -result * game.player
            
            for n in path:
                n.visits += 1
                n.value_sum += leaf_value
                leaf_value = -leaf_value

        return root.children
