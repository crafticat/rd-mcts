"""
Value-only network architecture (no policy head).
This is the correct implementation for testing pure value network + Thompson Sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.framework import ValueHead, ValueRepresentation


class ValueOnlyNetwork(nn.Module):
    """Network with only value head (no policy head)."""
    
    def __init__(self, value_head: ValueHead):
        super().__init__()
        # Shared backbone
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64)
            ) for _ in range(4)
        ])
        
        # Value head only (no policy head)
        self.val_conv = nn.Conv2d(64, 32, 1)
        self.val_fc = nn.Linear(32 * 6 * 7, 256)
        
        self.value_head = value_head
    
    def forward(self, x):
        """Forward pass returns only value representation (no policy)."""
        x = x.view(-1, 1, 6, 7)
        x = F.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            residual = x
            x = F.relu(block(x) + residual)
        
        # Value head only
        v = F.relu(self.val_conv(x))
        v = v.view(v.size(0), -1)
        v_features = F.relu(self.val_fc(v))
        
        value_rep = self.value_head(v_features)
        
        return value_rep
