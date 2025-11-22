"""
Modular framework for testing different value representations and search algorithms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import copy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUPPORT_SIZE = 51
V_MIN = -1.0
V_MAX = 1.0
ATOMS = np.linspace(V_MIN, V_MAX, SUPPORT_SIZE)

def project_gaussian(mean, std):
    """Creates a discrete distribution from Gaussian parameters."""
    std = max(std, 1e-4)
    dist = np.exp(-0.5 * ((ATOMS - mean)/std)**2)
    return dist / np.sum(dist)

def dist_stats(dist):
    """Returns mean and std_dev of a categorical distribution."""
    mean = np.sum(dist * ATOMS)
    var = np.sum(dist * (ATOMS - mean)**2)
    return mean, np.sqrt(var)

@dataclass
class SearchConfig:
    """Configuration for search algorithms."""
    c_puct: float = 1.0
    lambda_penalty: float = 1.0
    use_thompson: bool = True
    use_penalty: bool = True

class ValueRepresentation:
    """Container for different value representations."""
    def __init__(self, data: Any, rep_type: str):
        self.data = data
        self.rep_type = rep_type

class ValueHead(nn.Module, ABC):
    """Abstract base class for value heads."""
    
    @abstractmethod
    def forward(self, features: torch.Tensor) -> ValueRepresentation:
        """Forward pass returning value representation."""
        pass
    
    @abstractmethod
    def loss(self, value_rep: ValueRepresentation, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss for this value representation."""
        pass
    
    @abstractmethod
    def to_scalar(self, value_rep: ValueRepresentation) -> float:
        """Convert value representation to scalar."""
        pass
    
    @abstractmethod
    def to_distribution(self, value_rep: ValueRepresentation) -> np.ndarray:
        """Convert value representation to distribution over ATOMS."""
        pass

class CategoricalValueHead(ValueHead):
    """Categorical distribution value head (C51-style)."""
    
    def __init__(self, input_dim: int = 256):
        super().__init__()
        self.fc = nn.Linear(input_dim, SUPPORT_SIZE)
    
    def forward(self, features: torch.Tensor) -> ValueRepresentation:
        logits = self.fc(features)
        probs = F.softmax(logits, dim=1)
        return ValueRepresentation(probs, "categorical")
    
    def loss(self, value_rep: ValueRepresentation, targets: torch.Tensor) -> torch.Tensor:
        target_dists = []
        for v in targets.cpu().numpy():
            target_dist = project_gaussian(v, 0.1)
            target_dists.append(target_dist)
        target_dists_tensor = torch.FloatTensor(np.array(target_dists)).to(DEVICE)
        
        probs = value_rep.data
        return -torch.mean(torch.sum(target_dists_tensor * torch.log(probs + 1e-8), dim=1))
    
    def to_scalar(self, value_rep: ValueRepresentation) -> float:
        probs = value_rep.data.detach().cpu().numpy()[0]
        return np.sum(probs * ATOMS)
    
    def to_distribution(self, value_rep: ValueRepresentation) -> np.ndarray:
        return value_rep.data.detach().cpu().numpy()[0]

class GaussianValueHead(ValueHead):
    """Gaussian parametric value head (predicts mean and std)."""
    
    def __init__(self, input_dim: int = 256):
        super().__init__()
        self.fc = nn.Linear(input_dim, 2)
    
    def forward(self, features: torch.Tensor) -> ValueRepresentation:
        params = self.fc(features)
        mu = torch.tanh(params[:, 0:1])
        log_sigma = params[:, 1:2]
        sigma = torch.clamp(torch.exp(log_sigma), 0.05, 1.0)
        return ValueRepresentation((mu, sigma), "gaussian")
    
    def loss(self, value_rep: ValueRepresentation, targets: torch.Tensor) -> torch.Tensor:
        mu, sigma = value_rep.data
        
        mse_loss = torch.mean((mu.squeeze() - targets) ** 2)
        
        sigma_penalty = torch.mean((sigma - 0.3) ** 2)
        
        return mse_loss + 0.01 * sigma_penalty
    
    def to_scalar(self, value_rep: ValueRepresentation) -> float:
        mu, sigma = value_rep.data
        return mu.detach().item()
    
    def to_distribution(self, value_rep: ValueRepresentation) -> np.ndarray:
        mu, sigma = value_rep.data
        return project_gaussian(mu.detach().item(), sigma.detach().item())

class ScalarValueHead(ValueHead):
    """Scalar value head (standard AlphaZero-style)."""
    
    def __init__(self, input_dim: int = 256):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
    
    def forward(self, features: torch.Tensor) -> ValueRepresentation:
        value = torch.tanh(self.fc(features))
        return ValueRepresentation(value, "scalar")
    
    def loss(self, value_rep: ValueRepresentation, targets: torch.Tensor) -> torch.Tensor:
        value = value_rep.data.squeeze()
        return torch.mean((value - targets) ** 2)
    
    def to_scalar(self, value_rep: ValueRepresentation) -> float:
        return value_rep.data.detach().item()
    
    def to_distribution(self, value_rep: ValueRepresentation) -> np.ndarray:
        value = value_rep.data.detach().item()
        return project_gaussian(value, 0.05)

class UnifiedNetwork(nn.Module):
    """Unified network with shared backbone and pluggable value head."""
    
    def __init__(self, value_head: ValueHead):
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
        
        self.value_head = value_head
    
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
        v_features = F.relu(self.val_fc(v))
        
        value_rep = self.value_head(v_features)
        
        return F.log_softmax(pi, dim=1), value_rep

class SearchNode:
    """Generic node for MCTS search."""
    def __init__(self, prior: float):
        self.visits = 0
        self.prior = prior
        self.children = {}
        self.dist = np.ones(SUPPORT_SIZE) / SUPPORT_SIZE
        self.value_sum = 0.0
    
    def get_value(self):
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

class Search(ABC):
    """Abstract base class for search algorithms."""
    
    def __init__(self, model: UnifiedNetwork, config: SearchConfig):
        self.model = model
        self.config = config
    
    @abstractmethod
    def search(self, root_state, simulations: int) -> Dict[int, SearchNode]:
        """Run search and return children of root."""
        pass

def convolve_max_and_penalize(child_dists, lambda_param):
    """Penalized max-convolution for distributional backpropagation."""
    if not child_dists:
        return np.ones(SUPPORT_SIZE) / SUPPORT_SIZE
    
    cdfs = [np.cumsum(d) for d in child_dists]
    max_cdf = np.ones(SUPPORT_SIZE)
    for cdf in cdfs:
        max_cdf *= cdf
    
    max_pdf = np.diff(max_cdf, prepend=0)
    max_pdf = max_pdf / np.sum(max_pdf)
    
    mu, sigma = dist_stats(max_pdf)
    penalized_mean = mu - (lambda_param * sigma)
    final_dist = project_gaussian(penalized_mean, sigma)
    
    return final_dist
