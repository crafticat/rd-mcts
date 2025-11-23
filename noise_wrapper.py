"""
Noise injection wrapper for adversarial testing of search algorithms.
"""

import torch
import numpy as np
from framework import UnifiedNetwork, ValueRepresentation

class NoisyValueWrapper:
    """
    Wraps a UnifiedNetwork and adds controlled noise to value predictions during search.
    This simulates a noisy/uncertain network to test when penalty mechanisms help.
    """
    
    def __init__(self, base_model: UnifiedNetwork, noise_std: float = 0.0):
        """
        Args:
            base_model: The underlying network to wrap
            noise_std: Standard deviation of Gaussian noise to add to value predictions
        """
        self.base_model = base_model
        self.noise_std = noise_std
        self.value_head = base_model.value_head
    
    def __call__(self, x):
        """Forward pass with noise injection."""
        return self.forward(x)
    
    def forward(self, x):
        """Forward pass with noise injection to value predictions."""
        # Get base predictions
        pi, value_rep = self.base_model(x)
        
        # Add noise to value representation based on type
        if self.noise_std > 0:
            if value_rep.rep_type == "scalar":
                # For scalar: add Gaussian noise directly
                noisy_value = value_rep.data + torch.randn_like(value_rep.data) * self.noise_std
                noisy_value = torch.clamp(noisy_value, -1.0, 1.0)
                value_rep = ValueRepresentation(noisy_value, "scalar")
                
            elif value_rep.rep_type == "gaussian":
                # For Gaussian: add noise to mean, increase sigma
                mu, sigma = value_rep.data
                noisy_mu = mu + torch.randn_like(mu) * self.noise_std
                noisy_mu = torch.clamp(noisy_mu, -1.0, 1.0)
                # Also increase uncertainty
                noisy_sigma = sigma + self.noise_std * 0.5
                noisy_sigma = torch.clamp(noisy_sigma, 0.05, 1.0)
                value_rep = ValueRepresentation((noisy_mu, noisy_sigma), "gaussian")
                
            elif value_rep.rep_type == "categorical":
                # For categorical: add noise to logits before softmax
                # We need to reconstruct logits from probs (approximate)
                probs = value_rep.data
                logits = torch.log(probs + 1e-8)
                noisy_logits = logits + torch.randn_like(logits) * self.noise_std
                noisy_probs = torch.softmax(noisy_logits, dim=1)
                value_rep = ValueRepresentation(noisy_probs, "categorical")
        
        return pi, value_rep
    
    def to(self, device):
        """Move to device."""
        self.base_model.to(device)
        return self
    
    def eval(self):
        """Set to eval mode."""
        self.base_model.eval()
        return self
    
    def train(self):
        """Set to train mode."""
        self.base_model.train()
        return self
    
    def parameters(self):
        """Get parameters."""
        return self.base_model.parameters()
    
    def state_dict(self):
        """Get state dict."""
        return self.base_model.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.base_model.load_state_dict(state_dict)

def create_noisy_model(base_model: UnifiedNetwork, noise_std: float) -> NoisyValueWrapper:
    """
    Create a noisy wrapper around a model.
    
    Args:
        base_model: The base model to wrap
        noise_std: Standard deviation of noise to add
        
    Returns:
        NoisyValueWrapper that adds noise during inference
    """
    return NoisyValueWrapper(base_model, noise_std)
