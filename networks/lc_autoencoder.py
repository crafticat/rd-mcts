"""
Autoencoder network for LC-MCTS.
The network outputs (μ, w) where w is a latent embedding learned via reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.framework import DEVICE


class LCAutoencoder(nn.Module):
    """
    Autoencoder network for LC-MCTS.
    
    Architecture:
    - Encoder: board → w (d-dimensional latent embedding)
    - Value head: w → μ (scalar value)
    - Decoder: w → reconstructed board
    
    Training loss: value_loss + λ_recon * reconstruction_loss
    """
    
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder: board → features → w
        self.enc_conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(64)
        
        self.enc_res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64)
            ) for _ in range(4)
        ])
        
        # Flatten and project to latent space
        self.enc_fc1 = nn.Linear(64 * 6 * 7, 256)
        self.enc_fc2 = nn.Linear(256, latent_dim)  # w embedding
        
        # Value head: w → μ
        self.val_fc1 = nn.Linear(latent_dim, 128)
        self.val_fc2 = nn.Linear(128, 1)
        
        # Decoder: w → board
        self.dec_fc1 = nn.Linear(latent_dim, 256)
        self.dec_fc2 = nn.Linear(256, 64 * 6 * 7)
        
        self.dec_deconv1 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(64)
        self.dec_deconv2 = nn.ConvTranspose2d(64, 1, 3, padding=1)
    
    def encode(self, x):
        """Encode board to latent embedding w."""
        x = x.view(-1, 1, 6, 7)
        x = F.relu(self.enc_bn1(self.enc_conv1(x)))
        
        for block in self.enc_res_blocks:
            residual = x
            x = F.relu(block(x) + residual)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.enc_fc1(x))
        w = self.enc_fc2(x)  # Latent embedding (no activation)
        
        return w
    
    def decode(self, w):
        """Decode latent embedding w to reconstructed board."""
        x = F.relu(self.dec_fc1(w))
        x = F.relu(self.dec_fc2(x))
        
        x = x.view(-1, 64, 6, 7)
        x = F.relu(self.dec_bn1(self.dec_deconv1(x)))
        x = torch.tanh(self.dec_deconv2(x))  # Output in [-1, 1]
        
        return x.view(-1, 6, 7)
    
    def value_head(self, w):
        """Compute value μ from latent embedding w."""
        x = F.relu(self.val_fc1(w))
        mu = torch.tanh(self.val_fc2(x)).squeeze(-1)  # Scalar in [-1, 1]
        
        return mu
    
    def forward(self, board):
        """
        Forward pass.
        
        Args:
            board: (batch_size, 6, 7) board state
        
        Returns:
            mu: (batch_size,) scalar value
            w: (batch_size, latent_dim) latent embedding
            board_recon: (batch_size, 6, 7) reconstructed board
        """
        w = self.encode(board)
        mu = self.value_head(w)
        board_recon = self.decode(w)
        
        return mu, w, board_recon
    
    def compute_loss(self, board, target_value, lambda_recon=1.0):
        """
        Compute training loss.
        
        Args:
            board: (batch_size, 6, 7) board state
            target_value: (batch_size,) target value
            lambda_recon: weight for reconstruction loss
        
        Returns:
            total_loss, value_loss, recon_loss
        """
        mu, w, board_recon = self.forward(board)
        
        # Value loss (MSE)
        value_loss = F.mse_loss(mu, target_value)
        
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(board_recon, board)
        
        # Total loss
        total_loss = value_loss + lambda_recon * recon_loss
        
        return total_loss, value_loss, recon_loss


class LCState:
    """
    State representation for LC-MCTS nodes.
    
    Each node stores:
    - mu: expected value (scalar)
    - w: structured uncertainty embedding (d-vector)
    - nu: unstructured uncertainty (scalar)
    """
    
    def __init__(self, mu, w, nu=0.0):
        self.mu = mu
        self.w = w if isinstance(w, torch.Tensor) else torch.tensor(w, dtype=torch.float32)
        self.nu = nu
    
    def to_numpy(self):
        """Convert to numpy for easier manipulation."""
        import numpy as np
        return (
            float(self.mu),
            self.w.cpu().numpy() if isinstance(self.w, torch.Tensor) else self.w,
            float(self.nu)
        )
    
    def total_variance(self):
        """Compute total variance: σ² = ||w||² + ν²"""
        w_var = torch.dot(self.w, self.w).item() if isinstance(self.w, torch.Tensor) else (self.w ** 2).sum()
        return w_var + self.nu ** 2
    
    def __repr__(self):
        w_norm = torch.norm(self.w).item() if isinstance(self.w, torch.Tensor) else ((self.w ** 2).sum() ** 0.5)
        return f"LCState(μ={self.mu:.3f}, ||w||={w_norm:.3f}, ν={self.nu:.3f})"


def test_autoencoder():
    """Test the autoencoder network."""
    print("="*80)
    print("TESTING LC AUTOENCODER")
    print("="*80)
    
    # Create network
    net = LCAutoencoder(latent_dim=32).to(DEVICE)
    
    # Create dummy board
    batch_size = 4
    board = torch.randn(batch_size, 6, 7).to(DEVICE)
    target_value = torch.randn(batch_size).to(DEVICE)
    
    # Forward pass
    mu, w, board_recon = net(board)
    
    print(f"\nInput shape: {board.shape}")
    print(f"μ shape: {mu.shape}")
    print(f"w shape: {w.shape}")
    print(f"Reconstructed board shape: {board_recon.shape}")
    
    # Compute loss
    total_loss, value_loss, recon_loss = net.compute_loss(board, target_value, lambda_recon=1.0)
    
    print(f"\nTotal loss: {total_loss.item():.4f}")
    print(f"Value loss: {value_loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    
    # Test LCState
    print("\nTesting LCState:")
    state = LCState(mu=0.5, w=w[0].detach(), nu=0.1)
    print(f"  {state}")
    print(f"  Total variance: {state.total_variance():.4f}")
    
    print("\n" + "="*80)
    print("✓ AUTOENCODER TEST PASSED")
    print("="*80)


if __name__ == "__main__":
    test_autoencoder()
