"""
RND (Random Network Distillation) models for uncertainty quantification.

Adapted from FIPER (Römer et al., 2025) for Pi0.5 VLA policy.
Simplified to handle only observation embeddings (no action predictions).
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np


class RND_OE(nn.Module):
    """
    Random Network Distillation for Observation Embeddings.
    
    This model learns to predict the output of a randomly initialized frozen network
    on in-distribution data. At inference, high prediction error indicates
    out-of-distribution inputs.
    
    Architecture:
        - Input: (2048,) SigLIP token embeddings from Pi0.5 vision encoder
        - Target Network: Frozen random network (2048 -> 512)
        - Predictor Network: Trainable network (2048 -> 512)
        - Loss: MSE between predictor and target outputs
    
    Usage:
        # Training
        model = RND_OE(obs_embedding_dim=2048, output_size=512)
        loss = model(token_embeddings)  # (batch_size,)
        
        # Inference
        with torch.no_grad():
            uncertainty = model(new_tokens)  # High values = OOD
    """
    
    def __init__(
        self,
        obs_embedding_dim: int = 2048,
        output_size: int = 512,
        rnd_loss: str = "mse",
        seed: Optional[int] = None,
    ):
        """
        Initialize RND model.
        
        Args:
            obs_embedding_dim: Dimension of input embeddings (default: 2048 for SigLIP)
            output_size: Dimension of RND feature space (default: 512)
            rnd_loss: Loss function type ('mse' or 'l2')
            seed: Random seed for network initialization (for reproducibility)
        """
        super().__init__()
        
        # Store config
        self.obs_embedding_dim = obs_embedding_dim
        self.output_size = output_size
        self.rnd_loss_type = rnd_loss
        self.seed = seed
        
        # Set seed for reproducible initialization
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Initialize networks
        self.target_network = self._build_target_network()
        self.predictor_network = self._build_predictor_network()
        
        # Initialize loss function
        if rnd_loss == "mse":
            self.rnd_mse_loss = nn.MSELoss(reduction="none")
            self.rnd_loss_function = lambda target, prediction: self.rnd_mse_loss(target, prediction).mean(dim=-1)
        elif rnd_loss == "l2":
            self.rnd_loss_function = nn.PairwiseDistance(p=2)
        else:
            raise ValueError(f"Unknown RND loss type: {rnd_loss}")
        
        # Initialize weights and freeze target network
        self._initialize_weights()
        self._freeze_target_network()
    
    def _build_target_network(self) -> nn.Module:
        """Build the frozen target network (from FIPER architecture)."""
        return nn.Sequential(
            nn.Linear(self.obs_embedding_dim, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, self.output_size),
        )
    
    def _build_predictor_network(self) -> nn.Module:
        """Build the trainable predictor network (from FIPER architecture)."""
        return nn.Sequential(
            nn.Linear(self.obs_embedding_dim, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.output_size),
        )
    
    def _initialize_weights(self):
        """Initialize network weights with orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, np.sqrt(2))
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def _freeze_target_network(self):
        """Freeze target network parameters (should never be updated)."""
        for param in self.target_network.parameters():
            param.requires_grad = False
    
    def forward(self, obs_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute RND prediction error (uncertainty).
        
        Args:
            obs_embeddings: Token embeddings of shape (batch_size, 2048)
        
        Returns:
            uncertainty: RND loss per sample of shape (batch_size,)
                        Higher values indicate more uncertain/OOD inputs
        """
        # Compute features from both networks
        target_features = self.target_network(obs_embeddings)      # (batch_size, output_size)
        predicted_features = self.predictor_network(obs_embeddings)  # (batch_size, output_size)
        
        # Compute prediction error as uncertainty score
        uncertainty = self.rnd_loss_function(predicted_features, target_features)
        
        return uncertainty
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for saving."""
        return {
            "obs_embedding_dim": self.obs_embedding_dim,
            "output_size": self.output_size,
            "rnd_loss": self.rnd_loss_type,
            "seed": self.seed,
        }
    
    def save_checkpoint(
        self,
        save_path: Path,
        epoch: int,
        best_val_loss: float,
        optimizer_state: Optional[Dict] = None,
        train_losses: Optional[list] = None,
        val_losses: Optional[list] = None,
        hyperparameters: Optional[Dict] = None,
    ):
        """
        Save model checkpoint with full metadata.
        
        Args:
            save_path: Path to save checkpoint file
            epoch: Training epoch number
            best_val_loss: Best validation loss achieved
            optimizer_state: Optimizer state dict (optional)
            train_losses: List of training losses per epoch
            val_losses: List of validation losses per epoch
            hyperparameters: Training hyperparameters
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            # Model state
            "state_dict": self.state_dict(),
            "model_config": self.get_config(),
            
            # Training state
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            
            # Training history
            "train_losses": train_losses or [],
            "val_losses": val_losses or [],
            
            # Hyperparameters for reproducibility
            "hyperparameters": hyperparameters or {},
        }
        
        # Add optimizer state if provided
        if optimizer_state is not None:
            checkpoint["optimizer_state"] = optimizer_state
        
        # Save checkpoint
        torch.save(checkpoint, save_path)
        
        # Save metadata as JSON for easy inspection
        metadata_path = save_path.parent / "metadata.json"
        metadata = {
            "model_config": self.get_config(),
            "epoch": epoch,
            "best_val_loss": float(best_val_loss),
            "hyperparameters": hyperparameters or {},
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load_checkpoint(cls, checkpoint_path: Path, device: str = "cpu") -> "RND_OE":
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load model on
        
        Returns:
            model: Loaded RND_OE model with trained weights
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Create model with saved config
        model_config = checkpoint["model_config"]
        model = cls(**model_config)
        
        # Load trained weights
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        
        return model


class RNDEnsemble(nn.Module):
    """
    Ensemble of RND models for multi-camera uncertainty estimation.
    
    Manages separate RND models for different camera views and combines
    their uncertainty scores.
    """
    
    def __init__(self, camera_models: Dict[str, RND_OE]):
        """
        Initialize RND ensemble.
        
        Args:
            camera_models: Dict mapping camera names to RND models
                          e.g., {'agentview': rnd_agentview, 'wrist': rnd_wrist}
        """
        super().__init__()
        self.camera_models = nn.ModuleDict(camera_models)
    
    def forward(self, camera_embeddings: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute uncertainty for all cameras.
        
        Args:
            camera_embeddings: Dict mapping camera names to token embeddings
                              e.g., {'agentview': (256, 2048), 'wrist': (256, 2048)}
        
        Returns:
            uncertainties: Dict with per-camera and combined uncertainty scores
        """
        uncertainties = {}
        
        # Compute uncertainty for each camera
        for camera, model in self.camera_models.items():
            if camera in camera_embeddings:
                uncertainties[camera] = model(camera_embeddings[camera])
        
        # Compute combined uncertainty (mean across cameras)
        if len(uncertainties) > 0:
            uncertainties['overall'] = torch.stack(list(uncertainties.values())).mean(dim=0)
        
        return uncertainties
    
    @classmethod
    def load_from_checkpoints(
        cls,
        checkpoint_paths: Dict[str, Path],
        device: str = "cpu"
    ) -> "RNDEnsemble":
        """
        Load ensemble from multiple checkpoint files.
        
        Args:
            checkpoint_paths: Dict mapping camera names to checkpoint paths
                             e.g., {'agentview': 'spatial_agentview/model.ckpt', ...}
            device: Device to load models on
        
        Returns:
            ensemble: Loaded RND ensemble
        """
        camera_models = {}
        for camera, ckpt_path in checkpoint_paths.items():
            camera_models[camera] = RND_OE.load_checkpoint(ckpt_path, device)
        
        return cls(camera_models)
