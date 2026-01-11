"""
Multi-Modal Encoder with Adaptive Fusion - FIXED VERSION
Handles behavior_dim correctly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class BehaviorEncoder(nn.Module):
    """
    Encode behavior features (clicks, purchases, ratings, etc.)
    
    FIXED: Proper handling of behavior_dim parameter
    """
    
    def __init__(self, 
                 behavior_dim: int = 32,  # Input dimension of behavior features
                 hidden_dim: int = 128,
                 output_dim: int = 256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(behavior_dim, hidden_dim),  # ✅ Use behavior_dim for input
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),  # LayerNorm instead of BatchNorm (works with any batch size)
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)  # LayerNorm instead of BatchNorm
        )
        
    def forward(self, x):
        """
        Args:
            x: Behavior features (batch_size, behavior_dim)
        Returns:
            encoded: (batch_size, output_dim)
        """
        return self.encoder(x)


class AdaptiveFusionModule(nn.Module):
    """
    Adaptive fusion module - learns per-user weights for each modality
    
    INNOVATION CHÍNH CỦA PROJECT!
    """
    
    def __init__(self, embedding_dim: int = 384):
        super().__init__()
        
        # Learn fusion weights based on combined embedding
        self.weight_generator = nn.Sequential(
            nn.Linear(embedding_dim * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3),
            nn.Softmax(dim=1)  # Weights sum to 1
        )
        
    def forward(self, 
                text_emb: torch.Tensor,
                image_emb: torch.Tensor, 
                behavior_emb: torch.Tensor,
                return_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Adaptive fusion of three modalities
        
        Args:
            text_emb: (batch_size, embedding_dim)
            image_emb: (batch_size, embedding_dim)
            behavior_emb: (batch_size, embedding_dim)
            return_weights: Whether to return fusion weights
            
        Returns:
            fused_embedding: (batch_size, embedding_dim)
            weights (optional): (batch_size, 3) - [w_text, w_image, w_behavior]
        """
        # Concatenate all embeddings
        concat_emb = torch.cat([text_emb, image_emb, behavior_emb], dim=1)
        
        # Generate adaptive weights
        weights = self.weight_generator(concat_emb)  # (batch_size, 3)
        
        # Weighted sum
        w_text = weights[:, 0:1]  # (batch_size, 1)
        w_image = weights[:, 1:2]
        w_behavior = weights[:, 2:3]
        
        fused = w_text * text_emb + w_image * image_emb + w_behavior * behavior_emb
        
        if return_weights:
            return fused, weights
        return fused


class MultiModalEncoder(nn.Module):
    """
    Complete multi-modal encoder with adaptive fusion
    
    FIXED: Proper dimension handling for all modalities
    """
    
    def __init__(self,
                 text_dim: int = 384,           # Sentence-Transformers output
                 image_dim: int = 2048,         # ResNet-50 output
                 behavior_dim: int = 32,        # ✅ Behavior features input dim
                 hidden_dim: int = 256,
                 output_dim: int = 384):        # Final unified embedding
        super().__init__()
        
        # Text projection (if needed)
        self.text_projection = nn.Identity() if text_dim == output_dim else nn.Linear(text_dim, output_dim)
        
        # Image projection
        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Behavior encoder
        self.behavior_encoder = BehaviorEncoder(
            behavior_dim=behavior_dim,      # ✅ Input: 32
            hidden_dim=hidden_dim,          # Hidden: 256
            output_dim=output_dim           # Output: 384
        )
        
        # Adaptive fusion
        self.fusion = AdaptiveFusionModule(embedding_dim=output_dim)
        
        self.output_dim = output_dim
        
    def forward(self, 
                text_emb: torch.Tensor,
                image_emb: torch.Tensor,
                behavior_features: torch.Tensor,
                return_weights: bool = False):
        """
        Forward pass
        
        Args:
            text_emb: (batch_size, text_dim=384)
            image_emb: (batch_size, image_dim=2048)
            behavior_features: (batch_size, behavior_dim=32)
            return_weights: Return fusion weights
            
        Returns:
            user_embedding: (batch_size, output_dim=384)
            fusion_weights (optional): (batch_size, 3)
        """
        # Project to common dimension
        text_proj = self.text_projection(text_emb)          # (batch, 384)
        image_proj = self.image_projection(image_emb)       # (batch, 384)
        behavior_proj = self.behavior_encoder(behavior_features)  # (batch, 384)
        
        # Adaptive fusion
        if return_weights:
            user_emb, weights = self.fusion(
                text_proj, image_proj, behavior_proj, 
                return_weights=True
            )
            return user_emb, weights
        else:
            user_emb = self.fusion(text_proj, image_proj, behavior_proj)
            return user_emb


# Test the fixed encoder
if __name__ == "__main__":
    print("Testing MultiModalEncoder with FIXED dimensions...")
    
    # Create encoder
    encoder = MultiModalEncoder(
        text_dim=384,
        image_dim=2048,
        behavior_dim=32,      # ✅ Correct input dim
        hidden_dim=256,
        output_dim=384
    )
    
    # Test with correct dimensions
    batch_size = 8
    text_emb = torch.randn(batch_size, 384)
    image_emb = torch.randn(batch_size, 2048)
    behavior_feat = torch.randn(batch_size, 32)  # ✅ 32 not 50
    
    # Forward pass
    print("\n=== Forward Pass Test ===")
    try:
        user_emb = encoder(text_emb, image_emb, behavior_feat)
        print(f"✅ User embedding shape: {user_emb.shape}")
        
        user_emb, weights = encoder(text_emb, image_emb, behavior_feat, return_weights=True)
        print(f"✅ Fusion weights shape: {weights.shape}")
        print(f"✅ Sample weights: {weights[0]}")
        print(f"✅ Weights sum to 1: {weights[0].sum().item():.4f}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()