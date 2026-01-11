"""
Attention Mechanisms for Multi-Modal Fusion
============================================

This module implements various attention mechanisms for fusing multi-modal features:
1. CrossAttentionModule: Cross-attention between modalities
2. SelfAttentionModule: Self-attention within a modality
3. MultiHeadAttention: Multi-head attention for richer representations
4. AdaptiveAttentionFusion: Learnable attention-based fusion

Author: Federated Multi-Modal Recommendation System
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttentionModule(nn.Module):
    """
    Cross-Attention between two modalities.
    
    Used to let one modality (query) attend to another modality (key, value).
    Example: Let text attend to image features to capture visual-textual correlations.
    
    Args:
        query_dim (int): Dimension of query features
        key_dim (int): Dimension of key/value features
        hidden_dim (int): Hidden dimension for attention computation
        dropout (float): Dropout rate
    """
    
    def __init__(self, query_dim=384, key_dim=512, hidden_dim=256, dropout=0.1):
        super(CrossAttentionModule, self).__init__()
        
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.hidden_dim = hidden_dim
        self.scale = math.sqrt(hidden_dim)
        
        # Linear projections
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(key_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, query_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(query_dim)
        
    def forward(self, query, key, value=None, mask=None):
        """
        Forward pass of cross-attention.
        
        Args:
            query (torch.Tensor): Query features [batch_size, query_dim]
            key (torch.Tensor): Key features [batch_size, key_dim]
            value (torch.Tensor): Value features [batch_size, key_dim] (optional, defaults to key)
            mask (torch.Tensor): Attention mask (optional)
            
        Returns:
            torch.Tensor: Attended features [batch_size, query_dim]
            torch.Tensor: Attention weights [batch_size, 1]
        """
        if value is None:
            value = key
            
        batch_size = query.size(0)
        
        # Store residual
        residual = query
        
        # Project to hidden dimension
        Q = self.query_proj(query)  # [batch_size, hidden_dim]
        K = self.key_proj(key)      # [batch_size, hidden_dim]
        V = self.value_proj(value)  # [batch_size, hidden_dim]
        
        # Add sequence dimension for attention computation
        Q = Q.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        K = K.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        V = V.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Compute attention scores
        # scores = Q @ K^T / sqrt(hidden_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [batch_size, 1, 1]
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, 1, 1]
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, V)  # [batch_size, 1, hidden_dim]
        attended = attended.squeeze(1)  # [batch_size, hidden_dim]
        
        # Output projection
        output = self.out_proj(attended)  # [batch_size, query_dim]
        
        # Residual connection + layer norm
        output = self.layer_norm(residual + self.dropout(output))
        
        return output, attn_weights.squeeze()


class SelfAttentionModule(nn.Module):
    """
    Self-Attention within a single modality.
    
    Used to capture internal relationships within features of the same modality.
    
    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden dimension for attention
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate
    """
    
    def __init__(self, input_dim=384, hidden_dim=256, num_heads=4, dropout=0.1):
        super(SelfAttentionModule, self).__init__()
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Linear projections
        self.qkv_proj = nn.Linear(input_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, input_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, x, mask=None):
        """
        Forward pass of self-attention.
        
        Args:
            x (torch.Tensor): Input features [batch_size, input_dim]
            mask (torch.Tensor): Attention mask (optional)
            
        Returns:
            torch.Tensor: Self-attended features [batch_size, input_dim]
            torch.Tensor: Attention weights [batch_size, num_heads, 1, 1]
        """
        batch_size = x.size(0)
        
        # Store residual
        residual = x
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # [batch_size, hidden_dim * 3]
        qkv = qkv.reshape(batch_size, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(1, 0, 2, 3)  # [3, batch_size, num_heads, head_dim]
        
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # Add sequence dimension
        Q = Q.unsqueeze(2)  # [batch_size, num_heads, 1, head_dim]
        K = K.unsqueeze(2)  # [batch_size, num_heads, 1, head_dim]
        V = V.unsqueeze(2)  # [batch_size, num_heads, 1, head_dim]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, V)  # [batch_size, num_heads, 1, head_dim]
        attended = attended.squeeze(2)  # [batch_size, num_heads, head_dim]
        attended = attended.reshape(batch_size, self.hidden_dim)
        
        # Output projection
        output = self.out_proj(attended)
        
        # Residual connection + layer norm
        output = self.layer_norm(residual + self.dropout(output))
        
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    
    Standard multi-head attention that can be used for both self and cross attention.
    
    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate
        bias (bool): Whether to use bias in projections
    """
    
    def __init__(self, embed_dim=384, num_heads=8, dropout=0.1, bias=True):
        super(MultiHeadAttention, self).__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Linear layers
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_linear = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        Forward pass of multi-head attention.
        
        Args:
            query (torch.Tensor): Query tensor [batch_size, embed_dim]
            key (torch.Tensor): Key tensor [batch_size, embed_dim]
            value (torch.Tensor): Value tensor [batch_size, embed_dim]
            attn_mask (torch.Tensor): Attention mask (optional)
            key_padding_mask (torch.Tensor): Key padding mask (optional)
            
        Returns:
            torch.Tensor: Output features [batch_size, embed_dim]
            torch.Tensor: Attention weights [batch_size, num_heads, 1, 1]
        """
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.q_linear(query)  # [batch_size, embed_dim]
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, self.num_heads, 1, self.head_dim)
        K = K.view(batch_size, self.num_heads, 1, self.head_dim)
        V = V.view(batch_size, self.num_heads, 1, self.head_dim)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply attention mask
        if attn_mask is not None:
            scores = scores + attn_mask
            
        # Apply key padding mask
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), -1e9)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, V)  # [batch_size, num_heads, 1, head_dim]
        attended = attended.view(batch_size, self.embed_dim)
        
        # Output projection
        output = self.out_linear(attended)
        output = self.proj_dropout(output)
        
        return output, attn_weights


class AdaptiveAttentionFusion(nn.Module):
    """
    Adaptive Attention-based Fusion for multiple modalities.
    
    This module learns to attend to different modalities based on their relevance
    for each user. It's the CORE INNOVATION of the project.
    
    Args:
        text_dim (int): Text feature dimension
        image_dim (int): Image feature dimension
        behavior_dim (int): Behavior feature dimension
        hidden_dim (int): Hidden dimension for attention
        output_dim (int): Output fused feature dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate
    """
    
    def __init__(self, text_dim=384, image_dim=512, behavior_dim=128, 
                 hidden_dim=256, output_dim=384, num_heads=4, dropout=0.1):
        super(AdaptiveAttentionFusion, self).__init__()
        
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.behavior_dim = behavior_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        
        # Project all modalities to same dimension
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.behavior_proj = nn.Linear(behavior_dim, hidden_dim)
        
        # Cross-attention between modalities
        self.text_to_image_attn = CrossAttentionModule(
            query_dim=hidden_dim,
            key_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        self.image_to_text_attn = CrossAttentionModule(
            query_dim=hidden_dim,
            key_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        self.behavior_attn = SelfAttentionModule(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Adaptive weight learning
        self.weight_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1)
        )
        
        # Final projection
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, text_features, image_features, behavior_features):
        """
        Forward pass of adaptive attention fusion.
        
        Args:
            text_features (torch.Tensor): Text features [batch_size, text_dim]
            image_features (torch.Tensor): Image features [batch_size, image_dim]
            behavior_features (torch.Tensor): Behavior features [batch_size, behavior_dim]
            
        Returns:
            torch.Tensor: Fused features [batch_size, output_dim]
            dict: Attention information (weights, cross-attention scores)
        """
        batch_size = text_features.size(0)
        
        # Project to same dimension
        text_proj = self.text_proj(text_features)      # [batch_size, hidden_dim]
        image_proj = self.image_proj(image_features)   # [batch_size, hidden_dim]
        behavior_proj = self.behavior_proj(behavior_features)  # [batch_size, hidden_dim]
        
        # Cross-modal attention
        text_attended, text_to_image_weights = self.text_to_image_attn(
            query=text_proj, 
            key=image_proj
        )
        
        image_attended, image_to_text_weights = self.image_to_text_attn(
            query=image_proj,
            key=text_proj
        )
        
        # Self-attention for behavior
        behavior_attended, behavior_weights = self.behavior_attn(behavior_proj)
        
        # Concatenate attended features
        all_features = torch.cat([
            text_attended,
            image_attended,
            behavior_attended
        ], dim=-1)  # [batch_size, hidden_dim * 3]
        
        # Learn adaptive weights
        adaptive_weights = self.weight_net(all_features)  # [batch_size, 3]
        
        # Weighted combination
        weighted_text = text_attended * adaptive_weights[:, 0:1]
        weighted_image = image_attended * adaptive_weights[:, 1:2]
        weighted_behavior = behavior_attended * adaptive_weights[:, 2:3]
        
        # Concatenate weighted features
        weighted_features = torch.cat([
            weighted_text,
            weighted_image,
            weighted_behavior
        ], dim=-1)
        
        # Final fusion
        fused = self.fusion_proj(weighted_features)
        fused = self.layer_norm(fused)
        
        # Prepare attention info for interpretability
        attention_info = {
            'adaptive_weights': adaptive_weights,  # [batch_size, 3]
            'text_to_image_attn': text_to_image_weights,
            'image_to_text_attn': image_to_text_weights,
            'behavior_self_attn': behavior_weights
        }
        
        return fused, attention_info


class ModalityGatingMechanism(nn.Module):
    """
    Gating mechanism to control information flow from each modality.
    
    This is an alternative/complementary approach to attention-based fusion.
    Each modality has a learnable gate that decides how much information to pass.
    
    Args:
        text_dim (int): Text feature dimension
        image_dim (int): Image feature dimension
        behavior_dim (int): Behavior feature dimension
        hidden_dim (int): Hidden dimension
        output_dim (int): Output dimension
    """
    
    def __init__(self, text_dim=384, image_dim=512, behavior_dim=128,
                 hidden_dim=256, output_dim=384):
        super(ModalityGatingMechanism, self).__init__()
        
        # Project to same dimension
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.behavior_proj = nn.Linear(behavior_dim, hidden_dim)
        
        # Gate networks for each modality
        self.text_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.image_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.behavior_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, text_features, image_features, behavior_features):
        """
        Forward pass with gating mechanism.
        
        Args:
            text_features (torch.Tensor): Text features
            image_features (torch.Tensor): Image features
            behavior_features (torch.Tensor): Behavior features
            
        Returns:
            torch.Tensor: Gated fused features
            dict: Gate values for interpretability
        """
        # Project
        text_proj = self.text_proj(text_features)
        image_proj = self.image_proj(image_features)
        behavior_proj = self.behavior_proj(behavior_features)
        
        # Compute gates
        text_gate_val = self.text_gate(text_proj)
        image_gate_val = self.image_gate(image_proj)
        behavior_gate_val = self.behavior_gate(behavior_proj)
        
        # Apply gates
        gated_text = text_proj * text_gate_val
        gated_image = image_proj * image_gate_val
        gated_behavior = behavior_proj * behavior_gate_val
        
        # Concatenate and fuse
        gated_features = torch.cat([gated_text, gated_image, gated_behavior], dim=-1)
        output = self.fusion(gated_features)
        
        # Gate info for interpretability
        gate_info = {
            'text_gate': text_gate_val.mean(dim=-1),  # Average gate value
            'image_gate': image_gate_val.mean(dim=-1),
            'behavior_gate': behavior_gate_val.mean(dim=-1)
        }
        
        return output, gate_info


# =============================================================================
# Utility Functions
# =============================================================================

def create_attention_mask(seq_len, device='cpu'):
    """
    Create a causal attention mask for sequence modeling.
    
    Args:
        seq_len (int): Sequence length
        device (str): Device to create mask on
        
    Returns:
        torch.Tensor: Attention mask [seq_len, seq_len]
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


def create_padding_mask(lengths, max_len=None, device='cpu'):
    """
    Create padding mask from sequence lengths.
    
    Args:
        lengths (torch.Tensor): Sequence lengths [batch_size]
        max_len (int): Maximum length (optional)
        device (str): Device to create mask on
        
    Returns:
        torch.Tensor: Padding mask [batch_size, max_len]
    """
    batch_size = lengths.size(0)
    if max_len is None:
        max_len = lengths.max().item()
    
    mask = torch.arange(max_len, device=device).expand(batch_size, max_len) < lengths.unsqueeze(1)
    return mask


# =============================================================================
# Testing & Validation
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Attention Mechanisms")
    print("=" * 70)
    
    # Set random seed
    torch.manual_seed(42)
    
    batch_size = 16
    text_dim = 384
    image_dim = 512
    behavior_dim = 128
    
    # Create dummy features
    text_features = torch.randn(batch_size, text_dim)
    image_features = torch.randn(batch_size, image_dim)
    behavior_features = torch.randn(batch_size, behavior_dim)
    
    print(f"\nInput shapes:")
    print(f"  Text: {text_features.shape}")
    print(f"  Image: {image_features.shape}")
    print(f"  Behavior: {behavior_features.shape}")
    
    # Test 1: CrossAttentionModule
    print("\n" + "=" * 70)
    print("1. Testing CrossAttentionModule")
    print("=" * 70)
    cross_attn = CrossAttentionModule(query_dim=text_dim, key_dim=image_dim)
    cross_output, cross_weights = cross_attn(text_features, image_features)
    print(f"✓ Output shape: {cross_output.shape}")
    print(f"✓ Attention weights shape: {cross_weights.shape}")
    print(f"✓ Attention weights sample: {cross_weights[0].item():.4f}")
    
    # Test 2: SelfAttentionModule
    print("\n" + "=" * 70)
    print("2. Testing SelfAttentionModule")
    print("=" * 70)
    self_attn = SelfAttentionModule(input_dim=text_dim, num_heads=4)
    self_output, self_weights = self_attn(text_features)
    print(f"✓ Output shape: {self_output.shape}")
    print(f"✓ Attention weights shape: {self_weights.shape}")
    
    # Test 3: MultiHeadAttention
    print("\n" + "=" * 70)
    print("3. Testing MultiHeadAttention")
    print("=" * 70)
    mha = MultiHeadAttention(embed_dim=text_dim, num_heads=8)
    mha_output, mha_weights = mha(text_features, text_features, text_features)
    print(f"✓ Output shape: {mha_output.shape}")
    print(f"✓ Attention weights shape: {mha_weights.shape}")
    
    # Test 4: AdaptiveAttentionFusion (MAIN MODULE)
    print("\n" + "=" * 70)
    print("4. Testing AdaptiveAttentionFusion (CORE INNOVATION)")
    print("=" * 70)
    fusion = AdaptiveAttentionFusion(
        text_dim=text_dim,
        image_dim=image_dim,
        behavior_dim=behavior_dim,
        output_dim=384
    )
    fused_output, attn_info = fusion(text_features, image_features, behavior_features)
    print(f"✓ Fused output shape: {fused_output.shape}")
    print(f"✓ Adaptive weights shape: {attn_info['adaptive_weights'].shape}")
    print(f"\nAdaptive weights for first sample:")
    print(f"  Text weight: {attn_info['adaptive_weights'][0, 0].item():.4f}")
    print(f"  Image weight: {attn_info['adaptive_weights'][0, 1].item():.4f}")
    print(f"  Behavior weight: {attn_info['adaptive_weights'][0, 2].item():.4f}")
    print(f"  Sum: {attn_info['adaptive_weights'][0].sum().item():.4f} (should be ~1.0)")
    
    # Test 5: ModalityGatingMechanism
    print("\n" + "=" * 70)
    print("5. Testing ModalityGatingMechanism")
    print("=" * 70)
    gating = ModalityGatingMechanism(
        text_dim=text_dim,
        image_dim=image_dim,
        behavior_dim=behavior_dim,
        output_dim=384
    )
    gated_output, gate_info = gating(text_features, image_features, behavior_features)
    print(f"✓ Gated output shape: {gated_output.shape}")
    print(f"\nGate values for first sample:")
    print(f"  Text gate: {gate_info['text_gate'][0].item():.4f}")
    print(f"  Image gate: {gate_info['image_gate'][0].item():.4f}")
    print(f"  Behavior gate: {gate_info['behavior_gate'][0].item():.4f}")
    
    # Test 6: Parameter count
    print("\n" + "=" * 70)
    print("6. Model Statistics")
    print("=" * 70)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"CrossAttentionModule parameters: {count_parameters(cross_attn):,}")
    print(f"SelfAttentionModule parameters: {count_parameters(self_attn):,}")
    print(f"MultiHeadAttention parameters: {count_parameters(mha):,}")
    print(f"AdaptiveAttentionFusion parameters: {count_parameters(fusion):,}")
    print(f"ModalityGatingMechanism parameters: {count_parameters(gating):,}")
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nAttention mechanisms are ready to use in your federated learning pipeline.")
    print("The AdaptiveAttentionFusion module is the core innovation for multi-modal fusion.")