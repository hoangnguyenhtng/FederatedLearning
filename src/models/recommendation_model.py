"""
Recommendation Model với FedPer Architecture

FedPer (Federated Personalization):
- Base layers: Shared với server (federated learning)
- Personal head: Giữ riêng ở client (KHÔNG share)

Structure:
├── Shared Base (gửi cho server)
│   ├── Multi-modal encoder
│   └── Shared recommendation layers (3 layers)
│
└── Personal Head (giữ riêng)
    └── Final prediction layers (2 layers)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class SharedRecommendationBase(nn.Module):
    """
    Base model - ĐƯỢC SHARE VỚI SERVER
    
    Nhiệm vụ: Học representation chung cho tất cả users
    """
    
    def __init__(self, 
                 input_dim: int = 384,
                 hidden_dims: list = [512, 256, 128],
                 dropout: float = 0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build shared layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),  # LayerNorm instead of BatchNorm (works with any batch size)
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.shared_network = nn.Sequential(*layers)
        self.output_dim = prev_dim
        
    def forward(self, x):
        """
        Args:
            x: User embedding (batch_size, input_dim)
        Returns:
            shared_features: (batch_size, output_dim)
        """
        return self.shared_network(x)


class PersonalHead(nn.Module):
    """
    Personal head - KHÔNG SHARE, giữ riêng ở client
    """
    
    def __init__(self,
                 input_dim: int = 128,
                 hidden_dims: list = [64, 32],
                 num_classes: int = 5,  # ✅ ĐỔI TÊN: num_items → num_classes
                 dropout: float = 0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build personal layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, num_classes))  # ✅ Dùng num_classes
        
        self.personal_network = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: Shared features (batch_size, input_dim)
        Returns:
            logits: (batch_size, num_classes)  # ✅ Đổi comment
        """
        return self.personal_network(x)


class FedPerRecommender(nn.Module):
    """
    Complete FedPer Recommendation Model
    """
    
    def __init__(self,
                 multimodal_encoder,
                 shared_hidden_dims: list = [512, 256, 128],
                 personal_hidden_dims: list = [64, 32],
                 num_classes: int = 5,  # ✅ ĐỔI TÊN: num_items → num_classes
                 dropout: float = 0.2):
        super().__init__()
        
        self.multimodal_encoder = multimodal_encoder
        
        # Shared base
        self.shared_base = SharedRecommendationBase(
            input_dim=384,
            hidden_dims=shared_hidden_dims,
            dropout=dropout
        )
        
        # Personal head
        self.personal_head = PersonalHead(
            input_dim=shared_hidden_dims[-1],
            hidden_dims=personal_hidden_dims,
            num_classes=num_classes,  # ✅ Dùng num_classes
            dropout=dropout
        )
        
    def forward(self, text_emb, image_emb, behavior_features, 
                return_fusion_weights=False):
        """
        Forward pass
        
        Args:
            text_emb: (batch_size, 384)
            image_emb: (batch_size, 384)
            behavior_features: (batch_size, 50)
            return_fusion_weights: Return fusion weights
            
        Returns:
            logits: (batch_size, num_classes)  # ✅ Đổi comment
            fusion_weights (optional): (batch_size, 3)
        """
        # Step 1: Multi-modal fusion
        if return_fusion_weights:
            user_emb, fusion_weights = self.multimodal_encoder(
                text_emb, image_emb, behavior_features, return_weights=True
            )
        else:
            user_emb = self.multimodal_encoder(
                text_emb, image_emb, behavior_features
            )
        
        # Step 2: Shared base
        shared_features = self.shared_base(user_emb)
        
        # Step 3: Personal head
        logits = self.personal_head(shared_features)
        
        if return_fusion_weights:
            return logits, fusion_weights
        return logits
    
    def get_shared_parameters(self):
        """
        Lấy parameters của shared layers (để gửi cho server)
        """
        shared_params = {}
        
        # Multimodal encoder parameters (có thể share hoặc không)
        for name, param in self.multimodal_encoder.named_parameters():
            shared_params[f"multimodal.{name}"] = param
        
        # Shared base parameters (bắt buộc share)
        for name, param in self.shared_base.named_parameters():
            shared_params[f"shared_base.{name}"] = param
            
        return shared_params
    
    def get_personal_parameters(self):
        """
        Lấy parameters của personal layers (KHÔNG share)
        """
        personal_params = {}
        
        for name, param in self.personal_head.named_parameters():
            personal_params[f"personal_head.{name}"] = param
            
        return personal_params
    
    def set_shared_parameters(self, shared_params: Dict[str, torch.Tensor]):
        """
        Update shared parameters từ server
        """
        current_state = self.state_dict()
        
        for name, param in shared_params.items():
            if name in current_state:
                current_state[name] = param
        
        self.load_state_dict(current_state, strict=False)
    
    def predict_top_k(self, text_emb, image_emb, behavior_features, k=5):  # ✅ k=5 thay vì k=10

        self.eval()
        with torch.no_grad():
            logits = self.forward(text_emb, image_emb, behavior_features)
            scores = F.softmax(logits, dim=1)
            top_k_scores, top_k_classes = torch.topk(scores, k, dim=1)
    
        return top_k_classes, top_k_scores


# Example usage
if __name__ == "__main__":
    from src.models.multimodal_encoder import MultiModalEncoder
    
    # Initialize models
    multimodal_encoder = MultiModalEncoder()
    model = FedPerRecommender(
        multimodal_encoder=multimodal_encoder,
        num_items=10000
    )
    
    # Create dummy data
    batch_size = 8
    text_emb = torch.randn(batch_size, 384)
    image_emb = torch.randn(batch_size, 384)
    behavior_feat = torch.randn(batch_size, 50)
    
    # Forward pass
    logits, weights = model(text_emb, image_emb, behavior_feat, 
                           return_fusion_weights=True)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Fusion weights: {weights[0]}")
    
    # Get shared và personal parameters
    shared_params = model.get_shared_parameters()
    personal_params = model.get_personal_parameters()
    
    print(f"\nShared parameters: {len(shared_params)} tensors")
    print(f"Personal parameters: {len(personal_params)} tensors")
    
    # Predict top-K
    top_items, top_scores = model.predict_top_k(
        text_emb, image_emb, behavior_feat, k=10
    )
    print(f"\nTop-10 items shape: {top_items.shape}")
    print(f"Top-10 scores shape: {top_scores.shape}")