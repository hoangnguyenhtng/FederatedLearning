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
    
    ✅ Improved: Added LayerNorm for training stability
    """
    
    def __init__(self,
                 input_dim: int = 128,
                 hidden_dims: list = [128, 64],
                 num_classes: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build personal layers with LayerNorm for stability
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),  # GELU works better than ReLU for classification
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.personal_network = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: Shared features (batch_size, input_dim)
        Returns:
            logits: (batch_size, num_classes)
        """
        return self.personal_network(x)


class FedPerRecommender(nn.Module):
    """
    Complete FedPer Recommendation Model
    """
    
    def __init__(self,
                 multimodal_encoder,
                 shared_hidden_dims: list = [512, 256, 128],
                 personal_hidden_dims: list = [128, 64],
                 num_classes: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        
        self.multimodal_encoder = multimodal_encoder
        
        # Shared base
        self.shared_base = SharedRecommendationBase(
            input_dim=384,
            hidden_dims=shared_hidden_dims,
            dropout=dropout
        )
        
        # ✅ Skip connection: project multimodal output to shared_base output dim
        # This gives personal head access to both shared representation AND raw fusion
        self.skip_projection = nn.Linear(384, shared_hidden_dims[-1])
        self.skip_norm = nn.LayerNorm(shared_hidden_dims[-1])
        
        # Personal head (takes shared_features + skip connection)
        self.personal_head = PersonalHead(
            input_dim=shared_hidden_dims[-1],  # Same dim after skip addition
            hidden_dims=personal_hidden_dims,
            num_classes=num_classes,
            dropout=dropout
        )
        
    def forward(self, text_emb, image_emb, behavior_features, 
                return_fusion_weights=False):
        """
        Forward pass with residual skip connection
        
        Args:
            text_emb: (batch_size, 384)
            image_emb: (batch_size, 2048)
            behavior_features: (batch_size, 32)
            return_fusion_weights: Return fusion weights
            
        Returns:
            logits: (batch_size, num_classes)
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
        
        # Step 3: ✅ Residual skip connection from multimodal output
        # personal_input = shared_features + projected(multimodal_output)
        skip = self.skip_projection(user_emb)
        personal_input = self.skip_norm(shared_features + skip)
        
        # Step 4: Personal head
        logits = self.personal_head(personal_input)
        
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
            shared_params[f"multimodal_encoder.{name}"] = param
        
        # Shared base parameters (bắt buộc share)
        for name, param in self.shared_base.named_parameters():
            shared_params[f"shared_base.{name}"] = param
        
        # Skip connection parameters (shared)
        for name, param in self.skip_projection.named_parameters():
            shared_params[f"skip_projection.{name}"] = param
        for name, param in self.skip_norm.named_parameters():
            shared_params[f"skip_norm.{name}"] = param
            
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
        num_classes=5  # Rating prediction: 5 classes (1-5 → 0-4)
    )
    
    # Create dummy data
    batch_size = 8
    text_emb = torch.randn(batch_size, 384)     # SentenceTransformer output
    image_emb = torch.randn(batch_size, 2048)    # ResNet-50 / image proxy output
    behavior_feat = torch.randn(batch_size, 32)  # Behavior features
    
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