"""
Models Module
Multi-modal encoders and recommendation models
"""

from .multimodal_encoder import (
    BehaviorEncoder,
    AdaptiveFusionModule,
    MultiModalEncoder
)
from .recommendation_model import (
    SharedRecommendationBase,
    PersonalHead,
    FedPerRecommender
)

# attention_mechanism is optional (may not exist)
try:
    from .attention_mechanism import (
        CrossAttentionModule,
        SelfAttentionModule,
        MultiHeadAttention,
        AdaptiveAttentionFusion,
        ModalityGatingMechanism
    )
except ImportError:
    CrossAttentionModule = None
    SelfAttentionModule = None
    MultiHeadAttention = None
    AdaptiveAttentionFusion = None
    ModalityGatingMechanism = None

__all__ = [
    # Encoders
    'BehaviorEncoder',
    'AdaptiveFusionModule',
    'MultiModalEncoder',
    
    # Recommenders
    'SharedRecommendationBase',
    'PersonalHead',
    'FedPerRecommender',
    
    # Attention (optional)
    'CrossAttentionModule',
    'SelfAttentionModule',
    'MultiHeadAttention',
    'AdaptiveAttentionFusion',
    'ModalityGatingMechanism'
]