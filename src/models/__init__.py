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
from .attention_mechanism import (
    CrossAttentionModule,
    SelfAttentionModule,
    MultiHeadAttention,
    AdaptiveAttentionFusion,
    ModalityGatingMechanism
)

__all__ = [
    # Encoders
    'BehaviorEncoder',
    'AdaptiveFusionModule',
    'MultiModalEncoder',
    
    # Recommenders
    'SharedRecommendationBase',
    'PersonalHead',
    'FedPerRecommender',
    
    # Attention
    'CrossAttentionModule',
    'SelfAttentionModule',
    'MultiHeadAttention',
    'AdaptiveAttentionFusion',
    'ModalityGatingMechanism'
]