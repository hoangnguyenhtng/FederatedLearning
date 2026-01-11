from src.models.recommendation_model import FedPerRecommender
from src.models.multimodal_encoder import MultiModalEncoder

def create_model(model_config: dict) -> FedPerRecommender:
    """
    Create FedPerRecommender model with proper initialization.
    
    Args:
        model_config: Dictionary containing model configuration
        
    Returns:
        FedPerRecommender: Initialized model
    """
    # Step 1: Create MultiModalEncoder first
    multimodal_encoder = MultiModalEncoder(
        text_dim=model_config.get("text_embedding_dim", 384),
        image_dim=model_config.get("image_embedding_dim", 2048),
        behavior_dim=model_config.get("behavior_embedding_dim", 32),
        hidden_dim=model_config.get("hidden_dim", 256),
        output_dim=384  # Output dimension for FedPerRecommender
    )
    
    # Step 2: Create FedPerRecommender with encoder
    shared_dims = model_config.get("shared_hidden_dims", [512, 256, 128])
    personal_dims = model_config.get("personal_hidden_dims", [64, 32])
    # Rating prediction: 5 classes (ratings 1-5, mapped to 0-4)
    num_classes = model_config.get("num_classes", 5)  # Changed from 10000 to 5
    dropout = model_config.get("dropout", 0.2)
    
    return FedPerRecommender(
        multimodal_encoder=multimodal_encoder,  # ✅ Pass encoder object
        shared_hidden_dims=shared_dims,
        personal_hidden_dims=personal_dims,
        num_items=num_classes,  # num_items parameter name, but value is num_classes (5)
        dropout=dropout
    )
