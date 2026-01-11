"""
Image Processor: Xử lý dữ liệu ảnh (product images)
Sử dụng ResNet-50 pre-trained làm feature extractor
"""
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from typing import List, Union
import numpy as np
from pathlib import Path

class ImageProcessor:
    """
    Xử lý image data và tạo embeddings
    
    Flow:
    1. Load ảnh → Resize → Normalize
    2. Pass qua ResNet-50 (pre-trained) → Extract features
    3. Project về 384-dim (cùng dimension với text)
    """
    
    def __init__(self, output_dim: int = 384, device: str = None):
        """
        Args:
            output_dim: Dimension của output embeddings
            device: 'cuda' hoặc 'cpu'
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dim = output_dim
        
        # Load ResNet-50 pre-trained
        resnet = models.resnet50(pretrained=True)
        
        # Loại bỏ FC layer cuối (classifier)
        # Chỉ giữ feature extractor
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_extractor.eval()  # Set to evaluation mode
        self.feature_extractor.to(self.device)
        
        # Projection layer: 2048-dim (ResNet output) → 384-dim
        self.projection = nn.Linear(2048, output_dim)
        self.projection.to(self.device)
        
        # Image preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet statistics
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def preprocess_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        Load và preprocess 1 ảnh
        
        Args:
            image_path: Đường dẫn đến file ảnh
            
        Returns:
            tensor: Shape (1, 3, 224, 224)
        """
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        return tensor
    
    @torch.no_grad()
    def encode_single(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Encode 1 ảnh thành embedding
        
        Args:
            image_path: Đường dẫn ảnh
            
        Returns:
            embedding: numpy array shape (384,)
        """
        tensor = self.preprocess_image(image_path).to(self.device)
        
        # Extract features từ ResNet
        features = self.feature_extractor(tensor)  # Shape: (1, 2048, 1, 1)
        features = features.flatten(1)  # Shape: (1, 2048)
        
        # Project về output_dim
        embedding = self.projection(features)  # Shape: (1, 384)
        
        return embedding.cpu().numpy()[0]
    
    @torch.no_grad()
    def encode_batch(self, image_paths: List[Union[str, Path]], 
                     batch_size: int = 32) -> np.ndarray:
        """
        Encode nhiều ảnh cùng lúc (batch processing)
        
        Args:
            image_paths: List đường dẫn ảnh
            batch_size: Số ảnh xử lý cùng lúc
            
        Returns:
            embeddings: numpy array shape (num_images, 384)
        """
        embeddings = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            
            # Load batch images
            batch_tensors = [
                self.preprocess_image(path) 
                for path in batch_paths
            ]
            batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)
            
            # Extract features
            features = self.feature_extractor(batch_tensor)
            features = features.flatten(1)
            
            # Project
            batch_embeddings = self.projection(features)
            embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """Lưu embeddings"""
        np.save(filepath, embeddings)
        
    def load_embeddings(self, filepath: str) -> np.ndarray:
        """Load embeddings"""
        return np.load(filepath)


# Example usage
if __name__ == "__main__":
    processor = ImageProcessor()
    
    print(f"Device: {processor.device}")
    print(f"Output dimension: {processor.output_dim}")
    
    # Test với sample image (tạo dummy image)
    dummy_image = Image.new('RGB', (224, 224), color='red')
    dummy_image.save('test_image.jpg')
    
    embedding = processor.encode_single('test_image.jpg')
    print(f"Image embedding shape: {embedding.shape}")
    print(f"Sample embedding (first 5 dims): {embedding[:5]}")