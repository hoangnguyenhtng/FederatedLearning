"""
Text Processor: Xử lý dữ liệu text (product descriptions, reviews, titles)
Sử dụng sentence-transformers để tạo embeddings
"""
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np

class TextProcessor:
    """
    Xử lý text data và tạo embeddings
    
    Input: Raw text (product titles, descriptions)
    Output: Text embeddings (384-dim vectors)
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Args:
            model_name: Pre-trained model từ sentence-transformers
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Chuyển text thành embeddings
        
        Args:
            texts: List các đoạn text
            batch_size: Số samples xử lý cùng lúc
            
        Returns:
            embeddings: numpy array shape (num_texts, embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def process_dataset(self, data: Dict) -> Dict:
        """
        Xử lý toàn bộ dataset
        
        Args:
            data: Dict chứa 'item_ids', 'titles', 'descriptions'
            
        Returns:
            processed_data: Dict với thêm 'text_embeddings'
        """
        # Kết hợp title và description
        combined_texts = [
            f"{title}. {desc}" 
            for title, desc in zip(data['titles'], data['descriptions'])
        ]
        
        # Tạo embeddings
        embeddings = self.encode(combined_texts)
        
        data['text_embeddings'] = embeddings
        return data
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """Lưu embeddings ra file"""
        np.save(filepath, embeddings)
        
    def load_embeddings(self, filepath: str) -> np.ndarray:
        """Load embeddings từ file"""
        return np.load(filepath)


# Example usage
if __name__ == "__main__":
    processor = TextProcessor()
    
    # Test với sample data
    sample_texts = [
        "Wireless Bluetooth Headphones with Noise Cancellation",
        "Comfortable Running Shoes for Marathon Training",
        "Premium Coffee Beans - Dark Roast"
    ]
    
    embeddings = processor.encode(sample_texts)
    print(f"Text embeddings shape: {embeddings.shape}")
    print(f"Sample embedding (first 5 dims): {embeddings[0][:5]}")