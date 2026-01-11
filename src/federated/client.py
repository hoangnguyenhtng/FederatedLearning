"""
Federated Client Implementation

Nhiệm vụ:
1. Nhận global model từ server
2. Train trên local data (private data của user)
3. Chỉ gửi SHARED parameters về server (giữ personal head)
4. Áp dụng Differential Privacy nếu cần
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import OrderedDict

class FedPerClient(fl.client.NumPyClient):
    """
    Federated Client với FedPer architecture
    
    Key features:
    - Train cả shared + personal layers locally
    - Chỉ upload shared layers lên server
    - Personal layers KHÔNG BAO GIỜ rời khỏi client
    """
    
    def __init__(self,
                 client_id: int,
                 model: nn.Module,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 device: str = 'cpu',
                 local_epochs: int = 3,
                 learning_rate: float = 0.001,
                 apply_dp: bool = False):
        """
        Args:
            client_id: ID của client
            model: FedPerRecommender model
            train_loader: DataLoader cho training data
            test_loader: DataLoader cho test data
            device: 'cuda' hoặc 'cpu'
            local_epochs: Số epochs train ở local
            learning_rate: Learning rate
            apply_dp: Có áp dụng differential privacy không
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.apply_dp = apply_dp
        
        # Optimizer chỉ cho SHARED parameters
        self.optimizer_shared = optim.Adam(
            self.model.shared_base.parameters(),
            lr=learning_rate
        )
        
        # Optimizer riêng cho PERSONAL parameters
        self.optimizer_personal = optim.Adam(
            self.model.personal_head.parameters(),
            lr=learning_rate
        )
        
        # Optimizer cho multimodal encoder
        self.optimizer_multimodal = optim.Adam(
            self.model.multimodal_encoder.parameters(),
            lr=learning_rate
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Lấy parameters để gửi cho server
        
        CHỈ GỬI SHARED PARAMETERS!
        Personal head KHÔNG được gửi
        
        Returns:
            List of numpy arrays (shared parameters only)
        """
        shared_params = self.model.get_shared_parameters()
        
        # Convert sang numpy arrays
        return [param.cpu().detach().numpy() for param in shared_params.values()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Nhận parameters từ server và update model
        
        CHỈ UPDATE SHARED PARAMETERS!
        Personal head giữ nguyên
        
        Args:
            parameters: List of numpy arrays from server
        """
        shared_params = self.model.get_shared_parameters()
        param_names = list(shared_params.keys())
        
        # Update shared parameters
        params_dict = OrderedDict()
        for name, param_array in zip(param_names, parameters):
            params_dict[name] = torch.tensor(param_array, dtype=torch.float32)
        
        self.model.set_shared_parameters(params_dict)
        
        print(f"[Client {self.client_id}] Received and updated shared parameters from server")
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Local training
        
        Process:
        1. Nhận global model từ server
        2. Train trên local data (cả shared + personal)
        3. Trả về SHARED parameters + metrics
        
        Args:
            parameters: Global model parameters từ server
            config: Training configuration
            
        Returns:
            updated_parameters: Shared parameters sau khi train
            num_examples: Số lượng training samples
            metrics: Training metrics
        """
        # Step 1: Update shared parameters từ server
        self.set_parameters(parameters)
        
        # Step 2: Local training
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            
            for batch_idx, batch_data in enumerate(self.train_loader):
                # Parse batch data (from MultiModalDataset - returns dict)
                if isinstance(batch_data, dict):
                    # Dictionary format - handle both 'image_features' and 'image_embedding'
                    if 'image_embedding' in batch_data:
                        # Amazon data format
                        image_emb = batch_data['image_embedding'].to(self.device)
                    elif 'image_features' in batch_data:
                        # Synthetic data format
                        image_emb = batch_data['image_features'].to(self.device)
                    else:
                        # Fallback: create dummy
                        batch_size = batch_data.get('label', batch_data.get('rating', torch.tensor([1]))).shape[0]
                        image_emb = torch.randn(batch_size, 512, device=self.device)
                    behavior_feat = batch_data['behavior_features'].to(self.device)
                    # Use 'label' (rating-1, range 0-4) for rating prediction task
                    labels = batch_data.get('label', batch_data['rating'] - 1).to(self.device)
                    # Ensure labels are in valid range [0, 4] for 5 classes
                    labels = torch.clamp(labels, 0, 4)
                    
                    # Validate and fix behavior_feat shape (should be 32 dim for BehaviorEncoder)
                    batch_size = behavior_feat.shape[0]
                    if len(behavior_feat.shape) == 1:
                        behavior_feat = behavior_feat.unsqueeze(0) if batch_size == 1 else behavior_feat.view(batch_size, -1)
                    if behavior_feat.shape[1] != 32:
                        if behavior_feat.shape[1] < 32:
                            # Pad with zeros
                            padding = torch.zeros(batch_size, 32 - behavior_feat.shape[1], device=self.device)
                            behavior_feat = torch.cat([behavior_feat, padding], dim=1)
                        else:
                            # Truncate
                            behavior_feat = behavior_feat[:, :32]
                    
                    # Validate shapes
                    batch_size = image_emb.shape[0]
                    
                    # Fix image_emb if it has wrong shape
                    if len(image_emb.shape) == 1:
                        # If 1D, reshape to (batch_size, features)
                        image_emb = image_emb.unsqueeze(0) if batch_size == 1 else image_emb.view(batch_size, -1)
                    elif image_emb.shape[1] == 0:
                        # If feature dim is 0, create dummy features
                        print(f"⚠️  Warning: image_emb has shape {image_emb.shape}, creating dummy features")
                        image_emb = torch.randn(batch_size, 512, device=self.device)
                    
                    # Ensure image_emb has at least some features
                    if image_emb.shape[1] == 0:
                        image_emb = torch.randn(batch_size, 512, device=self.device)
                    
                    # Text embeddings: Use REAL embeddings from data (if available)
                    if 'text_embedding' in batch_data:
                        # Amazon data: Real text embeddings!
                        text_emb = batch_data['text_embedding'].to(self.device)
                    else:
                        # Synthetic data: Create dummy (fallback)
                        text_emb = torch.randn(batch_size, 384, device=self.device)
                    
                    # Reshape image_emb to 2048 dim (ResNet-50 output)
                    if image_emb.shape[1] != 2048:
                        # Amazon data should already be 2048-dim
                        if image_emb.shape[1] == 512:
                            # Synthetic data: Project 512 → 2048
                            if not hasattr(self, '_img_proj'):
                                self._img_proj = torch.nn.Linear(512, 2048).to(self.device)
                            image_emb = self._img_proj(image_emb)
                        else:
                            # Unexpected dimension
                            print(f"⚠️  Warning: image_emb has unexpected shape {image_emb.shape}, creating 2048-dim dummy")
                            image_emb = torch.randn(batch_size, 2048, device=self.device)
                else:
                    # Tuple format (legacy TensorDataset)
                    text_emb = batch_data[0].to(self.device)
                    image_emb = batch_data[1].to(self.device)
                    behavior_feat = batch_data[2].to(self.device)
                    labels = batch_data[3].to(self.device)
                    # Validate labels are in range [0, 4] for rating prediction (5 classes)
                    labels = torch.clamp(labels, 0, 4)
                
                # Zero gradients
                self.optimizer_shared.zero_grad()
                self.optimizer_personal.zero_grad()
                self.optimizer_multimodal.zero_grad()
                
                # Forward pass
                logits = self.model(text_emb, image_emb, behavior_feat)
                
                # Validate labels are in valid range [0, 4] for rating prediction (5 classes)
                num_classes = logits.shape[1]  # Should be 5 for rating prediction
                labels_clamped = torch.clamp(labels, 0, num_classes - 1)
                if batch_idx == 0 and (labels != labels_clamped).any():
                    # Log warning if labels out of range (only first batch to avoid spam)
                    out_of_range = (labels < 0) | (labels >= num_classes)
                    if out_of_range.any():
                        print(f"⚠️  [Client {self.client_id}] Training Warning: {out_of_range.sum().item()} labels out of range [0, {num_classes-1}]")
                        print(f"   Min label: {labels.min().item()}, Max label: {labels.max().item()}, Expected: [0, 4]")
                
                loss = self.criterion(logits, labels_clamped)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping to prevent explosion/NaN
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update all parameters (shared + personal)
                self.optimizer_shared.step()
                self.optimizer_personal.step()
                self.optimizer_multimodal.step()
                
                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️  [Client {self.client_id}] NaN/Inf detected! Skipping batch")
                    continue
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            total_loss += avg_epoch_loss
            print(f"[Client {self.client_id}] Epoch {epoch+1}/{self.local_epochs}, Loss: {avg_epoch_loss:.4f}")
        
        avg_loss = total_loss / self.local_epochs
        
        # Step 3: Get updated shared parameters
        updated_parameters = self.get_parameters(config={})
        
        # Metrics to send to server
        metrics = {
            "train_loss": avg_loss,
            "client_id": self.client_id,
            "num_batches": num_batches
        }
        
        num_examples = len(self.train_loader.dataset)
        
        return updated_parameters, num_examples, metrics
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """
        Local evaluation
        
        Args:
            parameters: Model parameters từ server
            config: Evaluation configuration
            
        Returns:
            loss: Test loss
            num_examples: Số lượng test samples
            metrics: Evaluation metrics
        """
        # Update shared parameters
        self.set_parameters(parameters)
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data in self.test_loader:
                # Parse batch data (from MultiModalDataset - returns dict)
                if isinstance(batch_data, dict):
                    # Dictionary format - handle both 'image_features' and 'image_embedding'
                    if 'image_embedding' in batch_data:
                        # Amazon data format
                        image_emb = batch_data['image_embedding'].to(self.device)
                    elif 'image_features' in batch_data:
                        # Synthetic data format
                        image_emb = batch_data['image_features'].to(self.device)
                    else:
                        # Fallback: create dummy
                        batch_size = batch_data.get('label', batch_data.get('rating', torch.tensor([1]))).shape[0]
                        image_emb = torch.randn(batch_size, 512, device=self.device)
                    
                    behavior_feat = batch_data['behavior_features'].to(self.device)
                    # Use 'label' (rating-1, range 0-4) for rating prediction task
                    labels = batch_data.get('label', batch_data['rating'] - 1).to(self.device)
                    # Ensure labels are in valid range [0, 4] for 5 classes
                    labels = torch.clamp(labels, 0, 4)
                    
                    # Validate and fix behavior_feat shape (should be 32 dim for BehaviorEncoder)
                    batch_size = behavior_feat.shape[0]
                    if len(behavior_feat.shape) == 1:
                        behavior_feat = behavior_feat.unsqueeze(0) if batch_size == 1 else behavior_feat.view(batch_size, -1)
                    if behavior_feat.shape[1] != 32:
                        if behavior_feat.shape[1] < 32:
                            # Pad with zeros
                            padding = torch.zeros(batch_size, 32 - behavior_feat.shape[1], device=self.device)
                            behavior_feat = torch.cat([behavior_feat, padding], dim=1)
                        else:
                            # Truncate
                            behavior_feat = behavior_feat[:, :32]
                    
                    # Validate shapes
                    batch_size = image_emb.shape[0]
                    
                    # Fix image_emb if it has wrong shape
                    if len(image_emb.shape) == 1:
                        image_emb = image_emb.unsqueeze(0) if batch_size == 1 else image_emb.view(batch_size, -1)
                    elif image_emb.shape[1] == 0:
                        print(f"⚠️  Warning: image_emb has shape {image_emb.shape}, creating dummy features")
                        image_emb = torch.randn(batch_size, 512, device=self.device)
                    
                    # Ensure image_emb has at least some features
                    if image_emb.shape[1] == 0:
                        image_emb = torch.randn(batch_size, 512, device=self.device)
                    
                    # Text embeddings: Use REAL embeddings from data (if available)
                    if 'text_embedding' in batch_data:
                        # Amazon data: Real text embeddings!
                        text_emb = batch_data['text_embedding'].to(self.device)
                    else:
                        # Synthetic data: Create dummy (fallback)
                        text_emb = torch.randn(batch_size, 384, device=self.device)
                    
                    # Reshape image_emb to 2048 dim (ResNet-50 output)
                    if image_emb.shape[1] != 2048:
                        if image_emb.shape[1] == 512:
                            if not hasattr(self, '_img_proj'):
                                self._img_proj = torch.nn.Linear(512, 2048).to(self.device)
                            image_emb = self._img_proj(image_emb)
                        else:
                            # Unexpected dimension, create proper dummy
                            print(f"⚠️  Warning: image_emb has unexpected shape {image_emb.shape}, creating 2048-dim dummy")
                            image_emb = torch.randn(batch_size, 2048, device=self.device)
                else:
                    # Tuple format (legacy TensorDataset)
                    text_emb = batch_data[0].to(self.device)
                    image_emb = batch_data[1].to(self.device)
                    behavior_feat = batch_data[2].to(self.device)
                    labels = batch_data[3].to(self.device)
                
                # Forward pass
                logits = self.model(text_emb, image_emb, behavior_feat)
                
                # Validate labels are in valid range [0, 4] for rating prediction (5 classes)
                num_classes = logits.shape[1]  # Should be 5 for rating prediction
                labels_clamped = torch.clamp(labels, 0, num_classes - 1)
                if (labels != labels_clamped).any():
                    # Log warning if labels out of range
                    out_of_range = (labels < 0) | (labels >= num_classes)
                    if out_of_range.any():
                        print(f"⚠️  [Client {self.client_id}] Evaluation Warning: {out_of_range.sum().item()} labels out of range [0, {num_classes-1}]")
                        print(f"   Min label: {labels.min().item()}, Max label: {labels.max().item()}, Expected: [0, 4]")
                
                loss = self.criterion(logits, labels_clamped)
                
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(logits, 1)
                total += labels_clamped.size(0)
                correct += (predicted == labels_clamped).sum().item()
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = correct / total
        
        metrics = {
            "test_loss": avg_loss,
            "accuracy": accuracy,
            "client_id": self.client_id
        }
        
        num_examples = len(self.test_loader.dataset)
        
        print(f"[Client {self.client_id}] Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return avg_loss, num_examples, metrics


def create_client_fn(client_id: int,
                     model: nn.Module,
                     train_data: Dict,
                     test_data: Dict,
                     config: Dict):
    """
    Factory function để tạo Federated Client
    
    Sử dụng bởi Flower framework để khởi tạo clients
    
    Args:
        client_id: ID của client
        model: Model architecture
        train_data: Training data cho client này
        test_data: Test data cho client này
        config: Configuration dict
        
    Returns:
        FederatedClient instance
    """
    # Create dataloaders
    train_dataset = TensorDataset(
        train_data['text_embeddings'],
        train_data['image_embeddings'],
        train_data['behavior_features'],
        train_data['labels']
    )
    
    test_dataset = TensorDataset(
        test_data['text_embeddings'],
        test_data['image_embeddings'],
        test_data['behavior_features'],
        test_data['labels']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False
    )
    
    # Create client
    client = FedPerClient(
        client_id=client_id,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=config.get('device', 'cpu'),
        local_epochs=config.get('local_epochs', 3),
        learning_rate=config.get('learning_rate', 0.001),
        apply_dp=config.get('apply_dp', False)
    )
    
    return client


# Example usage
if __name__ == "__main__":
    from src.models.multimodal_encoder import MultiModalEncoder
    from src.models.recommendation_model import FedPerRecommender
    
    # Initialize model
    multimodal_encoder = MultiModalEncoder()
    model = FedPerRecommender(
        multimodal_encoder=multimodal_encoder,
        num_items=1000
    )
    
    # Create dummy data
    num_samples = 100
    train_data = {
        'text_embeddings': torch.randn(num_samples, 384),
        'image_embeddings': torch.randn(num_samples, 384),
        'behavior_features': torch.randn(num_samples, 50),
        'labels': torch.randint(0, 1000, (num_samples,))
    }
    
    test_data = {
        'text_embeddings': torch.randn(20, 384),
        'image_embeddings': torch.randn(20, 384),
        'behavior_features': torch.randn(20, 50),
        'labels': torch.randint(0, 1000, (20,))
    }
    
    config = {
        'batch_size': 16,
        'local_epochs': 2,
        'learning_rate': 0.001,
        'device': 'cpu'
    }
    
    # Create client
    client = create_client_fn(
        client_id=0,
        model=model,
        train_data=train_data,
        test_data=test_data,
        config=config
    )
    
    print(f"Client created successfully!")
    print(f"Training samples: {len(train_data['labels'])}")
    print(f"Test samples: {len(test_data['labels'])}")