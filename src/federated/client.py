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
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import numpy as np
from collections import OrderedDict
import math


class FocalLoss(nn.Module):
    """Focal Loss for extreme class imbalance (62.7% majority class).
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    gamma=2.0: Down-weights easy (majority) examples that model already classifies well.
    alpha: Per-class weights from inverse frequency.
    """
    
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = alpha.float()
        else:
            self.alpha = None
    
    def forward(self, inputs, targets):
        # Apply label smoothing
        num_classes = inputs.shape[1]
        
        # Compute log softmax for numerical stability
        log_p = torch.nn.functional.log_softmax(inputs, dim=1)
        p = torch.exp(log_p)
        
        # Gather the log probabilities for the target classes
        log_p_target = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_target = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal modulation: (1 - p_t)^gamma
        focal_weight = (1 - p_target) ** self.gamma
        
        # Alpha weighting
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_target = alpha.gather(0, targets)
            focal_weight = focal_weight * alpha_target
        
        # Label smoothing: blend between hard target and uniform
        if self.label_smoothing > 0:
            smooth_loss = -log_p.mean(dim=1)  # uniform part
            hard_loss = -log_p_target          # hard target part
            loss = (1 - self.label_smoothing) * hard_loss + self.label_smoothing * smooth_loss
            loss = focal_weight * loss
        else:
            loss = -focal_weight * log_p_target
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


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
                 apply_dp: bool = False,
                 weight_decay: float = 1e-4,
                 num_rounds: int = 100,
                 state_path: Optional[Union[str, Path]] = None,
                 personalize_epochs_eval: int = 2,
                 loss_type: str = "weighted_ce",
                 gradient_clip: float = 2.0,
                 fedprox_mu: float = 0.0):
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
        def _f(v, default):
            if isinstance(v, str):
                return float(v.strip())
            return float(v if v is not None else default)

        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.gradient_clip = _f(gradient_clip, 2.0)
        self.local_epochs = int(local_epochs)
        self.learning_rate = _f(learning_rate, 1e-3)
        self.apply_dp = apply_dp
        self.weight_decay = _f(weight_decay, 1e-4)
        self.num_rounds = int(num_rounds)
        self.personalize_epochs_eval = int(personalize_epochs_eval)
        self.state_path = Path(state_path) if state_path else None
        learning_rate = self.learning_rate
        weight_decay = self.weight_decay
        self._round = 0  # Track current round for scheduler
        self.fedprox_mu = _f(fedprox_mu, 0.0)
        
        # Optimizer chỉ cho SHARED parameters
        self.optimizer_shared = optim.Adam(
            self.model.shared_base.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay  # ✅ L2 regularization
        )
        
        # Optimizer riêng cho PERSONAL parameters (higher LR for faster adaptation)
        self.optimizer_personal = optim.Adam(
            self.model.personal_head.parameters(),
            lr=learning_rate * 2.0,  # ✅ Personal head needs higher LR since it resets each round
            weight_decay=weight_decay
        )
        
        # Optimizer cho multimodal encoder
        self.optimizer_multimodal = optim.Adam(
            self.model.multimodal_encoder.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # ✅ FIX: NO class weights in loss — WeightedRandomSampler already balances batches.
        # Using both sampler + weighted loss causes double compensation → overfits minority class.
        loss_type = str(loss_type).lower()
        if loss_type == "focal":
            self.criterion = FocalLoss(
                alpha=None,              # No class weights (sampler handles balance)
                gamma=2.0,               # Focus on hard examples
                label_smoothing=0.02,
            )
        else:
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=0.02,    # No weight= param (sampler handles balance)
            )
        print(f"[Client {self.client_id}] Loss: {loss_type} (no class weights — sampler balances batches)")

        if self.state_path and self.state_path.exists():
            self.load_personal_state(self.state_path)

    def _state_path_for(self, path: Optional[Union[str, Path]] = None) -> Optional[Path]:
        target = Path(path) if path is not None else self.state_path
        return target

    def save_personal_state(self, path: Optional[Union[str, Path]] = None) -> None:
        """Persist personal head (+ local adapters) across Flower client_fn calls."""
        target = self._state_path_for(path)
        if target is None:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        personal = {
            name: param.detach().cpu()
            for name, param in self.model.get_personal_parameters().items()
        }
        payload = {
            "personal": personal,
            "round": self._round,
            "optimizer_personal": self.optimizer_personal.state_dict(),
        }
        if hasattr(self, "_img_proj"):
            payload["img_proj"] = self._img_proj.state_dict()
        torch.save(payload, target)

    def load_personal_state(self, path: Optional[Union[str, Path]] = None) -> bool:
        """Warm-start personal head from disk (FedPer state must survive between rounds)."""
        target = self._state_path_for(path)
        if target is None or not target.exists():
            return False
        try:
            payload = torch.load(target, map_location=self.device, weights_only=False)
        except TypeError:
            payload = torch.load(target, map_location=self.device)

        personal = payload.get("personal") or {}
        current = self.model.state_dict()
        for name, param in personal.items():
            if name in current:
                current[name] = param.to(self.device)
        self.model.load_state_dict(current, strict=False)

        opt_state = payload.get("optimizer_personal")
        if opt_state is not None:
            try:
                self.optimizer_personal.load_state_dict(opt_state)
            except Exception:
                pass

        img_proj_state = payload.get("img_proj")
        if img_proj_state is not None:
            if not hasattr(self, "_img_proj"):
                self._img_proj = torch.nn.Linear(512, 2048).to(self.device)
            self._img_proj.load_state_dict(img_proj_state)

        self._round = int(payload.get("round", self._round))
        return True

    def _parse_batch(self, batch_data):
        """Shared batch parsing for fit / evaluate / personalize."""
        if isinstance(batch_data, dict):
            if "image_embedding" in batch_data:
                image_emb = batch_data["image_embedding"].to(self.device)
            elif "image_features" in batch_data:
                image_emb = batch_data["image_features"].to(self.device)
            else:
                batch_size = batch_data.get("label", batch_data.get("rating", torch.tensor([1]))).shape[0]
                image_emb = torch.randn(batch_size, 512, device=self.device)

            behavior_feat = batch_data["behavior_features"].to(self.device)
            labels = batch_data.get("label", batch_data["rating"] - 1).to(self.device)
            labels = torch.clamp(labels, 0, 2)  # 3-class sentiment

            batch_size = behavior_feat.shape[0]
            if len(behavior_feat.shape) == 1:
                behavior_feat = behavior_feat.unsqueeze(0) if batch_size == 1 else behavior_feat.view(batch_size, -1)
            if behavior_feat.shape[1] != 32:
                if behavior_feat.shape[1] < 32:
                    padding = torch.zeros(batch_size, 32 - behavior_feat.shape[1], device=self.device)
                    behavior_feat = torch.cat([behavior_feat, padding], dim=1)
                else:
                    behavior_feat = behavior_feat[:, :32]

            batch_size = image_emb.shape[0]
            if len(image_emb.shape) == 1:
                image_emb = image_emb.unsqueeze(0) if batch_size == 1 else image_emb.view(batch_size, -1)
            elif image_emb.shape[1] == 0:
                image_emb = torch.randn(batch_size, 512, device=self.device)

            if "text_embedding" in batch_data:
                text_emb = batch_data["text_embedding"].to(self.device)
            else:
                text_emb = torch.randn(batch_size, 384, device=self.device)

            if image_emb.shape[1] != 2048:
                if image_emb.shape[1] == 512:
                    if not hasattr(self, "_img_proj"):
                        self._img_proj = torch.nn.Linear(512, 2048).to(self.device)
                    image_emb = self._img_proj(image_emb)
                else:
                    image_emb = torch.randn(batch_size, 2048, device=self.device)
        else:
            text_emb = batch_data[0].to(self.device)
            image_emb = batch_data[1].to(self.device)
            behavior_feat = batch_data[2].to(self.device)
            labels = torch.clamp(batch_data[3].to(self.device), 0, 2)  # 3-class sentiment

        return text_emb, image_emb, behavior_feat, labels

    def _personalize_head(self, epochs: int) -> None:
        """Adapt personal head to latest shared weights (standard FedPer eval protocol)."""
        if epochs <= 0:
            return

        shared_names = set(self.model.get_shared_parameters().keys())
        for name, param in self.model.named_parameters():
            param.requires_grad = name not in shared_names

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_data in self.train_loader:
                text_emb, image_emb, behavior_feat, labels = self._parse_batch(batch_data)
                self.optimizer_personal.zero_grad()
                logits = self.model(text_emb, image_emb, behavior_feat)
                labels_clamped = torch.clamp(labels, 0, logits.shape[1] - 1)
                loss = self.criterion(logits, labels_clamped)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.personal_head.parameters(), max_norm=self.gradient_clip)
                self.optimizer_personal.step()
                epoch_loss += loss.item()
            print(
                f"[Client {self.client_id}] Personalize epoch {epoch + 1}/{epochs}, "
                f"Loss: {epoch_loss / max(len(self.train_loader), 1):.4f}"
            )

        for param in self.model.parameters():
            param.requires_grad = True

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
            # ✅ FIX: Move tensor to correct device (critical for GPU training)
            params_dict[name] = torch.tensor(param_array, dtype=torch.float32).to(self.device)
        
        self.model.set_shared_parameters(params_dict)
        
        print(f"[Client {self.client_id}] Received and updated shared parameters from server")
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:

        # Step 1: Update shared parameters từ server
        self.set_parameters(parameters)

        # ✅ FIX: KHÔNG tạo lại optimizer mỗi round!
        # Tạo lại optimizer mỗi round sẽ reset Adam's running mean/variance (m, v)
        # → Model học như SGD không có warm-up → accuracy stuck tại random baseline
        # Optimizers đã được init trong __init__ và được giữ nguyên qua các rounds

        # ✅ Warmup + Cosine LR Decay
        # Phase 1 (rounds 1-10): Linear warmup from 10% to 100% LR
        # Phase 2 (rounds 11+): Cosine decay from 100% to 10% LR
        server_round = config.get('server_round', 1)
        num_rounds = self.num_rounds
        warmup_rounds = 10
        
        if server_round <= warmup_rounds:
            # Linear warmup
            warmup_factor = server_round / warmup_rounds
            current_lr = self.learning_rate * warmup_factor
        else:
            # Cosine decay after warmup
            progress = (server_round - warmup_rounds) / max(1, num_rounds - warmup_rounds)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            current_lr = self.learning_rate * 0.1 + (self.learning_rate - self.learning_rate * 0.1) * cosine_factor
        
        for opt in [self.optimizer_shared, self.optimizer_multimodal]:
            for param_group in opt.param_groups:
                param_group['lr'] = current_lr
        # Personal head gets 2x LR for faster adaptation (resets each round)
        for param_group in self.optimizer_personal.param_groups:
            param_group['lr'] = current_lr * 2.0
        print(f"[Client {self.client_id}] Round {server_round}/{num_rounds} | LR: {current_lr:.6f} (personal: {current_lr*2:.6f})")

        # Step 2: Local training
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # ✅ FedProx: save global params snapshot for proximal term
        global_params_snapshot = None
        if self.fedprox_mu > 0:
            global_params_snapshot = {
                name: param.detach().clone()
                for name, param in self.model.get_shared_parameters().items()
            }
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            
            for batch_idx, batch_data in enumerate(self.train_loader):
                text_emb, image_emb, behavior_feat, labels = self._parse_batch(batch_data)
                
                # Zero gradients
                self.optimizer_shared.zero_grad()
                self.optimizer_personal.zero_grad()
                self.optimizer_multimodal.zero_grad()
                
                # Forward pass
                logits = self.model(text_emb, image_emb, behavior_feat)
                
                # Validate labels are in valid range for 3-class sentiment
                num_classes = logits.shape[1]  # 3 classes: Negative/Neutral/Positive
                labels_clamped = torch.clamp(labels, 0, num_classes - 1)
                if batch_idx == 0 and (labels != labels_clamped).any():
                    # Log warning if labels out of range (only first batch to avoid spam)
                    out_of_range = (labels < 0) | (labels >= num_classes)
                    if out_of_range.any():
                        print(f"⚠️  [Client {self.client_id}] Training Warning: {out_of_range.sum().item()} labels out of range [0, {num_classes-1}]")
                        print(f"   Min label: {labels.min().item()}, Max label: {labels.max().item()}, Expected: [0, 2]")
                
                loss = self.criterion(logits, labels_clamped)
                
                # ✅ FedProx: add proximal term μ/2 * ||w - w_global||²
                if global_params_snapshot is not None:
                    prox_term = 0.0
                    for name, param in self.model.get_shared_parameters().items():
                        if name in global_params_snapshot:
                            prox_term += ((param - global_params_snapshot[name]) ** 2).sum()
                    loss = loss + (self.fedprox_mu / 2.0) * prox_term
                
                # ✅ FIX: Check for NaN/Inf BEFORE backward (prevent gradient corruption)
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️  [Client {self.client_id}] NaN/Inf loss detected! Skipping batch")
                    self.optimizer_shared.zero_grad()
                    self.optimizer_personal.zero_grad()
                    self.optimizer_multimodal.zero_grad()
                    continue
                
                # Backward pass (safe — loss is finite)
                loss.backward()
                
                # Gradient clipping to prevent explosion/NaN (relaxed for imbalanced data)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)
                
                # Update all parameters (shared + personal)
                self.optimizer_shared.step()
                self.optimizer_personal.step()
                self.optimizer_multimodal.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            total_loss += avg_epoch_loss
            print(f"[Client {self.client_id}] Epoch {epoch+1}/{self.local_epochs}, Loss: {avg_epoch_loss:.4f}")
        
        avg_loss = total_loss / self.local_epochs
        self._round += 1
        self.save_personal_state()
        
        # Step 3: Get updated shared parameters
        updated_parameters = self.get_parameters(config={})
        
        # Metrics to send to server
        metrics = {
            "train_loss": avg_loss,
            "client_id": self.client_id,
            "num_batches": num_batches,
            "learning_rate": current_lr,  # ✅ Monitor LR decay
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

        personalize_epochs = int(
            config.get("personalize_epochs", self.personalize_epochs_eval)
        )
        self._personalize_head(personalize_epochs)
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data in self.test_loader:
                text_emb, image_emb, behavior_feat, labels = self._parse_batch(batch_data)
                
                # Forward pass
                logits = self.model(text_emb, image_emb, behavior_feat)
                
                # Validate labels are in valid range for 3-class sentiment
                num_classes = logits.shape[1]  # 3 classes: Negative/Neutral/Positive
                labels_clamped = torch.clamp(labels, 0, num_classes - 1)
                
                loss = self.criterion(logits, labels_clamped)
                
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(logits, 1)
                total += labels_clamped.size(0)
                correct += (predicted == labels_clamped).sum().item()
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = correct / total
        
        metrics = {
            "test_loss": float(avg_loss),
            "accuracy": float(accuracy),
            "client_id": self.client_id,
        }
        
        num_examples = len(self.test_loader.dataset)
        
        print(f"[Client {self.client_id}] Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        self.save_personal_state()
        
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
        apply_dp=config.get('apply_dp', False),
        gradient_clip=config.get('gradient_clip', 2.0)
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
        num_classes=5  # Rating prediction: 5 classes (1-5 → 0-4)
    )
    
    # Create dummy data
    num_samples = 100
    train_data = {
        'text_embeddings': torch.randn(num_samples, 384),
        'image_embeddings': torch.randn(num_samples, 2048),  # ✅ ResNet-50 output dim
        'behavior_features': torch.randn(num_samples, 32),   # ✅ 32-dim behavior
        'labels': torch.randint(0, 5, (num_samples,))        # ✅ Rating 0-4
    }
    
    test_data = {
        'text_embeddings': torch.randn(20, 384),
        'image_embeddings': torch.randn(20, 2048),
        'behavior_features': torch.randn(20, 32),
        'labels': torch.randint(0, 5, (20,))
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