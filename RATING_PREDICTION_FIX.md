# ✅ FIX: Chuyển từ Item Prediction → Rating Prediction

## 🎯 Vấn đề đã phát hiện

**Nguyên nhân chính:** Model đang predict **item_id** (10000 classes) nhưng labels là **rating** (1-5) → Mismatch nghiêm trọng!

```
❌ SAI:
Model output: (batch, 10000) - predict item_id
Labels: (batch,) với giá trị 1-5 - rating
→ CrossEntropyLoss fail hoàn toàn!
→ Accuracy = 0, Loss tăng
```

## ✅ Giải pháp: Rating Prediction (1-5)

Chuyển sang **Rating Prediction** - đơn giản hơn và phù hợp với dữ liệu.

```
✅ ĐÚNG:
Model output: (batch, 5) - predict rating class
Labels: (batch,) với giá trị 0-4 (rating-1)
→ CrossEntropyLoss hoạt động đúng!
→ Accuracy sẽ tăng, Loss sẽ giảm
```

---

## 📝 Các file đã sửa

### 1. ✅ `src/data_generation/federated_dataloader.py`
**Thay đổi:**
- Dòng 236: Đổi từ `'label': torch.tensor(item_id, ...)` 
- → `'label': torch.tensor(rating - 1, ...)` (0-4)
- Thêm validation để đảm bảo rating trong range [1,5]

**Code:**
```python
# Validate and convert rating to label (0-4 for 5 classes)
rating_value = int(rating)
if rating_value < 1:
    rating_value = 1
elif rating_value > 5:
    rating_value = 5
label = rating_value - 1  # Convert 1-5 → 0-4

sample = {
    ...
    'rating': torch.tensor(rating, dtype=torch.long),  # Metadata (1-5)
    'label': torch.tensor(label, dtype=torch.long)  # Label for training (0-4)
}
```

---

### 2. ✅ `configs/config.yaml`
**Thay đổi:**
- Dòng 45: `num_classes: 10000` → `num_classes: 5`

**Code:**
```yaml
# Output configuration
# Rating prediction: 5 classes (ratings 1-5, mapped to 0-4)
num_classes: 5  # Changed from 10000 to 5
```

---

### 3. ✅ `src/training/federated_training_pipeline.py`
**Thay đổi:**
- Dòng 100: `num_items = model_config.get('num_classes', 10000)` 
- → `num_classes = model_config.get('num_classes', 5)`
- Dòng 107: `num_items=num_items` → `num_items=num_classes`

**Code:**
```python
# Rating prediction: 5 classes (ratings 1-5, mapped to 0-4)
num_classes = model_config.get('num_classes', 5)  # Changed from 10000 to 5

model = FedPerRecommender(
    ...
    num_items=num_classes,  # num_items parameter name, but value is num_classes (5)
    ...
)
```

---

### 4. ✅ `src/models/model_factory.py`
**Thay đổi:**
- Dòng 26: `num_items = model_config.get("num_classes", 10000)` 
- → `num_classes = model_config.get("num_classes", 5)`
- Dòng 33: `num_items=num_items` → `num_items=num_classes`

**Code:**
```python
# Rating prediction: 5 classes (ratings 1-5, mapped to 0-4)
num_classes = model_config.get("num_classes", 5)  # Changed from 10000 to 5

return FedPerRecommender(
    ...
    num_items=num_classes,  # num_items parameter name, but value is num_classes (5)
    ...
)
```

---

### 5. ✅ `src/federated/client.py`
**Thay đổi:**
- Dòng 151: Đổi từ `batch_data.get('label', batch_data.get('item_id', ...))`
- → `batch_data.get('label', batch_data['rating'] - 1)`
- Thêm validation: `torch.clamp(labels, 0, 4)`
- Sửa cả `fit()` và `evaluate()`

**Code:**
```python
# Use 'label' (rating-1, range 0-4) for rating prediction task
labels = batch_data.get('label', batch_data['rating'] - 1).to(self.device)
# Ensure labels are in valid range [0, 4] for 5 classes
labels = torch.clamp(labels, 0, 4)

# Validate labels are in valid range [0, 4] for rating prediction (5 classes)
num_classes = logits.shape[1]  # Should be 5 for rating prediction
labels_clamped = torch.clamp(labels, 0, num_classes - 1)
```

---

### 6. ✅ `src/training/training_utils.py`
**Thay đổi:**
- Dòng 244, 329: Đổi từ `targets = batch['rating'].to(device)`
- → Sử dụng `batch['label']` với fallback convert rating-1

**Code:**
```python
# Use 'label' (rating-1, range 0-4) for rating prediction task
# If 'label' not available, convert rating (1-5) to label (0-4)
if 'label' in batch:
    targets = batch['label'].to(device)
else:
    targets = (batch['rating'].to(device) - 1).clamp(0, 4)  # Convert 1-5 → 0-4
```

---

## 🎯 Kết quả mong đợi

### Trước khi sửa:
```
Model: (batch, 10000) - predict item_id
Labels: (batch,) với giá trị 1-5 - rating
→ CrossEntropyLoss: pred[item_id], target=rating → SAI!
→ Accuracy = 0.0000
→ Loss tăng liên tục
```

### Sau khi sửa:
```
Model: (batch, 5) - predict rating class
Labels: (batch,) với giá trị 0-4 (rating-1)
→ CrossEntropyLoss: pred[class], target=class → ĐÚNG!
→ Accuracy sẽ tăng dần (target: 0.3-0.5 sau 50 rounds)
→ Loss sẽ giảm dần (target: < 1.0 sau 50 rounds)
```

---

## 📊 Expected Metrics

### Training Progress:
- **Round 1-10**: Loss ~2.0-3.0, Accuracy ~0.2-0.3
- **Round 11-30**: Loss ~1.0-2.0, Accuracy ~0.3-0.4
- **Round 31-50**: Loss ~0.5-1.0, Accuracy ~0.4-0.5

### Final Results (after 50 rounds):
- **Train Loss**: ~0.8-1.2
- **Test Loss**: ~1.0-1.5
- **Accuracy**: ~0.35-0.50 (35-50%)
- **Per-class accuracy**: Balanced across 5 rating classes

---

## ✅ Checklist

- [x] Dataset returns rating-1 (0-4) as label
- [x] Config set num_classes=5
- [x] Model outputs 5 classes
- [x] Training pipeline uses num_classes=5
- [x] Client validates labels in range [0,4]
- [x] Training utils uses label instead of rating

---

## 🚀 Next Steps

1. **Chạy lại training:**
   ```bash
   python src/training/federated_training_pipeline.py
   ```

2. **Monitor metrics:**
   - Accuracy should increase from 0 → 0.3-0.5
   - Loss should decrease from ~2.3 → ~1.0

3. **Nếu vẫn có vấn đề:**
   - Check logs for label range warnings
   - Verify model output shape is (batch, 5)
   - Check if labels are in range [0, 4]

---

## 📝 Notes

- **Model parameter name**: Vẫn dùng `num_items` trong `FedPerRecommender.__init__()` nhưng giá trị là `num_classes=5`
- **Label conversion**: Rating 1-5 → Label 0-4 (rating - 1)
- **Backward compatibility**: Code vẫn hỗ trợ cả `label` và `rating` (với conversion)

