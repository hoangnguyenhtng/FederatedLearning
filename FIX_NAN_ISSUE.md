# 🚨 NaN LOSS ISSUE - ROOT CAUSE & FIX

## 🐛 PROBLEM

```
Loss: nan
Accuracy: 0.04-0.10 (4-10%)
All rounds: Loss = nan
```

## 🔍 ROOT CAUSE

**NaN (Not a Number)** xảy ra do **Gradient Explosion**:

1. **Learning rate quá cao** (0.0001 có thể vẫn cao với data mới)
2. **Không có gradient clipping** → gradients explode → weights become NaN
3. **Data normalization issues** → embeddings có scale khác nhau

### Why This Happens:

```
Text embeddings: ~[-1, 1] (normalized by SentenceTransformer)
Image embeddings: ~[-10, 10] (dummy features, not normalized)
Behavior features: ~[0, 1] (normalized)

→ Mixed scales → Unstable gradients → NaN!
```

## ✅ FIXES APPLIED

### Fix 1: Gradient Clipping (CRITICAL)

```python
# In src/federated/client.py
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
```

**Effect**: Prevents gradients from exploding

### Fix 2: NaN Detection

```python
# Skip batches with NaN
if torch.isnan(loss) or torch.isinf(loss):
    print(f"⚠️  NaN detected! Skipping batch")
    continue
```

### Fix 3: Lower Learning Rate (RECOMMENDED)

```yaml
# configs/config.yaml
training:
  learning_rate: 0.00001  # 10x lower (was 0.0001)
```

## 🚀 SOLUTION: Re-train với Fixes

### Option A: Quick Fix (Gradient Clipping Only)

```powershell
# Already applied! Just re-train:
python src\training\federated_training_pipeline.py
```

**Expected**: Loss should be numeric (not NaN)

---

### Option B: Best Fix (Lower LR + Clipping)

```powershell
# 1. Edit configs/config.yaml
#    Change: learning_rate: 0.00001  (line 105)

# 2. Re-train
python src\training\federated_training_pipeline.py
```

**Expected**: 
- Loss: ~1.5 → ~0.8 (decreasing)
- Accuracy: 40-50%

---

## 📊 EXPECTED RESULTS AFTER FIX

### Before (NaN):
```
Round 1: Loss = nan, Accuracy = 0.04
Round 50: Loss = nan, Accuracy = 0.08
```

### After (Fixed):
```
Round 1: Loss = 1.52, Accuracy = 0.25
Round 10: Loss = 1.20, Accuracy = 0.35
Round 30: Loss = 0.90, Accuracy = 0.45
Round 50: Loss = 0.70, Accuracy = 0.50-55%
```

## 🔧 ADDITIONAL FIXES (If Still NaN)

### Fix 4: Normalize Image Embeddings

Edit `src/data_generation/process_amazon_data.py`:

```python
# After creating image_embedding:
image_embedding = image_embedding / (np.linalg.norm(image_embedding) + 1e-8)
image_embedding = image_embedding.astype(np.float32)
```

### Fix 5: Check Data Quality

```powershell
python test_dataloader.py
```

Verify:
- ✅ No NaN in embeddings
- ✅ No Inf values
- ✅ Reasonable ranges

### Fix 6: Use Synthetic Data (Fallback)

If Amazon data keeps causing NaN:

```powershell
# 1. Rename Amazon data (backup)
Rename-Item data\amazon_2023_processed data\amazon_2023_processed_backup

# 2. Train with synthetic (works but lower accuracy)
python src\training\federated_training_pipeline.py
```

## 🎯 RECOMMENDED ACTION

### Step 1: Lower Learning Rate (2 minutes)

Edit `configs/config.yaml` line 105:

```yaml
learning_rate: 0.00001  # Change from 0.0001
```

### Step 2: Re-train (30 minutes)

```powershell
python src\training\federated_training_pipeline.py
```

### Step 3: Monitor

Watch for:
- ✅ Loss is numeric (not NaN)
- ✅ Loss decreases over rounds
- ✅ Accuracy increases

If still NaN after 5 rounds → Stop and try Fix 4

## 📝 WHY SYNTHETIC DATA WORKED BEFORE

Synthetic data had:
- ✅ Consistent scales (all normalized)
- ✅ Smaller values
- ✅ No extreme outliers

Amazon data has:
- ⚠️ Mixed scales (text vs image vs behavior)
- ⚠️ Larger variance
- ⚠️ Possible outliers

**Solution**: Normalize everything + clip gradients!

## 🔬 DEBUG COMMANDS

### Check for NaN in data:

```powershell
python -c "import pandas as pd; import numpy as np; df = pd.read_pickle('data/amazon_2023_processed/client_0/data.pkl'); print('Text NaN:', np.isnan(df['text_embedding'].iloc[0]).any()); print('Image NaN:', np.isnan(df['image_embedding'].iloc[0]).any()); print('Behavior NaN:', np.isnan(df['behavior_features'].iloc[0]).any())"
```

### Check embedding ranges:

```powershell
python -c "import pandas as pd; import numpy as np; df = pd.read_pickle('data/amazon_2023_processed/client_0/data.pkl'); text = np.array(df['text_embedding'].iloc[0]); image = np.array(df['image_embedding'].iloc[0]); print('Text range:', text.min(), text.max()); print('Image range:', image.min(), image.max())"
```

## ✅ SUMMARY

| Issue | Cause | Fix | Status |
|-------|-------|-----|--------|
| NaN Loss | Gradient explosion | Gradient clipping | ✅ Applied |
| NaN Loss | High LR | Lower to 0.00001 | ⚠️ Need to edit config |
| Low Accuracy | NaN weights | Fix NaN first | 🔄 In progress |

## 🚀 NEXT STEPS

1. ✅ **Edit config.yaml** (learning_rate: 0.00001)
2. ✅ **Re-train** (python src\training\federated_training_pipeline.py)
3. ✅ **Monitor** first 5 rounds - should see numeric loss
4. ✅ **Wait** 30 minutes for full training
5. ✅ **Check results** - expect 45-55% accuracy

---

**Date**: January 5, 2026  
**Status**: Fixes applied, ready to re-train

