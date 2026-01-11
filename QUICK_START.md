# ⚡ QUICK START GUIDE

## 🎯 3 SCENARIOS

---

### ✅ **SCENARIO 1: Dùng Amazon Data (RECOMMENDED)**

```powershell
# Bước 1: Download Amazon data (10 phút)
PowerShell -ExecutionPolicy Bypass -File download_amazon_data.ps1

# Bước 2: Process data (40-60 phút)
python src\data_generation\process_amazon_data.py

# Bước 3: Train! (30-45 phút)
python src\training\federated_training_pipeline.py
```

**Expected Results:**
- ✅ Accuracy: **60-70%** (vs 30% với synthetic)
- ✅ Loss: **~0.5** (vs 1.555 với synthetic)
- ✅ Model học được từ REAL features!

**Pipeline tự động detect Amazon data** - KHÔNG cần thay đổi code!

---

### ⚠️ **SCENARIO 2: Chưa có Data - Dùng Synthetic (Current)**

```powershell
# Chạy training trực tiếp với synthetic data
python src\training\federated_training_pipeline.py
```

**Expected Results:**
- ⚠️ Accuracy: **30%** (random noise issue)
- ⚠️ Loss: **~1.555** (flat, không học)
- ❌ Model KHÔNG học được (text & image là random!)

**Recommend**: Chuyển sang Scenario 1 (Amazon data)

---

### 🔧 **SCENARIO 3: Fix Synthetic Data (Quick Patch)**

Nếu muốn cải thiện synthetic data (không tốt bằng Amazon nhưng OK hơn hiện tại):

```powershell
# Quick fix: Tạo consistent embeddings
python quick_fix_synthetic.py

# Train
python src\training\federated_training_pipeline.py
```

**Expected Results:**
- ✅ Accuracy: **40-50%** (better than 30%)
- ✅ Loss: **~1.0** (decreases slowly)
- ⚠️ Vẫn không tốt bằng Amazon data

---

## 🔍 AUTO-DETECTION LOGIC

Pipeline **TỰ ĐỘNG CHỌN** dataset theo thứ tự ưu tiên:

```
1. Check: data/amazon_2023_processed/client_*/data.pkl
   → Nếu có: Use Amazon data ✅
   
2. Check: data/simulated_clients/client_*/
   → Nếu có: Use synthetic data ⚠️
   
3. Không có gì:
   → Error: Please download/generate data ❌
```

**KHÔNG CẦN THAY ĐỔI CODE!** Pipeline tự động detect.

---

## 📊 COMPARISON

| Scenario | Accuracy | Time to Setup | Recommendation |
|----------|----------|---------------|----------------|
| **1. Amazon** | 60-70% | ~1.5 hours | ⭐⭐⭐⭐⭐ BEST |
| 2. Synthetic (current) | 30% | 0 (already done) | ⭐ Poor |
| 3. Fixed Synthetic | 40-50% | 5 minutes | ⭐⭐ OK |

---

## 🚀 RECOMMENDED PATH

### For Quick Test (Today)

```powershell
# Process 10K Amazon samples (faster)
python src\data_generation\process_amazon_data.py

# Train
python src\training\federated_training_pipeline.py
```

**Time**: ~1.5 hours total  
**Result**: See real improvement immediately!

---

### For Best Results (Overnight)

```powershell
# 1. Edit process_amazon_data.py
#    Change: SAMPLE_SIZE = None (line 371)

# 2. Run overnight
python src\data_generation\process_amazon_data.py

# 3. Next day: Train
python src\training\federated_training_pipeline.py
```

**Time**: ~8-12 hours processing + 1-2 hours training  
**Result**: 70-75% accuracy, ready for thesis!

---

## 🔎 CHECK STATUS

### Kiểm tra dataset có sẵn

```powershell
# Check Amazon data
Test-Path data\amazon_2023_processed\client_0\data.pkl

# Check synthetic data
Test-Path data\simulated_clients\client_0\interactions.csv
```

### Xem training output

```powershell
# Pipeline sẽ báo đang dùng dataset gì:
# "🎉 Using AMAZON REVIEWS 2023 dataset (Real features!)"
# hoặc
# "⚠️ Using SYNTHETIC data (contains random noise!)"
```

---

## 💡 FAQ

### Q: Tôi có PHẢI download Amazon data không?

**A**: KHÔNG bắt buộc, nhưng HIGHLY RECOMMENDED vì:
- Accuracy tăng 2x (30% → 60-70%)
- Model học được real patterns
- Ready for thesis/paper

### Q: Tôi đã có synthetic data, có bị mất không?

**A**: KHÔNG! Synthetic data vẫn giữ nguyên. Pipeline ưu tiên Amazon, nhưng fallback về synthetic nếu không có.

### Q: Có cần sửa config.yaml không?

**A**: KHÔNG! Pipeline tự động detect. Chỉ cần chạy:
```powershell
python src\training\federated_training_pipeline.py
```

### Q: Download Amazon data mất bao lâu?

**A**: 
- Download: ~5-10 phút (300MB)
- Process 10K samples: ~40-60 phút
- Process full (701K): ~8-12 giờ (overnight)

### Q: Training mất bao lâu?

**A**:
- CPU: ~30-45 phút (50 rounds)
- GPU: ~15-20 phút (50 rounds)

---

## 🎯 BOTTOM LINE

### Bạn muốn gì?

**Fast Test**: Chạy với synthetic ngay (30% accuracy) ⚠️

**Better Results**: Process 10K Amazon (~1.5h) → 60% accuracy ✅

**Best Results**: Process full Amazon (~10h) → 70-75% accuracy ⭐

---

**TL;DR**: Chỉ cần chạy:

```powershell
python src\training\federated_training_pipeline.py
```

Pipeline sẽ:
1. ✅ Tự động tìm Amazon data (nếu có)
2. ⚠️ Fallback về synthetic (nếu không)
3. ❌ Error nếu không có gì

**KHÔNG CẦN THAY ĐỔI GÌ!**

---

**Date**: January 5, 2026  
**Status**: ✅ AUTO-DETECTION READY

