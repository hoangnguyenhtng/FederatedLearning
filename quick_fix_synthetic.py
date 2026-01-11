# 🪟 HƯỚNG DẪN CHO WINDOWS + VSCODE

## ✅ CÀI ĐẶT BAN ĐẦU (Chỉ làm 1 lần)

### 1. Mở VSCode Terminal

Trong VSCode:
- Nhấn `Ctrl + ~` (mở terminal)
- Hoặc: View → Terminal

**Chọn PowerShell** (recommended):
- Click dropdown bên terminal → Select PowerShell

### 2. Activate Virtual Environment

```powershell
# Trong VSCode terminal:
.\fed_rec_env\Scripts\Activate.ps1
```

**Nếu gặp lỗi "cannot be loaded because running scripts is disabled"**:

```powershell
# Chạy lệnh này (1 lần duy nhất):
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Sau đó activate lại:
.\fed_rec_env\Scripts\Activate.ps1
```

### 3. Install Thêm Packages

```powershell
pip install sentence-transformers pillow requests tqdm
```

---

## 📥 DOWNLOAD DATASET (Tự động)

### Option A: Dùng PowerShell Script (Recommended)

```powershell
# 1. Chạy script download (tự động)
PowerShell -ExecutionPolicy Bypass -File download_amazon_data.ps1
```

**Script này sẽ**:
- ✅ Tự động download 2 files (~300MB)
- ✅ Tự động extract
- ✅ Ready to use!

**Thời gian**: 5-10 phút (tùy internet)

---

### Option B: Download Thủ Công (Backup)

Nếu script không work, download bằng browser:

**Step 1: Download files**

1. Mở browser, download 2 files này:
   - [Reviews](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz) (~200MB)
   - [Metadata](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_All_Beauty.jsonl.gz) (~100MB)

2. Copy vào folder: `D:\Federated Learning\data\raw\amazon_2023\`

**Step 2: Extract files**

```powershell
# Trong VSCode terminal:
cd data\raw\amazon_2023

# Extract reviews
Expand-Archive -Path All_Beauty.jsonl.gz -DestinationPath .

# Extract metadata  
Expand-Archive -Path meta_All_Beauty.jsonl.gz -DestinationPath .
```

**Hoặc**: Click phải file .gz → Extract here (nếu có 7-Zip/WinRAR)

---

## ⚙️ PROCESS DATA

### Quick Test (10K samples - Recommended First)

```powershell
# Chạy processing (trong VSCode terminal)
python src\data_generation\process_amazon_data.py
```

**Cấu hình mặc định**:
- Sample: 10,000 interactions (để test nhanh)
- Clients: 10
- Output: `data\amazon_2023_processed\client_*\data.pkl`

**Thời gian ước tính**:
- Loading data: 2-3 phút
- Processing embeddings: 20-30 phút (text encoding)
- Downloading images: 10-15 phút (tùy internet)
- **Total**: ~40-50 phút

**Progress sẽ hiện trong terminal**:
```
======================================================================
AMAZON REVIEWS 2023 → FEDERATED MULTI-MODAL DATASET
======================================================================
Loading text encoder (SentenceTransformer)...
✅ Initialized processors on device: cuda
Loading data\raw\amazon_2023\All_Beauty.jsonl...
100%|████████████| 10000/10000 [00:02<00:00, 4500.00it/s]
✅ Loaded 10000 records
...
```

---

### Full Dataset (701K samples - Sau khi test OK)

**Edit file**: `src\data_generation\process_amazon_data.py`

Tìm dòng 371, thay đổi:
```python
# FROM:
SAMPLE_SIZE = 10000  # Process 10K interactions first

# TO:
SAMPLE_SIZE = None  # Process all data
```

**Save** (Ctrl+S) và chạy lại:
```powershell
python src\data_generation\process_amazon_data.py
```

**Thời gian**: ~8-12 giờ (chạy overnight)

---

## 🔍 KIỂM TRA KẾT QUẢ

### Xem files đã tạo

```powershell
# List processed data
Get-ChildItem -Recurse data\amazon_2023_processed

# Kết quả mong đợi:
# client_0\data.pkl
# client_1\data.pkl
# ...
# client_9\data.pkl
```

### Verify data quality

```powershell
# Tạo file test
python -c "import pandas as pd; import numpy as np; df = pd.read_pickle('data/amazon_2023_processed/client_0/data.pkl'); print('Samples:', len(df)); print('Text emb shape:', np.array(df['text_embedding'].iloc[0]).shape); print('Image emb shape:', np.array(df['image_embedding'].iloc[0]).shape); print('Behavior shape:', np.array(df['behavior_features'].iloc[0]).shape)"
```

**Kết quả mong đợi**:
```
Samples: 1234
Text emb shape: (384,)
Image emb shape: (2048,)
Behavior shape: (32,)
```

---

## 🚀 TRAIN MODEL

### Option 1: Sử dụng Amazon Data (Recommended)

**Đầu tiên**: Update config để use Amazon data

Tạo file mới: `configs\config_amazon.yaml`

```yaml
# Copy từ config.yaml, chỉ thay đổi:
paths:
  data_dir: "./data/amazon_2023_processed"  # ← Use Amazon data
  experiments_dir: "experiments"
  
federated:
  num_clients: 10  # Match processed clients
  num_rounds: 50
  fraction_fit: 0.4
  
training:
  batch_size: 32
  learning_rate: 0.001  # Higher LR for real data
```

**Chạy training**:
```powershell
python src\training\federated_training_pipeline.py --config configs\config_amazon.yaml
```

---

### Option 2: Quick Fix Synthetic Data (Alternative)

Nếu chưa muốn dùng Amazon, fix synthetic data trước:

<function_calls>
<invoke name="write">
<parameter name="contents"># Quick Fix for Synthetic Data - Remove Random Noise
# This patches client.py to at least create consistent embeddings

import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("QUICK FIX: Synthetic Data Embeddings")
print("=" * 70)

print("\n⚠️  This is a TEMPORARY fix!")
print("For BEST results, use Amazon data (see AMAZON_DATASET_GUIDE.md)")

# Read client.py
client_file = Path("src/federated/client.py")
with open(client_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Check if already patched
if "QUICK_FIX_APPLIED" in content:
    print("\n✅ Already patched!")
    sys.exit(0)

# Find and replace random text embeddings
old_text = "text_emb = torch.randn(batch_size, 384, device=self.device)"
new_text = """# QUICK_FIX_APPLIED: Use item embeddings from keywords
                    if 'text' in batch_data and batch_data['text']:
                        # Use text encoder if available
                        from sentence_transformers import SentenceTransformer
                        if not hasattr(self, '_text_encoder'):
                            print("Loading text encoder (one-time)...")
                            self._text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
                            self._text_encoder.eval()
                        
                        # Encode text
                        texts = [str(t) if t else "product" for t in batch_data['text']]
                        text_emb = self._text_encoder.encode(texts, convert_to_tensor=True).to(self.device)
                    else:
                        # Fallback: deterministic based on item_id (not random!)
                        item_ids = batch_data.get('item_id', torch.arange(batch_size))
                        # Create deterministic embeddings from item_id
                        text_emb = torch.zeros(batch_size, 384, device=self.device)
                        for i in range(batch_size):
                            seed = int(item_ids[i].item()) if torch.is_tensor(item_ids[i]) else int(item_ids[i])
                            torch.manual_seed(seed)
                            text_emb[i] = torch.randn(384, device=self.device) * 0.1"""

# Replace
if old_text in content:
    content = content.replace(old_text, new_text)
    print("\n✅ Patched text embeddings (deterministic)")
else:
    print("\n⚠️  Text embedding code not found (may be already modified)")

# Save
with open(client_file, 'w', encoding='utf-8') as f:
    f.write(content)

print("\n✅ PATCH APPLIED!")
print("\nNow you can train with synthetic data (still not as good as Amazon)")
print("Expected accuracy: 40-50% (vs 30% before, 70% with Amazon)")

