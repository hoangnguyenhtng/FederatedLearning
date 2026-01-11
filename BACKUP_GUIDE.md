# 💾 DANH SÁCH BACKUP QUAN TRỌNG

> File này liệt kê những folder/file quan trọng bạn NÊN backup trước khi reset máy

---

## ✅ ĐÃ ĐƯỢC LƯU TRÊN GITHUB

Những file sau đã được push lên GitHub và sẽ tự động có khi clone lại:

### 📝 Source Code
- ✅ `src/` - Toàn bộ source code
- ✅ `configs/` - Config files
- ✅ `notebooks/` - Jupyter notebooks

### 📚 Documentation
- ✅ `README.md` - Tổng quan dự án
- ✅ `SETUP_NHANH.txt` - Hướng dẫn setup nhanh
- ✅ `HUONG_DAN_CAI_DAT_LAI.md` - Hướng dẫn chi tiết
- ✅ `CHECKLIST_CAI_DAT.md` - Checklist cài đặt
- ✅ `QUICK_START.md` - Quick start guide
- ✅ Các file `.md` khác (reports, guides)

### 🔧 Scripts & Tools
- ✅ `requirements.txt` - Dependencies
- ✅ `setup_env.bat` - Setup script
- ✅ `download_amazon_data.ps1` - Download scripts
- ✅ `test_*.py` - Test scripts
- ✅ `.gitignore` - Git configuration

**➡️ Không cần backup! Chỉ cần clone lại từ GitHub.**

---

## ⚠️ KHÔNG CÓ TRÊN GITHUB - CẦN BACKUP

### 1. Môi Trường Ảo
```
❌ fed_rec_env/
```
**Quyết định**: 
- ❌ KHÔNG CẦN backup (quá nặng: 2GB)
- ✅ Tái tạo lại sau khi reset: `pip install -r requirements.txt`

---

### 2. Dữ Liệu Raw
```
❌ data/raw/
   └── amazon_2023/
       ├── All_Beauty.jsonl
       ├── Amazon_Fashion.jsonl
       ├── Baby_Products.jsonl
       ├── Video_Games.jsonl
       └── meta_*.jsonl
```

**Quyết định**:
- ⚠️ TÙY CHỌN backup (nếu không muốn download lại)
- Dung lượng: ~500MB-2GB
- ✅ Có thể download lại: `download_amazon_data.ps1`

**Khuyến nghị**: 
- Nếu mạng nhanh: KHÔNG cần backup, download lại (~10 phút)
- Nếu mạng chậm: Backup để tiết kiệm thời gian

---

### 3. Dữ Liệu Đã Xử Lý
```
⚠️ data/processed/
⚠️ data/amazon_2023_processed/
   ├── client_0/data.pkl
   ├── client_1/data.pkl
   └── ...
   └── client_39/data.pkl
```

**Quyết định**:
- ⚠️ **KHUYẾN NGHỊ BACKUP** (nếu đã process xong)
- Dung lượng: ~1-3GB
- Lý do: 
  - Mất 1-3 giờ để process lại
  - Kết quả deterministic (giống nhau nếu process lại)

**Khuyến nghị**: 
- ✅ BACKUP nếu đã process data thành công
- Lưu vào USB/External HDD/Cloud

**Cách backup**:
```powershell
# Nén data processed
Compress-Archive -Path "data\amazon_2023_processed" -DestinationPath "D:\Backup\fedlearn_data_processed.zip"

# Hoặc copy trực tiếp
Copy-Item -Recurse "data\amazon_2023_processed" -Destination "E:\Backup\"
```

---

### 4. Model Checkpoints & Results
```
⚠️ experiments/
   └── fedper_multimodal_v1/
       ├── results.json          # Metrics
       ├── global_model.pt       # Trained model
       ├── training_history.png  # Plots
       └── client_*/local_model.pt
```

**Quyết định**:
- ⚠️ **BẮT BUỘC BACKUP** (nếu có kết quả training quan trọng)
- Dung lượng: ~100MB-500MB
- Lý do:
  - Mất 30-60 phút để train lại
  - Kết quả có thể khác nhau (non-deterministic)
  - Quan trọng cho thesis/paper

**Khuyến nghị**: 
- ✅ **BACKUP NGAY** các experiments thành công
- Lưu nhiều nơi: Local + Cloud (Google Drive/OneDrive)

**Cách backup**:
```powershell
# Nén experiments
Compress-Archive -Path "experiments" -DestinationPath "D:\Backup\fedlearn_experiments.zip"

# Upload lên Google Drive/OneDrive
# Hoặc push lên GitHub (tạo branch riêng cho results)
```

---

### 5. Pretrained Models (nếu có)
```
⚠️ models/pretrained/
   ├── resnet50_weights.pth
   └── bert_model/
```

**Quyết định**:
- ⚠️ TÙY CHỌN backup
- Dung lượng: ~500MB-2GB
- ✅ Có thể download lại từ HuggingFace/PyTorch

**Khuyến nghị**: KHÔNG cần backup, download lại khi cần

---

### 6. Docker Volumes
```
❌ configs/volumes/
   ├── etcd/
   ├── milvus/
   └── minio/
```

**Quyết định**:
- ❌ KHÔNG CẦN backup
- Lý do: Runtime data, tự động tạo lại khi chạy Docker

---

### 7. Notebooks với Kết Quả
```
⚠️ notebooks/
   ├── 01_data_exploration.ipynb  # Nếu có cells đã chạy
   ├── 02_model_development.ipynb
   └── 03_evaluation.ipynb
```

**Quyết định**:
- ✅ ĐÃ CÓ trên GitHub (code)
- ⚠️ BACKUP nếu có outputs/visualizations quan trọng

**Khuyến nghị**: 
- Export sang HTML/PDF nếu có kết quả quan trọng
- Git sẽ lưu code, nhưng có thể mất cell outputs

---

## 📋 CHECKLIST BACKUP

Đánh dấu những gì bạn muốn backup:

### Bắt buộc (nếu có)
- [ ] **experiments/** - Kết quả training
- [ ] **results.json** - Metrics
- [ ] **global_model.pt** - Trained model

### Khuyến nghị (tiết kiệm thời gian)
- [ ] **data/amazon_2023_processed/** - Data đã process (1-3GB)
- [ ] Notebook outputs quan trọng

### Tùy chọn (có thể download lại)
- [ ] **data/raw/amazon_2023/** - Raw data (500MB-2GB)
- [ ] **models/pretrained/** - Pretrained models

### KHÔNG cần backup
- [ ] ~~fed_rec_env/~~ - Môi trường ảo
- [ ] ~~__pycache__/~~ - Python cache
- [ ] ~~configs/volumes/~~ - Docker volumes
- [ ] ~~*.pyc, *.log~~ - Temporary files

---

## 💾 HƯỚNG DẪN BACKUP

### Option 1: Backup sang External Drive

```powershell
# Tạo thư mục backup
New-Item -ItemType Directory -Path "E:\FedLearn_Backup"

# Backup experiments (BẮT BUỘC)
Copy-Item -Recurse "experiments" -Destination "E:\FedLearn_Backup\experiments"

# Backup processed data (KHUYẾN NGHỊ)
Copy-Item -Recurse "data\amazon_2023_processed" -Destination "E:\FedLearn_Backup\data_processed"

# Backup raw data (TÙY CHỌN)
Copy-Item -Recurse "data\raw" -Destination "E:\FedLearn_Backup\data_raw"
```

### Option 2: Nén và Backup

```powershell
# Nén experiments
Compress-Archive -Path "experiments" `
  -DestinationPath "D:\Backup\fedlearn_experiments_$(Get-Date -Format 'yyyyMMdd').zip"

# Nén processed data
Compress-Archive -Path "data\amazon_2023_processed" `
  -DestinationPath "D:\Backup\fedlearn_data_$(Get-Date -Format 'yyyyMMdd').zip"
```

### Option 3: Upload lên Cloud

```powershell
# Google Drive Desktop: Copy vào folder sync
Copy-Item -Recurse "experiments" -Destination "$env:USERPROFILE\Google Drive\FedLearn_Backup\"

# OneDrive: Copy vào OneDrive folder
Copy-Item -Recurse "experiments" -Destination "$env:USERPROFILE\OneDrive\FedLearn_Backup\"
```

### Option 4: Git LFS (cho experiments)

```powershell
# Tạo branch riêng cho results (KHÔNG merge vào main)
git checkout -b results-backup
git add experiments/
git commit -m "Backup: Training results $(Get-Date -Format 'yyyy-MM-dd')"
git push origin results-backup

# Quay lại main branch
git checkout main
```

---

## 📊 TỔNG DUNG LƯỢNG

| Item | Size | Backup? | Priority |
|------|------|---------|----------|
| Source code (GitHub) | ~50 MB | ✅ Done | - |
| fed_rec_env/ | ~2 GB | ❌ No | - |
| data/raw/ | 500 MB - 2 GB | ⚠️ Optional | Low |
| data/processed/ | 1-3 GB | ✅ Yes | High |
| experiments/ | 100-500 MB | ✅ Yes | **CRITICAL** |
| models/pretrained/ | 500 MB - 2 GB | ❌ No | - |

**Tổng cần backup**: ~1.5 - 4 GB (tùy lựa chọn)

---

## 🔄 SAU KHI RESET - RESTORE BACKUP

### Bước 1: Clone dự án từ GitHub
```powershell
git clone https://github.com/hoangnguyenhtng/FederatedLearning.git
cd FederatedLearning
```

### Bước 2: Setup môi trường
```powershell
python -m venv fed_rec_env
.\fed_rec_env\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Bước 3: Restore data (nếu đã backup)
```powershell
# Restore processed data
Copy-Item -Recurse "E:\FedLearn_Backup\data_processed\*" -Destination "data\amazon_2023_processed\"

# Restore experiments
Copy-Item -Recurse "E:\FedLearn_Backup\experiments\*" -Destination "experiments\"
```

### Bước 4: Hoặc download/process lại (nếu không backup)
```powershell
# Download raw data
PowerShell -ExecutionPolicy Bypass -File download_amazon_data.ps1

# Process data
python src\data_generation\process_amazon_data.py
```

---

## 📝 GHI CHÚ

### Ưu tiên backup theo thứ tự:

1. **experiments/** - Kết quả training (BẮT BUỘC nếu có)
2. **data/amazon_2023_processed/** - Data đã process (Khuyến nghị)
3. **data/raw/** - Raw data (Tùy chọn, có thể download lại)

### Thời gian restore:

- **Với backup**: ~15 phút (copy files)
- **Không backup**: ~2-3 giờ (download + process data)

### Lưu backup ở đâu?

✅ **Khuyến nghị**:
- Local: External HDD/USB (fast restore)
- Cloud: Google Drive/OneDrive (safe backup)
- Git branch: `results-backup` (version control)

❌ **Không nên**:
- Chỉ local (mất khi hỏng HDD)
- Chỉ cloud (chậm khi restore)

➡️ **Best practice**: Backup ở 2 nơi (local + cloud)

---

## ✅ CHECKLIST TRƯỚC KHI RESET

- [ ] Push tất cả code changes lên GitHub
- [ ] Backup experiments/ (nếu có kết quả quan trọng)
- [ ] Backup data/amazon_2023_processed/ (nếu muốn tiết kiệm thời gian)
- [ ] Export notebook outputs quan trọng
- [ ] Lưu file config.yaml đã customize
- [ ] Note lại các settings/credentials quan trọng
- [ ] Verify backup files không bị corrupt
- [ ] Lưu backup ở nhiều nơi (local + cloud)
- [ ] Document lại các experiments đã chạy
- [ ] Screenshot các kết quả quan trọng

---

**Tạo ngày**: 12/01/2026  
**Version**: 1.0  
**Status**: ✅ Ready to backup

---

**💡 Tip**: Nếu không chắc, backup tất cả! Storage rẻ, thời gian training đắt.

**⚠️ Nhớ**: Code trên GitHub an toàn, chỉ cần backup DATA và RESULTS!
