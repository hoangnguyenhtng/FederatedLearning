# 🚀 HƯỚNG DẪN CÀI ĐẶT LẠI DỰ ÁN

> Hướng dẫn chi tiết để setup lại dự án Federated Learning sau khi reset máy tính

---

## 📋 YÊU CẦU HỆ THỐNG

### Phần mềm cần cài đặt:

1. **Python 3.9+** 
   - Download: https://www.python.org/downloads/
   - ⚠️ **QUAN TRỌNG**: Tick vào "Add Python to PATH" khi cài đặt!

2. **Git**
   - Download: https://git-scm.com/download/win
   - Chọn các tùy chọn mặc định khi cài đặt

3. **Visual Studio Code** (khuyến nghị)
   - Download: https://code.visualstudio.com/

### Phần cứng khuyến nghị:
- RAM: 8GB+ (16GB khuyến nghị)
- Ổ cứng trống: 10GB+
- GPU: NVIDIA GPU với CUDA (tùy chọn, giúp training nhanh hơn)

---

## 🔧 CÀI ĐẶT BƯỚC 1: CLONE DỰ ÁN

### 1.1. Mở PowerShell hoặc Command Prompt

Nhấn `Win + X` → chọn "Windows PowerShell" hoặc "Terminal"

### 1.2. Chọn thư mục lưu dự án

```powershell
# Ví dụ: Lưu vào ổ D:\
cd D:\

# Hoặc lưu vào Documents
cd ~\Documents
```

### 1.3. Clone repository từ GitHub

```powershell
git clone https://github.com/hoangnguyenhtng/FederatedLearning.git
```

### 1.4. Vào thư mục dự án

```powershell
cd FederatedLearning
```

✅ **Checkpoint**: Bạn đã có folder `FederatedLearning` với đầy đủ code!

---

## 🐍 CÀI ĐẶT BƯỚC 2: TẠO MÔI TRƯỜNG ẢO

### 2.1. Tạo môi trường ảo Python

```powershell
python -m venv fed_rec_env
```

⏱️ Quá trình này mất khoảng 1-2 phút.

### 2.2. Kích hoạt môi trường ảo

**Trên Windows PowerShell:**
```powershell
.\fed_rec_env\Scripts\Activate.ps1
```

**Nếu gặp lỗi "execution policy"**, chạy lệnh này trước:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Trên Windows CMD:**
```cmd
fed_rec_env\Scripts\activate.bat
```

✅ **Checkpoint**: Bạn sẽ thấy `(fed_rec_env)` ở đầu dòng lệnh!

---

## 📦 CÀI ĐẶT BƯỚC 3: CÀI ĐẶT THƯ VIỆN

### 3.1. Nâng cấp pip

```powershell
python -m pip install --upgrade pip
```

### 3.2. Cài đặt tất cả dependencies

```powershell
pip install -r requirements.txt
```

⏱️ Quá trình này mất khoảng 10-15 phút tùy tốc độ mạng.

### 3.3. Kiểm tra cài đặt thành công

```powershell
python test_imports.py
```

✅ **Checkpoint**: Nếu không có lỗi, tất cả thư viện đã được cài đặt thành công!

---

## 📊 CÀI ĐẶT BƯỚC 4: TẢI DỮ LIỆU

### 4.1. Tải dữ liệu Amazon (KHUYẾN NGHỊ)

**Cách 1: Tải dữ liệu nhỏ (Fast - 10 phút)**
```powershell
PowerShell -ExecutionPolicy Bypass -File download_amazon_data.ps1
```

**Cách 2: Tải nhiều category (Medium - 30 phút)**
```powershell
PowerShell -ExecutionPolicy Bypass -File download_amazon_multi_category.ps1
```

**Cách 3: Tải toàn bộ dataset (Full - 1-2 giờ)**
```powershell
PowerShell -ExecutionPolicy Bypass -File download_full_amazon_data.ps1
```

### 4.2. Xử lý dữ liệu

```powershell
python src\data_generation\process_amazon_data.py
```

⏱️ Quá trình này mất khoảng 40-60 phút (tùy kích thước dataset).

✅ **Checkpoint**: Kiểm tra folder `data/amazon_2023_processed/` có chứa các file `client_*/data.pkl`

---

## 🏃 CHẠY THỬ DỰ ÁN

### 5.1. Kiểm tra dữ liệu

```powershell
python check_data_distribution.py
```

### 5.2. Test dataloader

```powershell
python test_dataloader.py
```

### 5.3. Chạy training (Federated Learning)

```powershell
python src\training\federated_training_pipeline.py
```

⏱️ Quá trình training mất khoảng 30-45 phút trên CPU.

✅ **Checkpoint**: Model sẽ được lưu trong folder `experiments/`

---

## 🔍 KIỂM TRA KẾT QUẢ

### Xem kết quả training

```powershell
# Kết quả được lưu tại:
experiments\fedper_multimodal_v1\

# Chứa các file:
- results.json          # Metrics (accuracy, loss)
- global_model.pt       # Model đã train
- training_history.png  # Biểu đồ training
```

### Kết quả mong đợi:

| Dataset | Accuracy | Loss | Training Time |
|---------|----------|------|---------------|
| Amazon Data (10K) | 60-70% | ~0.5 | 30-45 phút |
| Amazon Data (Full) | 70-75% | ~0.3 | 1-2 giờ |
| Synthetic Data | 30-40% | ~1.5 | 15-20 phút |

---

## 🐛 XỬ LÝ LỖI THƯỜNG GẶP

### Lỗi 1: `python` không được nhận diện

**Nguyên nhân**: Python chưa được thêm vào PATH

**Giải pháp**:
1. Gỡ cài đặt Python
2. Cài lại và nhớ tick "Add Python to PATH"
3. Hoặc thêm Python vào PATH thủ công:
   - Mở "Environment Variables"
   - Thêm đường dẫn Python vào PATH (ví dụ: `C:\Python39\`)

### Lỗi 2: "execution policy" khi chạy PowerShell script

**Giải pháp**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Lỗi 3: Lỗi khi cài PyTorch

**Giải pháp**: Cài thủ công PyTorch phù hợp với hệ thống:

**CPU only** (không có GPU NVIDIA):
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**CUDA 11.8** (có GPU NVIDIA):
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Lỗi 4: Hết RAM khi processing data

**Giải pháp**: Giảm SAMPLE_SIZE trong file `src\data_generation\process_amazon_data.py`

Mở file và sửa dòng 371:
```python
SAMPLE_SIZE = 5000  # Giảm từ 10000 xuống 5000
```

### Lỗi 5: Không tìm thấy module khi import

**Giải pháp**: Chạy script từ thư mục gốc của dự án:
```powershell
cd D:\FederatedLearning
python src\training\federated_training_pipeline.py
```

---

## 📂 CẤU TRÚC THƯ MỤC SAU KHI SETUP

```
FederatedLearning/
│
├── fed_rec_env/              # Môi trường ảo (tạo bởi bạn)
│
├── data/
│   ├── raw/                  # Dữ liệu thô Amazon (sau khi download)
│   ├── processed/            # Dữ liệu đã xử lý
│   └── amazon_2023_processed/ # Dữ liệu Amazon đã process
│
├── src/                      # Source code
│   ├── data_generation/
│   ├── data_processing/
│   ├── federated/
│   ├── models/
│   └── training/
│
├── experiments/              # Kết quả training
├── configs/                  # Config files
├── notebooks/                # Jupyter notebooks
│
├── requirements.txt          # Danh sách thư viện
├── .gitignore               # File loại trừ Git
│
└── HUONG_DAN_CAI_DAT_LAI.md # File này!
```

---

## 🎯 CHECKLIST HOÀN THÀNH

Đánh dấu ✅ khi hoàn thành từng bước:

- [ ] Cài đặt Python 3.9+
- [ ] Cài đặt Git
- [ ] Clone repository từ GitHub
- [ ] Tạo môi trường ảo `fed_rec_env`
- [ ] Kích hoạt môi trường ảo (thấy `(fed_rec_env)` ở đầu dòng)
- [ ] Cài đặt dependencies từ `requirements.txt`
- [ ] Test imports thành công (`python test_imports.py`)
- [ ] Download dữ liệu Amazon
- [ ] Process dữ liệu thành công
- [ ] Chạy test dataloader thành công
- [ ] Chạy training và có kết quả

---

## 🚀 QUICK START (TÓM TẮT)

Nếu đã quen, chỉ cần chạy các lệnh sau:

```powershell
# 1. Clone project
git clone https://github.com/hoangnguyenhtng/FederatedLearning.git
cd FederatedLearning

# 2. Setup môi trường
python -m venv fed_rec_env
.\fed_rec_env\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# 3. Download & process data
PowerShell -ExecutionPolicy Bypass -File download_amazon_data.ps1
python src\data_generation\process_amazon_data.py

# 4. Training!
python src\training\federated_training_pipeline.py
```

⏱️ **Tổng thời gian**: ~2-3 giờ (bao gồm download và processing)

---

## 📞 HỖ TRỢ

### Tài liệu khác:
- `QUICK_START.md` - Hướng dẫn chạy nhanh
- `README.md` - Tổng quan dự án
- `TRAINING_EVALUATION_REPORT.md` - Báo cáo kết quả training

### Kiểm tra log:
Nếu gặp lỗi, kiểm tra:
- Console output trong terminal
- File log trong folder `logs/` (nếu có)

### Tips:
- Luôn chạy từ thư mục gốc của dự án
- Nhớ activate môi trường ảo trước khi chạy
- Nếu gặp lỗi, đọc kỹ error message - thường nó cho biết thiếu gì

---

## ⚡ LƯU Ý QUAN TRỌNG

### ❗ Không push các file sau lên Git:
- `fed_rec_env/` - Môi trường ảo (quá nặng)
- `data/raw/` - Dữ liệu thô (quá nặng)
- `data/processed/` - Dữ liệu đã xử lý (quá nặng)
- `experiments/` - Model checkpoints (quá nặng)

> File `.gitignore` đã được cấu hình sẵn để tự động loại trừ!

### 💡 Best Practices:
1. **Commit code thường xuyên** nhưng KHÔNG commit data/models
2. **Sử dụng môi trường ảo** cho mỗi dự án Python
3. **Backup kết quả training** quan trọng ra nơi khác
4. **Document thay đổi** trong commit messages

---

## 🎓 NEXT STEPS

Sau khi setup xong, bạn có thể:

1. **Khám phá Notebooks**: 
   ```powershell
   jupyter notebook
   # Mở file notebooks/01_data_exploration.ipynb
   ```

2. **Thử nghiệm với config khác nhau**:
   - Sửa `configs/config.yaml`
   - Thay đổi số clients, rounds, learning rate, v.v.

3. **Phát triển thêm**:
   - Thêm model mới trong `src/models/`
   - Thử aggregation strategy khác trong `src/federated/`
   - Implement differential privacy trong `src/federated/privacy.py`

4. **Đánh giá kết quả**:
   ```powershell
   python src\training\evaluate_federated_model.py
   ```

---

## 📊 BENCHMARK REFERENCE

Để so sánh kết quả training của bạn:

| Metric | Baseline | Good | Excellent |
|--------|----------|------|-----------|
| Test Accuracy | 50-60% | 65-70% | 75%+ |
| Test Loss | < 0.8 | < 0.5 | < 0.3 |
| Training Time (50 rounds) | 45-60 min | 30-40 min | < 30 min |
| Convergence | Round 40+ | Round 30 | Round 20 |

---

**Tạo ngày**: 12/01/2026  
**Phiên bản**: 1.0  
**Trạng thái**: ✅ Ready to use

---

### 🎉 Chúc bạn setup thành công!

Nếu có câu hỏi, hãy kiểm tra các file markdown khác trong dự án hoặc xem lại error messages trong terminal.

**Happy Coding! 🚀**
