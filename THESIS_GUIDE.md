# 📚 HƯỚNG DẪN SỬ DỤNG DỰ ÁN CHO TỐT NGHIỆP

## 🎯 TỔNG QUAN DỰ ÁN

**Tên dự án**: Hệ thống Đề xuất Đa phương thức sử dụng Federated Learning  
**Mô tả**: Xây dựng hệ thống recommendation với dữ liệu đa phương thức (text, image, behavior) sử dụng Federated Learning để bảo vệ privacy

---

## 📁 CẤU TRÚC DỰ ÁN

```
Federated Learning/
├── src/                          # Source code chính
│   ├── models/                   # Model architectures
│   ├── training/                 # Training pipelines
│   ├── federated/                # Federated learning logic
│   ├── data_generation/          # Data processing
│   ├── api/                      # FastAPI server
│   └── dashboard/                # Streamlit dashboard
│
├── configs/                      # Configuration files
│   └── config.yaml               # Main config
│
├── experiments/                  # Training results
│   └── fedper_multimodal_v1/    # Experiment outputs
│
├── data/                         # Data directory
│   ├── raw/                      # Raw data
│   └── processed/                # Processed data
│
├── run_pipeline.py              # Script tổng hợp
├── requirements.txt              # Dependencies
└── README.md                     # Main documentation
```

---

## 🚀 CÁCH SỬ DỤNG

### 1. Setup Môi trường

```powershell
# Activate virtual environment
.\fed_rec_env\Scripts\Activate.ps1

# Install dependencies (nếu chưa có)
pip install -r requirements.txt
```

### 2. Chạy Training

```powershell
# Cách 1: Dùng script tổng hợp
python run_pipeline.py --mode train

# Cách 2: Chạy trực tiếp
python src/training/federated_training_pipeline.py
```

### 3. Đánh giá Model

```powershell
# Cách 1: Dùng script tổng hợp
python run_pipeline.py --mode evaluate

# Cách 2: Chạy trực tiếp
python src/training/evaluate_federated_model.py
```

### 4. Chạy API Server

```powershell
# Cách 1: Dùng script tổng hợp
python run_pipeline.py --mode api

# Cách 2: Chạy trực tiếp
uvicorn src.api.fastapi_app:app --host 0.0.0.0 --port 8000
```

### 5. Chạy Dashboard

```powershell
# Cách 1: Dùng script tổng hợp
python run_pipeline.py --mode dashboard

# Cách 2: Chạy trực tiếp
streamlit run src/dashboard/explainable_ui.py
```

### 6. Chạy Tất Cả (Training + Evaluation)

```powershell
python run_pipeline.py --mode all
```

---

## 📊 KẾT QUẢ TRAINING

### Xem kết quả

Kết quả training được lưu trong `experiments/fedper_multimodal_v1/`:

- `models/global_model_final.pt` - Model đã train
- `metrics/training_history.json` - Lịch sử training
- `metrics/training_curves.png` - Biểu đồ loss

### Đánh giá kết quả

Xem file `EVALUATION_REPORT.md` để biết chi tiết về kết quả training.

---

## 🔧 CẤU HÌNH

Chỉnh sửa `configs/config.yaml` để thay đổi:
- Số clients, số rounds
- Learning rate, batch size
- Model architecture
- Data paths

---

## 📈 METRICS

Các metrics được tính:
- **Accuracy**: Độ chính xác dự đoán rating
- **Precision/Recall**: Độ chính xác và recall
- **NDCG@10**: Normalized Discounted Cumulative Gain
- **MRR**: Mean Reciprocal Rank

---

## 🎓 CHO TỐT NGHIỆP

### Điểm mạnh của dự án:

1. ✅ **Federated Learning**: Bảo vệ privacy của user
2. ✅ **Multi-modal**: Sử dụng text, image, behavior features
3. ✅ **FedPer Architecture**: Personalized federated learning
4. ✅ **Real-world Dataset**: Amazon Reviews 2023
5. ✅ **Complete Pipeline**: Training, Evaluation, API, Dashboard

### Các chức năng chính:

1. **Training Pipeline** (`src/training/federated_training_pipeline.py`)
   - Federated training với Flower framework
   - Support FedAvg, FedProx algorithms
   - Auto-save checkpoints và metrics

2. **Evaluation** (`src/training/evaluate_federated_model.py`)
   - Đánh giá model trên tất cả clients
   - Tính các metrics chi tiết
   - Visualization kết quả

3. **API** (`src/api/fastapi_app.py`)
   - RESTful API cho recommendations
   - Explainability features
   - Integration với Milvus vector DB

4. **Dashboard** (`src/dashboard/explainable_ui.py`)
   - Streamlit UI
   - Interactive visualizations
   - Real-time recommendations

---

## 📝 TÀI LIỆU THAM KHẢO

- **README.md**: Tài liệu chính
- **QUICK_START.md**: Hướng dẫn nhanh
- **EVALUATION_REPORT.md**: Báo cáo kết quả
- **HUONG_DAN_CAI_DAT_LAI.md**: Hướng dẫn setup

---

## 🐛 XỬ LÝ LỖI

### Lỗi thường gặp:

1. **Import Error**: Đảm bảo đã activate virtual environment
2. **CUDA Error**: Model sẽ tự động fallback về CPU
3. **Data Not Found**: Kiểm tra đường dẫn trong config.yaml

---

**Chúc bạn tốt nghiệp thành công! 🎓**
