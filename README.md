# 🚀 Federated Learning for Recommendation Systems

Hệ thống đề xuất sản phẩm sử dụng Federated Learning với dữ liệu đa phương thức (Multi-modal) từ Amazon Reviews 2023.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📚 Tài Liệu Hướng Dẫn

### ⭐ Bắt đầu từ đây:
1. **[PROJECT_COMPLETE_GUIDE.md](PROJECT_COMPLETE_GUIDE.md)** - ⭐⭐⭐ **HƯỚNG DẪN TỔNG HỢP TOÀN BỘ DỰ ÁN** (Đọc đầu tiên!)
2. **[THESIS_COMPLETION_PLAN.md](THESIS_COMPLETION_PLAN.md)** - ⭐ Kế hoạch chi tiết hoàn thiện đồ án
3. **[REAL_WORLD_APPLICATION.md](REAL_WORLD_APPLICATION.md)** - ⭐ Ứng dụng thực tế của dự án

### 🎓 Cho Tốt Nghiệp

1. **[THESIS_GUIDE.md](THESIS_GUIDE.md)** - Hướng dẫn sử dụng dự án cho tốt nghiệp
2. **[EVALUATION_REPORT.md](EVALUATION_REPORT.md)** - Báo cáo đánh giá kết quả training
3. **[run_pipeline.py](run_pipeline.py)** - Script tổng hợp để chạy toàn bộ pipeline

### ✅ Đọc 1 file là đủ (khuyến nghị)

1. **[SETUP_NHANH.txt](SETUP_NHANH.txt)** - Đọc đầu tiên! Hướng dẫn 5 phút
2. **[HUONG_DAN_CAI_DAT_LAI.md](HUONG_DAN_CAI_DAT_LAI.md)** - Hướng dẫn chi tiết từng bước
3. **[CHECKLIST_CAI_DAT.md](CHECKLIST_CAI_DAT.md)** - Checklist để theo dõi tiến độ

### 📖 Tài liệu chi tiết (chỉ khi cần)

- **[QUICK_START.md](QUICK_START.md)** - Hướng dẫn chạy nhanh dự án
- **[TRAINING_EVALUATION_REPORT.md](TRAINING_EVALUATION_REPORT.md)** - Báo cáo kết quả training
- **[PROJECT_ANALYSIS_REPORT.md](PROJECT_ANALYSIS_REPORT.md)** - Phân tích kiến trúc dự án

---

## ⚡ Quick Start (5 lệnh)

```powershell
# 1. Clone project
git clone https://github.com/hoangnguyenhtng/FederatedLearning.git
cd FederatedLearning

# 2. Setup environment
python -m venv fed_rec_env
.\fed_rec_env\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Download & process data
PowerShell -ExecutionPolicy Bypass -File download_amazon_data.ps1
python src\data_generation\process_amazon_data.py

# 4. Train!
python src\training\federated_training_pipeline.py
```

⏱️ **Tổng thời gian**: ~3 giờ

---

## 🎯 Tính Năng Chính

### ✨ Federated Learning
- **Phân tán dữ liệu**: Mô phỏng nhiều clients (mặc định **10** trong `configs/config.yaml`, có thể tăng) với dữ liệu Non-IID
- **Aggregation**: FedAvg, FedProx algorithms
- **Privacy**: Differential Privacy support (Opacus)

### 🎨 Multi-modal Learning
- **Text**: Embeddings 384-dim (Sentence-Transformers / pipeline xử lý Amazon)
- **Image**: ResNet-50 features (2048-dim) từ ảnh sản phẩm
- **Behavioral**: Vector hành vi (32-dim)
- **Fusion**: Adaptive fusion (trọng số text/image/behavior học được) trong `multimodal_encoder.py`

### 📊 Real-world Dataset
- **Amazon Reviews 2023**: 4 categories (Beauty, Fashion, Baby, Games)
- **700K+ reviews** với text, images, ratings
- **Preprocessing pipeline** tự động

### 🔧 Advanced Features
- **FedPer**: Personalized federated learning
- **Attention / fusion**: Module attention trong `attention_mechanism.py`; fusion chính dùng adaptive weights trong encoder
- **Vector DB**: Milvus integration cho item retrieval (tùy chọn)
- **Dashboard**: Streamlit UI for explainable AI

---

## 📁 Cấu Trúc Dự Án

```
FederatedLearning/
│
├── 📄 SETUP_NHANH.txt              # Đọc đầu tiên!
├── 📄 HUONG_DAN_CAI_DAT_LAI.md     # Hướng dẫn chi tiết
├── 📄 CHECKLIST_CAI_DAT.md         # Checklist setup
│
├── src/
│   ├── data_generation/            # Data processing & generation
│   │   ├── process_amazon_data.py  # Process Amazon reviews
│   │   └── federated_dataloader.py # Federated data loaders
│   │
│   ├── models/                     # Model architectures
│   │   ├── recommendation_model.py # Main recommendation model
│   │   ├── multimodal_encoder.py   # Multi-modal fusion
│   │   └── attention_mechanism.py  # Attention layers
│   │
│   ├── federated/                  # Federated learning logic
│   │   ├── server.py               # FL server
│   │   ├── client.py               # FL client
│   │   ├── aggregator.py           # Aggregation strategies
│   │   └── privacy.py              # Differential privacy
│   │
│   └── training/                   # Training pipelines
│       ├── federated_training_pipeline.py  # Main training
│       ├── local_trainer.py        # Local training
│       └── evaluate_federated_model.py     # Evaluation
│
├── configs/
│   ├── config.yaml                 # Main configuration
│   ├── config_thesis.yaml          # Thesis experiments
│   └── docker-compose.yml          # Milvus setup
│
├── notebooks/                      # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_development.ipynb
│   └── 03_evaluation.ipynb
│
├── data/                           # Data directory (not in Git)
│   ├── raw/                        # Raw Amazon data
│   ├── processed/                  # Processed data
│   └── amazon_2023_processed/      # FL-ready data
│
└── experiments/                    # Training results (not in Git)
    └── fedper_multimodal_v1/
        ├── results.json
        ├── global_model.pt
        └── training_history.png
```

---

## 🎓 Kết Quả Thực Nghiệm

### Performance Benchmarks

| Dataset | Accuracy | Loss | Training Time |
|---------|----------|------|---------------|
| Amazon (10K samples) | 60-70% | ~0.5 | 30-45 min |
| Amazon (Full 700K) | 70-75% | ~0.3 | 1-2 hours |
| Synthetic Data | 30-40% | ~1.5 | 15-20 min |

### Model Architecture
- **Input**: Text (384-dim) + Image (2048-dim) + Behavior (32-dim)
- **Encoder**: `MultiModalEncoder` → unified 384-dim + fusion weights (3 chiều)
- **FedPer**: Shared base + personal head; **training**: dự đoán rating **5 lớp** (0–4)
- **API demo**: có thể dùng `num_classes = số item` cho top-K; cần đồng bộ với checkpoint khi deploy

---

## 🔧 Yêu Cầu Hệ Thống

### Software
- Python 3.9+
- PyTorch 2.1.0+
- CUDA 11.8+ (optional, for GPU)

### Hardware
- **Minimum**: 8GB RAM, 10GB storage
- **Recommended**: 16GB RAM, 20GB storage, NVIDIA GPU

### Dependencies
Xem đầy đủ trong `requirements.txt`:
- `torch`, `torchvision` - Deep learning
- `transformers` - BERT models
- `flwr` - Federated learning framework
- `pymilvus` - Vector database
- `streamlit` - Dashboard

---

## 🚀 Workflow

### 1️⃣ Data Preparation
```powershell
# Download Amazon reviews
PowerShell -ExecutionPolicy Bypass -File download_amazon_data.ps1

# Process into FL-ready format
python src\data_generation\process_amazon_data.py
```

### 2️⃣ Training
```powershell
# Federated training (số round theo config.yaml, mặc định 100)
python src\training\federated_training_pipeline.py
```

### 3️⃣ Evaluation
```powershell
# Evaluate model
python src\training\evaluate_federated_model.py

# Check results
cat experiments\fedper_multimodal_v1\results.json
```

### 4️⃣ Visualization (Optional)
```powershell
# Launch dashboard
streamlit run src\dashboard\explainable_ui.py
```

---

## 📊 Configuration

Chỉnh sửa `configs/config.yaml` (cấu trúc thực tế):

```yaml
federated:
  num_clients: 10
  num_rounds: 100
  fraction_fit: 0.6
  fraction_evaluate: 1.0

model:
  text_embedding_dim: 384
  image_embedding_dim: 2048
  behavior_embedding_dim: 32
  num_classes: 5   # rating 1–5 → 0–4 khi train FL

training:
  batch_size: 32
  local_epochs: 5
  learning_rate: 0.0001
```

---

## 🐛 Xử Lý Lỗi

### Lỗi: `python` not recognized
```powershell
# Cài lại Python với "Add to PATH"
# Hoặc thêm Python vào PATH thủ công
```

### Lỗi: Execution Policy
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Lỗi: Out of Memory
```python
# Sửa file src/data_generation/process_amazon_data.py
SAMPLE_SIZE = 5000  # Giảm từ 10000
```

### Lỗi: Module not found
```powershell
# Kiểm tra môi trường ảo đã active
.\fed_rec_env\Scripts\Activate.ps1

# Cài lại dependencies
pip install -r requirements.txt
```
Xem thêm trong `PROJECT_GUIDE.md` (mục Troubleshooting).

---

## 📚 Nghiên Cứu & Tham Khảo

### Algorithms Implemented
- **FedAvg**: McMahan et al. (2017)
- **FedProx**: Li et al. (2020)
- **FedPer**: Arivazhagan et al. (2019)
- **Differential Privacy**: Abadi et al. (2016)

### Datasets
- **Amazon Reviews 2023**: McAuley et al. (2023)
- Categories: All_Beauty, Amazon_Fashion, Baby_Products, Video_Games

### Multi-modal Fusion
- **Text**: BERT (Devlin et al., 2019)
- **Image**: ResNet-50 (He et al., 2016)
- **Fusion**: Multi-head Attention (Vaswani et al., 2017)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Hoang Nguyen**

- GitHub: [@hoangnguyenhtng](https://github.com/hoangnguyenhtng)
- Repository: [FederatedLearning](https://github.com/hoangnguyenhtng/FederatedLearning)

---

## 🙏 Acknowledgments

- **Flower (Flwr)** - Federated learning framework
- **Hugging Face** - Transformers library
- **Amazon** - Reviews 2023 dataset
- **PyTorch** - Deep learning framework

---

## 📞 Support

Có câu hỏi? Xem:

1. **[SETUP_NHANH.txt](SETUP_NHANH.txt)** - Quick start
2. **[HUONG_DAN_CAI_DAT_LAI.md](HUONG_DAN_CAI_DAT_LAI.md)** - Detailed guide
3. **[CHECKLIST_CAI_DAT.md](CHECKLIST_CAI_DAT.md)** - Installation checklist
4. **Issues** - Open an issue on GitHub

---

## 🔄 Version History

- **v1.0** (Jan 2026) - Initial release
  - Multi-modal federated learning
  - Amazon Reviews 2023 integration
  - FedPer implementation
  - Comprehensive documentation

---

## 🎯 Roadmap

- [ ] Add more aggregation algorithms (FedNova, FedOpt)
- [ ] Implement vertical federated learning
- [ ] Add more datasets support
- [ ] Optimize for edge devices
- [ ] Deploy as production service
- [ ] Add unit tests
- [ ] Docker containerization

---

## ⭐ Star History

If you find this project useful, please consider giving it a star ⭐

---

**📅 Last Updated**: March 28, 2026  
**🔖 Version**: 1.0.1  
**✅ Status**: Core ready (đồng bộ README với `config.yaml` và pipeline hiện tại)

---

Tuần 1 (Ngày 1–7): “Stabilize & Validate” — kiểm tra lại model/pipeline là ưu tiên số 1
Chạy lại end-to-end 2 lần với Amazon (10k samples trước):
Process → Train → Evaluate → API → Dashboard
Ghi lại: seed/config, thời gian chạy, file output
Sanity checks bắt buộc:
Loss giảm, metrics không “ảo”
Không NaN/Inf
Kiểm tra cân bằng nhãn (rating 1–5) và confusion matrix (nếu cần thêm)
Chốt cấu hình thesis:
1 config “demo nhanh” (ít rounds, nhanh)
1 config “final” (đủ rounds để báo cáo)
Fix những điểm còn yếu:
Nếu recommendation còn “kém thuyết phục”: tăng candidate pool, cải thiện cách tạo behavior features, hoặc dùng Milvus retrieval (nếu bạn bật Milvus)
Deliverables cuối tuần:

1 lần chạy “final” có checkpoint + amazon_evaluation_summary.json
Ảnh/plot training curves + bảng metrics tổng
Tuần 2 (Ngày 8–14): “Experiments that matter” — đủ số liệu để bảo vệ
Làm tối thiểu 3 thí nghiệm (đủ để hội đồng thấy có so sánh):

Baseline 1: FedAvg (tắt FedPer/personal head hoặc chạy strategy FedAvg nếu đã có)
Ablation modalities: tắt lần lượt Text / Image / Behavior (3 run nhỏ, có thể ít rounds)
Non-IID sensitivity: thay alpha (ví dụ 0.1 vs 0.5 vs 1.0) chạy ít rounds để vẽ trend
Deliverables cuối tuần:

1 bảng so sánh (NDCG@10, MRR, Accuracy, Loss) + 1–2 biểu đồ
Kết luận rõ: FedPer hơn baseline bao nhiêu, modality nào quan trọng, non-IID ảnh hưởng thế nào
Tuần 3 (Ngày 15–21): “Polish & Defense-ready” — demo web + slide + rehearsal
Demo e-commerce (khả thi trong 3 tuần):
Nếu bạn có thời gian frontend: dựng React/Vite + các trang: Catalog (pagination), Product detail, Cart, Recommendations + Explain toggle
Nếu không: Streamlit dashboard + FastAPI docs vẫn đủ, nhưng cần polish UI/flow demo (kịch bản 3–5 phút)
Chuẩn hóa repo & tái lập:
PROJECT_GUIDE.md có “Demo script bảo vệ” + “Test plan”
Chốt commands chạy 1 máy khác vẫn được
Slide & bảo vệ:
15–20 slides: problem → method → system → experiments → demo → conclusion/limitations
Chuẩn bị Q&A: privacy claim, trade-off accuracy, non-IID realism, explainability
Deliverables cuối tuần:

Demo chạy ổn (kể cả offline) + video backup
Slide hoàn chỉnh + luyện demo 3 lần