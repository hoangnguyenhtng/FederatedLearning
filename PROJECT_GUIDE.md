# Project Guide: Training, Evaluation, Demo

Tài liệu này là **file hướng dẫn chính** cho đồ án tốt nghiệp. Mục tiêu là chạy được pipeline hoàn chỉnh:

```text
Amazon Reviews 2023 → xử lý dữ liệu → FedPer training → evaluation → API → dashboard/demo
```

## 1. Trạng Thái Dự Án

### Đã có

- Mô hình `FedPerRecommender` với shared layers và personal head.
- `MultiModalEncoder` cho text, image, behavior.
- Federated training bằng Flower simulation.
- Amazon dataloader theo client.
- Evaluation theo từng client.
- FastAPI và Streamlit dashboard demo.

### Cần chốt trước bảo vệ

- Chạy lại training với lượng dữ liệu đủ.
- Lưu checkpoint và evaluation report.
- Chạy demo API/dashboard từ checkpoint đã train.
- Ghi lại kết quả cuối cùng vào slide/luận văn.

## 2. Chuẩn Bị Môi Trường

Chạy từ thư mục gốc repo:

```powershell
python -m venv fed_rec_env
.\fed_rec_env\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Kiểm tra nhanh:

```powershell
python run_pipeline.py --help
```

Nếu thiếu package như `fastapi`, `flwr`, `opacus`, `streamlit`, hãy chạy lại:

```powershell
pip install -r requirements.txt
```

## 3. Lượng Dữ Liệu Khuyến Nghị Cho Đồ Án

### Mốc tối thiểu để debug nhanh

- `SAMPLE_SIZE = 10000`
- `num_clients = 10`
- `num_rounds = 10-20`
- Mục đích: kiểm tra pipeline, không dùng làm kết quả cuối.

### Mốc khuyến nghị cho báo cáo tốt nghiệp

- `SAMPLE_SIZE = 50000` đến `100000`
- `num_clients = 10-20`
- `num_rounds = 50-100`
- `local_epochs = 3-5`
- `batch_size = 16` nếu máy 16GB RAM; `32` nếu máy khỏe hơn.

### Mốc mở rộng nếu còn thời gian/GPU

- `SAMPLE_SIZE = None` để dùng tối đa dữ liệu đã tải.
- Chỉ nên chạy qua đêm và sau khi pipeline 50k/100k đã ổn.

## 4. Xử Lý Dữ Liệu Amazon

Tải dữ liệu:

```powershell
PowerShell -ExecutionPolicy Bypass -File download_amazon_data.ps1
```

Xử lý dữ liệu:

```powershell
python src\data_generation\process_amazon_data.py
```

Output cần có:

```text
data/amazon_2023_processed/
├── client_0/data.pkl
├── client_1/data.pkl
├── ...
├── items_global.csv
└── users_global.csv
```

Nếu muốn tăng dữ liệu, chỉnh trong `src/data_generation/process_amazon_data.py`:

```python
SAMPLE_SIZE = 50000  # hoặc 100000, hoặc None
NUM_CLIENTS = 10     # có thể tăng 20 nếu máy chịu được
ALPHA = 0.5          # Non-IID mức vừa
```

## 5. Cấu Hình Training

File chính:

```text
configs/config.yaml
```

Cấu hình khuyến nghị cho máy 16GB RAM:

```yaml
federated:
  num_clients: 10
  num_rounds: 100
  fraction_fit: 0.2
  fraction_evaluate: 0.5
  min_fit_clients: 2
  min_evaluate_clients: 5

training:
  batch_size: 16
  local_epochs: 5
  learning_rate: 0.0001
  gradient_clip: 1.0
```

Nếu Ray/Windows bị crash:

- Giảm `num_rounds` để test.
- Giảm `batch_size` xuống `8`.
- Giảm `fraction_fit` xuống `0.2`.
- Đóng app nặng trước khi train.

## 6. Training Model

Chạy trực tiếp:

```powershell
python src\training\federated_training_pipeline.py
```

Hoặc dùng wrapper:

```powershell
python run_pipeline.py --mode train
```

Output sau training:

```text
experiments/fedper_multimodal_v1/
├── models/global_model_final.pt
├── metrics/training_history.json
└── metrics/training_curves.png
```

Checklist sau training:

- Có file `global_model_final.pt`.
- Loss giảm qua các round.
- Không có NaN/Inf.
- Log báo đang dùng Amazon data, không phải synthetic fallback.

## 7. Evaluation Sau Khi Train

Chạy:

```powershell
python src\training\evaluate_federated_model.py
```

Hoặc:

```powershell
python run_pipeline.py --mode evaluate
```

Output:

```text
experiments/fedper_multimodal_v1/evaluation/
├── evaluation_report.json
├── client_results.csv
└── evaluation_results.png
```

Các chỉ số cần đưa vào luận văn:

- `mean_accuracy`
- `mean_loss`
- `precision`
- `recall`
- `ndcg@10`
- `mrr`

## 8. Các Bước Sau Khi Train

### Bước 1: Kiểm tra checkpoint

```powershell
Test-Path experiments\fedper_multimodal_v1\models\global_model_final.pt
```

### Bước 2: Chạy evaluation

```powershell
python src\training\evaluate_federated_model.py
```

### Bước 3: Chạy API

```powershell
python src\api\fastapi_app.py
```

Kiểm tra:

- `http://localhost:8000/health`
- `http://localhost:8000/docs`

API sẽ tự tìm checkpoint mới nhất trong `experiments/**/models/global_model_final.pt`.

### Bước 4: Chạy dashboard

Mở terminal khác:

```powershell
.\fed_rec_env\Scripts\Activate.ps1
streamlit run src\dashboard\explainable_ui.py
```

Dashboard gọi API tại:

```text
http://localhost:8000
```

## 9. Demo Kịch Bản Bảo Vệ

1. Mở dashboard.
2. Chọn user id.
3. Gọi recommendation.
4. Hiển thị top-k sản phẩm.
5. Bật phần explainability:
   - text contribution
   - image contribution
   - behavior contribution
6. Mở API docs để chứng minh backend thật.
7. Mở thư mục `experiments/` để chỉ checkpoint và report.

## 10. Kiểm Tra Dự Án Trước Khi Nộp

Chạy kiểm tra cú pháp:

```powershell
python -c "import ast, pathlib; [ast.parse(p.read_text(encoding='utf-8-sig')) for p in pathlib.Path('.').rglob('*.py') if 'fed_rec_env' not in str(p)] ; print('syntax ok')"
```

Chạy quick test nếu muốn kiểm tra pipeline nhỏ:

```powershell
python run_quick_test.py
```

Kiểm tra docs:

```powershell
Get-ChildItem *.md
```

Repo nên chỉ còn:

```text
README.md
PROJECT_GUIDE.md
```

## 11. Kết Quả Cần Chốt Cho Luận Văn

Trước khi viết phần thực nghiệm, hãy lưu lại:

- Config cuối cùng (`configs/config.yaml`).
- Dataset size (`SAMPLE_SIZE`, số clients, số interactions thực tế).
- Training time.
- Final loss.
- Mean accuracy.
- NDCG@10, MRR.
- Screenshot dashboard.
- Screenshot API `/docs` hoặc `/health`.
- File checkpoint và evaluation report.

## 12. Nếu Còn 3 Tuần

### Tuần 1

- Chạy ổn pipeline với 10k/50k.
- Fix lỗi data/model nếu có.
- Chốt config final.

### Tuần 2

- Train final 50k-100k.
- Chạy evaluation.
- Tạo bảng/biểu đồ cho luận văn.

### Tuần 3

- Polish dashboard/demo.
- Viết slide.
- Luyện kịch bản bảo vệ.
- Quay video demo backup.
