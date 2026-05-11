## PROJECT GUIDE (1 file) — Federated Multi-Modal Recommendation (FedPer + Amazon 2023)

Tài liệu này thay cho phần lớn các “note” rời rạc. Mục tiêu: **1 luồng chạy end-to-end** + **những điểm cần biết để bảo vệ**.

---

## 1) TL;DR — chạy end-to-end (Windows)

```powershell
# 0) venv + dependencies
python -m venv fed_rec_env
.\fed_rec_env\Scripts\Activate.ps1
pip install -r requirements.txt

# 1) Download + process Amazon → tạo data federated clients + catalog
PowerShell -ExecutionPolicy Bypass -File download_amazon_data.ps1
python src\data_generation\process_amazon_data.py

# 2) Train Federated (FedPer)
python src\training\federated_training_pipeline.py

# 3) Evaluate (Amazon)
python src\training\evaluate_federated_model.py --config configs\config.yaml --amazon_dir data\amazon_2023_processed

# 4) Run API + Dashboard demo
python src\api\fastapi_app.py
streamlit run src\dashboard\explainable_ui.py
```

---

## 2) Kiến trúc hệ thống (giữ nguyên FedPer)

- **Data**: Amazon Reviews 2023 → xử lý thành `data/amazon_2023_processed/client_*/data.pkl`
- **Model**:
  - `MultiModalEncoder` (text 384, image 2048, behavior 32) → fused embedding (384)
  - `FedPerRecommender`:
    - **Shared**: `multimodal_encoder` + `shared_base`
    - **Personal**: `personal_head` (không upload lên server)
- **Federated Learning**: Flower simulation (`fl.simulation.start_simulation`)
  - Server strategy: `FedPerStrategy` (aggregate shared params)
  - Client: `FedPerClient` (train local, upload shared only)

---

## 3) Dữ liệu Amazon: output bạn cần để demo “giống thương mại điện tử”

Sau khi chạy `process_amazon_data.py`, bạn sẽ có:
- **Federated clients**:
  - `data/amazon_2023_processed/client_0/data.pkl` … `client_{N-1}/data.pkl`
- **Catalog cho website/API**:
  - `data/amazon_2023_processed/items_global.csv`
  - `data/amazon_2023_processed/users_global.csv`

Ghi chú:
- Nếu bạn bật `skip_image_download=True`, ảnh sẽ dùng **dummy embedding nhưng deterministic theo item** (không random mỗi lần chạy) → đủ ổn cho demo/đồ án.

---

## 4) Train: checkpoint & artifacts

Training chạy từ:
- `src/training/federated_training_pipeline.py`

Artifacts:
- Checkpoint: `experiments/<experiment_name>/models/global_model_final.pt`
- Metrics/history: `experiments/<experiment_name>/metrics/training_history.json`

`experiment_name` lấy từ `configs/config.yaml` → `experiment.name`.

---

## 5) Evaluate (Amazon)

```powershell
python src\training\evaluate_federated_model.py --config configs\config.yaml --amazon_dir data\amazon_2023_processed
```

Outputs:
- `experiments/<experiment_name>/evaluation/amazon_client_results.csv`
- `experiments/<experiment_name>/evaluation/amazon_evaluation_summary.json`

---

## 6) API + Demo UI

### 6.1 FastAPI

```powershell
python src\api\fastapi_app.py
```

- Docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`
- API ưu tiên Amazon processed + tự load checkpoint nếu có.

### 6.2 Streamlit dashboard

```powershell
streamlit run src\dashboard\explainable_ui.py
```

---

## 7) “Giống thực tế” (để bảo vệ chắc)

- **Không dùng random feature** trong demo: API đã chuyển sang **deterministic** và ưu tiên dùng **embeddings thật** từ Amazon processed.
- **Kịch bản e-commerce**: item catalog + search/filter/pagination + recommend + explain.
- **Giải thích**: dùng fusion weights (text/image/behavior) từ model.

---

## 8) Troubleshooting (ngắn gọn)

- **Ray/Windows crash**: giảm `federated.num_clients`, giảm `training.batch_size`, đóng bớt app khác; chạy lại.
- **NaN/Inf**: data Amazon đã được `nan_to_num`, nếu vẫn gặp hãy giảm `learning_rate` và bật `training.gradient_clip`.
- **Import/deps**: chạy đúng venv và `pip install -r requirements.txt`.

---

## 9) Ghi chú

Repo đã được “dọn note” để gọn: tài liệu chính là file này. Nếu cần bổ sung thêm mục nào (thesis outline, checklist bảo vệ, v.v.) thì thêm trực tiếp vào `PROJECT_GUIDE.md`.

