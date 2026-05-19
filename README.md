# Federated Multi-Modal Recommendation System

Hệ thống đề xuất đa phương thức sử dụng **Federated Learning (FedPer)** trên dữ liệu **Amazon Reviews 2023**. Dự án hướng tới đồ án tốt nghiệp hệ kỹ sư: có pipeline xử lý dữ liệu, training/evaluation, API và dashboard demo.

## Đọc Gì?

- Tài liệu chính: [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md)
- File này chỉ là tổng quan nhanh.

## Tính Năng Chính

- **FedPer**: shared layers được aggregate qua server, personal head giữ tại client.
- **Multi-modal input**: text embedding, image embedding, behavior features.
- **Amazon Reviews 2023**: xử lý thành dữ liệu federated theo client.
- **Evaluation**: xuất report theo client và summary cho luận văn.
- **Demo**: FastAPI + Streamlit dashboard.

## Chạy Nhanh

```powershell
python -m venv fed_rec_env
.\fed_rec_env\Scripts\Activate.ps1
pip install -r requirements.txt

PowerShell -ExecutionPolicy Bypass -File download_amazon_data.ps1
python src\data_generation\process_amazon_data.py
python src\training\federated_training_pipeline.py
python src\training\evaluate_federated_model.py
python src\api\fastapi_app.py
streamlit run src\dashboard\explainable_ui.py
```

## Cấu Trúc

```text
FederatedLearning/
├── README.md
├── PROJECT_GUIDE.md
├── requirements.txt
├── run_pipeline.py
├── configs/
│   ├── config.yaml
│   └── config_multi_category.yaml
├── src/
│   ├── api/
│   ├── dashboard/
│   ├── data_generation/
│   ├── data_processing/
│   ├── federated/
│   ├── models/
│   ├── training/
│   └── vector_db/
├── data/                 # Không commit: raw/processed Amazon data
└── experiments/          # Không commit: checkpoints, metrics, evaluation reports
```

## Output Quan Trọng

- Data federated: `data/amazon_2023_processed/client_*/data.pkl`
- Catalog demo: `data/amazon_2023_processed/items_global.csv`
- Users demo: `data/amazon_2023_processed/users_global.csv`
- Model: `experiments/<experiment_name>/models/global_model_final.pt`
- Evaluation: `experiments/<experiment_name>/evaluation/evaluation_report.json`

## Ghi Chú

- Cấu hình chính nằm ở `configs/config.yaml`.
- Nếu máy 16GB RAM, giữ `batch_size=16`, `fraction_fit=0.2` để giảm lỗi Ray/Windows.
- Toàn bộ hướng dẫn training đủ cho đồ án nằm trong `PROJECT_GUIDE.md`.
