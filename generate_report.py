#!/usr/bin/env python3
"""
Tạo báo cáo đánh giá + biểu đồ sau khi train (evaluation_report.json, client_results.csv, evaluation_results.png).

Tài liệu cũ (vd. HUONG_DAN_CHAY_VSCODE.md) gọi `generate_report.py` — script này gọi
`src.training.evaluate_federated_model.main`.

Chạy từ thư mục gốc project, venv đã kích hoạt:

  python generate_report.py

Nên truyền **cùng config** với lúc train (vd. 40 client). Mặc định của evaluate là configs/config.yaml (10 client):

  python generate_report.py --config configs/config_multi_category.yaml --experiment-dir experiments/<tên_thư_mục_train>

Nếu không truyền --experiment-dir, evaluator chọn thư mục experiment mới nhất khớp fedper_* trong experiments/.

Nếu bạn chỉ chạy `python generate_report.py` với config.yaml nhưng dữ liệu có 40 file client_*/data.pkl, code evaluator sẽ **tự tăng** số client lên 40 khi tạo report (tránh chỉ 10 dòng trong CSV).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Đảm bảo import src.*
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.evaluate_federated_model import main  # noqa: E402


if __name__ == "__main__":
    main()
