"""
Launcher API (tương thích hướng dẫn cũ gọi `python src/api/server.py`).

Ứng dụng thật: `src.api.fastapi_app:app` (FastAPI + Uvicorn).
Chạy từ thư mục gốc project:

  python src/api/server.py

Hoặc (khuyến nghị, rõ ràng hơn):

  python -m uvicorn src.api.fastapi_app:app --host 0.0.0.0 --port 8000

Auto-reload (dev only, dễ làm ngắt WebSocket): set FED_REC_API_RELOAD=1 rồi chạy lại.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import os

if __name__ == "__main__":
    import uvicorn

    from src.api.fastapi_app import get_uvicorn_server_kwargs

    print("=" * 70)
    print("FEDERATED MULTI-MODAL RECOMMENDATION API (via server.py)")
    print("=" * 70)
    print("Docs: http://localhost:8000/docs")
    _reload = os.environ.get("FED_REC_API_RELOAD", "").strip().lower() in ("1", "true", "yes")
    if _reload:
        print("Reload: ON (FED_REC_API_RELOAD=1) — WebSocket có thể ngắt khi file đổi.")
    else:
        print("Reload: OFF — bật auto-reload: set FED_REC_API_RELOAD=1")
    print("=" * 70)

    uvicorn.run("src.api.fastapi_app:app", **get_uvicorn_server_kwargs(reload=_reload))
