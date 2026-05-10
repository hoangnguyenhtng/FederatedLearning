#!/usr/bin/env python3
"""
Kiểm tra chất lượng / thống kê dữ liệu Amazon Reviews 2023 cho MỘT PHẦN category
(Plan 1 — không cần nạp toàn bộ corpus vào RAM).

Chạy từ thư mục gốc project:
  python src/data_generation/inspect_amazon_subset.py --config configs/config_multi_category.yaml
  python src/data_generation/inspect_amazon_subset.py --quick-sample 20000   # nhanh, chỉ N dòng đầu

Mặc định raw: một lượt quét cả file — histogram rating đầy đủ, độ dài title+text (khớp bộ lọc processor).

Chế độ:
  --mode raw        Chỉ raw JSONL (+ meta)
  --mode processed  Chỉ client_*/data.pkl
  --mode all        Cả hai (mặc định)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def load_yaml_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_lines(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def sample_raw_jsonl(
    path: Path,
    max_lines: int,
    min_review_chars: int = 10,
) -> Dict[str, Any]:
    """Đọc tối đa max_lines dòng đầu để thống kê (không load cả file vào list)."""
    ratings: List[float] = []
    text_lens: List[int] = []
    missing_text = 0
    short_text = 0
    empty_images = 0
    with_images = 0
    lines_used = 0

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if lines_used >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            lines_used += 1
            r = obj.get("rating")
            if r is not None:
                ratings.append(float(r))
            text = (obj.get("text") or "") or ""
            if not str(text).strip():
                missing_text += 1
            else:
                L = len(str(text))
                text_lens.append(L)
                if L < min_review_chars:
                    short_text += 1
            imgs = obj.get("images")
            if isinstance(imgs, list):
                if len(imgs) == 0:
                    empty_images += 1
                else:
                    with_images += 1

    return {
        "lines_sampled": lines_used,
        "rating_counts": dict(Counter(round(x) for x in ratings)),
        "rating_mean": sum(ratings) / len(ratings) if ratings else None,
        "text_len_mean": sum(text_lens) / len(text_lens) if text_lens else None,
        "text_len_min": min(text_lens) if text_lens else None,
        "text_len_max": max(text_lens) if text_lens else None,
        "rows_missing_text": missing_text,
        "rows_short_text_lt_min": short_text,
        "min_review_chars": min_review_chars,
        "rows_empty_images": empty_images,
        "rows_with_images": with_images,
    }


def scan_jsonl_stream(
    path: Path,
    min_review_chars: int = 10,
    max_physical_lines: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Một lượt đọc file: histogram rating (1–5), độ dài title+text (cùng chuẩn với processor),
    ảnh trong trường review. Tiết kiệm RAM (không giữ danh sách).
    """
    rating_counts: Counter = Counter()
    json_errors = 0
    empty_lines = 0
    rows_missing_body = 0
    rows_short = 0
    empty_images = 0
    with_images = 0
    n_len = 0
    mean_len = 0.0
    m2 = 0.0
    min_len: Optional[int] = None
    max_len: Optional[int] = None
    physical_lines = 0

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if max_physical_lines is not None and physical_lines >= max_physical_lines:
                break
            physical_lines += 1
            line = line.strip()
            if not line:
                empty_lines += 1
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                json_errors += 1
                continue

            r = obj.get("rating")
            if r is not None:
                try:
                    ri = int(round(float(r)))
                    if 1 <= ri <= 5:
                        rating_counts[ri] += 1
                except (TypeError, ValueError):
                    pass

            title = (obj.get("title") or "") or ""
            text = (obj.get("text") or "") or ""
            combined = f"{title} {text}".strip()
            if not combined:
                rows_missing_body += 1
            else:
                L = len(combined)
                n_len += 1
                if min_len is None or L < min_len:
                    min_len = L
                if max_len is None or L > max_len:
                    max_len = L
                delta = L - mean_len
                mean_len += delta / n_len
                delta2 = L - mean_len
                m2 += delta * delta2
                if L < min_review_chars:
                    rows_short += 1

            imgs = obj.get("images")
            if isinstance(imgs, list):
                if len(imgs) == 0:
                    empty_images += 1
                else:
                    with_images += 1

    variance = (m2 / n_len) if n_len > 1 else 0.0
    total_rated = sum(rating_counts.values())
    return {
        "scan_mode": "full_file" if max_physical_lines is None else f"first_{max_physical_lines}_lines",
        "physical_lines": physical_lines,
        "empty_lines": empty_lines,
        "json_decode_errors": json_errors,
        "rating_counts_star_1_to_5": {str(k): int(v) for k, v in sorted(rating_counts.items())},
        "rows_with_rating_1_to_5": total_rated,
        "combined_title_text_len_mean": mean_len if n_len else None,
        "combined_title_text_len_std": variance ** 0.5 if n_len > 1 else None,
        "combined_title_text_len_min": min_len,
        "combined_title_text_len_max": max_len,
        "rows_nonempty_combined": n_len,
        "rows_missing_title_and_text": rows_missing_body,
        "rows_combined_shorter_than_min": rows_short,
        "min_review_chars_used": min_review_chars,
        "rows_empty_images_field": empty_images,
        "rows_with_images_in_review": with_images,
    }


def inspect_meta_jsonl(meta_path: Path, max_lines: int) -> Dict[str, Any]:
    if not meta_path.is_file():
        return {"exists": False}
    total = count_lines(meta_path)
    keys_sample: Optional[List[str]] = None
    n = 0
    with open(meta_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if n >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if keys_sample is None:
                keys_sample = sorted(obj.keys())
            n += 1
    return {"exists": True, "line_count": total, "sample_keys": keys_sample, "sample_lines_read": n}


def build_client_category_map(categories: List[str], clients_per_category: int) -> Dict[int, str]:
    m: Dict[int, str] = {}
    cid = 0
    for cat in categories:
        for _ in range(clients_per_category):
            m[cid] = cat
            cid += 1
    return m


def check_tensor_shapes_row(row: Any, expected_text: int, expected_image: int, expected_behavior: int) -> Dict[str, Any]:
    import numpy as np

    def _shape(x: Any) -> Tuple:
        if hasattr(x, "shape"):
            return tuple(x.shape)
        if isinstance(x, (list, tuple)):
            a = np.asarray(x)
            return tuple(a.shape)
        return ()

    te = row.get("text_embedding", row.get("text_embeddings"))
    ie = row.get("image_embedding", row.get("image_embeddings"))
    bf = row.get("behavior_features")

    return {
        "text_embedding_shape": _shape(te),
        "image_embedding_shape": _shape(ie),
        "behavior_features_shape": _shape(bf),
        "text_ok": _shape(te)[-1] == expected_text if _shape(te) else False,
        "image_ok": _shape(ie)[-1] == expected_image if _shape(ie) else False,
        "behavior_ok": _shape(bf)[-1] == expected_behavior if _shape(bf) else False,
    }


def inspect_processed_client(
    pkl_path: Path,
    client_id: int,
    category_guess: str,
    model_cfg: dict,
    max_rows_scan: int,
) -> Dict[str, Any]:
    import numpy as np
    import pandas as pd

    df = pd.read_pickle(pkl_path)
    n = len(df)
    text_dim = int(model_cfg.get("text_embedding_dim", 384))
    image_dim = int(model_cfg.get("image_embedding_dim", 2048))
    behavior_dim = int(model_cfg.get("behavior_dim", model_cfg.get("behavior_embedding_dim", 32)))

    sample_idx = [0, n // 2, n - 1] if n >= 3 else list(range(n))
    shape_checks = []
    for i in sample_idx:
        shape_checks.append(check_tensor_shapes_row(df.iloc[i], text_dim, image_dim, behavior_dim))

    # Rating / label (chỉ trên prefix để tiết kiệm)
    tail = min(n, max_rows_scan)
    sub = df.iloc[:tail]
    rating_counts = {}
    label_counts = {}
    if "rating" in sub.columns:
        rating_counts = sub["rating"].value_counts().sort_index().to_dict()
    if "label" in sub.columns:
        label_counts = sub["label"].value_counts().sort_index().to_dict()

    cat_col = sub["category"].value_counts().to_dict() if "category" in sub.columns else {}

    # NaN trong embedding mẫu (một vài dòng)
    nan_report = []
    for i in sample_idx[:2]:
        row = df.iloc[i]
        for name, col in [
            ("text_embedding", "text_embedding"),
            ("image_embedding", "image_embedding"),
            ("behavior_features", "behavior_features"),
        ]:
            if col not in row:
                nan_report.append({col: "missing"})
                continue
            v = row[col]
            arr = np.asarray(v, dtype=np.float64)
            nan_report.append(
                {
                    "column": col,
                    "nan_count": int(np.isnan(arr).sum()),
                    "inf_count": int(np.isinf(arr).sum()),
                }
            )

    users = int(sub["user_id"].nunique()) if "user_id" in sub.columns else None

    return {
        "client_id": client_id,
        "category_expected": category_guess,
        "num_rows": n,
        "rows_scanned": tail,
        "num_unique_users_in_scan": users,
        "category_value_counts_in_scan": cat_col,
        "rating_distribution_in_scan": {str(k): int(v) for k, v in rating_counts.items()},
        "label_distribution_in_scan": {str(k): int(v) for k, v in label_counts.items()},
        "shape_checks_sample_rows": shape_checks,
        "nan_inf_sample": nan_report,
        "columns": list(df.columns),
    }


def main() -> int:
    root = _project_root()
    parser = argparse.ArgumentParser(description="Inspect Amazon subset (raw + processed)")
    parser.add_argument("--config", type=str, default="configs/config_multi_category.yaml")
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Lọc category (mặc định: lấy từ config). Ví dụ: All_Beauty Video_Games",
    )
    parser.add_argument("--mode", choices=("raw", "processed", "all"), default="all")
    parser.add_argument(
        "--quick-sample",
        type=int,
        default=0,
        help="Nếu > 0: chỉ phân tích N dòng vật lý đầu (nhanh). 0 = quét cả file (mặc định).",
    )
    parser.add_argument("--meta-max-sample", type=int, default=5, help="Số dòng meta đọc để lấy keys mẫu")
    parser.add_argument("--processed-max-rows", type=int, default=100_000, help="Giới hạn số hàng quét mỗi client PKL")
    parser.add_argument("--json-out", type=str, default=None, help="Ghi báo cáo JSON (tùy chọn)")
    args = parser.parse_args()

    cfg_path = root / args.config
    if not cfg_path.is_file():
        print(f"❌ Không tìm thấy config: {cfg_path}")
        return 1

    cfg = load_yaml_config(cfg_path)
    all_categories: List[str] = list(cfg.get("categories") or [])
    # Raw: dùng đúng tên category cho tên file; có thể truyền --categories ngoài config
    if args.categories:
        categories = list(args.categories)
    else:
        categories = list(all_categories)

    if not categories:
        print("❌ Không có category nào để kiểm tra (config thiếu `categories` hoặc chưa truyền --categories).")
        return 1

    if all_categories:
        unknown = [c for c in categories if c not in all_categories]
        if unknown:
            print(f"⚠️  Category không nằm trong config.categories: {unknown}")
            print(f"   (Vẫn thử đọc raw {unknown} nếu có file .jsonl tương ứng.)")

    paths = cfg.get("paths") or {}
    raw_dir = root / paths.get("data_raw", "data/raw/amazon_2023")
    processed_dir = root / paths.get("data_processed", "data/processed/multi_category")
    fed = cfg.get("federated") or {}
    dg = cfg.get("data_generation") or {}
    clients_per_category = int(dg.get("clients_per_category", 10))
    map_categories = all_categories if all_categories else categories
    num_clients_cfg = int(fed.get("num_clients", len(map_categories) * clients_per_category))
    client_cat = build_client_category_map(map_categories, clients_per_category)

    model_cfg = cfg.get("model") or {}
    proc_cfg = cfg.get("processing") or {}
    min_review_chars = int(proc_cfg.get("min_review_length", 10))

    report: Dict[str, Any] = {
        "config": str(cfg_path),
        "categories_inspected": categories,
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
        "raw": {},
        "processed": {},
    }

    print("=" * 70)
    print("AMAZON SUBSET DATA INSPECTION (Plan 1)")
    print("=" * 70)
    print(f"Config: {cfg_path}")
    print(f"Categories: {categories}")
    print(f"Raw dir: {raw_dir}")
    print(f"Processed dir: {processed_dir}")
    print("")

    if args.mode in ("raw", "all"):
        print("--- RAW JSONL ---")
        for cat in categories:
            rev = raw_dir / f"{cat}.jsonl"
            meta = raw_dir / f"meta_{cat}.jsonl"
            block: Dict[str, Any] = {"reviews_path": str(rev), "meta_path": str(meta)}
            if not rev.is_file():
                block["reviews"] = {"exists": False, "error": "file not found"}
                print(f"\n[{cat}] ❌ Thiếu file review: {rev}")
            else:
                max_lines = args.quick_sample if args.quick_sample > 0 else None
                if max_lines:
                    print(f"\n[{cat}] Quick sample: {max_lines:,} dòng đầu (min_review_length={min_review_chars})...")
                    stats = scan_jsonl_stream(rev, min_review_chars=min_review_chars, max_physical_lines=max_lines)
                    block["reviews"] = {"exists": True, "stream_stats": stats}
                else:
                    print(f"\n[{cat}] Quét một lượt cả file (có thể vài phút nếu file rất lớn)...")
                    print(f"   Tiêu chí độ dài: title+text, min_review_length={min_review_chars} (khớp processor)")
                    stats = scan_jsonl_stream(rev, min_review_chars=min_review_chars, max_physical_lines=None)
                    block["reviews"] = {"exists": True, "stream_stats": stats}
                rc = stats.get("rating_counts_star_1_to_5") or {}
                print(f"   Dòng vật lý (trong phạm vi quét): {stats['physical_lines']:,}")
                print(f"   Rating ★ (trong phạm vi quét): {rc}")
                print(
                    f"   Độ dài title+text (mean / std / min / max): "
                    f"{stats.get('combined_title_text_len_mean')} / "
                    f"{stats.get('combined_title_text_len_std')} / "
                    f"{stats.get('combined_title_text_len_min')} / "
                    f"{stats.get('combined_title_text_len_max')}"
                )
                print(
                    f"   Thiếu title+text: {stats['rows_missing_title_and_text']:,} | "
                    f"Ngắn hơn {min_review_chars} ký tự: {stats['rows_combined_shorter_than_min']:,}"
                )
                print(
                    f"   Ảnh (trường review) rỗng / có ảnh: "
                    f"{stats['rows_empty_images_field']:,} / {stats['rows_with_images_in_review']:,}"
                )
                if stats.get("json_decode_errors", 0):
                    print(f"   ⚠️  JSON lỗi: {stats['json_decode_errors']:,}")

            block["meta"] = inspect_meta_jsonl(meta, max_lines=args.meta_max_sample)
            if block["meta"].get("exists"):
                print(f"   Meta: {block['meta']['line_count']:,} dòng | keys mẫu: {block['meta'].get('sample_keys')}")
            else:
                print(f"   Meta: ❌ Không có {meta}")

            report["raw"][cat] = block

    if args.mode in ("processed", "all"):
        print("\n--- PROCESSED PKL (client_*/data.pkl) ---")
        if not processed_dir.is_dir():
            print(f"❌ Thư mục processed không tồn tại: {processed_dir}")
        else:
            found_any = False
            # Chỉ các client thuộc category được chọn
            wanted_clients = [
                cid for cid, c in client_cat.items() if c in categories and cid < num_clients_cfg
            ]
            if not wanted_clients:
                wanted_clients = list(range(num_clients_cfg))

            for cid in wanted_clients:
                cat_guess = client_cat.get(cid, "?")
                if cat_guess not in categories:
                    continue
                pkl = processed_dir / f"client_{cid}" / "data.pkl"
                if not pkl.is_file():
                    continue
                found_any = True
                print(f"\n[client_{cid}] ({cat_guess}) {pkl}")
                try:
                    info = inspect_processed_client(
                        pkl, cid, cat_guess, model_cfg, max_rows_scan=args.processed_max_rows
                    )
                    report["processed"][f"client_{cid}"] = info
                    print(f"   Hàng: {info['num_rows']:,} | users (trong scan): {info['num_unique_users_in_scan']}")
                    print(f"   Label (scan): {info['label_distribution_in_scan']}")
                    print(f"   Shape checks: {info['shape_checks_sample_rows']}")
                    print(f"   NaN/Inf mẫu: {info['nan_inf_sample']}")
                except Exception as e:
                    print(f"   ❌ Lỗi đọc PKL: {e}")
                    report["processed"][f"client_{cid}"] = {"error": str(e)}

            if not found_any:
                print(
                    "\n⚠️  Không thấy file data.pkl nào cho các category đã chọn.\n"
                    "   Metadata có thể còn nhưng client_* đã bị xóa — cần chạy lại bộ xử lý multi-category.\n"
                    f"   python src/data_generation/process_amazon_multi_category.py --config {args.config}"
                )

    if args.json_out:
        out_path = root / args.json_out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n✅ Đã ghi JSON: {out_path}")

    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
