"""
Federated Learning — Full Demo Runner
Chạy toàn bộ pipeline từ đầu đến cuối cho bảo vệ đồ án tốt nghiệp.

Tối ưu cho: Dell G15 5511 — RTX 3050, 16GB RAM

Usage:
    python run_full_demo.py              # Chạy full pipeline
    python run_full_demo.py --skip-train # Bỏ qua training (nếu đã có model)
    python run_full_demo.py --demo-only  # Chỉ mở demo website
"""

import sys
import os
import time
import argparse
import subprocess
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

class C:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"

def header(text):
    print(f"\n{C.BOLD}{C.CYAN}{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}{C.END}\n")

def success(text):
    print(f"{C.GREEN}✅ {text}{C.END}")

def warn(text):
    print(f"{C.YELLOW}⚠️  {text}{C.END}")

def error(text):
    print(f"{C.RED}❌ {text}{C.END}")

def info(text):
    print(f"{C.CYAN}ℹ️  {text}{C.END}")


def check_environment():
    header("BƯỚC 1: Kiểm Tra Môi Trường")
    
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    success(f"Python: {py_ver}")
    
    packages = {
        'torch': 'PyTorch',
        'flwr': 'Flower (Federated Learning)',
        'sentence_transformers': 'SentenceTransformers',
        'fastapi': 'FastAPI',
        'opacus': 'Opacus (Differential Privacy)',
    }
    
    all_ok = True
    for pkg, name in packages.items():
        try:
            mod = __import__(pkg)
            ver = getattr(mod, '__version__', '?')
            success(f"{name}: {ver}")
        except ImportError:
            error(f"{name}: CHƯA CÀI ĐẶT → pip install {pkg}")
            all_ok = False
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            success(f"GPU: {gpu_name} ({vram:.1f}GB VRAM)")
        else:
            warn("GPU: Không có CUDA → sẽ dùng CPU (chậm hơn nhưng vẫn chạy được)")
    except Exception:
        warn("Không thể kiểm tra GPU")
    
    if not all_ok:
        error("Một số packages chưa cài. Chạy: pip install -r requirements.txt")
        sys.exit(1)
    
    return True


def generate_data():
    header("BƯỚC 2: Tạo Dữ Liệu Demo")
    
    data_dir = PROJECT_ROOT / "data" / "amazon_2023_processed"
    client_0 = data_dir / "client_0" / "data.pkl"
    
    if client_0.exists():
        success(f"Dữ liệu đã có tại: {data_dir}")
        num_clients = len(list(data_dir.glob("client_*")))
        info(f"Số clients: {num_clients}")
        return True
    
    info("Chưa có dữ liệu → Đang tạo demo data...")
    
    try:
        from src.data_generation.generate_demo_data import generate_demo_data
        generate_demo_data()
        success("Tạo demo data thành công!")
        return True
    except Exception as e:
        error(f"Tạo data thất bại: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_training(num_rounds=5):
    header(f"BƯỚC 3: Huấn Luyện Federated Learning ({num_rounds} rounds)")
    
    experiments_dir = PROJECT_ROOT / "experiments"
    checkpoints = list(experiments_dir.glob("**/global_model_final.pt"))
    
    if checkpoints:
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        age_hours = (time.time() - latest.stat().st_mtime) / 3600
        success(f"Model checkpoint đã có: {latest}")
        info(f"Được tạo cách đây: {age_hours:.1f} giờ")
        
        if age_hours < 24:
            info("Model còn mới → bỏ qua training")
            return True
        else:
            info("Model cũ → sẽ train lại")
    
    info(f"Bắt đầu training {num_rounds} rounds...")
    info("Cấu hình: 10 clients, batch_size=32, local_epochs=3")
    
    try:
        import yaml
        import torch
        
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        config['federated']['num_rounds'] = num_rounds
        config['federated']['min_fit_clients'] = 3
        config['federated']['min_evaluate_clients'] = 3
        config['federated']['min_available_clients'] = 5
        config['training']['local_epochs'] = 3
        config['training']['batch_size'] = 32
        
        from src.training.federated_training_pipeline import FederatedTrainingPipeline
        
        pipeline = FederatedTrainingPipeline(config=config)
        history = pipeline.train()
        
        success(f"Training hoàn thành! ({num_rounds} rounds)")
        return True
        
    except Exception as e:
        error(f"Training thất bại: {e}")
        import traceback
        traceback.print_exc()
        warn("Bạn vẫn có thể mở demo website (dữ liệu hardcoded)")
        return False


def run_evaluation():
    header("BƯỚC 4: Đánh Giá Model")
    
    try:
        from src.training.evaluate_federated_model import FederatedEvaluator
        
        evaluator = FederatedEvaluator(config_path="configs/config.yaml")
        model = evaluator.load_model()
        results = evaluator.evaluate_all_clients(model)
        
        if results:
            report = evaluator.generate_report(results)
            evaluator.visualize_results(results)
            
            overall = report.get("overall_metrics", {})
            success(f"Accuracy: {overall.get('mean_accuracy', 0):.4f} (±{overall.get('std_accuracy', 0):.4f})")
            success(f"Loss: {overall.get('mean_loss', 0):.4f}")
            return True
        else:
            warn("Không có kết quả evaluation")
            return False
            
    except Exception as e:
        warn(f"Evaluation bị lỗi (không nghiêm trọng): {e}")
        return False


def open_demo():
    header("BƯỚC 5: Mở Demo Website")
    
    demo_path = PROJECT_ROOT / "demo" / "index.html"
    
    if not demo_path.exists():
        error(f"Demo không tìm thấy: {demo_path}")
        return False
    
    demo_url = demo_path.as_uri()
    info(f"Mở: {demo_url}")
    webbrowser.open(demo_url)
    success("Demo đã mở trong trình duyệt!")
    return True


def launch_api():
    header("BƯỚC 6: Khởi Động API Server")
    
    info("API sẽ chạy ở background tại http://localhost:8000")
    info("Nhấn Ctrl+C để dừng API server")
    
    try:
        subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "src.api.fastapi_app:app",
             "--host", "0.0.0.0", "--port", "8000", "--reload"],
            cwd=str(PROJECT_ROOT),
        )
        success("API server đã khởi động tại http://localhost:8000")
        info("Endpoints: /health, /recommend, /stats, /metrics")
        return True
    except Exception as e:
        warn(f"Không thể khởi động API: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Federated Learning — Full Demo Runner cho Đồ Án Tốt Nghiệp"
    )
    parser.add_argument("--skip-train", action="store_true",
                        help="Bỏ qua bước training (dùng model cũ)")
    parser.add_argument("--demo-only", action="store_true",
                        help="Chỉ mở demo website")
    parser.add_argument("--with-api", action="store_true",
                        help="Khởi động API server cùng demo")
    parser.add_argument("--rounds", type=int, default=5,
                        help="Số rounds training (default: 5)")
    args = parser.parse_args()

    header("FEDERATED LEARNING — ĐỒ ÁN TỐT NGHIỆP")
    print(f"  Project: {PROJECT_ROOT}")
    print(f"  Mode: {'Demo only' if args.demo_only else 'Full pipeline'}")
    print()

    if args.demo_only:
        open_demo()
        if args.with_api:
            launch_api()
        return

    # Full pipeline
    start_time = time.time()

    # Step 1: Check environment
    check_environment()

    # Step 2: Generate data
    if not generate_data():
        error("Không thể tạo dữ liệu. Dừng lại.")
        sys.exit(1)

    # Step 3: Training
    if not args.skip_train:
        run_training(num_rounds=args.rounds)
    else:
        info("Bỏ qua training (--skip-train)")

    # Step 4: Evaluation
    run_evaluation()

    # Step 5: Open demo
    open_demo()

    # Step 6: API (if requested)
    if args.with_api:
        launch_api()

    # Summary
    elapsed = time.time() - start_time
    header("HOÀN THÀNH!")
    print(f"  ⏱️  Tổng thời gian: {elapsed:.1f}s ({elapsed/60:.1f} phút)")
    print()
    print(f"  📊 Demo website: {PROJECT_ROOT / 'demo' / 'index.html'}")
    if args.with_api:
        print(f"  🌐 API server: http://localhost:8000")
        print(f"  📝 API docs: http://localhost:8000/docs")
    print()
    print(f"  {C.BOLD}Hướng dẫn bảo vệ đồ án:{C.END}")
    print(f"  1. Tab 'Tổng Quan' — giới thiệu hệ thống")
    print(f"  2. Tab 'Bảo Mật' — demo Privacy Inspector (điểm nhấn!)")
    print(f"  3. Tab 'Multi-Modal' — giải thích Adaptive Fusion")
    print(f"  4. Tab 'Kết Quả' — biểu đồ training loss")
    print(f"  5. Tab 'Kiến Trúc' — cấu trúc hệ thống + tech stack")
    print()


if __name__ == "__main__":
    main()
