"""
Script tổng hợp để chạy toàn bộ pipeline Federated Learning
Sử dụng cho tốt nghiệp
"""

import sys
from pathlib import Path
import argparse
import yaml
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Federated Learning Pipeline')
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'evaluate', 'api', 'dashboard', 'all'],
        help='Mode to run: train, evaluate, api, dashboard, or all'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("="*70)
    logger.info("FEDERATED LEARNING PIPELINE")
    logger.info("="*70)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Config: {config_path}")
    logger.info("")
    
    if args.mode == 'train' or args.mode == 'all':
        logger.info("🚀 Starting Training...")
        from src.training import federated_training_pipeline as ftp
        ftp.main(config_path=str(config_path.resolve()))
    
    if args.mode == 'evaluate' or args.mode == 'all':
        logger.info("📊 Starting Evaluation...")
        from src.training.evaluate_federated_model import main as eval_main
        eval_main()
    
    if args.mode == 'api' or args.mode == 'all':
        logger.info("🌐 Starting API Server...")
        api_cfg = config.get('api') or {}
        host = api_cfg.get('host', '0.0.0.0')
        port = api_cfg.get('port', 8000)
        logger.info(f"API will run at http://localhost:{port}")
        logger.info("Press Ctrl+C to stop")
        import uvicorn
        from src.api.fastapi_app import app
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=api_cfg.get('reload', False)
        )
    
    if args.mode == 'dashboard' or args.mode == 'all':
        dash_cfg = config.get('dashboard') or {}
        port = dash_cfg.get('port', 8501)
        logger.info("📊 Starting Dashboard...")
        logger.info(f"Dashboard will run at http://localhost:{port}")
        logger.info("Press Ctrl+C to stop")
        import subprocess
        subprocess.run([
            'streamlit', 'run',
            'src/dashboard/explainable_ui.py',
            '--server.port', str(port)
        ])
    
    logger.info("="*70)
    logger.info("✅ Pipeline completed!")
    logger.info("="*70)


if __name__ == "__main__":
    main()
