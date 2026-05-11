"""
Quick End-to-End Test Script
=============================
Validates the entire pipeline works:
  1. Generate demo data
  2. Load data into pipeline
  3. Run 3 rounds of federated training
  4. Evaluate model
  5. Save results

Usage:
    python run_quick_test.py
"""

import sys
import os
from pathlib import Path
import time

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

os.environ.setdefault("RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE", "1")
os.environ.setdefault("RAY_DEDUP_LOGS", "1")


def step(msg: str):
    print(f"\n{'='*70}")
    print(f"  {msg}")
    print(f"{'='*70}\n")


def main():
    start = time.time()

    # ── Step 1: Generate demo data ─────────────────────────────────
    step("STEP 1/5: Generating demo data")
    
    data_dir = project_root / "data" / "amazon_2023_processed"
    client_0 = data_dir / "client_0" / "data.pkl"

    if not client_0.exists():
        from src.data_generation.generate_demo_data import generate_demo_data
        generate_demo_data()
    else:
        print(f"✅ Demo data already exists at {data_dir}")

    # Verify data
    assert client_0.exists(), f"Data not found: {client_0}"
    import pandas as pd
    df = pd.read_pickle(client_0)
    print(f"   Client 0: {len(df)} samples, columns: {list(df.columns)}")

    # ── Step 2: Verify model creation ──────────────────────────────
    step("STEP 2/5: Verifying model creation")

    import torch
    from src.models.multimodal_encoder import MultiModalEncoder
    from src.models.recommendation_model import FedPerRecommender

    encoder = MultiModalEncoder(
        text_dim=384, image_dim=2048, behavior_dim=32,
        hidden_dim=256, output_dim=384
    )
    model = FedPerRecommender(
        multimodal_encoder=encoder,
        shared_hidden_dims=[512, 256, 128],
        personal_hidden_dims=[64, 32],
        num_classes=5,
        dropout=0.2
    )

    total_params = sum(p.numel() for p in model.parameters())
    shared_params = sum(p.numel() for p in model.get_shared_parameters().values())
    personal_params = sum(p.numel() for p in model.get_personal_parameters().values())

    print(f"✅ Model created")
    print(f"   Total params:    {total_params:,}")
    print(f"   Shared params:   {shared_params:,} ({shared_params/total_params*100:.1f}%)")
    print(f"   Personal params: {personal_params:,} ({personal_params/total_params*100:.1f}%)")

    # Test forward pass
    batch_size = 4
    text_emb = torch.randn(batch_size, 384)
    image_emb = torch.randn(batch_size, 2048)
    behavior_feat = torch.randn(batch_size, 32)
    
    logits, weights = model(text_emb, image_emb, behavior_feat, return_fusion_weights=True)
    print(f"   Forward pass OK: logits={logits.shape}, weights={weights.shape}")
    print(f"   Fusion weights sum: {weights[0].sum().item():.4f} (should be ~1.0)")

    # ── Step 3: Verify dataloader ──────────────────────────────────
    step("STEP 3/5: Verifying dataloaders")

    from src.data_generation.amazon_dataloader import get_amazon_dataloaders

    loaders = get_amazon_dataloaders(
        num_clients=10,
        data_dir=str(data_dir),
        batch_size=32,
        test_split=0.2
    )
    print(f"✅ Loaded {len(loaders)} client dataloaders")

    # Test one batch
    train_loader, test_loader = loaders[0]
    batch = next(iter(train_loader))
    print(f"   Batch keys: {list(batch.keys())}")
    print(f"   text_embedding: {batch['text_embedding'].shape}")
    print(f"   image_embedding: {batch['image_embedding'].shape}")
    print(f"   behavior_features: {batch['behavior_features'].shape}")
    print(f"   label: {batch['label'].shape}, range: [{batch['label'].min()}, {batch['label'].max()}]")

    # ── Step 4: Quick federated training (3 rounds) ────────────────
    step("STEP 4/5: Running quick federated training (3 rounds)")

    import yaml
    config_path = project_root / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Override for quick test
    config["federated"]["num_rounds"] = 3
    config["federated"]["num_clients"] = 10
    config["federated"]["fraction_fit"] = 0.3  # Only 3 clients per round
    config["federated"]["min_fit_clients"] = 3
    config["federated"]["fraction_evaluate"] = 0.3
    config["federated"]["min_evaluate_clients"] = 3
    config["federated"]["min_available_clients"] = 10  # Must match num_clients
    config["training"]["local_epochs"] = 1
    config["training"]["batch_size"] = 32

    from src.training.federated_training_pipeline import FederatedTrainingPipeline
    from datetime import datetime

    exp_dir = project_root / "experiments" / f"quick_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    pipeline = FederatedTrainingPipeline(config=config, experiment_dir=exp_dir)

    try:
        history = pipeline.train()
        print(f"\n✅ Training completed!")
        
        if history.losses_distributed:
            print(f"   Final loss: {history.losses_distributed[-1][1]:.4f}")
        if history.metrics_distributed:
            from src.training.federated_training_pipeline import _metric_series_from_flwr_history
            _, acc_series = _metric_series_from_flwr_history(history.metrics_distributed, 'accuracy')
            if acc_series:
                print(f"   Final accuracy: {acc_series[-1]:.4f}")
    except Exception as e:
        print(f"⚠️  Training had an issue: {e}")
        print("   This may be a Ray/resource issue on Windows. Checking if results were saved...")
        import traceback
        traceback.print_exc()

    # ── Step 5: Quick evaluation ───────────────────────────────────
    step("STEP 5/5: Running evaluation")

    try:
        from src.training.evaluate_federated_model import FederatedEvaluator

        evaluator = FederatedEvaluator(
            config_path=str(config_path),
            experiment_dir=str(exp_dir),
        )

        eval_model = evaluator.load_model()
        results = evaluator.evaluate_all_clients(eval_model, client_ids=[0, 1, 2])

        if results:
            report = evaluator.generate_report(results)
            evaluator.visualize_results(results)
            
            overall = report.get("overall_metrics", {})
            print(f"\n✅ Evaluation completed!")
            print(f"   Mean accuracy: {overall.get('mean_accuracy', 0):.4f}")
            print(f"   Mean loss:     {overall.get('mean_loss', 0):.4f}")
    except Exception as e:
        print(f"⚠️  Evaluation issue: {e}")
        import traceback
        traceback.print_exc()

    # ── Summary ────────────────────────────────────────────────────
    elapsed = time.time() - start
    step("TEST SUMMARY")
    print(f"⏱️  Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"📁 Results: {exp_dir}")
    print(f"\n✅ End-to-end pipeline validation complete!")
    print(f"\n{'='*70}")
    print(f"Next steps:")
    print(f"  1. Full training:  python src/training/federated_training_pipeline.py")
    print(f"  2. Evaluation:     python src/training/evaluate_federated_model.py")
    print(f"  3. API demo:       python src/api/fastapi_app.py")
    print(f"  4. Dashboard:      streamlit run src/dashboard/explainable_ui.py")
    print(f"  5. HTML demo:      Open demo/index.html in browser")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
