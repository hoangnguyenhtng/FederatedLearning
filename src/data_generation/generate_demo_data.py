"""
Generate synthetic demo data for testing the pipeline without downloading Amazon data.

Creates realistic-looking multi-modal data for 10 clients with Non-IID distribution,
matching the exact format expected by amazon_dataloader.py.

Usage:
    python src/data_generation/generate_demo_data.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

NUM_CLIENTS = 10
SAMPLES_PER_CLIENT = 200  # Total: ~2000 samples (fast training)
OUTPUT_DIR = project_root / "data" / "amazon_2023_processed"
SEED = 42

# Product categories (simulating Amazon)
CATEGORIES = ["Beauty", "Fashion", "Baby", "Games"]
BRANDS = {
    "Beauty": ["L'Oreal", "Maybelline", "Neutrogena", "CeraVe", "Olay"],
    "Fashion": ["Nike", "Adidas", "Zara", "H&M", "Levi's"],
    "Baby": ["Pampers", "Huggies", "Gerber", "Fisher-Price", "Graco"],
    "Games": ["Nintendo", "Sony", "Microsoft", "EA", "Ubisoft"],
}
PRODUCTS = {
    "Beauty": ["Face Cream", "Lipstick", "Sunscreen", "Shampoo", "Serum", "Mascara", "Foundation", "Moisturizer"],
    "Fashion": ["T-Shirt", "Jeans", "Sneakers", "Jacket", "Dress", "Hoodie", "Shorts", "Backpack"],
    "Baby": ["Diapers", "Baby Food", "Stroller", "Car Seat", "Blanket", "Bottle", "Pacifier", "Toy Set"],
    "Games": ["Controller", "Headset", "Game Card", "Console", "VR Set", "Racing Game", "RPG Game", "Puzzle Game"],
}


def generate_demo_data():
    """Generate synthetic federated data mimicking Amazon Reviews format."""
    np.random.seed(SEED)
    print("=" * 70)
    print("GENERATING DEMO DATA FOR FEDERATED LEARNING")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Build item catalog ─────────────────────────────────────────
    items = []
    item_id = 0
    for cat in CATEGORIES:
        for product in PRODUCTS[cat]:
            for brand in BRANDS[cat]:
                items.append({
                    "item_id": f"ITEM_{item_id:05d}",
                    "item_title": f"{brand} {product}",
                    "item_category": cat,
                    "item_brand": brand,
                    "item_price": round(np.random.uniform(5, 200), 2),
                    "item_image_url": None,
                })
                item_id += 1
    items_df = pd.DataFrame(items)
    num_items = len(items_df)
    print(f"📦 Created {num_items} items across {len(CATEGORIES)} categories")

    # ── Pre-compute item embeddings ────────────────────────────────
    # Text embedding: deterministic from item name hash (384-dim)
    rng = np.random.RandomState(42)
    item_text_embs = {}
    item_image_embs = {}
    proj_matrix = rng.randn(384, 2048).astype(np.float32) * 0.05

    for _, row in items_df.iterrows():
        iid = row["item_id"]
        seed = hash(iid) % (2**31)
        r = np.random.RandomState(seed)
        text_emb = r.randn(384).astype(np.float32) * 0.5
        # Normalize
        text_emb = text_emb / (np.linalg.norm(text_emb) + 1e-8)
        item_text_embs[iid] = text_emb
        # Image proxy from text
        image_emb = (text_emb @ proj_matrix).astype(np.float32)
        item_image_embs[iid] = image_emb

    # ── Generate per-client data (Non-IID via Dirichlet) ────────────
    # Each client has category preferences
    alpha = 0.5  # Non-IID parameter
    cat_proportions = np.random.dirichlet([alpha] * len(CATEGORIES), size=NUM_CLIENTS)

    all_client_dfs = {}
    total_samples = 0

    for client_id in range(NUM_CLIENTS):
        n_samples = SAMPLES_PER_CLIENT + np.random.randint(-50, 50)
        n_samples = max(n_samples, 100)

        # Sample categories according to Dirichlet distribution
        cat_counts = np.random.multinomial(n_samples, cat_proportions[client_id])

        interactions = []
        user_pool = [f"USER_{client_id}_{u:03d}" for u in range(max(n_samples // 10, 5))]

        for cat_idx, cat in enumerate(CATEGORIES):
            n_cat = cat_counts[cat_idx]
            if n_cat == 0:
                continue

            # Get items in this category
            cat_items = items_df[items_df["item_category"] == cat]
            if len(cat_items) == 0:
                continue

            for _ in range(n_cat):
                item_row = cat_items.sample(1).iloc[0]
                user_id = np.random.choice(user_pool)
                iid = item_row["item_id"]

                # Generate rating (skewed toward 4-5 for more realistic data)
                rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.10, 0.15, 0.35, 0.35])
                label = rating - 1  # 0-4

                # Text embedding: user-specific variation of item embedding
                user_seed = hash(user_id) % (2**31)
                user_noise = np.random.RandomState(user_seed).randn(384).astype(np.float32) * 0.1
                text_emb = item_text_embs[iid] + user_noise
                text_emb = text_emb / (np.linalg.norm(text_emb) + 1e-8)

                # Image embedding
                image_emb = item_image_embs[iid].copy()

                # Behavior features (32-dim)
                behavior = np.zeros(32, dtype=np.float32)
                behavior[0] = float(item_row["item_price"]) / 200.0
                behavior[1] = float(rating) / 5.0
                behavior[2] = np.random.uniform(0.1, 1.0)  # click probability
                behavior[3] = np.random.uniform(0, 1)  # purchase prob
                behavior[4] = float(np.random.randint(1, 100)) / 100.0  # view count norm
                for i in range(5, 32):
                    behavior[i] = np.random.uniform(0, 1)

                interactions.append({
                    "user_id": user_id,
                    "item_id": iid,
                    "item_title": item_row["item_title"],
                    "item_category": item_row["item_category"],
                    "item_brand": item_row["item_brand"],
                    "item_price": item_row["item_price"],
                    "item_image_url": None,
                    "rating": rating,
                    "label": label,
                    "text_embedding": text_emb.tolist(),
                    "image_embedding": image_emb.tolist(),
                    "behavior_features": behavior.tolist(),
                    "timestamp": 1700000000 + np.random.randint(0, 10000000),
                })

        client_df = pd.DataFrame(interactions)
        all_client_dfs[client_id] = client_df
        total_samples += len(client_df)

        # Dominant category for this client
        dominant_cat = CATEGORIES[np.argmax(cat_proportions[client_id])]
        dominant_pct = cat_proportions[client_id].max() * 100

        print(f"  Client {client_id}: {len(client_df)} samples, "
              f"{client_df['user_id'].nunique()} users, "
              f"dominant: {dominant_cat} ({dominant_pct:.0f}%)")

    # ── Save ────────────────────────────────────────────────────────
    print(f"\n💾 Saving to {OUTPUT_DIR}...")

    for client_id, client_df in all_client_dfs.items():
        client_dir = OUTPUT_DIR / f"client_{client_id}"
        client_dir.mkdir(exist_ok=True)
        client_df.to_pickle(client_dir / "data.pkl")

    # Save global tables
    all_df = pd.concat(list(all_client_dfs.values()), ignore_index=True)

    # Items table
    item_cols = ["item_id", "item_title", "item_category", "item_brand", "item_price", "item_image_url"]
    items_global = all_df[item_cols].drop_duplicates(subset=["item_id"]).reset_index(drop=True)
    items_global.to_csv(OUTPUT_DIR / "items_global.csv", index=False)

    # Users table
    users_global = (
        all_df.groupby("user_id")
        .agg(num_interactions=("item_id", "count"), avg_rating_given=("rating", "mean"))
        .reset_index()
    )
    users_global.to_csv(OUTPUT_DIR / "users_global.csv", index=False)

    print(f"\n✅ DEMO DATA GENERATED SUCCESSFULLY!")
    print(f"   Total samples: {total_samples}")
    print(f"   Clients: {NUM_CLIENTS}")
    print(f"   Items: {len(items_global)}")
    print(f"   Users: {len(users_global)}")
    print(f"   Output: {OUTPUT_DIR}")
    print(f"\nNext: python src/training/federated_training_pipeline.py")


if __name__ == "__main__":
    generate_demo_data()
