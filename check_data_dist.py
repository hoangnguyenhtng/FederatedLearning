"""Quick data distribution check for diagnosis."""
import pandas as pd
import numpy as np
from pathlib import Path

data_dir = Path("data/processed/multi_category")

for cid in range(5):
    pkl = data_dir / f"client_{cid}" / "data.pkl"
    if not pkl.exists():
        print(f"Client {cid}: NOT FOUND at {pkl}")
        continue
    df = pd.read_pickle(pkl)
    labels = df["label"].values if "label" in df.columns else (df["rating"].values - 1)
    
    if cid == 0:
        print(f"Client 0 columns: {list(df.columns)}")
        print(f"Client 0 shape: {df.shape}")
        te = df.iloc[0]["text_embedding"]
        ie = df.iloc[0]["image_embedding"]
        bf = df.iloc[0]["behavior_features"]
        print(f"text_embedding dim: {len(te)}")
        print(f"image_embedding dim: {len(ie)}")
        print(f"behavior_features dim: {len(bf)}")
        # Check for zeros/NaN in image embeddings
        img_embs = np.stack(df["image_embedding"].values[:100])
        print(f"Image emb mean: {img_embs.mean():.6f}, std: {img_embs.std():.6f}")
        all_zero_rows = (np.abs(img_embs).sum(axis=1) == 0).sum()
        print(f"Image emb all-zero rows: {all_zero_rows}/100")
        # Text embeddings quality
        txt_embs = np.stack(df["text_embedding"].values[:100])
        print(f"Text emb mean: {txt_embs.mean():.6f}, std: {txt_embs.std():.6f}")
        # Behavior features quality
        bfs = np.stack(df["behavior_features"].values[:100])
        print(f"Behavior feat mean: {bfs.mean():.6f}, std: {bfs.std():.6f}")
        print()
    
    dist = pd.Series(labels).value_counts().sort_index()
    pcts = (dist / dist.sum() * 100).round(1)
    print(f"Client {cid} ({len(df)} samples): {dict(zip(dist.index, pcts.values))} | majority={dist.max()/dist.sum():.3f}")

# Overall distribution
print("\n--- Overall (all 40 clients) ---")
all_labels = []
total_samples = 0
for cid in range(40):
    pkl = data_dir / f"client_{cid}" / "data.pkl"
    if pkl.exists():
        df = pd.read_pickle(pkl)
        labels = df["label"].values if "label" in df.columns else (df["rating"].values - 1)
        all_labels.extend(labels)
        total_samples += len(df)

print(f"Total samples: {total_samples}")
dist = pd.Series(all_labels).value_counts().sort_index()
pcts = (dist / dist.sum() * 100).round(1)
print(f"Label distribution:\n{dist}")
print(f"Percentages:\n{pcts}")
print(f"Majority class accuracy: {dist.max() / dist.sum():.4f}")
