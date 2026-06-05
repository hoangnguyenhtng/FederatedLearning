"""
Thesis Data Generator – Federated Multi-Modal Recommendation System
====================================================================
Generates HIGH-QUALITY synthetic data for 40 federated clients across
4 Amazon-style domains:

  Domain 0 – All_Beauty      → clients  0–9   (10 clients)
  Domain 1 – Video_Games     → clients 10–19  (10 clients)
  Domain 2 – Amazon_Fashion  → clients 20–29  (10 clients)
  Domain 3 – Baby_Products   → clients 30–39  (10 clients)

Each client: ~3,000 samples  →  Total: ~120,000 samples

Text is encoded by the REAL SentenceTransformer (all-MiniLM-L6-v2) so
embeddings carry genuine semantic meaning – same model used at inference.

Output: data/amazon_2023_processed/client_<i>/data.pkl
        (compatible with AmazonDataset / get_amazon_dataloaders)
"""

import os
import sys
import time
import random
import pickle
import hashlib
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import torch

# ── Project root ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── SentenceTransformer ───────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer

# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

NUM_CLIENTS      = 40
CLIENTS_PER_DOM  = 10          # 10 clients per domain
SAMPLES_PER_CLIENT = 3_000     # target samples per client
ALPHA_DIRICHLET  = 0.5         # non-IID strength (lower = more non-IID)
SEED             = 42
OUTPUT_DIR       = PROJECT_ROOT / "data" / "amazon_2023_processed"

# ── Amazon-style rating prior (J-shaped, realistic but trainable) ─────────────
# label: 0=1★, 1=2★, 2=3★, 3=4★, 4=5★
BASE_LABEL_PROBS = np.array([0.06, 0.10, 0.16, 0.30, 0.38])   # sums to 1.0

# ── Domain definitions ────────────────────────────────────────────────────────
DOMAINS = {
    "All_Beauty": {
        "client_ids": list(range(0, 10)),
        "label_bias": np.array([0.00,  0.00,  0.02,  0.05, -0.07]),  # Beauty → more 5★
        "brands":  ["L'Oreal","Neutrogena","Cetaphil","The Ordinary","CeraVe",
                    "Olay","Maybelline","NYX","e.l.f.","Revlon","Dove","Pantene"],
        "categories": ["Skincare","Makeup","Hair Care","Body Lotion","Sunscreen",
                        "Foundation","Mascara","Serum","Moisturizer","Shampoo"],
        "price_range": (5, 80),
        "review_templates": [
            "This {product} is absolutely amazing! My skin feels {adj} after just {days} days.",
            "I've been using this {product} for {months} months and love the results. {adj} quality.",
            "{brand} {product} is my go-to skincare staple. The formula is lightweight and {adj}.",
            "Incredible {product}! Solved my {issue} problem completely. Highly recommend.",
            "Average {product}, nothing special. The consistency is {adj} but scent is overwhelming.",
            "This {product} broke me out badly. Not suitable for sensitive skin.",
            "Best {product} under $30! Better than expensive luxury brands.",
            "Love this {product}. Apply every morning and my {concern} has improved significantly.",
            "Repurchasing my third bottle of this {product}. {adj} texture, absorbs quickly.",
            "Decent {product} for the price. Does what it claims. {adj} packaging.",
            "Game changer {product} for my daily routine. Skin looks {adj} and glowing.",
            "The {product} from {brand} smells wonderful and feels {adj} on the skin.",
            "Perfect lightweight {product} for {season} weather. Non-greasy formula.",
            "This {product} has SPF which is a bonus. {adj} coverage and moisturizes well.",
            "I bought this {product} after seeing reviews online. Not disappointed at all.",
        ],
        "adjectives": ["smooth","hydrating","lightweight","nourishing","refreshing",
                        "silky","effective","gentle","powerful","creamy","soothing"],
        "products":   ["moisturizer","serum","cleanser","toner","sunscreen","foundation",
                        "mascara","shampoo","conditioner","face wash","eye cream","lip balm"],
        "issues":     ["dryness","acne","dark spots","oiliness","redness","wrinkles"],
        "concerns":   ["texture","tone","hydration","brightness","pores"],
    },
    "Video_Games": {
        "client_ids": list(range(10, 20)),
        "label_bias": np.array([-0.02,  0.02,  0.02, -0.01, -0.01]),  # Games → more polarized
        "brands":  ["Sony","Microsoft","Nintendo","Razer","SteelSeries","Logitech",
                    "HyperX","Corsair","ASUS","Alienware","Turtle Beach","Astro"],
        "categories": ["Action","RPG","FPS","Sports","Racing","Strategy","Indie",
                        "Controller","Headset","Gaming Chair","Keyboard","Mouse"],
        "price_range": (15, 250),
        "review_templates": [
            "This {product} is a masterpiece! Spent {hours} hours and still can't stop playing.",
            "Absolutely love {product}. The gameplay is {adj} and story is compelling.",
            "{brand} {product} delivers an {adj} gaming experience. Worth every penny.",
            "The graphics on {product} are {adj}. Runs smoothly at 60fps.",
            "Disappointed with {product}. Too many bugs on release. Needs more patches.",
            "Best {genre} game I've played this year. {product} is truly {adj}.",
            "This {product} controller feels {adj} in hand. Perfect grip and response.",
            "Multiplayer on {product} is incredibly {adj}. Team up with friends easily.",
            "The storyline in {product} kept me hooked. {adj} character development.",
            "Not worth full price. Wait for a sale on {product}.",
            "Incredible soundtrack and {adj} visuals in {product}. Immersive experience.",
            "This gaming {product} from {brand} has transformed my setup. {adj} build quality.",
            "Addictive gameplay loop in {product}. Spent {hours} hours in one sitting.",
            "Great value bundle. {product} includes everything needed to get started.",
            "The co-op mode in {product} is {adj}. Playing with friends is a blast.",
        ],
        "adjectives": ["incredible","immersive","responsive","smooth","lag-free",
                        "epic","addictive","stunning","competitive","tactical","next-level"],
        "products":   ["game","controller","headset","keyboard","mouse","gaming chair",
                        "console","game pass","DLC","expansion","racing wheel","webcam"],
        "genres":     ["action","RPG","FPS","strategy","simulation","sports"],
        "hours":      ["20", "50", "100", "200", "30", "15", "75"],
    },
    "Amazon_Fashion": {
        "client_ids": list(range(20, 30)),
        "label_bias": np.array([0.01,  0.01, -0.01,  0.01, -0.02]),  # Fashion → more neutral
        "brands":  ["Nike","Adidas","Levi's","H&M","Zara","Gap","Calvin Klein",
                    "Tommy Hilfiger","Ralph Lauren","Gucci","Puma","Under Armour"],
        "categories": ["T-Shirt","Jeans","Dress","Sneakers","Jacket","Hoodie",
                        "Skirt","Blouse","Pants","Coat","Boots","Sandals"],
        "price_range": (15, 300),
        "review_templates": [
            "The {product} fits {adj} and the material is high quality. Very comfortable.",
            "Ordered size {size} and it fits perfectly. The {color} color is {adj}.",
            "This {product} from {brand} is {adj}. Great for both casual and formal occasions.",
            "Love the cut of this {product}. Very flattering and {adj} fabric.",
            "Received compliments on this {product} immediately. Looks {adj} in person.",
            "The {product} runs small, order a size up. Material feels {adj} though.",
            "Perfect {product} for {season}. Lightweight and {adj} for warm weather.",
            "This {product} is exactly as described. {adj} stitching and durable material.",
            "Great everyday {product}. Machine washable and stays {adj} after washing.",
            "The {product} color is slightly different from photos but still {adj}.",
            "Affordable {product} for the quality. {brand} never disappoints.",
            "This {product} is a wardrobe staple. Classic design and {adj} fit.",
            "Very satisfied with this {product} purchase. Fast shipping and {adj} packaging.",
            "The {product} material is breathable and {adj} for all-day wear.",
            "Bought this {product} for an event and got {adj} feedback on the style.",
        ],
        "adjectives": ["comfortable","stylish","elegant","durable","soft","breathable",
                        "well-fitted","trendy","classic","premium","chic","versatile"],
        "products":   ["dress","jeans","t-shirt","jacket","sneakers","hoodie",
                        "blouse","skirt","coat","pants","boots","cardigan"],
        "sizes":      ["S","M","L","XL","32","34","36","38"],
        "colors":     ["black","navy","white","grey","beige","olive","burgundy"],
        "seasons":    ["summer","spring","fall","winter"],
    },
    "Baby_Products": {
        "client_ids": list(range(30, 40)),
        "label_bias": np.array([-0.01,  0.00, -0.01,  0.01,  0.01]),  # Baby → trust high ratings
        "brands":  ["Pampers","Huggies","Graco","Baby Bjorn","Fisher-Price","Chicco",
                    "Ergobaby","UPPAbaby","Britax","Diono","Munchkin","Tommee Tippee"],
        "categories": ["Diapers","Stroller","Car Seat","Baby Monitor","Baby Carrier",
                        "High Chair","Breast Pump","Bottle","Baby Food","Baby Clothes"],
        "price_range": (10, 500),
        "review_templates": [
            "This {product} is a lifesaver for new parents! {adj} quality and easy to use.",
            "Our baby loves the {product} from {brand}. Keeps them {adj} and comfortable.",
            "The {product} is {adj} to assemble and very sturdy. Great safety features.",
            "Best {product} we've purchased for our baby. {adj} construction and design.",
            "This {product} has made our routine so much easier. {adj} and reliable.",
            "Very satisfied with the {product} quality. Our {baby_age} loves it.",
            "The {product} is easy to clean and {adj}. Essential baby gear.",
            "Bought this {product} based on recommendations. Not disappointed at all.",
            "This {product} exceeded expectations. {adj} build and worth the price.",
            "The {product} broke after {weeks} weeks. Poor quality for the price.",
            "Perfect {product} for traveling with baby. Compact and {adj} design.",
            "This {product} from {brand} is {adj} and fits our lifestyle perfectly.",
            "Baby sleeps much better since using this {product}. {adj} comfort level.",
            "Great {product} for the price range. Does everything advertised.",
            "This {product} is essential in our nursery. {adj} and functional design.",
        ],
        "adjectives": ["safe","durable","easy-to-use","comfortable","reliable",
                        "sturdy","portable","practical","well-designed","gentle","hypoallergenic"],
        "products":   ["diaper","stroller","car seat","baby monitor","carrier",
                        "high chair","bottle","breast pump","baby food","onesie","toy","crib"],
        "baby_ages":  ["3-month-old","6-month-old","newborn","1-year-old","toddler"],
        "weeks":      ["2","3","4","6","8"],
    },
}

# ── Helper: deterministic hash-based float ─────────────────────────────────────
def _hash_float(s: str, i: int = 0) -> float:
    h = int(hashlib.md5(f"{s}_{i}".encode()).hexdigest(), 16)
    return (h % 100000) / 100000.0


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_domain_texts(domain_name: str, domain_cfg: dict,
                           n_samples: int, rng: np.random.Generator) -> list:
    """Generate n_samples review texts for a domain."""
    templates  = domain_cfg["review_templates"]
    adjectives = domain_cfg["adjectives"]
    products   = domain_cfg["products"]
    brands     = domain_cfg["brands"]
    categories = domain_cfg["categories"]

    texts = []
    for _ in range(n_samples):
        t   = rng.choice(templates)
        adj = rng.choice(adjectives)
        prd = rng.choice(products)
        br  = rng.choice(brands)
        cat = rng.choice(categories)

        # Fill generic placeholders
        text = t.format(
            product=prd, adj=adj, brand=br, category=cat,
            # domain-specific optional fields (ignored if not in template)
            days   = str(rng.integers(3, 30)),
            months = str(rng.integers(1, 12)),
            hours  = rng.choice(domain_cfg.get("hours", ["20"])),
            size   = rng.choice(domain_cfg.get("sizes", ["M"])),
            color  = rng.choice(domain_cfg.get("colors", ["black"])),
            season = rng.choice(domain_cfg.get("seasons", ["summer"])),
            issue  = rng.choice(domain_cfg.get("issues", ["dryness"])),
            concern= rng.choice(domain_cfg.get("concerns", ["texture"])),
            genre  = rng.choice(domain_cfg.get("genres",  ["action"])),
            baby_age = rng.choice(domain_cfg.get("baby_ages", ["6-month-old"])),
            weeks  = rng.choice(domain_cfg.get("weeks", ["4"])),
        )
        texts.append(text)
    return texts


# ═══════════════════════════════════════════════════════════════════════════════
# LABEL GENERATION  (Non-IID via Dirichlet)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_labels_noniid(n_samples: int, client_local_id: int,
                             base_probs: np.ndarray, alpha: float,
                             rng: np.random.Generator) -> np.ndarray:
    """
    Each client gets its own label distribution drawn from Dirichlet(alpha).
    This produces realistic non-IID variation across clients.
    """
    # Client-specific distribution centred on base_probs
    concentration = base_probs * alpha * 10 + 0.5   # avoid zeros
    client_probs  = rng.dirichlet(concentration)
    labels = rng.choice(5, size=n_samples, p=client_probs)
    return labels.astype(np.int64)


# ═══════════════════════════════════════════════════════════════════════════════
# BEHAVIOR FEATURES  (32-dim, meaningful)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_behavior_features(n_samples: int, domain_cfg: dict,
                                 labels: np.ndarray,
                                 rng: np.random.Generator) -> np.ndarray:
    """Generate 32-dim behavior features correlated with labels."""
    lo, hi = domain_cfg["price_range"]
    B = np.zeros((n_samples, 32), dtype=np.float32)

    # Feature 0: avg_rating of item  (correlated with label)
    item_avg_rating = np.clip(labels * 0.8 + rng.normal(2.5, 0.5, n_samples), 1, 5)
    B[:, 0] = item_avg_rating.astype(np.float32)

    # Feature 1: log(rating_count)
    rating_count = np.exp(rng.uniform(2, 9, n_samples))
    B[:, 1] = np.log1p(rating_count).astype(np.float32)

    # Feature 2: price (normalized)
    prices = rng.uniform(lo, hi, n_samples)
    B[:, 2] = np.log1p(prices).astype(np.float32)

    # Feature 3: helpful_votes (higher for extreme ratings)
    helpful = np.abs(labels - 2.0) * rng.uniform(0, 5, n_samples)
    B[:, 3] = np.log1p(helpful).astype(np.float32)

    # Feature 4: verified_purchase (0/1)
    B[:, 4] = rng.choice([0, 1], size=n_samples, p=[0.15, 0.85]).astype(np.float32)

    # Feature 5: normalized rating (0–1)
    B[:, 5] = (item_avg_rating / 5.0).astype(np.float32)

    # Feature 6: normalized log rating count
    B[:, 6] = (B[:, 1] / B[:, 1].max()).astype(np.float32)

    # Feature 7: normalized log price
    B[:, 7] = (B[:, 2] / B[:, 2].max()).astype(np.float32)

    # Features 8–15: user/item hashes (deterministic noise)
    for i in range(8, 16):
        B[:, i] = rng.uniform(0, 1, n_samples).astype(np.float32)

    # Features 16–31: domain-specific interaction patterns
    for i in range(16, 32):
        scale = rng.uniform(0.1, 1.0)
        B[:, i] = (np.sin(labels * (i + 1) * 0.3) * scale +
                   rng.uniform(-0.1, 0.1, n_samples)).astype(np.float32)

    # Clip and clean
    B = np.clip(B, -10, 10)
    B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
    return B


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_all_data():
    print("=" * 70)
    print("  THESIS DATA GENERATOR")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"  Clients: {NUM_CLIENTS} ({CLIENTS_PER_DOM} per domain)")
    print(f"  Samples/client: {SAMPLES_PER_CLIENT:,}")
    print(f"  Total target:   {NUM_CLIENTS * SAMPLES_PER_CLIENT:,}")
    print(f"  Output:         {OUTPUT_DIR}")
    print()

    rng_global = np.random.default_rng(SEED)

    # -- Load SentenceTransformer --
    print("[INFO] Loading SentenceTransformer (all-MiniLM-L6-v2)...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = encoder.to(device)
    print(f"   Device: {device}")

    # -- Deterministic projection matrix 384->2048 (image proxy) --
    rng_proj = np.random.RandomState(42)
    IMG_PROJ = rng_proj.randn(384, 2048).astype(np.float32) * 0.05
    print("[INFO] Image projection matrix ready (384->2048)")
    print()

    # ── Create output directory ───────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_generated = 0
    start_time = time.time()

    for domain_name, domain_cfg in DOMAINS.items():
        client_ids = domain_cfg["client_ids"]
        label_probs = np.clip(BASE_LABEL_PROBS + domain_cfg["label_bias"], 0.01, 1.0)
        label_probs = label_probs / label_probs.sum()

        print(f"{'='*70}")
        print(f"  DOMAIN: {domain_name}")
        print(f"  Clients: {client_ids[0]}–{client_ids[-1]}")
        print(f"  Label distribution: {np.round(label_probs, 3)}")
        print(f"{'='*70}")

        for local_id, client_id in enumerate(client_ids):
            client_seed = SEED + client_id * 1000
            rng = np.random.default_rng(client_seed)

            # slight variation per client
            n = SAMPLES_PER_CLIENT + rng.integers(-200, 200)

            print(f"\n  Client {client_id:2d} (domain local {local_id})  →  {n:,} samples")

            # 1. Generate texts
            texts = generate_domain_texts(domain_name, domain_cfg, n, rng)

            # 2. Encode texts (batch for speed)
            batch_size = 512
            text_embeddings = encoder.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )   # shape (n, 384)

            # 3. Image proxy embeddings
            image_embeddings = (text_embeddings @ IMG_PROJ)   # (n, 2048)
            # Add small domain-specific noise to differentiate modalities
            image_embeddings += rng.normal(0, 0.02, image_embeddings.shape).astype(np.float32)

            # 4. Labels (non-IID per client)
            labels = generate_labels_noniid(n, local_id, label_probs, ALPHA_DIRICHLET, rng)

            # 5. Behavior features
            behavior = generate_behavior_features(n, domain_cfg, labels, rng)

            # 6. Fake user/item IDs (unique per client)
            user_ids  = [f"user_{domain_name[:3]}_{client_id}_{i:04d}" for i in range(n)]
            item_ids  = [f"item_{domain_name[:3]}_{rng.integers(0, n//3):04d}" for _ in range(n)]
            ratings   = labels + 1    # 1–5
            timestamps = rng.integers(1_600_000_000, 1_750_000_000, n)

            # ── Compute label distribution summary ────────────────────────
            ld = np.bincount(labels, minlength=5)
            pct = ld / ld.sum() * 100
            print(f"     Labels: " +
                  " ".join([f"★{i+1}:{ld[i]}({pct[i]:.0f}%)" for i in range(5)]))

            # 7. Build DataFrame
            records = []
            for i in range(n):
                records.append({
                    "user_id"          : user_ids[i],
                    "item_id"          : item_ids[i],
                    "item_title"       : texts[i][:80],
                    "item_category"    : domain_name,
                    "item_brand"       : rng.choice(domain_cfg["brands"]),
                    "item_price"       : float(rng.uniform(*domain_cfg["price_range"])),
                    "item_image_url"   : None,
                    "rating"           : int(ratings[i]),
                    "label"            : int(labels[i]),
                    "text_embedding"   : text_embeddings[i].tolist(),
                    "image_embedding"  : image_embeddings[i].tolist(),
                    "behavior_features": behavior[i].tolist(),
                    "timestamp"        : int(timestamps[i]),
                })

            df = pd.DataFrame(records)

            # 8. Save
            client_dir = OUTPUT_DIR / f"client_{client_id}"
            client_dir.mkdir(exist_ok=True)
            df.to_pickle(client_dir / "data.pkl")
            print(f"     Saved → {client_dir/'data.pkl'}  ({len(df):,} rows)")

            total_generated += len(df)

    # ── Global tables ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  Saving global items & users tables...")
    try:
        all_dfs = []
        for i in range(NUM_CLIENTS):
            p = OUTPUT_DIR / f"client_{i}" / "data.pkl"
            if p.exists():
                all_dfs.append(pd.read_pickle(p))
        if all_dfs:
            all_df = pd.concat(all_dfs, ignore_index=True)

            item_cols = ["item_id","item_title","item_category","item_brand","item_price","item_image_url"]
            items_df  = all_df[item_cols].drop_duplicates("item_id").reset_index(drop=True)
            items_df.to_csv(OUTPUT_DIR / "items_global.csv", index=False)
            print(f"  [OK] items_global.csv ({len(items_df):,} items)")

            users_df = (all_df.groupby("user_id")
                        .agg(num_interactions=("item_id","count"),
                             avg_rating_given=("rating","mean"),
                             last_timestamp  =("timestamp","max"))
                        .reset_index())
            users_df.to_csv(OUTPUT_DIR / "users_global.csv", index=False)
            print(f"  [OK] users_global.csv ({len(users_df):,} users)")
    except Exception as e:
        print(f"  [WARN] Could not save global tables: {e}")

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  [DONE] DATA GENERATION COMPLETE")
    print(f"  Total samples : {total_generated:,}")
    print(f"  Time elapsed  : {elapsed/60:.1f} minutes")
    print(f"  Output dir    : {OUTPUT_DIR}")
    print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    generate_all_data()
