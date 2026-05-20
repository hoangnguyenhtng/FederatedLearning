import os
import json
import pickle
import pandas as pd

print('='*60)
print('KIEM TRA DU LIEU TOAN BO PROJECT')
print('='*60)

# 1. Check simulated_clients (demo data - 10 clients)
print('\n[1] SIMULATED CLIENTS (Demo Data - 10 clients):')
total_sim = 0
has_category = False
for i in range(10):
    path = f'data/simulated_clients/client_{i}/interactions.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        total_sim += len(df)
        if i == 0:
            print(f'  Columns: {list(df.columns)}')
            has_category = 'item_category' in df.columns
            print(f'  item_category exists: {has_category}')
            print(f'  Sample rows: {len(df)}')

# Check items_global.csv in simulated_clients
items_path = 'data/simulated_clients/items_global.csv'
print(f'  items_global.csv in simulated_clients: {os.path.exists(items_path)}')
print(f'  Total simulated samples: {total_sim}')

# 2. Check multi_category processed data (real Amazon data)
print('\n[2] MULTI_CATEGORY PROCESSED (Real Amazon Data - 40 clients):')
meta_path = 'data/processed/multi_category/processing_metadata.json'
if os.path.exists(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
    print(f'  Num clients: {meta["num_clients"]}')
    print(f'  Categories: {meta["categories"]}')
    print(f'  Total samples: {meta["total_samples"]:,}')

# Check one client pkl size
pkl_path = 'data/processed/multi_category/client_0/data.pkl'
if os.path.exists(pkl_path):
    size_mb = os.path.getsize(pkl_path) / 1024 / 1024
    print(f'  client_0/data.pkl size: {size_mb:.1f} MB')
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        print(f'  client_0 type: {type(data)}')
        if isinstance(data, dict):
            print(f'  client_0 keys: {list(data.keys())}')
            for k, v in data.items():
                if hasattr(v, '__len__') and not isinstance(v, str):
                    print(f'    {k}: {len(v)} items')
                elif hasattr(v, "shape"):
                    print(f'    {k}: shape {v.shape}')
                else:
                    print(f'    {k}: {v}')
        elif isinstance(data, pd.DataFrame):
            print(f'  client_0 DataFrame shape: {data.shape}')
            print(f'  Columns: {list(data.columns)}')
    except Exception as e:
        print(f'  Error loading pkl: {e}')

# 3. Check amazon_2023_processed
print('\n[3] AMAZON 2023 PROCESSED (data/amazon_2023_processed):')
total_az = 0
for i in range(10):
    path = f'data/amazon_2023_processed/client_{i}/data.pkl'
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                d = pickle.load(f)
            if isinstance(d, pd.DataFrame):
                total_az += len(d)
                if i == 0:
                    print(f'  client_0 columns: {list(d.columns)}')
                    print(f'  client_0 shape: {d.shape}')
            elif isinstance(d, dict):
                for k, v in d.items():
                    if hasattr(v, '__len__') and not isinstance(v, str):
                        total_az += len(v)
                        if i == 0:
                            print(f'  client_0 key={k} len={len(v)}')
                        break
        except Exception as e:
            print(f'  Error client_{i}: {e}')
if total_az > 0:
    print(f'  Total samples loaded: {total_az}')
else:
    print('  No data loaded from amazon_2023_processed')

# 4. Count total clients pkl in multi_category
print('\n[4] COUNTING ALL CLIENT PKLS (multi_category):')
mc_path = 'data/processed/multi_category'
total_clients = 0
existing_pkls = []
for item in os.listdir(mc_path):
    client_path = os.path.join(mc_path, item, 'data.pkl')
    if os.path.isdir(os.path.join(mc_path, item)) and os.path.exists(client_path):
        size_mb = os.path.getsize(client_path) / 1024 / 1024
        existing_pkls.append((item, size_mb))
        total_clients += 1

print(f'  Total client folders with data.pkl: {total_clients}')
total_size = sum(s for _, s in existing_pkls)
print(f'  Total data size: {total_size:.1f} MB')

print('\n' + '='*60)
print('SUMMARY')
print('='*60)
print(f'  Demo data (simulated_clients): {total_sim} samples, 10 clients')
print(f'  Real data (multi_category): 220,000 samples, 40 clients (from metadata)')
print(f'  Real data clients with pkl: {total_clients}')
