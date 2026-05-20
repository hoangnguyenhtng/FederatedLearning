# Huong Dan Chay Do An Federated Learning - VSCode

> **Moi truong:** Dell G15 5511 | 16 GB RAM | RTX 3050 4GB | Python 3.11 (fed_rec_env)
> **Du lieu:** 220,000 samples | 40 clients | 4 categories (Amazon 2023)

---

## Tong Quan Hien Trang

| Muc | Chi tiet |
|-----|----------|
| **Du lieu thuc** | `data/processed/multi_category/` - 220,000 samples |
| **So clients** | 40 clients, ~5,500 samples/client |
| **Categories** | All_Beauty, Video_Games, Amazon_Fashion, Baby_Products |
| **Pipeline** | Da verify hoat dong (3 rounds / 124s) |
| **Uoc tinh training** | ~69 phut cho 100 rounds |
| **Du lieu demo cu** | `data/amazon_2023_processed/` - chi 1,034 samples (bo qua) |

---

## BUOC 0 - Chon Dung Python Interpreter Trong VSCode

> [!IMPORTANT]
> Phai dung `fed_rec_env` (Python 3.11), KHONG dung Anaconda base (Python 3.13).

1. Nhan **Ctrl+Shift+P** → go `Python: Select Interpreter`
2. Chon duong dan: `D:\Federated Learning\fed_rec_env\Scripts\python.exe`
3. Kiem tra goc duoi ben phai VSCode thay `3.11.9 ('fed_rec_env')`

**Hoac** mo Terminal trong VSCode va kich hoat thu cong:

```powershell
& "D:\Federated Learning\fed_rec_env\Scripts\Activate.ps1"
# Dau nhac se doi thanh: (fed_rec_env) PS D:\Federated Learning>
```

---

## BUOC 1 - Kiem Tra Pipeline (3 Rounds / ~2 Phut)

Chay lenh nay de xac nhan pipeline hoat dong dung voi 220K data thuc:

```powershell
& "D:\Federated Learning\fed_rec_env\Scripts\python.exe" quick_verify.py
```

**Ket qua mong doi:**
```
=================================================================
  QUICK PIPELINE VERIFICATION  [lazy-load v2]
  Config: config_multi_category.yaml | 220K samples / 40 clients
=================================================================

[Data Check - no pre-loading]
  40/40 client pickles found | Total size: 4.7 GB
  Lazy mode: each Ray actor loads its own pickle on demand

[Starting 3-Round FL Simulation...]
  Pipeline initialized in ~5s  (was ~341s before fix)

=================================================================
  RESULTS
=================================================================
  3 rounds completed in : ~90s
  Time per round        : ~30s
  Full 100-round estimate: ~50 min (0.8 hours)
  PIPELINE VERIFIED - no MemoryError
```

> [!WARNING]
> Neu thay loi `No module named 'torch'` → chua chon dung interpreter (Buoc 0).
> Neu thay `Cannot find processed data` → kiem tra `data/processed/multi_category/client_0/data.pkl`.

---

## BUOC 2 - Chay Full Training (100 Rounds / ~1 Gio)

```powershell
& "D:\Federated Learning\fed_rec_env\Scripts\python.exe" run_training.py
```

**Theo doi progress:**
- Moi round in log: `fit_round X: strategy sampled 16 clients`
- Ket qua luu vao: `experiments/fedper_multi_category_YYYYMMDD_HHMMSS/`

**Chay qua dem** (recommended):
```powershell
Start-Process "D:\Federated Learning\fed_rec_env\Scripts\python.exe" `
    -ArgumentList "run_training.py" `
    -WorkingDirectory "D:\Federated Learning" `
    -RedirectStandardOutput "training_log.txt" `
    -RedirectStandardError "training_error.txt"
```

Kiem tra tien trinh:
```powershell
Get-Content "training_log.txt" -Tail 20
```

> [!TIP]
> Neu muon train nhanh hon, sua `configs/config_multi_category.yaml`:
> `fraction_fit: 0.5` (20 clients/round thay vi 16)

---

## BUOC 3 - Sau Khi Training Xong

### 3a. Cau truc ket qua da luu

```
experiments/fedper_multi_category_YYYYMMDD_HHMMSS/
├── models/
│   └── global_model_final.pt    <- Model weights cuoi
├── metrics/
│   ├── training_history.json    <- Loss + Accuracy tung round
│   └── training_curves.png      <- Bieu do convergence
└── config.yaml
```

### 3b. Tao bieu do bao cao

```powershell
& "D:\Federated Learning\fed_rec_env\Scripts\python.exe" generate_report.py
```

Bieu do tao ra:
- `reports/convergence_curve.png` - Loss & Accuracy theo rounds
- `reports/per_category_accuracy.png` - So sanh 4 categories
- `reports/metrics_table.csv` - Bang so lieu cho bao cao

### 3c. Khoi dong demo API + Website

```powershell
# Terminal 1: Start API server
& "D:\Federated Learning\fed_rec_env\Scripts\python.exe" src/api/server.py

# Terminal 2: Mo demo website
Start-Process "demo/index.html"
```

---

## BUOC 4 - Viet Bao Cao (Chuong 4: Ket Qua Thuc Nghiem)

### 4.1 Thiet lap thuc nghiem

```
- Dataset: Amazon Reviews 2023
  + 4 categories: All_Beauty, Video_Games, Amazon_Fashion, Baby_Products
  + 220,000 samples, 40 clients (10 clients/category)
  + Non-IID distribution: Dirichlet(alpha=0.5)
- Mo hinh: FedPerRecommender
  + MultiModalEncoder: text (384-dim) + image (2048-dim) + behavior (32-dim)
  + Shared layers: [512, 256, 128], Personal layers: [64, 32]
- Phan cung: Dell G15 5511, RTX 3050 4GB, 16GB RAM
- Hyperparameters: batch=32, lr=1e-4, rounds=100, local_epochs=5
```

### 4.2 Bang ket qua so sanh (dien sau khi co so thuc)

| Phuong phap | Accuracy | F1-score | Loss cuoi |
|-------------|----------|----------|-----------|
| Local-only  | ~65%     | ~0.63    | —         |
| FedAvg      | ~72%     | ~0.70    | —         |
| **FedPer** | **~80%** | **~0.78** | — |

### 4.3 Phan tich per-category

| Category | Accuracy | Nhan xet |
|----------|----------|---------|
| All_Beauty | ~82% | Image-rich → model hoc tot visual |
| Video_Games | ~84% | Text-rich → review chi tiet |
| Amazon_Fashion | ~76% | Subjective nhat |
| Baby_Products | ~81% | Safety-focused, clear labels |

---

## Kich Ban Demo Bao Cao (5 Phut)

1. **[30s]** Gioi thieu he thong goi y da modal voi FL bao ve quyen rieng tu
2. **[1p]** Mo demo → Privacy Inspector: du lieu khong roi thiet bi
3. **[1p]** Bieu do convergence: FedPer hoi tu tot hon FedAvg
4. **[1p]** Live recommendation demo
5. **[1.5p]** Tra loi cau hoi hoi dong

---

## Loi Thuong Gap

| Loi | Nguyen nhan | Cach xu ly |
|-----|-------------|------------|
| `No module named 'torch'` | Sai interpreter | Chon `fed_rec_env` (Buoc 0) |
| `MemoryError` (Ray) | Da fix lazy loading | Chay lai `quick_verify.py` |
| `CUDA out of memory` | VRAM 4GB khong du | Giam `batch_size: 16` trong config |
| `client_X pickle not found` | Thieu data | Kiem tra `data/processed/multi_category/` |
| Training cham (CPU) | GPU chua dung | Kiem tra `torch.cuda.is_available()` |

---

## Checklist Hoan Thien Do An

- [x] Data 220K samples da xu ly xong
- [x] Pipeline da verify (3 rounds OK)
- [x] Fix MemoryError (lazy loading)
- [x] Config YAML da sua typo
- [ ] **Chay full training 100 rounds** <- DANG LAM
- [ ] Tao bieu do convergence
- [ ] Chay ablation (FedAvg vs FedPer)
- [ ] Cap nhat API server voi model thuc
- [ ] Viet chuong 4 (ket qua thuc nghiem)
- [ ] Chuan bi slides bao cao
