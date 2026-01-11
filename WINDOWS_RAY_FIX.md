# 🔧 FIX: Windows Ray Access Violation Error

## Vấn đề
**Windows fatal exception: access violation** trong Ray khi chạy Flower simulation.

## Nguyên nhân
1. **Memory pressure**: Ray cần nhiều memory trên Windows
2. **Too many concurrent operations**: Quá nhiều clients chạy đồng thời
3. **Ray bug trên Windows**: Known issue với Ray trên Windows

## Giải pháp đã áp dụng

### 1. Giảm số clients concurrent
**File:** `configs/config.yaml`
```yaml
federated:
  fraction_fit: 0.4      # Giảm từ 0.6 → 0.4 (4 clients/round thay vì 6)
  fraction_evaluate: 0.3  # Giảm từ 0.5 → 0.3 (3 clients/round thay vì 5)
  min_fit_clients: 2     # Giảm từ 3 → 2
  min_evaluate_clients: 2 # Giảm từ 3 → 2
```

### 2. Giảm batch size
**File:** `configs/config.yaml`
```yaml
training:
  batch_size: 16  # Giảm từ 32 → 16 (giảm memory per batch)
```

### 3. Giảm memory per client
**File:** `src/training/federated_training_pipeline.py`
```python
client_resources = {
    "num_cpus": 1,
    "num_gpus": 0.0,
    "memory": 500 * 1024 * 1024  # 500MB per client (giảm từ default)
}
```

### 4. Set Ray environment variables
**File:** `src/training/federated_training_pipeline.py`
```python
os.environ.setdefault("RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE", "1")
os.environ.setdefault("RAY_DEDUP_LOGS", "1")
```

## Giải pháp thay thế (nếu vẫn crash)

### Option 1: Giảm số clients tổng
```yaml
federated:
  num_clients: 5  # Giảm từ 10 → 5
```

### Option 2: Giảm batch size hơn nữa
```yaml
training:
  batch_size: 8  # Giảm từ 16 → 8
```

### Option 3: Giảm local epochs
```yaml
training:
  local_epochs: 2  # Giảm từ 3 → 2
```

### Option 4: Dùng threading backend (không dùng Ray)
**Lưu ý:** Flower VCE yêu cầu Ray, nhưng có thể dùng threading cho testing:
```python
# Không dùng simulation, dùng threading thay thế
# (Cần refactor code)
```

## Monitoring

Sau khi apply fixes, monitor:
1. **Memory usage**: Task Manager → Memory
2. **Ray logs**: Check for memory warnings
3. **Training stability**: Có crash sau nhiều rounds không?

## Kết quả mong đợi

- ✅ Giảm memory pressure
- ✅ Ít concurrent operations
- ✅ Training ổn định hơn
- ⚠️ Training chậm hơn (trade-off)

## Nếu vẫn crash

1. **Restart Ray**: `ray stop` và chạy lại
2. **Giảm num_clients**: Từ 10 → 5
3. **Giảm batch_size**: Từ 16 → 8
4. **Chạy trên Linux/WSL**: Ray ổn định hơn trên Linux

