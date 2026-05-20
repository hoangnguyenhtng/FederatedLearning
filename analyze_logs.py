import json, statistics

with open('experiments/fedper_multi_category_20260517_141543/evaluation/evaluation_report.json') as f:
    report = json.load(f)
with open('experiments/fedper_multi_category_20260517_141543/metrics/training_history.json') as f:
    history = json.load(f)

clients = report['per_client']

# ---- Group analysis (client 0-29 vs 30-39) ----
group_a = [c for c in clients if c['client_id'] < 30]
group_b = [c for c in clients if c['client_id'] >= 30]

def avg(lst, key): return sum(x[key] for x in lst) / len(lst)

print("=== CLIENT GROUP ANALYSIS ===")
print(f"Clients  0-29 (n={len(group_a)}): acc={avg(group_a,'accuracy'):.3f}  loss={avg(group_a,'loss'):.4f}  nDCG@10={avg(group_a,'ndcg@10'):.4f}")
print(f"Clients 30-39 (n={len(group_b)}): acc={avg(group_b,'accuracy'):.3f}  loss={avg(group_b,'loss'):.4f}  nDCG@10={avg(group_b,'ndcg@10'):.4f}")
print(f"Accuracy gap : {avg(group_a,'accuracy') - avg(group_b,'accuracy'):+.3f}  <-- suspicious if large")

# ---- Accuracy stats ----
all_accs  = [c['accuracy'] for c in clients]
all_loss  = [c['loss']     for c in clients]
all_ndcg  = [c['ndcg@10'] for c in clients]
all_prec  = [c['precision'] for c in clients]
all_rec   = [c['recall']   for c in clients]
all_mrr   = [c['mrr']      for c in clients]

print("\n=== OVERALL METRIC STATS ===")
for name, vals in [("accuracy", all_accs), ("loss", all_loss), ("nDCG@10", all_ndcg),
                   ("precision", all_prec), ("recall", all_rec), ("MRR", all_mrr)]:
    print(f"  {name:12s}: mean={statistics.mean(vals):.4f}  std={statistics.stdev(vals):.4f}  min={min(vals):.4f}  max={max(vals):.4f}")

# ---- Loss uniformity ----
loss_range = max(all_loss) - min(all_loss)
print(f"\n=== LOSS UNIFORMITY ===")
print(f"  All client losses in [{min(all_loss):.4f}, {max(all_loss):.4f}]  range={loss_range:.4f}")
if loss_range < 0.02:
    print("  ⚠️  CRITICAL: Loss is near-identical across ALL clients => model not personalizing!")
    print("      The personal head is not learning client-specific patterns.")

# ---- nDCG vs accuracy paradox ----
print("\n=== nDCG@10 vs ACCURACY PARADOX ===")
print("  Best nDCG@10 clients:")
for c in sorted(clients, key=lambda x: x['ndcg@10'], reverse=True)[:5]:
    cid = c['client_id']
    print(f"    Client {cid:2d}: nDCG@10={c['ndcg@10']:.4f}  acc={c['accuracy']:.4f}  precision={c['precision']:.4f}")

print("  Worst accuracy clients:")
for c in sorted(clients, key=lambda x: x['accuracy'])[:5]:
    cid = c['client_id']
    print(f"    Client {cid:2d}: acc={c['accuracy']:.4f}  nDCG@10={c['ndcg@10']:.4f}  loss={c['loss']:.4f}")

# ---- Accuracy trend ----
accs = history['metrics_distributed']['accuracy']
acc_vals = [v for r, v in accs]
print("\n=== ACCURACY TREND (100 rounds) ===")
checkpoints = [0, 9, 19, 29, 49, 69, 89, 99]
for i in checkpoints:
    r, v = accs[i]
    print(f"  Round {r:3d}: {v:.4f} ({v*100:.1f}%)")
print(f"  Net change round1->100: {acc_vals[-1]-acc_vals[0]:+.4f}")
print(f"  Max ever reached      : {max(acc_vals):.4f} at round {accs[acc_vals.index(max(acc_vals))][0]}")

# ---- test_loss vs train_loss mirror check ----
train_losses = history['losses_distributed']
test_losses  = history['metrics_distributed']['test_loss']
print("\n=== TRAIN LOSS vs TEST_LOSS MIRROR CHECK ===")
print("  If test_loss == train_loss => evaluation uses TRAIN set (data leak!)")
match_count = sum(1 for (r1,l1),(r2,l2) in zip(train_losses, test_losses) if abs(l1-l2) < 1e-9)
print(f"  Exact matches: {match_count}/100  => {'⚠️ test_loss = train_loss (BUG!)' if match_count > 50 else 'OK, different values'}")
