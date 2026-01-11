"""
Quick script to re-process Amazon data with NaN fixes
"""
import os
import shutil
from pathlib import Path

# 1. Backup old data
old_data_dir = Path("data/amazon_2023_processed")
if old_data_dir.exists():
    backup_dir = Path("data/amazon_2023_processed_backup_with_nan")
    print(f"📦 Backing up old data to: {backup_dir}")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    shutil.copytree(old_data_dir, backup_dir)
    
    # Remove old data
    print(f"🗑️  Removing old data: {old_data_dir}")
    shutil.rmtree(old_data_dir)

# 2. Re-run processing with fixed code
print("\n" + "="*70)
print("🔧 RE-PROCESSING AMAZON DATA (with NaN fixes)")
print("="*70)

# Use virtual environment python
venv_python = Path("fed_rec_env/Scripts/python.exe")
if venv_python.exists():
    print(f"✅ Using venv: {venv_python}")
    os.system(f"{venv_python} src/data_generation/process_amazon_data.py")
else:
    print("⚠️  Virtual env not found, using system python")
    os.system("python src/data_generation/process_amazon_data.py")

print("\n" + "="*70)
print("✅ Re-processing complete!")
print("="*70)
print("\nNext steps:")
print("  1. Run: python check_data_nan.py   (verify no NaN)")
print("  2. Run: python src\\training\\federated_training_pipeline.py")

