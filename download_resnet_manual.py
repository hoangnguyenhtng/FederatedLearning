"""
Manual download ResNet-50 weights
For environments with network issues
"""

import urllib.request
import os
from pathlib import Path

print("=" * 70)
print("MANUAL RESNET-50 WEIGHT DOWNLOAD")
print("=" * 70)

# URL and cache path
url = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
cache_file = cache_dir / "resnet50-0676ba61.pth"

# Create directory
cache_dir.mkdir(parents=True, exist_ok=True)

print(f"\nDownloading from: {url}")
print(f"Saving to: {cache_file}")
print(f"\nThis may take a few minutes (~100MB)...")

try:
    # Download with progress
    def reporthook(blocknum, blocksize, totalsize):
        readsofar = blocknum * blocksize
        if totalsize > 0:
            percent = readsofar * 1e2 / totalsize
            s = f"\r{percent:5.1f}% {readsofar:,} / {totalsize:,} bytes"
            print(s, end='')
            if readsofar >= totalsize:
                print()
        else:
            print(f"\rRead {readsofar:,} bytes", end='')
    
    urllib.request.urlretrieve(url, cache_file, reporthook)
    
    print(f"\n✅ Successfully downloaded!")
    print(f"   File: {cache_file}")
    print(f"   Size: {cache_file.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"\n✅ Now you can run process_amazon_data.py with real images!")
    print(f"   Set: skip_image_download=False")
    
except Exception as e:
    print(f"\n❌ Download failed: {e}")
    print(f"\n💡 Alternative:")
    print(f"   1. Download manually from browser: {url}")
    print(f"   2. Save to: {cache_file}")
    print(f"   3. Or use skip_image_download=True (dummy features)")

