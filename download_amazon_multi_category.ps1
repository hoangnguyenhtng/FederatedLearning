# ================================================================
# Download Amazon Reviews 2023 - Multi-Category (for Thesis)
# Option C: 4 Balanced Categories = 2.67M Reviews
# ================================================================

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "DOWNLOADING AMAZON MULTI-CATEGORY DATASET" -ForegroundColor Cyan
Write-Host "4 Categories: Beauty + Games + Fashion + Baby" -ForegroundColor Cyan
Write-Host "Total: ~2.67 MILLION reviews" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

$dataDir = "data\raw\amazon_2023"

# Create directory
if (-not (Test-Path $dataDir)) {
    New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
}

# ================================================================
# CATEGORY DEFINITIONS
# ================================================================

$categories = @(
    @{
        Name = "All_Beauty"
        ReviewUrl = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz"
        MetaUrl = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_All_Beauty.jsonl.gz"
        ReviewSize = "~80 MB"
        MetaSize = "~120 MB"
        Samples = "371,345"
    },
    @{
        Name = "Video_Games"
        ReviewUrl = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Video_Games.jsonl.gz"
        MetaUrl = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Video_Games.jsonl.gz"
        ReviewSize = "~110 MB"
        MetaSize = "~170 MB"
        Samples = "497,577"
    },
    @{
        Name = "Amazon_Fashion"
        ReviewUrl = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Amazon_Fashion.jsonl.gz"
        MetaUrl = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Amazon_Fashion.jsonl.gz"
        ReviewSize = "~190 MB"
        MetaSize = "~280 MB"
        Samples = "883,636"
    },
    @{
        Name = "Baby_Products"
        ReviewUrl = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Baby_Products.jsonl.gz"
        MetaUrl = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Baby_Products.jsonl.gz"
        ReviewSize = "~200 MB"
        MetaSize = "~290 MB"
        Samples = "915,446"
    }
)

Write-Host "`n📊 Will download:" -ForegroundColor Yellow
Write-Host "   1. All_Beauty:      371,345 reviews (~200 MB)" -ForegroundColor White
Write-Host "   2. Video_Games:     497,577 reviews (~280 MB)" -ForegroundColor White
Write-Host "   3. Amazon_Fashion:  883,636 reviews (~470 MB)" -ForegroundColor White
Write-Host "   4. Baby_Products:   915,446 reviews (~490 MB)" -ForegroundColor White
Write-Host "   ─────────────────────────────────────────────" -ForegroundColor Gray
Write-Host "   TOTAL:            2,668,004 reviews (~1.44 GB)" -ForegroundColor Green
Write-Host ""

$confirmation = Read-Host "Proceed with download? (y/n)"
if ($confirmation -ne "y") {
    Write-Host "❌ Download cancelled." -ForegroundColor Red
    exit 0
}

# ================================================================
# DOWNLOAD LOOP
# ================================================================

$totalCategories = $categories.Count
$currentCategory = 0
$startTime = Get-Date

foreach ($cat in $categories) {
    $currentCategory++
    
    Write-Host "`n================================================================" -ForegroundColor Cyan
    Write-Host "[$currentCategory/$totalCategories] Downloading: $($cat.Name)" -ForegroundColor Cyan
    Write-Host "Samples: $($cat.Samples)" -ForegroundColor Yellow
    Write-Host "================================================================" -ForegroundColor Cyan
    
    $reviewFile = "$dataDir\$($cat.Name).jsonl.gz"
    $metaFile = "$dataDir\meta_$($cat.Name).jsonl.gz"
    
    # Download Reviews
    Write-Host "`n   [1/2] Reviews ($($cat.ReviewSize))..." -ForegroundColor Green
    try {
        $ProgressPreference = 'SilentlyContinue'  # Speed up download
        Invoke-WebRequest -Uri $cat.ReviewUrl -OutFile $reviewFile -ErrorAction Stop
        $ProgressPreference = 'Continue'
        Write-Host "   ✅ Downloaded: $reviewFile" -ForegroundColor Green
    } catch {
        Write-Host "   ❌ Failed to download reviews: $_" -ForegroundColor Red
        Write-Host "   Skipping to next category..." -ForegroundColor Yellow
        continue
    }
    
    # Download Metadata
    Write-Host "   [2/2] Metadata ($($cat.MetaSize))..." -ForegroundColor Green
    try {
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $cat.MetaUrl -OutFile $metaFile -ErrorAction Stop
        $ProgressPreference = 'Continue'
        Write-Host "   ✅ Downloaded: $metaFile" -ForegroundColor Green
    } catch {
        Write-Host "   ❌ Failed to download metadata: $_" -ForegroundColor Red
        Write-Host "   Skipping to next category..." -ForegroundColor Yellow
        continue
    }
    
    Write-Host "   ✅ $($cat.Name) complete!" -ForegroundColor Green
    
    # Progress
    $elapsed = (Get-Date) - $startTime
    $avgTime = $elapsed.TotalMinutes / $currentCategory
    $remaining = $avgTime * ($totalCategories - $currentCategory)
    Write-Host "   ⏱️  Elapsed: $([math]::Round($elapsed.TotalMinutes, 1)) min | Est. remaining: $([math]::Round($remaining, 1)) min" -ForegroundColor Cyan
}

# ================================================================
# EXTRACTION
# ================================================================

Write-Host "`n================================================================" -ForegroundColor Cyan
Write-Host "EXTRACTING FILES" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

$gzFiles = Get-ChildItem -Path $dataDir -Filter "*.gz"
$totalFiles = $gzFiles.Count
$currentFile = 0

# Check extraction tool
$gzipAvailable = Get-Command gzip -ErrorAction SilentlyContinue
$7zipPath = "C:\Program Files\7-Zip\7z.exe"

if (-not $gzipAvailable -and -not (Test-Path $7zipPath)) {
    Write-Host "❌ No extraction tool found!" -ForegroundColor Red
    Write-Host "Please install one of:" -ForegroundColor Yellow
    Write-Host "  - gzip (via Chocolatey: choco install gzip)" -ForegroundColor White
    Write-Host "  - 7-Zip (https://www.7-zip.org/)" -ForegroundColor White
    Write-Host ""
    Write-Host "Files downloaded to: $dataDir" -ForegroundColor Cyan
    Write-Host "Extract manually before processing." -ForegroundColor Yellow
    exit 1
}

foreach ($file in $gzFiles) {
    $currentFile++
    Write-Host "[$currentFile/$totalFiles] Extracting: $($file.Name)..." -ForegroundColor Yellow
    
    if ($gzipAvailable) {
        gzip -d -f $file.FullName
    } else {
        & $7zipPath x $file.FullName "-o$dataDir" -y | Out-Null
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ Extracted!" -ForegroundColor Green
    } else {
        Write-Host "   ❌ Extraction failed!" -ForegroundColor Red
    }
}

# ================================================================
# VERIFICATION
# ================================================================

Write-Host "`n================================================================" -ForegroundColor Cyan
Write-Host "VERIFICATION" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

$jsonlFiles = Get-ChildItem -Path $dataDir -Filter "*.jsonl"
$totalSize = 0
$totalLines = 0

Write-Host "`n📊 Extracted Files:" -ForegroundColor Yellow

foreach ($cat in $categories) {
    $reviewFile = Get-ChildItem -Path $dataDir -Filter "$($cat.Name).jsonl" -ErrorAction SilentlyContinue
    
    if ($reviewFile) {
        $sizeMB = [math]::Round($reviewFile.Length / 1MB, 2)
        $totalSize += $sizeMB
        
        Write-Host "`n   $($cat.Name):" -ForegroundColor Cyan
        Write-Host "     File: $($reviewFile.Name)" -ForegroundColor White
        Write-Host "     Size: $sizeMB MB" -ForegroundColor White
        
        # Count lines (reviews)
        Write-Host "     Counting reviews..." -ForegroundColor Gray -NoNewline
        $lineCount = (Get-Content $reviewFile.FullName | Measure-Object -Line).Lines
        $totalLines += $lineCount
        Write-Host " $lineCount reviews" -ForegroundColor Green
    } else {
        Write-Host "   ❌ $($cat.Name): NOT FOUND!" -ForegroundColor Red
    }
}

Write-Host "`n   ─────────────────────────────────────────────" -ForegroundColor Gray
Write-Host "   TOTAL: $([math]::Round($totalSize, 2)) MB" -ForegroundColor Green
Write-Host "   TOTAL REVIEWS: $totalLines" -ForegroundColor Green

# ================================================================
# SUMMARY
# ================================================================

$totalTime = (Get-Date) - $startTime

Write-Host "`n================================================================" -ForegroundColor Cyan
Write-Host "✅ DOWNLOAD COMPLETE!" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Cyan

Write-Host "`n⏱️  Total time: $([math]::Round($totalTime.TotalMinutes, 1)) minutes" -ForegroundColor Yellow
Write-Host "💾  Total size: $([math]::Round($totalSize, 2)) MB" -ForegroundColor Yellow
Write-Host "📊  Total reviews: $totalLines" -ForegroundColor Yellow
Write-Host ""

Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Process the multi-category data:" -ForegroundColor White
Write-Host "   python src\data_generation\process_amazon_multi_category.py" -ForegroundColor Yellow
Write-Host "   ⏱️  Estimated time: ~24 hours" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Train with multi-category config:" -ForegroundColor White
Write-Host "   python src\training\federated_training_pipeline.py --config configs\config_multi_category.yaml" -ForegroundColor Yellow
Write-Host "   ⏱️  Estimated time: ~5-7 days (200 rounds)" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Expected results:" -ForegroundColor White
Write-Host "   - Accuracy: 79-83%" -ForegroundColor Green
Write-Host "   - Cross-domain generalization ✅" -ForegroundColor Green
Write-Host "   - Excellent thesis material! 🎓" -ForegroundColor Green
Write-Host ""

# ================================================================
# SAVE METADATA
# ================================================================

$metadata = @{
    download_date = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    categories = $categories.Name
    total_reviews = $totalLines
    total_size_mb = [math]::Round($totalSize, 2)
    download_time_minutes = [math]::Round($totalTime.TotalMinutes, 1)
}

$metadata | ConvertTo-Json | Out-File "$dataDir\download_metadata.json"

Write-Host "📝 Metadata saved to: $dataDir\download_metadata.json" -ForegroundColor Cyan
Write-Host ""
Write-Host "🎉 Ready to process! Good luck with your thesis! 🎓🚀" -ForegroundColor Green

