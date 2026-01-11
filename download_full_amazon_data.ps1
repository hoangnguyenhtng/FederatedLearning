# ================================================================
# Download FULL Amazon Reviews 2023 Dataset (for Thesis)
# ================================================================

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "DOWNLOADING FULL AMAZON REVIEWS 2023 DATASET" -ForegroundColor Cyan
Write-Host "For Graduation Thesis - Federated Learning Project" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

$dataDir = "data\raw\amazon_2023"

# Create directory
if (-not (Test-Path $dataDir)) {
    New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
}

# ================================================================
# OPTION 1: All_Beauty (FULL - ~371k reviews)
# ================================================================
Write-Host "`n[OPTION 1] All_Beauty - FULL Dataset (~371k reviews)" -ForegroundColor Yellow
Write-Host "This will download ~200MB of data" -ForegroundColor Yellow

$reviewUrl = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz"
$metaUrl = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_All_Beauty.jsonl.gz"

$reviewFile = "$dataDir\All_Beauty_FULL.jsonl.gz"
$metaFile = "$dataDir\meta_All_Beauty_FULL.jsonl.gz"

Write-Host "`n1. Downloading reviews (~80MB)..." -ForegroundColor Green
try {
    Invoke-WebRequest -Uri $reviewUrl -OutFile $reviewFile -ErrorAction Stop
    Write-Host "   ✅ Downloaded: $reviewFile" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Failed to download reviews: $_" -ForegroundColor Red
    exit 1
}

Write-Host "`n2. Downloading metadata (~120MB)..." -ForegroundColor Green
try {
    Invoke-WebRequest -Uri $metaUrl -OutFile $metaFile -ErrorAction Stop
    Write-Host "   ✅ Downloaded: $metaFile" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Failed to download metadata: $_" -ForegroundColor Red
    exit 1
}

# ================================================================
# Extract files
# ================================================================
Write-Host "`n3. Extracting files..." -ForegroundColor Green

# Check if gzip is available
$gzipAvailable = Get-Command gzip -ErrorAction SilentlyContinue

if ($gzipAvailable) {
    Write-Host "   Using gzip..." -ForegroundColor Cyan
    gzip -d -f $reviewFile
    gzip -d -f $metaFile
} else {
    Write-Host "   ⚠️  gzip not found, using 7-Zip alternative..." -ForegroundColor Yellow
    
    # Try 7-Zip
    $7zipPath = "C:\Program Files\7-Zip\7z.exe"
    if (Test-Path $7zipPath) {
        & $7zipPath x $reviewFile "-o$dataDir" -y
        & $7zipPath x $metaFile "-o$dataDir" -y
    } else {
        Write-Host "   ❌ Please install 7-Zip or gzip to extract .gz files" -ForegroundColor Red
        Write-Host "   Files are in: $dataDir" -ForegroundColor Yellow
        Write-Host "   Extract manually before proceeding." -ForegroundColor Yellow
        exit 1
    }
}

# Verify extraction
$reviewJsonl = "$dataDir\All_Beauty_FULL.jsonl"
$metaJsonl = "$dataDir\meta_All_Beauty_FULL.jsonl"

if ((Test-Path $reviewJsonl) -and (Test-Path $metaJsonl)) {
    Write-Host "   ✅ Extracted successfully!" -ForegroundColor Green
    
    # Get file sizes
    $reviewSize = (Get-Item $reviewJsonl).Length / 1MB
    $metaSize = (Get-Item $metaJsonl).Length / 1MB
    
    Write-Host "`n📊 Dataset Info:" -ForegroundColor Cyan
    Write-Host "   Reviews: $([math]::Round($reviewSize, 2)) MB" -ForegroundColor White
    Write-Host "   Metadata: $([math]::Round($metaSize, 2)) MB" -ForegroundColor White
    
    # Count lines (estimated reviews)
    Write-Host "`n   Counting reviews (this may take a minute)..." -ForegroundColor Yellow
    $reviewCount = (Get-Content $reviewJsonl | Measure-Object -Line).Lines
    Write-Host "   Total reviews: $reviewCount" -ForegroundColor Green
    
} else {
    Write-Host "   ❌ Extraction failed. Files not found." -ForegroundColor Red
    exit 1
}

Write-Host "`n================================================================" -ForegroundColor Cyan
Write-Host "✅ DOWNLOAD COMPLETE!" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Cyan

Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "1. Process the data:" -ForegroundColor White
Write-Host "   python src\data_generation\process_amazon_data_full.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "2. Train with thesis config:" -ForegroundColor White
Write-Host "   python src\training\federated_training_pipeline.py --config configs\config_thesis.yaml" -ForegroundColor Cyan
Write-Host ""
Write-Host "⏱️  Processing ~371k samples will take ~3-4 hours" -ForegroundColor Yellow
Write-Host "💾  Final processed data: ~2-3 GB" -ForegroundColor Yellow
Write-Host ""

# ================================================================
# OPTIONAL: Download additional categories
# ================================================================
Write-Host "`n================================================================" -ForegroundColor Cyan
Write-Host "OPTIONAL: Download More Categories (for larger thesis dataset)" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

Write-Host "`nAvailable categories:" -ForegroundColor Yellow
Write-Host "- Toys_and_Games (~1.6M reviews)" -ForegroundColor White
Write-Host "- Sports_and_Outdoors (~3.9M reviews)" -ForegroundColor White
Write-Host "- Digital_Music (~1.3M reviews)" -ForegroundColor White
Write-Host ""
Write-Host "To download more, edit this script and add category URLs." -ForegroundColor Yellow
Write-Host "See: https://amazon-reviews-2023.github.io/main.html" -ForegroundColor Cyan

