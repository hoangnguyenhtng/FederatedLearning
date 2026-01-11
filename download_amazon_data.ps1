# Download Amazon Reviews 2023 - All_Beauty Dataset
# For Windows PowerShell

Write-Host "=" -NoNewline; Write-Host "="*69
Write-Host "DOWNLOADING AMAZON REVIEWS 2023 - ALL_BEAUTY DATASET"
Write-Host "=" -NoNewline; Write-Host "="*69

# Create directory
$dataDir = "data\raw\amazon_2023"
New-Item -Path $dataDir -ItemType Directory -Force | Out-Null
Write-Host "`n✅ Created directory: $dataDir"

# URLs
$reviewsUrl = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz"
$metaUrl = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_All_Beauty.jsonl.gz"

# Download reviews
Write-Host "`n📥 Downloading reviews (~200MB)..."
$reviewsPath = "$dataDir\All_Beauty.jsonl.gz"
try {
    Invoke-WebRequest -Uri $reviewsUrl -OutFile $reviewsPath -UseBasicParsing
    Write-Host "✅ Downloaded: $reviewsPath"
} catch {
    Write-Host "❌ Failed to download reviews: $_"
    exit 1
}

# Download metadata
Write-Host "`n📥 Downloading metadata (~100MB)..."
$metaPath = "$dataDir\meta_All_Beauty.jsonl.gz"
try {
    Invoke-WebRequest -Uri $metaUrl -OutFile $metaPath -UseBasicParsing
    Write-Host "✅ Downloaded: $metaPath"
} catch {
    Write-Host "❌ Failed to download metadata: $_"
    exit 1
}

# Extract files
Write-Host "`n📦 Extracting files..."

# Extract reviews
try {
    $reviewsGz = [System.IO.File]::OpenRead($reviewsPath)
    $reviewsOut = [System.IO.File]::Create($reviewsPath.Replace(".gz", ""))
    $gzipStream = New-Object System.IO.Compression.GZipStream($reviewsGz, [System.IO.Compression.CompressionMode]::Decompress)
    $gzipStream.CopyTo($reviewsOut)
    $gzipStream.Close()
    $reviewsOut.Close()
    $reviewsGz.Close()
    Write-Host "✅ Extracted reviews"
} catch {
    Write-Host "❌ Failed to extract reviews: $_"
}

# Extract metadata
try {
    $metaGz = [System.IO.File]::OpenRead($metaPath)
    $metaOut = [System.IO.File]::Create($metaPath.Replace(".gz", ""))
    $gzipStream = New-Object System.IO.Compression.GZipStream($metaGz, [System.IO.Compression.CompressionMode]::Decompress)
    $gzipStream.CopyTo($metaOut)
    $gzipStream.Close()
    $metaOut.Close()
    $metaGz.Close()
    Write-Host "✅ Extracted metadata"
} catch {
    Write-Host "❌ Failed to extract metadata: $_"
}

Write-Host "`n✅ DOWNLOAD COMPLETE!"
Write-Host "`nFiles saved to: $dataDir"
Write-Host "  - All_Beauty.jsonl (~400MB)"
Write-Host "  - meta_All_Beauty.jsonl (~200MB)"
Write-Host "`nNext step: Run processing script"
Write-Host "  python src\data_generation\process_amazon_data.py"

