# Force all .txt label files (including all files in train and val folders) to UTF-8 (without BOM) encoding
# 1. Set your labels folder paths

# In the monkeyv7 project root directory, open PowerShell and run this script. This will permanently fix encoding issues for all label files.
$trainDir = "C:\Users\wangs\monkeyv7\datasets\wildlife\labels\train_pseudo"
$valDir   = "C:\Users\wangs\monkeyv7\datasets\wildlife\labels\val"

# Create a UTF-8 encoder without BOM
$utf8NoBOM = New-Object System.Text.UTF8Encoding $false  # $false means "no BOM"

# 2. Process Train directory
$trainFiles = Get-ChildItem -Path $trainDir -Filter "*.txt" -Recurse
Write-Host "--- Processing Train directory ($($trainFiles.Count) files) ---"

foreach ($file in $trainFiles) {
    try {
        # [System.IO.File] class provides more precise control over encoding
        $text = [System.IO.File]::ReadAllText($file.FullName)
        [System.IO.File]::WriteAllText($file.FullName, $text, $utf8NoBOM)
    } catch {
        Write-Error "  - Conversion failed: $($file.Name) - $($_.Exception.Message)"
    }
}

# 3. Process Val directory
$valFiles = Get-ChildItem -Path $valDir -Filter "*.txt" -Recurse
Write-Host "--- Processing Val directory ($($valFiles.Count) files) ---"

foreach ($file in $valFiles) {
    try {
        $text = [System.IO.File]::ReadAllText($file.FullName)
        [System.IO.File]::WriteAllText($file.FullName, $text, $utf8NoBOM)
    } catch {
        Write-Error "  - Conversion failed: $($file.Name) - $($_.Exception.Message)"
    }
}

Write-Host "---"
Write-Host "✅ [Encoding Conversion Complete] ✅"
Write-Host "All .txt label files have been converted to UTF-8 (without BOM) format."
Write-Host "You can now rerun the yolo train command."
