# Step 1. Extract TEST_ and PRODUCTION_ keys from .env
$envKeys = Get-Content .env | ForEach-Object {
    if ($_ -match '^(TEST_[A-Z0-9_]+)=') { $matches[1] }
    elseif ($_ -match '^(PRODUCTION_[A-Z0-9_]+)=') { $matches[1] }
}

# Step 2. Prepare results array
$result = @()

# Step 3. Scan project for each key
foreach ($key in $envKeys) {
    $matches = Get-ChildItem -Recurse -Include *.py,*.js,*.ts,*.env,*.json |
        Where-Object { $_.FullName -notmatch '\\(node_modules|dist|.git|.venv|docker)\\' } |
        Select-String $key -SimpleMatch

    if ($matches) {
        foreach ($m in $matches) {
            $result += [PSCustomObject]@{
                Key   = $key
                File  = $m.Path
                Line  = $m.LineNumber
                Match = $m.Line.Trim()
            }
        }
    }
    else {
        # If the key was never found, mark it as unused
        $result += [PSCustomObject]@{
            Key   = $key
            File  = "<unused>"
            Line  = ""
            Match = ""
        }
    }
}

# Step 4. Save results to CSV
$result | Export-Csv -Path "env_keys_usage.csv" -NoTypeInformation -Encoding UTF8
Write-Host "✅ Done. Results saved to env_keys_usage.csv"
