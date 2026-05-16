$ErrorActionPreference = "Continue"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Runner = Join-Path $ProjectRoot "scripts\runs\run_qwen_tam_prompt_sweep.py"
$PromptsFile = Join-Path $ProjectRoot "prompt_sets\prompt_sensitivity_v2.json"
$LogDir = Join-Path $ProjectRoot "outputs\prompt_sensitivity\_logs"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Images = @(
    @{ Id = 331352; Label = "bathroom_toilet" },
    @{ Id = 426253; Label = "microwave_bottle" },
    @{ Id = 30213;  Label = "kitchen_counter" },
    @{ Id = 555009; Label = "desk_monitor" },
    @{ Id = 393226; Label = "street_traffic" },
    @{ Id = 133645; Label = "bench_boat" }
)

foreach ($Image in $Images) {
    $RunName = "overnight_prompt_sensitivity_v2f_rawscanpath_$($Image.Label)"
    $LogPath = Join-Path $LogDir "$RunName.log"

    if (Test-Path $LogPath) {
        Remove-Item -LiteralPath $LogPath -Force
    }

    "[$(Get-Date -Format s)] START img_id=$($Image.Id) label=$($Image.Label)" | Tee-Object -FilePath $LogPath -Append

    & $PythonExe $Runner `
        --img-id $Image.Id `
        --image-label $Image.Label `
        --prompts-file $PromptsFile `
        --run-name $RunName `
        --max-new-tokens 192 `
        --final-layer-only `
        --scanpath-threshold-percentile 95 `
        --scanpath-min-hotspot-area 64 `
        --scanpath-topk-hotspots 3 `
        --scanpath-max-link-distance-ratio 0.18 `
        2>&1 | Tee-Object -FilePath $LogPath -Append

    if ($LASTEXITCODE -ne 0) {
        "[$(Get-Date -Format s)] FAILED img_id=$($Image.Id) label=$($Image.Label) exit_code=$LASTEXITCODE" | Tee-Object -FilePath $LogPath -Append
        exit $LASTEXITCODE
    }

    "[$(Get-Date -Format s)] END img_id=$($Image.Id) label=$($Image.Label)" | Tee-Object -FilePath $LogPath -Append
}
