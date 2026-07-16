# Bundle the node-transfer set into large tars under D:\transfer\ (Box-friendly:
# ~200 large files instead of ~285k small ones — Box API per-file overhead makes
# raw small-file uploads slower than the bytes). Plain tar, no compression
# (float16 chips and model weights barely compress; speed wins).
# Contents (see docs/NODE_HANDOFF.md "Box bulk-transfer path"):
#   chips/<scene>.tar        150 study-scene chip dirs (from frozen splits.json)
#   raw/<scene>.tar          39 dev+test raw scene dirs (whole-scene evals)
#   runs.tar                 checkpoints + results of finished cells
#   weights.tar              pinned FM checkpoints + license notes
#   labels.tar               xView3 label CSVs
#   MANIFEST.txt             file counts per bundle for post-transfer checks
# Resumable: existing non-empty tars are skipped.

$ErrorActionPreference = 'Stop'
Set-Location D:\JHU-xView3
$out = 'D:\transfer'
New-Item -ItemType Directory -Force "$out\chips", "$out\raw" | Out-Null

$splits = (Get-Content data\splits.json | ConvertFrom-Json).splits
$study = @($splits.train) + @($splits.dev) + @($splits.test) | Sort-Object
$evalScenes = @($splits.dev) + @($splits.test) | Sort-Object

$manifest = @()
function Bundle($tarPath, $workDir, $item) {
    if ((Test-Path $tarPath) -and ((Get-Item $tarPath).Length -gt 0)) {
        Write-Output "skip (exists): $tarPath"
        return
    }
    tar -cf $tarPath -C $workDir $item
    Write-Output "made: $tarPath"
}

foreach ($s in $study) {
    Bundle "$out\chips\$s.tar" 'data\chips' $s
    $script:manifest += "chips/$s.tar files=$((Get-ChildItem data\chips\$s -File).Count)"
}
foreach ($s in $evalScenes) {
    Bundle "$out\raw\$s.tar" 'data\raw\xview3\GRD' $s
    $script:manifest += "raw/$s.tar files=$((Get-ChildItem data\raw\xview3\GRD\$s -File).Count)"
}
Bundle "$out\runs.tar" '.' 'runs'
Bundle "$out\weights.tar" 'data' 'weights'
Bundle "$out\labels.tar" 'data\raw\xview3' 'labels'
$manifest += "runs.tar files=$((Get-ChildItem runs -Recurse -File).Count)"
$manifest += "weights.tar files=$((Get-ChildItem data\weights -Recurse -File).Count)"
$manifest += "labels.tar files=$((Get-ChildItem data\raw\xview3\labels -File).Count)"
$manifest | Out-File -Encoding ascii "$out\MANIFEST.txt"
"total: {0:N1} GB in $((Get-ChildItem $out -Recurse -File).Count) files" -f ((Get-ChildItem $out -Recurse -File | Measure-Object Length -Sum).Sum/1GB)
