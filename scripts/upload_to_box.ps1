# Upload the D:\transfer bundles to Box via rclone (resumable; re-run freely).
# PREREQ (one-time, needs a human in the browser):
#   & $rclone config create box box
#   -> a browser window opens; log into Box and click "Grant access to Box".
# Then run this script; it copies only what is missing/changed and verifies
# checksums per file. Safe to interrupt and re-run.

$rclone = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter rclone.exe |
    Select-Object -First 1 -ExpandProperty FullName

& $rclone copy D:\transfer box:JHU-xView3/transfer `
    --transfers 4 --checkers 8 `
    --box-upload-cutoff 64M --box-chunk-size 32M `
    --retries 10 --low-level-retries 20 `
    --progress --log-file D:\transfer_upload.log --log-level INFO

& $rclone check D:\transfer box:JHU-xView3/transfer --one-way --log-level NOTICE
