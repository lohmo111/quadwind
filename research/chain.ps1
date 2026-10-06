param([string]$waitFor, [string]$plan, [string]$log)
while (Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -like "*$waitFor*" }) { Start-Sleep 60 }
& "D:\RL\project\.venv\Scripts\python.exe" -u scripts\run_series.py --exp e7 --plan $plan --parallel 4 --threads 2 --total-steps 3000000 *> $log
