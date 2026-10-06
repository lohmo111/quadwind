param([string]$waitFor, [string]$plan, [string]$log)
# ждём, пока живы python-процессы с $waitFor в командной строке, затем запускаем очередь $plan
while (Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -like "*$waitFor*" }) { Start-Sleep 60 }
Start-Process -FilePath "D:\RL\project\.venv\Scripts\python.exe" -ArgumentList "-u","scripts\run_series.py","--exp","e7","--plan",$plan,"--parallel","4","--threads","2","--total-steps","3000000" -WorkingDirectory "D:\RL\project" -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError "$log.err" -Wait
