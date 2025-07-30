Write-Host "=== 🖥️ System Specs ===`n"

# CPU Info
Write-Host "CPU:"
Get-CimInstance Win32_Processor | ForEach-Object {
    Write-Host "  Name: $($_.Name)"
    Write-Host "  Cores: $($_.NumberOfCores)"
    Write-Host "  Logical processors: $($_.NumberOfLogicalProcessors)"
    Write-Host "  Max Clock Speed: $($_.MaxClockSpeed) MHz"
}

# RAM Info
Write-Host "`nRAM:"
Get-CimInstance Win32_ComputerSystem | ForEach-Object {
    $ramGB = [math]::Round($_.TotalPhysicalMemory / 1GB, 2)
    Write-Host "  Total Physical Memory: $ramGB GB"
}

# GPU Info
Write-Host "`nGPU:"
Get-CimInstance Win32_VideoController | ForEach-Object {
    Write-Host "  Name: $($_.Name)"
    $vramGB = [math]::Round($_.AdapterRAM / 1GB, 2)
    Write-Host "  VRAM: $vramGB GB"
    Write-Host "  Driver Version: $($_.DriverVersion)"
}

# OS Info
Write-Host "`nOS:"
Get-CimInstance Win32_OperatingSystem | ForEach-Object {
    Write-Host "  Caption: $($_.Caption)"
    Write-Host "  Version: $($_.Version)"
    Write-Host "  Architecture: $($_.OSArchitecture)"
    Write-Host "  Build Number: $($_.BuildNumber)"
}
