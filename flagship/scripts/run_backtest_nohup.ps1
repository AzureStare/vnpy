# PowerShell 脚本：使用 nohup 方式启动回测并输出日志
param(
    [string]$LabPath = "lab/flagship_alpha_momentum",
    [string]$Start = "2025-01-01",
    [string]$End = "2025-12-31",
    [string]$Interval = "minute",
    [int]$Capital = 1000000
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)
$LogDir = Join-Path $ProjectRoot "backtest_report"
$LogFile = Join-Path $LogDir "backtest_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

# 创建日志目录
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

Write-Host "=========================================="
Write-Host "启动回测任务"
Write-Host "=========================================="
Write-Host "Lab路径: $LabPath"
Write-Host "日期范围: $Start 到 $End"
Write-Host "K线周期: $Interval"
Write-Host "初始资金: $Capital"
Write-Host "日志文件: $LogFile"
Write-Host "=========================================="
Write-Host ""

# 设置环境变量
$env:PYTHONPATH = $ProjectRoot

# 激活虚拟环境并运行回测
$VenvActivate = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
$BacktestScript = Join-Path $ProjectRoot "flagship\backtest\flagship_alpha_momentum_backtest.py"

# 构建命令
$Command = @"
& `"$VenvActivate`"; python `"$BacktestScript`" --lab-path `"$LabPath`" --start `"$Start`" --end `"$End`" --interval `"$Interval`" --capital $Capital
"@

# 启动后台任务并输出到日志文件
$Job = Start-Job -ScriptBlock {
    param($cmd, $logFile)
    & powershell -Command $cmd *> $logFile
} -ArgumentList $Command, $LogFile

Write-Host "回测任务已启动，Job ID: $($Job.Id)"
Write-Host "日志文件: $LogFile"
Write-Host ""
Write-Host "使用以下命令监控日志:"
Write-Host "  Get-Content `"$LogFile`" -Wait -Tail 50"
Write-Host ""
Write-Host "或使用监控脚本:"
Write-Host "  python flagship/scripts/monitor_backtest_log.py `"$LogFile`""
Write-Host ""

# 保存 Job ID 到文件，方便后续管理
$JobInfoFile = Join-Path $LogDir "backtest_job_info.txt"
@{
    JobId = $Job.Id
    LogFile = $LogFile
    StartTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Parameters = @{
        LabPath = $LabPath
        Start = $Start
        End = $End
        Interval = $Interval
        Capital = $Capital
    }
} | ConvertTo-Json | Out-File $JobInfoFile -Encoding UTF8

Write-Host "任务信息已保存到: $JobInfoFile"

