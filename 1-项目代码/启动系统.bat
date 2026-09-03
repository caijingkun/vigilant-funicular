@echo off
chcp 65001 >nul 2>&1
title 高校教室能耗智能决策系统 - 启动中...

echo.
echo ============================================================
echo   基于AI的高校教室能耗预测与智能节能决策系统
echo   一键启动脚本
echo ============================================================
echo.

:: 检查Python是否安装
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到Python，请先安装Python 3.8以上版本。
    echo.
    echo 下载地址: https://www.python.org/downloads/
    echo 安装时请勾选 "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

:: 显示Python版本
echo [1/2] 检测到Python环境...
python --version
echo.

:: 检查项目目录
cd /d "%~dp0"
if not exist "run.py" (
    echo [错误] 未找到 run.py 文件，请确保在项目根目录运行。
    echo.
    pause
    exit /b 1
)

:: 启动系统
echo [2/2] 正在启动系统，请稍候...
echo.
python run.py

:: 如果异常退出
if %errorlevel% neq 0 (
    echo.
    echo [错误] 系统启动失败，请查看上方错误信息。
    echo.
    echo 常见问题:
    echo   1. 依赖未安装 - 运行: pip install -r requirements.txt
    echo   2. 端口被占用 - 系统会自动尝试其他端口
    echo   3. Python版本过低 - 需要3.8以上版本
    echo.
    pause
)

exit /b %errorlevel%
