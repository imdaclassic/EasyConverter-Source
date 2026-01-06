@echo off
title EASY YOLO Converter
color 0A

:MENU
cls
echo ===============================================
echo      EASY PORTABLE CONVERTER (v8/11/12)
echo ===============================================
echo.
echo Select your Hardware:
echo.
echo  1) Nvidia 30-40 Series (CUDA 12.x)
echo  2) Nvidia 50+ Series   (CUDA 12.8+)
echo  3) Nvidia GTX / Older  (CUDA 11.8)
echo  4) AMD / CPU Only
echo.

set /p choice="> Enter choice (1-4): "

if "%choice%"=="1" set TYPE=126
if "%choice%"=="2" set TYPE=128
if "%choice%"=="3" set TYPE=118
if "%choice%"=="4" set TYPE=amd

if not defined TYPE (
    echo Invalid choice. Try again.
    timeout /t 2 >nul
    goto MENU
)

cls
echo Launching Converter for Type: %TYPE%...
python run_converter.py --type=%TYPE%

pause
