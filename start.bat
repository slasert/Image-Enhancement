@echo off
cd /d "%~dp0"
echo Sunucu baslatiliyor...
echo Tarayicida su adresi ac: http://localhost:8000
echo Kapatmak icin bu pencereyi kapat.
echo.
python -m uvicorn main:app --reload
pause
