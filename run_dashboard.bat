@echo off
echo Starting Bridge Health Dashboard...
start "Flask API" "%~dp0venv\Scripts\python.exe" app/app.py
timeout /t 3
"%~dp0venv\Scripts\python.exe" -m streamlit run app/app_dashboard.py
pause
