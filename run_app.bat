@echo off
echo Installing requirements (first time only)...
pip install -r requirements.txt
if not exist stress_model.pkl python stress_detection.py
streamlit run app.py
pause
