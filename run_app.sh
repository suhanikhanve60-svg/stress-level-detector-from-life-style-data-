#!/bin/bash
pip install -r requirements.txt
[ -f stress_model.pkl ] || python3 stress_detection.py
streamlit run app.py
