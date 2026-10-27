"""Demo app. Run:  pip install streamlit  then  streamlit run app.py
(Run stress_detection.py first so stress_model.pkl exists.)"""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Stress Level Detector", page_icon="🧠")
st.title("🧠 Stress Level Detector")
st.caption("Enter your lifestyle habits and the model predicts your stress level.")

model = joblib.load("stress_model.pkl")

c1, c2 = st.columns(2)
age = c1.slider("Age", 15, 60, 21)
gender = c2.selectbox("Gender", ["Male", "Female", "Other"])
sleep = c1.slider("Sleep (hours/night)", 3.0, 10.0, 7.0, 0.5)
screen = c2.slider("Screen time (hours/day)", 1.0, 16.0, 6.0, 0.5)
caffeine = c1.slider("Caffeine (cups/day)", 0, 8, 2)
exercise = c2.slider("Exercise (min/week)", 0, 420, 120, 10)
work = c1.slider("Study/work (hours/day)", 1.0, 14.0, 7.0, 0.5)
social = c2.slider("Social time (hours/week)", 0.0, 30.0, 8.0, 0.5)
diet = st.selectbox("Diet quality", ["Poor", "Average", "Good"], index=1)

if st.button("Predict"):
    row = pd.DataFrame([{
        "age": age, "gender": gender, "sleep_hours": sleep,
        "screen_time_hours": screen, "caffeine_cups": caffeine,
        "exercise_min_per_week": exercise, "study_work_hours": work,
        "social_hours_per_week": social, "diet_quality": diet}])
    pred = model.predict(row)[0]
    proba = dict(zip(model.classes_, model.predict_proba(row)[0]))
    {"Low": st.success, "Medium": st.warning, "High": st.error}[pred](
        f"Predicted stress level: **{pred}**")
    st.bar_chart(pd.Series(proba))
    st.caption("This is a student ML project, not a medical diagnosis.")
