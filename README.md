# Stress Level Detection from Lifestyle Data
Machine Learning Classification Mini Project

Classifies a person's stress level (Low / Medium / High) from lifestyle habits:
sleep, screen time, caffeine, exercise, study/work hours, social time, diet, age, gender.

## Contents
| File | Purpose |
|---|---|
| `stress_detection.py` | Full pipeline: data, preprocessing, 4 models, evaluation, plots, saves model |
| `stress_detection.ipynb` | Same project as a Jupyter notebook (with inline plots) |
| `app.py` | Streamlit demo app (live predictions) |
| `stress_detector_prototype.html` | Offline demo: double-click to open in any browser, no install needed |
| `stress_data.csv` | Dataset (replace with your own survey data, same columns) |
| `stress_model.pkl` | Trained model used by `app.py` |
| `outputs/` | Charts and the model comparison table |
| `Project_Report.docx` | Written report for submission |
| `run_app.bat` / `run_app.sh` | One-click launchers for the Streamlit app |

## Quick start
**Easiest demo (no install):** double-click `stress_detector_prototype.html`.

**Full app:**
1. Install Python 3.9+ from python.org
2. Windows: double-click `run_app.bat`. Mac/Linux: `./run_app.sh`
   (or manually: `pip install -r requirements.txt`, `python stress_detection.py`, `streamlit run app.py`)

## Using your own survey data
Create `stress_data.csv` with these columns, then rerun `python stress_detection.py`:

`age, gender, sleep_hours, screen_time_hours, caffeine_cups, exercise_min_per_week, study_work_hours, social_hours_per_week, diet_quality, stress_level`

- gender: Male / Female / Other
- diet_quality: Poor / Average / Good
- stress_level: Low / Medium / High
- Missing values are allowed; they are filled automatically.

After retraining, update the numbers in the report from `outputs/model_comparison.csv`.

## Note
Educational project only, not a medical diagnosis tool.
