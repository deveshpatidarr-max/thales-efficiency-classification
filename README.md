# AI-Based Manufacturing Efficiency Classification

## What is in this folder
| File | What it does |
|---|---|
| `data/Thales_Group_Manufacturing.csv` | The dataset (100,000 machine readings) |
| `core.py` | Loads data, builds the engineered features |
| `train.py` | Trains 3 models, makes the charts, saves the best model |
| `app.py` | The Streamlit dashboard |
| `models/` | Saved model + results (created by `train.py`, already included) |
| `figures/` | Charts used in the paper |

## Step-by-step: run it on your computer

**Step 1 – Install Python.** Download Python 3.10 or newer from python.org. On Windows, tick "Add Python to PATH" in the installer.

**Step 2 – Unzip this project** to a folder, e.g. `Desktop/thales_project`.

**Step 3 – Open a terminal in that folder.**
Windows: open the folder, click the address bar, type `cmd`, press Enter.
Mac: right-click the folder → "New Terminal at Folder".

**Step 4 – Install the libraries (one time).**
```
pip install -r requirements.txt
```

**Step 5 – (Optional) retrain the models.** The trained model is already included, so you can skip this. To redo it (takes ~5–10 minutes):
```
python train.py
```

**Step 6 – Start the dashboard.**
```
streamlit run app.py
```
Your browser opens at http://localhost:8501. Stop it with Ctrl+C in the terminal.

## How to use the dashboard
- **Sidebar filters** (apply to every tab): machine, operation mode, network quality, time window.
- **Prediction tab:** pick a record (or type your own values). The big label is the predicted efficiency, the bar is the confidence. Move the **sensitivity sliders** to see what happens if speed drops 20% or errors rise 30%.
- **Machine insights:** which machines are mostly High / Low, and trends over days and hours. Select one machine in the sidebar to see just that machine.
- **Explainability:** global importance chart + a "why this record?" breakdown in plain English.
- **Operational monitoring:** efficiency by operation mode and network quality, and sensor vs network vs production impact.

## Publishing the dashboard online (free)
1. Create a free GitHub account and upload this folder to a new repository.
2. Go to share.streamlit.io, sign in with GitHub, click "New app", choose the repository and `app.py`, and deploy.
3. Copy the link you get into your submission.

## What to submit
1. `Research_Paper.docx` – paper (EDA, insights, recommendations)
2. `Executive_Summary.docx` – one-pager for stakeholders
3. The dashboard link (or this folder + a screen recording of the dashboard)
4. This code folder (GitHub link or zip)

## Important finding to mention when you present
In this dataset the label follows a rule on **production speed** and **error rate** only. Sensor and 6G network columns do not influence it. Say this openly – it shows you analysed the data carefully, and the paper explains it.
