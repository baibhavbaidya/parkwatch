# ParkWatch 🚦
### AI-Driven Parking Hotspot Detection & Congestion Impact Intelligence

> Turning 298,445 violation records into targeted enforcement intelligence for Bengaluru Traffic Police.

**Live Demo:** https://parkwatch-bb.streamlit.app/  
**Dataset:** Bengaluru Traffic Police parking violation records (Nov 2023 – Apr 2024)

---

## The Problem

Bengaluru's traffic police rely on patrol-based, reactive enforcement to tackle illegal parking. There is no data-driven system to identify *where* violations cluster, *how badly* they impact traffic flow, or *when* enforcement should be deployed. Officers are spread thin across the city with no prioritization.

---

## What ParkWatch Does

- **Detects parking violation hotspots** using DBSCAN spatial clustering on GPS coordinates — grouping nearby violations into real geographic clusters rather than splitting large junctions into dozens of artificial fragments
- **Scores each hotspot** with a Congestion Impact Score combining three independently validated signals:
  - Violation density (50%)
  - Estimated road capacity loss (30%) — based on traffic engineering principles
  - Repeat-vehicle ratio (20%) — entrenched offenders
- **Produces a ranked priority list** of enforcement zones with time-of-day breakdown for targeted deployment

---

## Results

- 298,445 violation records analyzed
- 2,104 distinct geo-clusters identified across Bengaluru
- 485 validated hotspots (≥20 violations)
- Top hotspots: Elite Junction (24,339), Safina Plaza (20,067), KR Market (16,264), Sagar Theatre (11,670)

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data Processing | Python, pandas, numpy |
| Spatial Clustering | scikit-learn (DBSCAN) |
| Dashboard | Streamlit, pydeck, Plotly |
| Data Hosting | Hugging Face Datasets |
| Deployment | Streamlit Community Cloud |

---

## Project Structure

```
parkwatch/
  app.py            ← Streamlit dashboard
  pipeline.py       ← Full data pipeline (cleaning, clustering, scoring)
  requirements.txt  ← Python dependencies
  README.md
```

> **Note:** The raw dataset (`jan_to_may_police_violation_anonymized791b166.csv`) and generated CSVs (`geo_cell_scores.csv`, `cleaned_violations.csv`) are not included in this repo due to file size limits. The dashboard downloads the generated CSVs automatically from Hugging Face on first load. To regenerate from the raw data, follow the instructions below.

---

## Instructions to Run

### Option 1 — View the live dashboard
Just open: https://parkwatch-bb.streamlit.app/

### Option 2 — Run locally

**Prerequisites:** Python 3.8+

```bash
# 1. Clone the repo
git clone https://github.com/baibhavbaidya/parkwatch
cd parkwatch

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run app.py
```

On first run, `geo_cell_scores.csv` and `cleaned_violations.csv` are downloaded automatically from Hugging Face. This may take 30–60 seconds on the first cold start. Subsequent runs use the cached local files and load instantly.

### Option 3 — Regenerate data from scratch

If you have the raw dataset:

```bash
# Place the raw CSV in a data/ subfolder
mkdir data
# Copy jan_to_may_police_violation_anonymized791b166.csv into data/

# Edit pipeline.py: update the path in load_and_clean() to point to your local file
# Then run:
python pipeline.py
```

This produces `geo_cell_scores.csv` and `cleaned_violations.csv` in the project root, which `app.py` will use automatically.

---

## How the Congestion Impact Score Works

The dataset contains only violation records — no direct traffic-flow sensor data. The score is an honest estimate, not a fake measurement, built from:

1. **Violation Density (50%)** — log-scaled count of recorded violations. High recurring count over 6 months = proven problem location.

2. **Estimated Capacity Loss (30%)** — based on Highway Capacity Manual-style lane-blockage analysis. Each violation type is assigned an estimated % of road capacity lost (e.g. parking in a main road = 45% capacity loss, footpath parking = 5%).

3. **Repeat-Vehicle Ratio (20%)** — fraction of violations from vehicles caught more than once at the same location. Flags entrenched enforcement gaps. Validated as genuinely independent from capacity loss (correlation: -0.06).

All three components validated: each correlates 0.47–0.51 with the final score. No single factor dominates, no redundancy.

---

## Limitations

- Congestion impact is an **estimate**, not a direct measurement — no speed/travel-time data available
- Historical dataset only (Nov 2023 – Apr 2024), not a real-time feed
- DBSCAN parameters (eps=120m, min_samples=3) and hotspot threshold (≥20 violations) are reasonable but not calibrated against ground truth

---

## Built For

HackerEarth Hackathon — Round 2  
Theme: Poor Visibility on Parking-Induced Congestion