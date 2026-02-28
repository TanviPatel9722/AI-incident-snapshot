# AI Incident Observatory (AIO)
A reproducible research notebook suite for exploring the AI Incident Database snapshot.

## What you get
- End-to-end, reproducible workflow (extract → validate → analyze → visualize)
- Clear documentation of assumptions + limitations
- Trend analysis (time, sector, intent, timing, entity)
- Taxonomy alignment (MIT + CSET + GMF)
- Co-occurrence network graphs (risk domains × technologies × failures)

## Quickstart (after downloading this zip)
1) **Unzip**
```bash
unzip ai-incident-observatory.zip
cd ai-incident-observatory
```

2) **Create a virtual environment (recommended)**
```bash
python -m venv .venv
# mac/linux:
source .venv/bin/activate
# windows (powershell):
# .venv\Scripts\Activate.ps1
```

3) **Install dependencies**
```bash
pip install -r requirements.txt
```

4) **Add your dataset snapshot**
Place your snapshot archive at:
- `data/backup-YYYYMMDDHHMMSS.tar.bz2`

For your current file name, you can copy it into `data/`:
```bash
cp /path/to/backup-20260223102103.tar.bz2 data/
```

5) **Extract + build processed tables**
```bash
python -m src.pipeline --snapshot data/backup-20260223102103.tar.bz2
```
This creates:
- `data/processed/*.parquet`
- `outputs/figures/*`

6) **Run notebooks**
```bash
jupyter lab
```
Open notebooks in order: `01_...` → `07_...`

## Notebook order
- `01_Data_Loading_Validation.ipynb`
- `02_Taxonomy_Alignment.ipynb`
- `03_Descriptive_Statistics.ipynb`
- `04_Technical_Failure_Analysis.ipynb`
- `05_Misuse_vs_Systemic_Failure.ipynb`
- `06_Post_Deployment_Analysis.ipynb`
- `07_Responsible_Interpretation.ipynb`

## Dashboard
Run an interactive dashboard:
```bash
streamlit run src/dashboard.py
```

Supported input modes:
- Snapshot URL (`.tar.bz2`)
- Uploaded snapshot archive (`.tar.bz2`)
- Uploaded custom CSV dataset (requires `incidents.csv` and `reports.csv`)

For richer cross-table analysis (including harm/impact views), also include optional files when available:
- `classifications_CSETv1.csv`, `classifications_MIT.csv`, `classifications_GMF.csv`
- `submissions.csv`, `quickadd.csv`, `duplicates.csv`
- `entities.csv`, `entity_relationships.csv`, `incident_links.csv`

## Reproducibility notes
- We log package versions and snapshot filename in `outputs/run_metadata.json`.
- All charts are generated from code (no manual edits).

## License
This repository is intended to be used with the AI Incident Database license included in your snapshot (`license.txt`).
