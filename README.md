# AI Incident Observatory

Reproducible notebook system and Streamlit dashboard for structured analysis of the AI Incident Database snapshot.

## What This Project Does
- Loads incident and report snapshot CSVs.
- Normalizes schema differences across snapshots.
- Aligns MIT, GMF, and CSET taxonomy tables at incident level.
- Produces reproducible descriptive analysis and trend visualizations.
- Documents assumptions, limitations, and responsible interpretation.

## Why It Matters
AI incident evidence is often fragmented across incident records, reports, and multiple taxonomies. This project provides a transparent workflow to:
- quantify harm patterns,
- assess taxonomy coverage,
- and support policy-relevant interpretation without overstating conclusions.

## Current Snapshot Metrics (Local `data/`, computed on February 28, 2026)
- Incidents: `1,367`
- Reports: `6,687`
- MIT coverage: `90.86%` of incidents (`1,242 / 1,367`)
- GMF coverage: `23.85%` of incidents (`326 / 1,367`)
- CSET v1 coverage: `15.65%` of incidents (`214 / 1,367`)
- Incident date range (best available incident date column): `1983-09-26` to `2026-02-17`

These values are snapshot-specific and recomputed from data each run.

## Repository Structure
```text
ai-incident-observatory/
├─ data/                                # Snapshot CSVs (or extracted archive contents)
├─ docs/
│  └─ ASSUMPTIONS_LIMITATIONS.md
├─ notebooks/
│  ├─ 01_Data_Loading_Validation.ipynb
│  ├─ 02_Taxonomy_Alignment.ipynb
│  ├─ 03_Descriptive_Statistics.ipynb
│  ├─ 04_Technical_Failure_Analysis.ipynb
│  ├─ 05_Misuse_vs_Systemic_Failure.ipynb
│  ├─ 06_Post_Deployment_Analysis.ipynb
│  └─ 07_Responsible_Interpretation.ipynb
├─ outputs/
│  └─ figures/                          # Exported figures and CSV summaries
├─ src/
│  ├─ io.py                             # Snapshot extract/load helpers
│  ├─ transform.py                      # Core joins and incident-level overview construction
│  ├─ notebook_utils.py                 # Shared notebook helpers
│  ├─ viz.py                            # Figure helpers for pipeline mode
│  ├─ pipeline.py                       # CLI preprocessing pipeline
│  └─ dashboard.py                      # Streamlit dashboard
├─ requirements.txt
└─ run_all.sh
```

## Quick Start
### 1. Create environment and install dependencies
```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Place snapshot files
At minimum for notebook workflow:
- `data/incidents.csv`
- `data/reports.csv`

Optional but recommended:
- `data/classifications_MIT.csv`
- `data/classifications_GMF.csv`
- `data/classifications_CSETv1.csv`
- `data/submissions.csv`
- `data/quickadd.csv`
- `data/duplicates.csv`

### 3. Run preprocessing pipeline (optional, CLI mode)
```bash
python -m src.pipeline --snapshot data/backup-YYYYMMDDHHMMSS.tar.bz2
```

### 4. Run notebooks in order
```bash
jupyter lab
```
Execution order:
1. `01_Data_Loading_Validation.ipynb`
2. `02_Taxonomy_Alignment.ipynb`
3. `03_Descriptive_Statistics.ipynb`
4. `04_Technical_Failure_Analysis.ipynb`
5. `05_Misuse_vs_Systemic_Failure.ipynb`
6. `06_Post_Deployment_Analysis.ipynb`
7. `07_Responsible_Interpretation.ipynb`

### 5. Run dashboard
```bash
streamlit run src/dashboard.py
```

Dashboard input modes:
- Snapshot URL (`.tar.bz2`)
- Uploaded snapshot archive (`.tar.bz2`)
- Uploaded CSV bundle (`incidents.csv` + `reports.csv`; optional taxonomy and auxiliary tables)

## Reproducibility and Reliability
- Shared helpers (`src/notebook_utils.py`) enforce consistent loading, incident-id normalization, and plotting.
- Optional tables are handled defensively; analysis degrades gracefully instead of crashing.
- Date columns are selected from candidate lists, then parsed with coercion.
- Dashboard and transforms use schema-flexible column detection and safe fallbacks.
- Outputs are deterministic given the same input snapshot and package versions.

## Documentation Files
- Full technical report: `DOCUMENTATION.md`
- Judge one-page summary: `SUBMISSION_SUMMARY.md`
- Responsible AI disclosure: `RESPONSIBLE_AI_STATEMENT.md`
- Additional assumptions/limitations: `docs/ASSUMPTIONS_LIMITATIONS.md`

## License Note
Use this repository with snapshot data in compliance with the included dataset license (`data/license.txt`).
