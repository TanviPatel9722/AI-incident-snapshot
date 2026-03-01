# Submission Summary (Judge Version)

## Project
**AI Incident Observatory** is a reproducible analytics system for the AI Incident Database. It combines a multi-notebook research workflow with an interactive dashboard to analyze incident records, report metadata, and taxonomy labels (MIT, GMF, CSET).

## Problem
AI incident evidence is fragmented across multiple tables and unevenly labeled taxonomies. Snapshot schemas can vary, which makes repeatable analysis difficult and error-prone. Teams often spend more effort on data cleanup than analysis, and results are hard to compare across runs.

## Solution
This system standardizes the workflow end-to-end:
1. Loads snapshot tables (archive or CSVs).
2. Normalizes schema and `incident_id` variations.
3. Builds incident-level enriched views across reports + taxonomy tables.
4. Produces deterministic descriptive and trend analyses.
   --The dashboard renders outputs derived from the same transformation layer used in the notebooks, ensuring consistency between interactive and reproducible          workflows.
6. Exports figures, CSV summaries, PDF reports, and filtered datasets.
7. Includes explicit responsible-interpretation guardrails.

## What Makes It Strong Technically
- Shared utility layer (`src/notebook_utils.py`) for reusable loading/normalization/plotting.
- Robust transforms (`src/transform.py`) with fallback logic when keys are missing.
- Snapshot-aware dashboard (`src/dashboard.py`) that enables only valid graphs for available columns/tables.
- Defensive checks before plots, crosstabs, joins, and date operations.
- Deterministic output generation and stable file naming.

## Snapshot Facts (Current Local Data)
Computed from the local `data/` snapshot on **February 28, 2026**:
- Incidents: **1,367**
- Reports: **6,687**
- MIT coverage: **90.86%** (1,242 incidents)
- GMF coverage: **23.85%** (326 incidents)
- CSET v1 coverage: **15.65%** (214 incidents)
- Incident date coverage range: **1983-09-26 to 2026-02-17**

These numbers are generated from code and update automatically for each snapshot.

## Outputs Delivered
- 7 reproducible notebooks (`01` to `07`) from validation to responsible interpretation.
- Figures in `outputs/figures/` (trend charts, taxonomy distributions, co-occurrence summaries).
- CSV exports for key crosstab outputs.
- Streamlit dashboard with:
  - snapshot URL or file upload,
  - dataset filtering,
  - dynamic graph generation,
  - single-graph PNG download,
  - PDF report generation.

## Responsible AI Position
- Descriptive analysis only; no causal claims.
- Explicit handling of reporting bias, labeling subjectivity, and partial taxonomy coverage.
- Coverage-aware interpretation emphasized throughout notebooks and documentation.
- AI-assisted development disclosed; conclusions remain data-derived and deterministic.

## Real-World Use Cases
- **Policy teams:** trend monitoring and evidence summaries with transparent caveats.
- **Researchers:** reproducible baseline for cross-snapshot studies.
- **Regulators:** structured oversight workflows.
- **Journalists/civil society:** auditable visual outputs with limitations clearly stated.

## Why This Is Hackathon-Ready
- Practical, runnable, and reproducible.
- Lowers the barrier for non-technical stakeholders via an interactive dashboard built on reproducible analytical foundations.
- Strong technical hygiene (modular, defensive, deterministic).
- Clear documentation for judges to rerun quickly.
- Responsible interpretation integrated into the system itself, not added as an afterthought.
