# AI Incident Observatory: Technical Documentation

## 1) Executive Summary

### Problem Context
AI incidents are documented across multiple related tables (incidents, reports, and taxonomy annotations), but these datasets are not uniform across snapshots. Field names vary, optional files may be missing, and taxonomy coverage is uneven. This creates a practical barrier for researchers, policy analysts, and journalists who need reproducible, transparent analysis.

The core problem is not only statistical analysis; it is reliable data preparation and responsible interpretation under incomplete, biased reporting conditions.

### What the System Does
This project provides a reproducible analytical workflow that:
- loads and validates AI Incident Database snapshot files,
- normalizes schema and incident identifiers across tables,
- aligns MIT, GMF, and CSET taxonomy views at incident level,
- generates descriptive and temporal analyses,
- exports figures/tables with deterministic, rerunnable logic,
- and documents assumptions and limitations explicitly.

The system is implemented as:
- a seven-notebook research workflow (`notebooks/01` to `07`),
- shared utility modules (`src/notebook_utils.py`, `src/transform.py`, `src/io.py`),
- and an optional Streamlit dashboard (`src/dashboard.py`) for interactive use.

### Why It Matters
Without schema-aware and reproducible infrastructure, analysts can draw inconsistent results from the same snapshot. This project addresses that by enforcing:
- consistent incident-level joins,
- transparent table availability checks,
- conservative handling of missing or ambiguous fields,
- and explicit non-causal framing of all visual outputs.

### Key Insights Produced (Snapshot-Specific Example)
For the local snapshot currently in `data/` (computed February 28, 2026):
- Incidents: `1,367`
- Reports: `6,687`
- MIT incident coverage: `90.86%` (`1,242/1,367`)
- GMF incident coverage: `23.85%` (`326/1,367`)
- CSET v1 incident coverage: `15.65%` (`214/1,367`)
- Best available incident-date range: `1983-09-26` to `2026-02-17`

These numbers show why coverage-aware interpretation is mandatory: broad conclusions can differ substantially depending on whether analysis is done on MIT-labeled incidents, GMF-labeled incidents, or CSET-labeled incidents.

### Why Reproducibility Is Central
The system is designed so judges or researchers can rerun analysis from raw snapshot files with minimal manual intervention. Reproducibility is achieved through:
- explicit notebook order,
- shared helper functions,
- deterministic transformations,
- defensive handling of optional tables,
- and stable output file naming conventions.

### Barrier Reduction for AI Risk Research
This workflow lowers the barrier by providing:
- standardized loading and schema normalization,
- documented assumptions in each notebook,
- pre-structured taxonomy analysis patterns,
- and a dashboard that adapts to available snapshot tables.

The result is a practical, credible entry point for structured AI incident analysis without requiring custom data engineering for every new snapshot.

---

## 2) System Architecture Overview

### End-to-End Flow
```text
Raw Snapshot (.tar.bz2 or CSV bundle)
        |
        v
Data Ingestion (src/io.py, notebook_utils.load_data)
        |
        v
Schema Normalization (column casing/spacing, incident_id normalization)
        |
        v
Cross-Table Linking (reports <-> incidents, optional duplicates canonicalization)
        |
        v
Taxonomy Alignment (MIT, GMF, CSET views + incident-level aggregates)
        |
        v
Notebook Analyses (01..07) and Dashboard Views
        |
        v
Figures / CSV outputs / PDF export / Interpretation notes
```

### Architecture Diagram (Component View)
```text
                               +------------------------------+
                               |   Snapshot Input Layer       |
                               |------------------------------|
                               | .tar.bz2 archive OR CSV set  |
                               +---------------+--------------+
                                               |
                                               v
                     +-------------------------+--------------------------+
                     |                 Data Access Layer                  |
                     |----------------------------------------------------|
                     | src/io.py: extract_snapshot, find_csv_root,        |
                     |            load_tables                              |
                     | src/notebook_utils.py: load_data                   |
                     +-------------------------+--------------------------+
                                               |
                                               v
                 +-----------------------------+-----------------------------+
                 |            Normalization + Transform Layer                |
                 |-----------------------------------------------------------|
                 | normalize_columns / normalize_incident_id                 |
                 | join_reports_to_incidents                                 |
                 | taxonomy_alignment                                        |
                 | build_incident_overview (MIT/GMF/CSET/evidence features) |
                 +-----------------------------+-----------------------------+
                                               |
                              +----------------+----------------+
                              |                                 |
                              v                                 v
           +------------------+-------------------+   +---------+------------------+
           | Notebook Analytics (01..07)          |   | Streamlit Dashboard        |
           |--------------------------------------|   |-----------------------------|
           | validation, trends, taxonomy,        |   | dynamic filters, charts,   |
           | failure analysis, interpretation     |   | CSV/PDF/PNG exports        |
           +------------------+-------------------+   +---------+------------------+
                              |                                 |
                              +----------------+----------------+
                                               |
                                               v
                        +----------------------+----------------------+
                        |              Output Layer                  |
                        |--------------------------------------------|
                        | outputs/figures/*.png, *.csv              |
                        | filtered exports, PDF report, summaries    |
                        +--------------------------------------------+
```

### Layer Explanation
- **Raw Snapshot Input**
  - Supports archive extraction and direct CSV loading.
  - Core required tables: `incidents.csv`, `reports.csv`.
  - Optional tables enrich analysis (MIT, GMF, CSET, submissions, quickadd, duplicates).

- **Ingestion and Validation**
  - `src/io.py` discovers and loads known files.
  - Unknown CSVs can still be loaded as generic tables.
  - Missing required files raise explicit errors.

- **Normalization and Joining**
  - `normalize_columns` and `normalize_incident_id` standardize schema.
  - `join_reports_to_incidents` merges report evidence and supports fallback counting when direct `incident_id` is missing in reports.
  - `build_incident_overview` constructs incident-level enriched data with optional taxonomy and evidence features.

- **Analysis**
  - Notebooks are organized from data validation to interpretation.
  - All notebooks preserve descriptive (not causal) framing.
  - Output artifacts are saved under `outputs/figures/`.

- **Interactive Layer**
  - `src/dashboard.py` provides snapshot URL upload, archive upload, and CSV upload.
  - Graph availability is dynamic and constrained to valid columns/tables.
  - Export support includes filtered CSV, PDF report, and single-graph PNG.

### Modular Components
- `src/io.py`: extraction, CSV root detection, table loading.
- `src/notebook_utils.py`: reusable notebook helpers.
- `src/transform.py`: robust cross-table transforms and overview table building.
- `src/pipeline.py`: CLI batch preprocessing and baseline figures.
- `src/dashboard.py`: interactive analysis and exports.

### Configuration Design
Each notebook defines a clear config block:
- `DATA_PATH`
- `OUTPUT_PATH`
- `TOP_N`
- `DATE_CANDIDATES`

This keeps runs transparent and easy to retarget to a new snapshot location.

### Snapshot Compatibility
Compatibility is achieved by:
- candidate-based date/field selection,
- incident-id normalization across naming variants,
- optional table handling,
- and explicit fallback behavior when columns are missing.

---

## 3) Reproducibility Framework

### Dependencies
From `requirements.txt`:
- `pandas>=2.1`
- `numpy>=1.26`
- `matplotlib>=3.8`
- `scikit-learn>=1.3`
- `networkx>=3.2`
- `pyarrow>=14.0`
- `jupyterlab>=4.0`
- `nbformat>=5.9`
- `tqdm>=4.66`
- `streamlit>=1.31`

### Environment Setup
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

### Execution Order
Primary research path:
1. `notebooks/01_Data_Loading_Validation.ipynb`
2. `notebooks/02_Taxonomy_Alignment.ipynb`
3. `notebooks/03_Descriptive_Statistics.ipynb`
4. `notebooks/04_Technical_Failure_Analysis.ipynb`
5. `notebooks/05_Misuse_vs_Systemic_Failure.ipynb`
6. `notebooks/06_Post_Deployment_Analysis.ipynb`
7. `notebooks/07_Responsible_Interpretation.ipynb`

Optional pipeline mode:
```bash
python -m src.pipeline --snapshot data/backup-YYYYMMDDHHMMSS.tar.bz2
```

Optional interactive mode:
```bash
streamlit run src/dashboard.py
```

### Deterministic Design Choices
- Fixed output naming conventions in notebooks and pipeline.
- Deterministic grouping/filtering operations.
- No ML model fitting or stochastic inference for conclusions.
- Where graph layout randomness exists (`cooccurrence_network`), `seed=42` is fixed.

### Schema Normalization Strategy
- Lowercase + underscore column normalization.
- Incident ID normalization from common variants (`incident_id`, `incident id`, `incidentid`, etc.).
- Candidate-based detection for date columns and taxonomy-like columns.

### Defensive Coding Practices
- Required table checks with explicit error messages.
- Optional table support with graceful skip behavior.
- Column existence checks before plotting/grouping/crosstab.
- Safe parsing (`errors="coerce"` for datetimes and numeric conversions).
- Guardrails for expensive text operations (for example, skipping explode on very long tag text).

### Snapshot Variation Handling
- Optional tables can be absent without crashing analysis.
- Alternative field names are discovered through helper functions.
- Dashboard graph options are enabled only if supporting columns exist and contain usable values.

---

## 4) Notebook-by-Notebook Documentation

## 4.1 `01_Data_Loading_Validation.ipynb`
- **Purpose**
  - Validate core tables and estimate taxonomy coverage before deeper analysis.
- **Question Answered**
  - Is the snapshot structurally usable, and how complete are taxonomy labels?
- **Inputs**
  - `incidents.csv`, `reports.csv`, `submissions.csv`, optional `classifications_MIT.csv`, `classifications_GMF.csv`, `classifications_CSETv1.csv`.
- **Outputs**
  - Coverage metrics printed in notebook.
  - Missingness summaries (`isna().mean()`).
- **Assumptions**
  - `incident_id` can be detected from common naming patterns.
  - Coverage is measured against incident rows present in the snapshot.
- **Limitations**
  - Coverage metrics do not indicate label quality.
  - Missingness does not distinguish “not applicable” from “not recorded.”
- **Responsible Interpretation Note**
  - High coverage in one taxonomy does not make it representative of all harm categories.

## 4.2 `02_Taxonomy_Alignment.ipynb` (Temporal Dynamics)
- **Purpose**
  - Analyze incident/report temporal trends and proxy reporting lag.
- **Question Answered**
  - How do incident and report counts evolve over time, and what does a lag proxy suggest?
- **Inputs**
  - `incidents.csv`, `reports.csv`, `submissions.csv`.
- **Outputs**
  - `10_incidents_per_year.png`
  - `11_reports_per_year.png`
  - `12_reports_monthly_rolling.png`
  - `13_reporting_lag_proxy_hist.png`
- **Assumptions**
  - Chosen date columns are the best available proxies from candidate sets.
  - URL merge between submissions and reports is a valid subset for lag proxy analysis.
- **Limitations**
  - Reporting lag is a proxy, not ground-truth latency.
  - Time trends are sensitive to reporting intensity and archive practices.
- **Responsible Interpretation Note**
  - Do not treat observed trend slopes as direct evidence of real-world harm incidence growth without external context.

## 4.3 `03_Descriptive_Statistics.ipynb`
- **Purpose**
  - Produce descriptive distributions for MIT, GMF, CSET, and report metadata.
- **Question Answered**
  - What are the dominant taxonomy categories and report corpus characteristics?
- **Inputs**
  - `incidents.csv`, `reports.csv`, optional `classifications_MIT.csv`, `classifications_GMF.csv`, `classifications_CSETv1.csv`.
- **Outputs**
  - MIT figures: `20_...` to `26_...`
  - GMF figures: `40_gmf_goal_top15.png`, `41_gmf_tech_top15.png`, `42_gmf_failure_top15.png`
  - CSET figures (conditional): `30_cset_*_top15.png`
  - Report metadata figures: `50_reports_top_source_domains.png`, `51_reports_languages.png`, `52_reports_tags.png`
- **Assumptions**
  - Top-N views are sufficient for exploratory interpretation.
  - Candidate keyword matching identifies intended GMF/CSET columns.
- **Limitations**
  - Long-tail categories are compressed by top-N plotting.
  - Tags can be noisy and inconsistently formatted.
- **Responsible Interpretation Note**
  - Frequency is not severity; high count categories are not automatically highest risk impact.

## 4.4 `04_Technical_Failure_Analysis.ipynb`
- **Purpose**
  - Focus on GMF technical failures and co-occurrence with goals/technologies.
- **Question Answered**
  - Which technical failures appear most frequently in GMF-labeled incidents?
- **Inputs**
  - `incidents.csv`, `classifications_GMF.csv`.
- **Outputs**
  - `60_gmf_failures_top15.png`
  - `61_gmf_failures_percent.png`
  - In-notebook crosstabs (failure by goal, failure by technology).
- **Assumptions**
  - Keyword-based failure/goal/technology column selection maps correctly.
  - GMF labels are internally consistent enough for descriptive summaries.
- **Limitations**
  - GMF coverage is partial.
  - Co-occurrence does not imply causal mechanism.
- **Responsible Interpretation Note**
  - Use these outputs to prioritize further investigation, not to assign definitive blame.

## 4.5 `05_Misuse_vs_Systemic_Failure.ipynb`
- **Purpose**
  - Join MIT and GMF labels at incident level to compare intent and failure mixtures.
- **Question Answered**
  - How do GMF failure patterns vary by MIT intent classes?
- **Inputs**
  - `incidents.csv`, `classifications_MIT.csv`, `classifications_GMF.csv`.
- **Outputs**
  - `70_intent_among_gmf.png`
  - `71_failure_by_intent_pct.csv`
- **Assumptions**
  - MIT intent and GMF failures are joinable by `incident_id`.
  - Inner join population is analytically meaningful.
- **Limitations**
  - Analysis excludes incidents missing either MIT or GMF labels.
  - Labeling conventions may vary across incident types and time.
- **Responsible Interpretation Note**
  - Differences by intent are descriptive of labeled subsets, not universal properties of all incidents.

## 4.6 `06_Post_Deployment_Analysis.ipynb`
- **Purpose**
  - Analyze MIT timing labels and risk-domain composition by timing class; include lag proxy.
- **Question Answered**
  - How do risk domains distribute across MIT timing categories?
- **Inputs**
  - `classifications_MIT.csv`, `reports.csv`, `submissions.csv`.
- **Outputs**
  - `80_mit_timing_counts.png`
  - `81_domain_by_timing_pct.csv`
  - `82_reporting_lag_proxy_hist.png`
- **Assumptions**
  - MIT `timing` labels are stable enough for descriptive cross-tab analysis.
  - URL merge again serves as a lag proxy subset.
- **Limitations**
  - Timing labels can reflect annotation subjectivity.
  - Lag proxy depends on subset linkability and date quality.
- **Responsible Interpretation Note**
  - Timing distributions should not be read as complete lifecycle risk estimates.

## 4.7 `07_Responsible_Interpretation.ipynb`
- **Purpose**
  - Consolidate dataset scope, strengths, caveats, and non-causal interpretation guidance.
- **Question Answered**
  - What should analysts conclude cautiously, and what should they avoid concluding?
- **Inputs**
  - `incidents.csv`, `reports.csv`, `submissions.csv`, optional MIT/GMF/CSET files.
- **Outputs**
  - Structured interpretation guidance in notebook narrative.
  - Snapshot overview card with coverage indicators.
- **Assumptions**
  - Coverage rates and metadata are adequate for framing limitations.
- **Limitations**
  - It is a narrative guidance notebook, not a statistical correction model.
- **Responsible Interpretation Note**
  - This notebook is the guardrail: descriptive outputs are evidence summaries, not causal claims.

---

## 5) Responsible AI and Data Ethics

### Reporting Bias
- Incidents are more likely to be captured when there is media visibility, documentation, or public reporting.
- Underreported geographies, languages, and sectors may be systematically missing.

### Dataset and Selection Bias
- The dataset is curated and post hoc.
- Near misses, confidential incidents, and non-public failures are likely underrepresented.

### Taxonomy Subjectivity
- MIT/GMF/CSET labels are human-generated categories.
- Taxonomy assignment can vary across annotators and update cycles.

### Post-Deployment Skew
- Public incidents are often documented after harms are observed.
- This can overrepresent deployment-stage failures relative to earlier lifecycle issues.

### Coverage Constraints
- Coverage is uneven across taxonomies in this snapshot:
  - MIT: `90.86%`
  - GMF: `23.85%`
  - CSET v1: `15.65%`
- Any cross-taxonomy comparison must state which subset is analyzed.

### Descriptive Analysis Is Not Causal Inference
- Frequency charts and crosstabs describe observed records.
- They do not establish causal effects or prevalence in all deployed AI systems.

### Anti-Sensationalism Guidance
- Avoid equating trend growth with uncontrolled harm growth without denominator context.
- Avoid ranking organizations or sectors as inherently “most harmful” from frequency-only charts.
- Pair outputs with caveat statements in policy communication.

---

## 6) Validation and Reliability Guarantees

### Schema Variation Handling
- Incident identifiers are normalized from common naming variants.
- Date columns are chosen from candidate sets with non-null checks.
- Taxonomy columns are discovered by keyword heuristics where needed (especially GMF/CSET).

### Missing-Column and Missing-Table Detection
- Required tables raise explicit errors.
- Optional tables resolve to `None` and downstream analyses skip safely.
- Plotting and crosstabs are guarded by precondition checks.

### Graceful Degradation
- If a taxonomy table is absent, related analysis sections are skipped with clear messages.
- Dashboard graph menu only shows charts with valid supporting data.
- Missing key fields trigger informative messages rather than notebook/dashboard crashes.

### Deterministic Outputs
- Stable grouping logic and fixed naming for generated artifacts.
- No stochastic model components driving conclusions.
- Fixed random seed in co-occurrence graph layout (`seed=42`).

### Snapshot-Agnostic Behavior
- New snapshots can be loaded with minimal changes if core files exist.
- Optional enrichment tables are opportunistic, not hard dependencies.

### Design Trade-Offs
- **Trade-off 1:** broad schema flexibility vs strict schema contracts.
  - Chosen approach favors practical robustness for real snapshot variation.
- **Trade-off 2:** descriptive transparency vs predictive complexity.
  - Chosen approach intentionally avoids opaque modeling to preserve interpretability.
- **Trade-off 3:** top-N readability vs long-tail completeness.
  - Chosen approach prioritizes interpretable figures while keeping raw data available.

---

## 7) Real-World Relevance

### For Policymakers
- Provides structured evidence on harm categories and temporal patterns.
- Makes taxonomy coverage explicit, reducing risk of overgeneralized conclusions.

### For Researchers
- Offers a reproducible baseline for cross-snapshot studies.
- Supports extension with additional taxonomies or domain-specific stratification.

### For Regulators and Oversight Teams
- Enables recurring monitoring with consistent methodology.
- Supports early identification of shifts in labeled risk domains.

### For Journalists and Civil Society
- Generates transparent, reproducible descriptive visuals.
- Helps frame responsible reporting with caveats on coverage and bias.

### Beyond Hackathon Feasibility
- Uses common Python tooling and plain CSV interfaces.
- Modular architecture supports operationalization in CI workflows or scheduled runs.
- Dashboard mode supports non-notebook stakeholders without changing core analytical logic.

---

## 8) Future Extensions (Responsible and Grounded)

- **Cross-snapshot drift summaries**
  - Compare category proportions and coverage changes over time.

- **Taxonomy consistency checks**
  - Validate whether similar incidents receive divergent taxonomy labels.

- **Structured quality diagnostics**
  - Add snapshot-level reports for missingness, duplicates, and column drift.

- **Programmatic report packaging**
  - Produce standardized publication bundles (figures + caveats + metadata).

- **Dashboard governance mode**
  - Add machine-readable audit metadata for each generated chart (source columns, filters, snapshot hash).

These extensions maintain descriptive and reproducible principles without introducing unsupported predictive claims.

---

## 9) AI Usage Disclosure (Hackathon Requirement)

### How AI Tools Were Used
AI-assisted coding tools were used for:
- code structuring and refactoring support,
- documentation drafting and formatting,
- and robustness improvements for schema variation handling.

### How AI Tools Were Not Used
- Analytical conclusions were not generated by AI models.
- Incident labels and metrics come directly from snapshot data and deterministic code execution.
- No generative model is used to infer hidden labels, causal effects, or policy conclusions.

### Reproducibility and Traceability
- Results are deterministic from input snapshot + code + dependency versions.
- Outputs can be rerun and audited notebook-by-notebook.
- AI assistance improved developer productivity, not analytical ground truth.

---

## Appendix: Representative Output Artifacts
- Temporal trends: `10_incidents_per_year.png`, `11_reports_per_year.png`, `12_reports_monthly_rolling.png`
- MIT distributions: `20_...` to `26_...`
- CSET and GMF distributions: `30_...`, `40_...`, `42_...`, `60_...`, `61_...`
- Interaction and post-deployment summaries: `70_intent_among_gmf.png`, `71_failure_by_intent_pct.csv`, `80_mit_timing_counts.png`, `81_domain_by_timing_pct.csv`, `82_reporting_lag_proxy_hist.png`

All figures are generated from code and can be reproduced by rerunning the notebooks in sequence.
