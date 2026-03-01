# Responsible AI Statement

## Scope and Intent
This project analyzes documented AI incidents using structured snapshot data from the AI Incident Database. Its purpose is to provide reproducible descriptive analysis, not predictive scoring or automated decision-making.

## Data and Coverage Realities
The dataset is a curated record of documented incidents and associated reports. It is not a complete census of all AI harms. In the current local snapshot (computed February 28, 2026), taxonomy coverage is uneven:
- MIT: 90.86% of incidents
- GMF: 23.85% of incidents
- CSET v1: 15.65% of incidents

Any interpretation must account for these coverage gaps.

## Bias and Limitation Acknowledgment
The system explicitly recognizes:
- **Reporting bias:** incidents with higher visibility are more likely to appear.
- **Selection bias:** underreporting of private, non-English, or less-publicized incidents.
- **Taxonomy subjectivity:** MIT/GMF/CSET labels are human-assigned and can vary.
- **Temporal bias:** reporting lag and archive updates can distort trend timing.

## Interpretation Boundaries
Outputs from this project are descriptive summaries of observed records.
- They do **not** establish causality.
- They do **not** estimate true real-world incident prevalence.
- They do **not** assign definitive blame to organizations or sectors.

Charts are intended to support investigation, not replace contextual analysis.

## Risk Controls Implemented in the System
- Schema normalization for robust cross-snapshot use.
- Defensive checks for missing tables/columns before analysis.
- Graceful degradation when optional taxonomies are unavailable.
- Explicit notebook section on responsible interpretation (`07_Responsible_Interpretation.ipynb`).
- Dynamic dashboard graph gating so unavailable data is not misrepresented.

## Transparency and Reproducibility
- All results are generated from local snapshot files with deterministic code paths.
- No hidden model inference is used to create conclusions.
- Outputs can be rerun notebook-by-notebook and audited by reviewers.

## AI Tooling Disclosure
AI-assisted tools were used for development support (refactoring, formatting, documentation, and robustness improvements).  
AI tools were **not** used to fabricate incident findings or generate analytical conclusions independent of data.

## Responsible Communication Commitment
When presenting results, this project commits to:
- stating taxonomy coverage and data limitations,
- avoiding sensational framing,
- distinguishing frequency from severity,
- and keeping causal claims out of descriptive outputs unless independently validated.
