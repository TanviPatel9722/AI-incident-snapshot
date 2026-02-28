from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from .io import extract_snapshot, find_csv_root, load_tables, save_run_metadata
from .transform import normalize_columns, parse_dates, join_reports_to_incidents, taxonomy_alignment
from .viz import bar_counts, incidents_over_time, cooccurrence_network

def get_versions() -> dict:
    import importlib
    pkgs = ["pandas","numpy","matplotlib","scikit_learn","networkx","pyarrow"]
    versions = {}
    for p in pkgs:
        try:
            mod = importlib.import_module(p if p != "scikit_learn" else "sklearn")
            versions[p] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[p] = "not_installed"
    return versions

def main():
    ap = argparse.ArgumentParser(description="Extract and preprocess AI Incident Database snapshot.")
    ap.add_argument("--snapshot", required=True, help="Path to backup-*.tar.bz2 snapshot")
    ap.add_argument("--out", default="data", help="Output base directory (default: data)")
    args = ap.parse_args()

    snapshot = Path(args.snapshot)
    out_base = Path(args.out)

    extracted_root = extract_snapshot(snapshot, out_base)
    csv_root = find_csv_root(extracted_root)

    tables = load_tables(csv_root)

    # Normalize + process
    incidents = normalize_columns(tables["incidents"])
    reports = normalize_columns(tables["reports"])

    incidents = parse_dates(incidents)
    incidents_enriched = join_reports_to_incidents(incidents, reports)

    processed_dir = out_base / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    incidents_enriched.to_parquet(processed_dir / "incidents_enriched.parquet", index=False)
    reports.to_parquet(processed_dir / "reports.parquet", index=False)

    mit = normalize_columns(tables.get("mit")) if tables.get("mit") is not None else None
    csetv1 = normalize_columns(tables.get("csetv1")) if tables.get("csetv1") is not None else None
    gmf = normalize_columns(tables.get("gmf")) if tables.get("gmf") is not None else None

    aligned = taxonomy_alignment(mit, csetv1, gmf)
    for k, df in aligned.items():
        df.to_parquet(processed_dir / f"{k}.parquet", index=False)

    # Figures
    fig_dir = Path("outputs/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Incidents over time: pick a date column if present
    date_candidates = [c for c in incidents_enriched.columns if "date" in c]
    if date_candidates:
        # choose the first date-like column with enough non-null values
        date_col = max(date_candidates, key=lambda c: incidents_enriched[c].notna().sum())
        incidents_over_time(incidents_enriched, date_col, fig_dir / "incidents_over_time.png")
    if "report_count" in incidents_enriched.columns:
        # quick diagnostic distribution
        bar_counts(incidents_enriched["report_count"], "Report count per incident (top bins)", fig_dir / "report_count.png", top_n=15)

    # Co-occurrence network from MIT risk core if available
    if "risk_core_mit" in aligned and not aligned["risk_core_mit"].empty:
        df = aligned["risk_core_mit"]
        cols = [c for c in ["risk_domain","entity","intent","timing"] if c in df.columns]
        if cols:
            cooccurrence_network(df, cols, fig_dir / "cooccurrence_network.png")

    save_run_metadata(Path("outputs/run_metadata.json"), snapshot, get_versions())
    print("✅ Done.")
    print(f"- Processed tables: {processed_dir}")
    print(f"- Figures: {fig_dir}")
    print("- Run metadata: outputs/run_metadata.json")

if __name__ == "__main__":
    main()
