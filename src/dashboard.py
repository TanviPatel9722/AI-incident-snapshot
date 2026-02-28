from __future__ import annotations

import tempfile
import urllib.parse
import urllib.request
import urllib.error
import importlib.util
from pathlib import Path

import pandas as pd
import streamlit as st

def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_io_mod = _load_module("aio_io", Path(__file__).with_name("io.py"))
_transform_mod = _load_module("aio_transform", Path(__file__).with_name("transform.py"))

extract_snapshot = _io_mod.extract_snapshot
find_csv_root = _io_mod.find_csv_root
load_tables = _io_mod.load_tables
join_reports_to_incidents = _transform_mod.join_reports_to_incidents
normalize_columns = _transform_mod.normalize_columns
parse_dates = _transform_mod.parse_dates
taxonomy_alignment = _transform_mod.taxonomy_alignment
build_incident_overview = _transform_mod.build_incident_overview


REQUIRED_CUSTOM_FILES = {"incidents.csv", "reports.csv"}


def _download_snapshot(url: str, target_dir: Path) -> Path:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Snapshot URL must start with http:// or https://")

    filename = Path(parsed.path).name or "snapshot.tar.bz2"
    if not filename.endswith(".tar.bz2"):
        filename = f"{filename}.tar.bz2"
    out_path = target_dir / filename
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            out_path.write_bytes(resp.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 403:
            raise PermissionError(
                "HTTP 403 Forbidden. The snapshot URL is not publicly downloadable. "
                "Use a direct public file URL, or use 'Upload snapshot archive'."
            ) from exc
        raise
    return out_path


def _save_uploaded_file(uploaded, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(uploaded.getbuffer())
    return target_path


def _load_custom_tables(uploaded_files: list) -> dict[str, pd.DataFrame]:
    file_map = {f.name: f for f in uploaded_files}
    missing = sorted(REQUIRED_CUSTOM_FILES - set(file_map))
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(f"Missing required CSV file(s): {missing_str}")

    tables: dict[str, pd.DataFrame] = {
        "incidents": pd.read_csv(file_map["incidents.csv"]),
        "reports": pd.read_csv(file_map["reports.csv"]),
    }
    consumed = {"incidents.csv", "reports.csv"}

    optional_map = {
        "classifications_MIT.csv": "mit",
        "classifications_GMF.csv": "gmf",
        "classifications_CSETv1.csv": "csetv1",
        "classifications_CSETv0.csv": "csetv0",
        "submissions.csv": "submissions",
        "quickadd.csv": "quickadd",
        "duplicates.csv": "duplicates",
        "entities.csv": "entities",
        "entity_relationships.csv": "entity_relationships",
        "incident_links.csv": "incident_links",
    }
    for fname, key in optional_map.items():
        if fname in file_map:
            tables[key] = pd.read_csv(file_map[fname])
            consumed.add(fname)
    for uploaded in uploaded_files:
        if uploaded.name in consumed:
            continue
        key = uploaded.name.rsplit(".", 1)[0].strip().lower().replace(" ", "_")
        if key not in tables:
            tables[key] = pd.read_csv(uploaded)
    return tables


def _build_processed_data(tables: dict[str, pd.DataFrame]) -> dict[str, object]:
    incidents = normalize_columns(tables["incidents"])
    reports = normalize_columns(tables["reports"])
    incidents = parse_dates(incidents)
    incidents_enriched = join_reports_to_incidents(incidents, reports)

    mit = normalize_columns(tables.get("mit")) if tables.get("mit") is not None else None
    csetv1 = (
        normalize_columns(tables.get("csetv1"))
        if tables.get("csetv1") is not None
        else None
    )
    gmf = normalize_columns(tables.get("gmf")) if tables.get("gmf") is not None else None
    aligned = taxonomy_alignment(mit, csetv1, gmf)
    overview = build_incident_overview(tables)

    return {
        "incidents_enriched": incidents_enriched,
        "reports": reports,
        "aligned": aligned,
        "overview": overview,
    }


def _best_date_column(df: pd.DataFrame) -> str | None:
    date_candidates = [c for c in df.columns if "date" in c.lower()]
    if not date_candidates:
        return None
    return max(date_candidates, key=lambda c: df[c].notna().sum())


def _clean_date_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    min_dt = pd.Timestamp("1980-01-01", tz="UTC")
    max_dt = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=366)
    parsed = parsed.where((parsed >= min_dt) & (parsed <= max_dt))
    return parsed


def _data_quality_summary(
    incidents: pd.DataFrame,
    reports: pd.DataFrame,
    date_col: str | None,
    clean_dates: pd.Series | None,
) -> pd.DataFrame:
    incident_dupes = (
        int(incidents["incident_id"].duplicated().sum()) if "incident_id" in incidents.columns else 0
    )
    report_dupes = int(reports["_id"].duplicated().sum()) if "_id" in reports.columns else 0
    missing_source = (
        float(reports["source_domain"].isna().mean() * 100) if "source_domain" in reports.columns else 0.0
    )
    invalid_date_rows = 0
    if date_col is not None and clean_dates is not None:
        raw_non_null = pd.to_datetime(incidents[date_col], errors="coerce", utc=True).notna().sum()
        cleaned_non_null = clean_dates.notna().sum()
        invalid_date_rows = int(raw_non_null - cleaned_non_null)
    return pd.DataFrame(
        {
            "Metric": [
                "Incident rows",
                "Report rows",
                "Duplicate incident_id rows",
                "Duplicate report _id rows",
                "Missing source_domain (%)",
                "Out-of-range/invalid incident dates",
            ],
            "Value": [
                f"{len(incidents):,}",
                f"{len(reports):,}",
                f"{incident_dupes:,}",
                f"{report_dupes:,}",
                f"{missing_source:.2f}",
                f"{invalid_date_rows:,}",
            ],
        }
    )


def _render_summary(data: dict[str, object]) -> None:
    incidents_enriched: pd.DataFrame = data["incidents_enriched"]  # type: ignore[assignment]
    reports: pd.DataFrame = data["reports"]  # type: ignore[assignment]
    aligned: dict[str, pd.DataFrame] = data["aligned"]  # type: ignore[assignment]
    overview: pd.DataFrame = data.get("overview", incidents_enriched)  # type: ignore[assignment]

    date_col = _best_date_column(overview)
    clean_dates = _clean_date_series(overview[date_col]) if date_col else None

    st.subheader("Filters")
    c1, c2, c3 = st.columns(3)

    start_date = None
    end_date = None
    if clean_dates is not None and clean_dates.notna().any():
        min_date = clean_dates.min().date()
        max_date = clean_dates.max().date()
        date_range = c1.date_input(
            "Incident date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range

    selected_domains: list[str] = []
    if "source_domain" in reports.columns:
        domain_vals = sorted(
            reports["source_domain"].dropna().astype(str).unique().tolist()
        )
        selected_domains = c2.multiselect(
            "Report source domains",
            options=domain_vals,
            default=[],
            help="Leave empty to include all domains.",
        )

    taxonomy_name = c3.selectbox(
        "Taxonomy filter table",
        options=["None"] + sorted(aligned.keys()),
    )

    filtered_incidents = overview.copy()
    filtered_reports = reports.copy()

    if date_col is not None and start_date is not None and end_date is not None:
        d0 = pd.Timestamp(start_date, tz="UTC")
        d1 = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        date_mask = clean_dates.between(d0, d1, inclusive="both") if clean_dates is not None else True
        filtered_incidents = filtered_incidents.loc[date_mask].copy()
        if date_col in filtered_incidents.columns:
            filtered_incidents[date_col] = _clean_date_series(filtered_incidents[date_col])

    if selected_domains and "source_domain" in filtered_reports.columns:
        filtered_reports = filtered_reports[
            filtered_reports["source_domain"].astype(str).isin(selected_domains)
        ].copy()

    if (
        selected_domains
        and "incident_id" in filtered_reports.columns
        and "incident_id" in filtered_incidents.columns
    ):
        keep_ids = set(filtered_reports["incident_id"].dropna().astype(str))
        filtered_incidents = filtered_incidents[
            filtered_incidents["incident_id"].astype(str).isin(keep_ids)
        ].copy()

    if taxonomy_name != "None":
        tax_df = aligned.get(taxonomy_name)
        if tax_df is not None and not tax_df.empty and "incident_id" in tax_df.columns:
            tax_cols = [c for c in tax_df.columns if c != "incident_id"]
            if tax_cols:
                tax_col = st.selectbox("Taxonomy field", options=tax_cols)
                tax_values = sorted(tax_df[tax_col].dropna().astype(str).unique().tolist())
                selected_tax_vals = st.multiselect(
                    "Taxonomy values",
                    options=tax_values,
                    default=[],
                    help="Leave empty to include all values in selected taxonomy field.",
                )
                if selected_tax_vals:
                    match_ids = set(
                        tax_df[tax_df[tax_col].astype(str).isin(selected_tax_vals)]["incident_id"]
                        .dropna()
                        .astype(str)
                    )
                    if "incident_id" in filtered_incidents.columns:
                        filtered_incidents = filtered_incidents[
                            filtered_incidents["incident_id"].astype(str).isin(match_ids)
                        ].copy()
                    if "incident_id" in filtered_reports.columns:
                        filtered_reports = filtered_reports[
                            filtered_reports["incident_id"].astype(str).isin(match_ids)
                        ].copy()
        else:
            st.info("Selected taxonomy table has no incident-level key; skipping taxonomy filtering.")

    st.subheader("Dataset Summary")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Incidents", f"{len(filtered_incidents):,}")
    m2.metric("Reports", f"{len(filtered_reports):,}")
    avg_reports = (
        filtered_incidents["report_count"].mean()
        if "report_count" in filtered_incidents.columns and len(filtered_incidents) > 0
        else 0.0
    )
    m3.metric("Avg Reports / Incident", f"{avg_reports:.2f}")
    m4.metric(
        "Median Reports / Incident",
        f"{filtered_incidents['report_count'].median():.2f}"
        if "report_count" in filtered_incidents.columns and len(filtered_incidents) > 0
        else "0.00",
    )
    m5.metric(
        "Total Evidence",
        f"{int(filtered_incidents['evidence_total'].sum()):,}"
        if "evidence_total" in filtered_incidents.columns
        else "0",
    )

    st.subheader("Analysis Summary")
    if date_col is not None and date_col in filtered_incidents.columns:
        valid_dates = _clean_date_series(filtered_incidents[date_col]).dropna()
        if not valid_dates.empty:
            st.write(
                f"Date coverage: {valid_dates.min().date()} to {valid_dates.max().date()} "
                f"({len(valid_dates):,} incidents with valid dates)."
            )
    if "source_domain" in filtered_reports.columns and not filtered_reports.empty:
        top_domain = (
            filtered_reports["source_domain"].dropna().astype(str).value_counts().head(1)
        )
        if not top_domain.empty:
            st.write(f"Most frequent source domain: `{top_domain.index[0]}` ({int(top_domain.iloc[0])} reports).")
    if "report_count" in filtered_incidents.columns and not filtered_incidents.empty:
        p90 = filtered_incidents["report_count"].quantile(0.9)
        st.write(f"Report count concentration: 90th percentile is {p90:.2f} reports/incident.")
    if "evidence_total" in filtered_incidents.columns and not filtered_incidents.empty:
        p90_e = filtered_incidents["evidence_total"].quantile(0.9)
        st.write(f"Evidence concentration: 90th percentile is {p90_e:.2f} evidence records/incident.")

    if date_col is not None and date_col in filtered_incidents.columns:
        ts = (
            filtered_incidents.assign(_plot_date=_clean_date_series(filtered_incidents[date_col]))
            .dropna(subset=["_plot_date"])
            .set_index("_plot_date")
            .resample("M")
            .size()
        )
        if not ts.empty:
            st.subheader("Incidents Over Time")
            st.line_chart(ts)

    if "source_domain" in filtered_reports.columns:
        st.subheader("Top Report Source Domains")
        top_domains = (
            filtered_reports["source_domain"].dropna().astype(str).value_counts().head(15)
        )
        if not top_domains.empty:
            st.bar_chart(top_domains.sort_values())

    st.subheader("Harm and Impact Analysis")
    h1, h2, h3 = st.columns(3)
    harmed_pct = (
        float(filtered_incidents["has_alleged_harmed_party"].mean() * 100)
        if "has_alleged_harmed_party" in filtered_incidents.columns and len(filtered_incidents) > 0
        else 0.0
    )
    h1.metric("Incidents With Alleged Harmed Party", f"{harmed_pct:.1f}%")
    h2.metric(
        "CSET Harm-Annotated Incidents",
        f"{int(filtered_incidents['cset_annotations'].fillna(0).gt(0).sum()):,}"
        if "cset_annotations" in filtered_incidents.columns
        else "0",
    )
    if "lives_lost_max" in filtered_incidents.columns:
        ll_series = pd.to_numeric(filtered_incidents["lives_lost_max"], errors="coerce").fillna(0)
    else:
        ll_series = pd.Series([0] * len(filtered_incidents))
    h3.metric(
        "Incidents With Lives Lost > 0",
        f"{int(ll_series.gt(0).sum()):,}",
    )

    if "harm_domain_top" in filtered_incidents.columns:
        harm_domain = (
            filtered_incidents["harm_domain_top"]
            .dropna()
            .astype(str)
            .str.strip()
        )
        harm_domain = harm_domain[harm_domain != ""]
        if not harm_domain.empty:
            st.write("Top Harm Domains (CSET)")
            st.bar_chart(harm_domain.value_counts().head(15).sort_values())

    if "ai_harm_level_top" in filtered_incidents.columns:
        harm_lvl = (
            filtered_incidents["ai_harm_level_top"]
            .dropna()
            .astype(str)
            .str.strip()
        )
        harm_lvl = harm_lvl[harm_lvl != ""]
        if not harm_lvl.empty:
            st.write("AI Harm Levels (CSET)")
            st.bar_chart(harm_lvl.value_counts().sort_values())

    st.subheader("Data Quality")
    quality_df = _data_quality_summary(overview, reports, date_col, clean_dates)
    st.dataframe(quality_df, hide_index=True, use_container_width=True)

    st.subheader("Downloads")
    d1, d2 = st.columns(2)
    d1.download_button(
        "Download Filtered Incidents CSV",
        data=filtered_incidents.to_csv(index=False).encode("utf-8"),
        file_name="filtered_incidents_enriched.csv",
        mime="text/csv",
    )
    d2.download_button(
        "Download Filtered Reports CSV",
        data=filtered_reports.to_csv(index=False).encode("utf-8"),
        file_name="filtered_reports.csv",
        mime="text/csv",
    )

    st.subheader("Preview")
    st.write("Incidents (first 10 rows)")
    st.dataframe(filtered_incidents.head(10), use_container_width=True)

    st.write("Reports (first 10 rows)")
    st.dataframe(filtered_reports.head(10), use_container_width=True)

    if aligned:
        st.write("Available taxonomy outputs")
        st.write(", ".join(sorted(aligned.keys())))


def main() -> None:
    st.set_page_config(page_title="AI Incident Observatory Dashboard", layout="wide")
    st.title("AI Incident Observatory Dashboard")
    st.caption("Load a snapshot URL or your own dataset to explore incidents and reports.")

    source_mode = st.radio(
        "Choose data source",
        [
            "Snapshot URL",
            "Upload snapshot archive (.tar.bz2)",
            "Upload custom CSV dataset",
        ],
    )

    if "loaded_data" not in st.session_state:
        st.session_state["loaded_data"] = None
    if "loaded_source" not in st.session_state:
        st.session_state["loaded_source"] = ""

    if source_mode == "Snapshot URL":
        with st.form("snapshot_url_form"):
            snapshot_url = st.text_input(
                "Snapshot URL",
                placeholder="https://.../backup-YYYYMMDDHHMMSS.tar.bz2",
            )
            run_btn = st.form_submit_button("Load Dataset")
        if run_btn:
            if not snapshot_url.strip():
                st.error("Enter a snapshot URL before loading.")
            else:
                try:
                    with st.spinner("Downloading and processing snapshot..."):
                        with tempfile.TemporaryDirectory() as tmp:
                            tmp_dir = Path(tmp)
                            snapshot_path = _download_snapshot(snapshot_url.strip(), tmp_dir)
                            extracted = extract_snapshot(snapshot_path, tmp_dir)
                            csv_root = find_csv_root(extracted)
                            tables = load_tables(csv_root)
                            st.session_state["loaded_data"] = _build_processed_data(tables)
                            st.session_state["loaded_source"] = f"Snapshot URL: {snapshot_url.strip()}"
                except Exception as exc:
                    st.error(f"Failed to load snapshot URL: {exc}")

    elif source_mode == "Upload snapshot archive (.tar.bz2)":
        with st.form("snapshot_upload_form"):
            uploaded_snapshot = st.file_uploader("Upload snapshot archive", type=["bz2"])
            run_btn = st.form_submit_button("Load Dataset")
        if run_btn:
            if uploaded_snapshot is None:
                st.error("Upload a snapshot archive before loading.")
            else:
                try:
                    with st.spinner("Processing uploaded snapshot..."):
                        with tempfile.TemporaryDirectory() as tmp:
                            tmp_dir = Path(tmp)
                            snapshot_path = _save_uploaded_file(
                                uploaded_snapshot, tmp_dir / uploaded_snapshot.name
                            )
                            extracted = extract_snapshot(snapshot_path, tmp_dir)
                            csv_root = find_csv_root(extracted)
                            tables = load_tables(csv_root)
                            st.session_state["loaded_data"] = _build_processed_data(tables)
                            st.session_state["loaded_source"] = f"Uploaded archive: {uploaded_snapshot.name}"
                except Exception as exc:
                    st.error(f"Failed to process uploaded snapshot: {exc}")

    else:
        st.write(
            "Upload at least `incidents.csv` and `reports.csv`. "
            "Optional: taxonomy files (`classifications_MIT.csv`, `classifications_GMF.csv`, "
            "`classifications_CSETv1.csv`, `classifications_CSETv0.csv`), plus "
            "`submissions.csv`, `quickadd.csv`, `duplicates.csv`, `entities.csv`, "
            "`entity_relationships.csv`, `incident_links.csv` for richer overview/harm analysis."
        )
        with st.form("custom_csv_form"):
            uploaded_files = st.file_uploader(
                "Upload CSV files",
                type=["csv"],
                accept_multiple_files=True,
            )
            run_btn = st.form_submit_button("Load Dataset")
        if run_btn:
            if not uploaded_files:
                st.error("Upload the required CSV files before loading.")
            else:
                try:
                    with st.spinner("Loading custom dataset..."):
                        tables = _load_custom_tables(uploaded_files)
                        st.session_state["loaded_data"] = _build_processed_data(tables)
                        st.session_state["loaded_source"] = "Uploaded custom CSV dataset"
                except Exception as exc:
                    st.error(f"Failed to load custom CSV dataset: {exc}")

    if st.session_state["loaded_data"] is not None:
        st.success("Dataset loaded successfully.")
        st.caption(f"Active dataset source: {st.session_state['loaded_source']}")
        _render_summary(st.session_state["loaded_data"])


if __name__ == "__main__":
    main()
