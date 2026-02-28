from __future__ import annotations

import tempfile
import urllib.parse
import urllib.request
import urllib.error
import importlib.util
from pathlib import Path
from io import BytesIO

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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


def _build_pdf_report(
    filtered_incidents: pd.DataFrame,
    filtered_reports: pd.DataFrame,
    date_col: str | None,
    selected_domains: list[str],
    taxonomy_name: str,
    selected_tax_vals: list[str],
) -> bytes:
    buf = BytesIO()

    with PdfPages(buf) as pdf:
        # Page 1: Executive summary
        fig = plt.figure(figsize=(11.69, 8.27))
        ax = fig.add_subplot(111)
        ax.axis("off")

        incidents_n = len(filtered_incidents)
        reports_n = len(filtered_reports)
        avg_reports = (
            filtered_incidents["report_count"].mean()
            if "report_count" in filtered_incidents.columns and incidents_n > 0
            else 0.0
        )
        med_reports = (
            filtered_incidents["report_count"].median()
            if "report_count" in filtered_incidents.columns and incidents_n > 0
            else 0.0
        )
        evidence_total = (
            int(filtered_incidents["evidence_total"].sum())
            if "evidence_total" in filtered_incidents.columns
            else 0
        )

        lines = [
            "AI Incident Observatory - Analysis Report",
            "",
            "Scope",
            f"- Incidents (filtered): {incidents_n:,}",
            f"- Reports (filtered): {reports_n:,}",
            f"- Avg reports/incident: {avg_reports:.2f}",
            f"- Median reports/incident: {med_reports:.2f}",
            f"- Total evidence records: {evidence_total:,}",
            "",
            "Applied filters",
            f"- Source domains filter: {', '.join(selected_domains) if selected_domains else 'All'}",
            f"- Taxonomy table: {taxonomy_name}",
            f"- Taxonomy values: {', '.join(selected_tax_vals) if selected_tax_vals else 'All'}",
        ]
        if date_col is not None and date_col in filtered_incidents.columns:
            valid_dates = pd.to_datetime(filtered_incidents[date_col], errors="coerce", utc=True).dropna()
            if not valid_dates.empty:
                lines.append(f"- Date range in filtered data: {valid_dates.min().date()} to {valid_dates.max().date()}")

        y = 0.95
        for i, line in enumerate(lines):
            fs = 20 if i == 0 else 12
            ax.text(0.03, y, line, fontsize=fs, va="top")
            y -= 0.05 if i == 0 else 0.04

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: Incidents over time
        if date_col is not None and date_col in filtered_incidents.columns:
            ts = (
                filtered_incidents.assign(_plot_date=_clean_date_series(filtered_incidents[date_col]))
                .dropna(subset=["_plot_date"])
                .set_index("_plot_date")
                .resample("M")
                .size()
            )
            if not ts.empty:
                fig = plt.figure(figsize=(11.69, 8.27))
                ax = fig.add_subplot(111)
                ts.plot(ax=ax, lw=1.7)
                ax.set_title("Incidents Over Time (Monthly)")
                ax.set_xlabel("Date")
                ax.set_ylabel("Incident Count")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        # Page 3: Top domains + harm (if present)
        fig = plt.figure(figsize=(11.69, 8.27))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        domain_counts = (
            filtered_reports["source_domain"].dropna().astype(str).value_counts().head(15)
            if "source_domain" in filtered_reports.columns
            else pd.Series(dtype=int)
        )
        if not domain_counts.empty:
            domain_counts.sort_values().plot(kind="barh", ax=ax1)
            ax1.set_title("Top Report Source Domains")
            ax1.set_xlabel("Count")
        else:
            ax1.axis("off")
            ax1.text(0.0, 0.5, "No source domain data available.", fontsize=12)

        harm_counts = (
            filtered_incidents["harm_domain_top"]
            .dropna()
            .astype(str)
            .str.strip()
            .value_counts()
            .head(15)
            if "harm_domain_top" in filtered_incidents.columns
            else pd.Series(dtype=int)
        )
        harm_counts = harm_counts[harm_counts.index != ""]
        if not harm_counts.empty:
            harm_counts.sort_values().plot(kind="barh", ax=ax2)
            ax2.set_title("Top Harm Domains (CSET)")
            ax2.set_xlabel("Count")
        else:
            ax2.axis("off")
            ax2.text(0.0, 0.5, "No harm-domain data available in filtered results.", fontsize=12)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 4: Distribution-focused graphs
        fig = plt.figure(figsize=(11.69, 8.27))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        if "report_count" in filtered_incidents.columns and not filtered_incidents.empty:
            rc = pd.to_numeric(filtered_incidents["report_count"], errors="coerce").fillna(0)
            ax1.hist(rc, bins=20)
            ax1.set_title("Report Count Distribution")
            ax1.set_xlabel("Reports per incident")
            ax1.set_ylabel("Incidents")
        else:
            ax1.axis("off")
            ax1.text(0.05, 0.5, "No report_count data", fontsize=10)

        if "evidence_total" in filtered_incidents.columns and not filtered_incidents.empty:
            ev = pd.to_numeric(filtered_incidents["evidence_total"], errors="coerce").fillna(0)
            ax2.hist(ev, bins=20)
            ax2.set_title("Evidence Total Distribution")
            ax2.set_xlabel("Evidence records per incident")
            ax2.set_ylabel("Incidents")
        else:
            ax2.axis("off")
            ax2.text(0.05, 0.5, "No evidence_total data", fontsize=10)

        if "ai_harm_level_top" in filtered_incidents.columns:
            harm_lvl = (
                filtered_incidents["ai_harm_level_top"]
                .dropna()
                .astype(str)
                .str.strip()
            )
            harm_lvl = harm_lvl[harm_lvl != ""]
            if not harm_lvl.empty:
                harm_lvl.value_counts().head(10).sort_values().plot(kind="barh", ax=ax3)
                ax3.set_title("AI Harm Levels (Top)")
                ax3.set_xlabel("Count")
            else:
                ax3.axis("off")
                ax3.text(0.05, 0.5, "No AI harm level data", fontsize=10)
        else:
            ax3.axis("off")
            ax3.text(0.05, 0.5, "No AI harm level column", fontsize=10)

        ll = (
            pd.to_numeric(filtered_incidents["lives_lost_max"], errors="coerce").fillna(0)
            if "lives_lost_max" in filtered_incidents.columns
            else pd.Series([], dtype=float)
        )
        inj = (
            pd.to_numeric(filtered_incidents["injuries_max"], errors="coerce").fillna(0)
            if "injuries_max" in filtered_incidents.columns
            else pd.Series([], dtype=float)
        )
        if not ll.empty or not inj.empty:
            ll_pos = int(ll.gt(0).sum()) if not ll.empty else 0
            inj_pos = int(inj.gt(0).sum()) if not inj.empty else 0
            neither = max(len(filtered_incidents) - ll_pos - inj_pos, 0)
            ax4.bar(["Lives Lost > 0", "Injuries > 0", "Neither"], [ll_pos, inj_pos, neither])
            ax4.set_title("Severe Outcome Signals")
            ax4.set_ylabel("Incident count")
        else:
            ax4.axis("off")
            ax4.text(0.05, 0.5, "No lives_lost/injuries columns", fontsize=10)

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    return buf.getvalue()


def _fig_to_png_bytes(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    return buf.getvalue()


def _build_single_graph_png(
    graph_key: str,
    filtered_incidents: pd.DataFrame,
    filtered_reports: pd.DataFrame,
    date_col: str | None,
) -> tuple[bytes | None, str, str]:
    if graph_key == "Incidents Over Time":
        if date_col is None or date_col not in filtered_incidents.columns:
            return None, "incidents_over_time.png", "No date column available."
        ts = (
            filtered_incidents.assign(_plot_date=_clean_date_series(filtered_incidents[date_col]))
            .dropna(subset=["_plot_date"])
            .set_index("_plot_date")
            .resample("M")
            .size()
        )
        if ts.empty:
            return None, "incidents_over_time.png", "No data available for incidents over time."
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ts.plot(ax=ax, lw=1.7)
        ax.set_title("Incidents Over Time (Monthly)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Incident Count")
        return _fig_to_png_bytes(fig), "incidents_over_time.png", ""

    if graph_key == "Top Report Source Domains":
        if "source_domain" not in filtered_reports.columns:
            return None, "top_report_source_domains.png", "source_domain column not available."
        counts = filtered_reports["source_domain"].dropna().astype(str).value_counts().head(15)
        if counts.empty:
            return None, "top_report_source_domains.png", "No source domain data available."
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        counts.sort_values().plot(kind="barh", ax=ax)
        ax.set_title("Top Report Source Domains")
        ax.set_xlabel("Count")
        return _fig_to_png_bytes(fig), "top_report_source_domains.png", ""

    if graph_key == "Top Harm Domains":
        if "harm_domain_top" not in filtered_incidents.columns:
            return None, "top_harm_domains.png", "harm_domain_top column not available."
        counts = (
            filtered_incidents["harm_domain_top"]
            .dropna()
            .astype(str)
            .str.strip()
            .value_counts()
            .head(15)
        )
        counts = counts[counts.index != ""]
        if counts.empty:
            return None, "top_harm_domains.png", "No harm domain data available."
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        counts.sort_values().plot(kind="barh", ax=ax)
        ax.set_title("Top Harm Domains (CSET)")
        ax.set_xlabel("Count")
        return _fig_to_png_bytes(fig), "top_harm_domains.png", ""

    if graph_key == "Report Count Distribution":
        if "report_count" not in filtered_incidents.columns:
            return None, "report_count_distribution.png", "report_count column not available."
        series = pd.to_numeric(filtered_incidents["report_count"], errors="coerce").fillna(0)
        if series.empty:
            return None, "report_count_distribution.png", "No report_count data available."
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.hist(series, bins=20)
        ax.set_title("Report Count Distribution")
        ax.set_xlabel("Reports per incident")
        ax.set_ylabel("Incidents")
        return _fig_to_png_bytes(fig), "report_count_distribution.png", ""

    if graph_key == "Evidence Total Distribution":
        if "evidence_total" not in filtered_incidents.columns:
            return None, "evidence_total_distribution.png", "evidence_total column not available."
        series = pd.to_numeric(filtered_incidents["evidence_total"], errors="coerce").fillna(0)
        if series.empty:
            return None, "evidence_total_distribution.png", "No evidence_total data available."
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.hist(series, bins=20)
        ax.set_title("Evidence Total Distribution")
        ax.set_xlabel("Evidence records per incident")
        ax.set_ylabel("Incidents")
        return _fig_to_png_bytes(fig), "evidence_total_distribution.png", ""

    if graph_key == "AI Harm Levels":
        if "ai_harm_level_top" not in filtered_incidents.columns:
            return None, "ai_harm_levels.png", "ai_harm_level_top column not available."
        counts = (
            filtered_incidents["ai_harm_level_top"]
            .dropna()
            .astype(str)
            .str.strip()
            .value_counts()
            .head(15)
        )
        counts = counts[counts.index != ""]
        if counts.empty:
            return None, "ai_harm_levels.png", "No AI harm level data available."
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        counts.sort_values().plot(kind="barh", ax=ax)
        ax.set_title("AI Harm Levels")
        ax.set_xlabel("Count")
        return _fig_to_png_bytes(fig), "ai_harm_levels.png", ""

    return None, "graph.png", "Unknown graph selection."


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
    selected_tax_vals: list[str] = []

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
    d1, d2, d3 = st.columns(3)
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
    pdf_bytes = _build_pdf_report(
        filtered_incidents=filtered_incidents,
        filtered_reports=filtered_reports,
        date_col=date_col,
        selected_domains=selected_domains,
        taxonomy_name=taxonomy_name,
        selected_tax_vals=selected_tax_vals,
    )
    d3.download_button(
        "Download PDF Report",
        data=pdf_bytes,
        file_name="ai_incident_analysis_report.pdf",
        mime="application/pdf",
    )
    graph_options = [
        "Incidents Over Time",
        "Top Report Source Domains",
        "Top Harm Domains",
        "Report Count Distribution",
        "Evidence Total Distribution",
        "AI Harm Levels",
    ]
    selected_graph = st.selectbox("Choose a single graph to download", options=graph_options)
    graph_bytes, graph_filename, graph_err = _build_single_graph_png(
        graph_key=selected_graph,
        filtered_incidents=filtered_incidents,
        filtered_reports=filtered_reports,
        date_col=date_col,
    )
    if graph_bytes is None:
        st.info(graph_err)
    else:
        st.download_button(
            "Download Selected Graph (PNG)",
            data=graph_bytes,
            file_name=graph_filename,
            mime="image/png",
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
