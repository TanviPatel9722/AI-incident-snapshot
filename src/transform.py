from __future__ import annotations
import ast
import pandas as pd

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def parse_dates(incidents: pd.DataFrame) -> pd.DataFrame:
    df = incidents.copy()
    # Many snapshots include "date" or "incident_date" variants.
    date_cols = [c for c in df.columns if "date" in c.lower()]
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    return df

def _safe_count_listlike(value) -> int:
    if pd.isna(value):
        return 0
    if isinstance(value, list):
        return len([x for x in value if pd.notna(x) and str(x).strip()])
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return 0
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return len([x for x in parsed if pd.notna(x) and str(x).strip()])
        except Exception:
            pass
        if "," in s:
            return len([x for x in s.split(",") if x.strip()])
        return 1
    return 0

def join_reports_to_incidents(incidents: pd.DataFrame, reports: pd.DataFrame) -> pd.DataFrame:
    inc = incidents.copy()
    rep = reports.copy()

    if "incident_id" not in inc.columns:
        raise KeyError("incidents table missing incident_id")

    # Common key in many exports is "incident_id" in reports.
    if "incident_id" not in rep.columns:
        alt = [c for c in rep.columns if c.lower().endswith("incident_id")]
        if alt:
            rep = rep.rename(columns={alt[0]: "incident_id"})
    if "incident_id" in rep.columns:
        rep_counts = (
            rep.dropna(subset=["incident_id"])
            .assign(incident_id=lambda d: d["incident_id"].astype(str))
            .groupby("incident_id")
            .size()
            .rename("report_count")
            .reset_index()
        )
        inc = inc.assign(incident_id=inc["incident_id"].astype(str))
        out = inc.merge(rep_counts, how="left", on="incident_id")
        out["report_count"] = out["report_count"].fillna(0).astype(int)
        return out

    # Fallback for snapshots where reports table has no incident key:
    # derive count from incidents.reports list-like field.
    if "reports" in inc.columns:
        out = inc.copy()
        out["report_count"] = out["reports"].apply(_safe_count_listlike).astype(int)
        return out

    raise KeyError("reports table missing incident_id and incidents table missing reports fallback column")

def taxonomy_alignment(mit: pd.DataFrame | None, csetv1: pd.DataFrame | None, gmf: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    """Return cleaned taxonomy tables + simple harmonized views."""
    out = {}
    if mit is not None:
        out["mit"] = mit.copy()
    if csetv1 is not None:
        out["csetv1"] = csetv1.copy()
    if gmf is not None:
        out["gmf"] = gmf.copy()

    # Minimal harmonized "risk core" view (incident_id + key dimensions when present)
    core_cols = []
    if mit is not None:
        m = mit.copy()
        # Normalize common field names if present
        # We keep columns as-is but try to expose standard names
        rename_map = {}
        for cand, std in [
            ("risk_domain", "risk_domain"),
            ("domain", "risk_domain"),
            ("entity", "entity"),
            ("intent", "intent"),
            ("timing", "timing"),
            ("incident_id", "incident_id"),
        ]:
            if cand in m.columns:
                rename_map[cand] = std
        m = m.rename(columns=rename_map)
        wanted = [c for c in ["incident_id","risk_domain","entity","intent","timing"] if c in m.columns]
        core_cols.extend([c for c in wanted if c not in core_cols])
        out["risk_core_mit"] = m[wanted].drop_duplicates()

    return out


def _normalize_incident_id_column(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    out = df.copy()
    rename_map = {}
    for c in out.columns:
        low = c.strip().lower().replace(" ", "_")
        if low in {"incident_id", "incidentid", "incident_number"}:
            rename_map[c] = "incident_id"
        if low == "incident_id" or low == "incident_id_":
            rename_map[c] = "incident_id"
    if "incident id" in out.columns and "incident_id" not in out.columns:
        rename_map["incident id"] = "incident_id"
    out = out.rename(columns=rename_map)
    if "incident_id" in out.columns:
        incident_col = out["incident_id"]
        if isinstance(incident_col, pd.DataFrame):
            incident_col = incident_col.bfill(axis=1).iloc[:, 0]
            out = out.loc[:, ~out.columns.duplicated()]
        out["incident_id"] = incident_col.astype(str).str.strip()
    return out


def _safe_top_value(series: pd.Series) -> str:
    vc = series.dropna().astype(str).str.strip()
    vc = vc[vc != ""]
    if vc.empty:
        return ""
    return vc.value_counts().idxmax()


def _aggregate_cset_harm(csetv1: pd.DataFrame | None) -> pd.DataFrame:
    if csetv1 is None or csetv1.empty:
        return pd.DataFrame(columns=["incident_id"])
    cset = _normalize_incident_id_column(normalize_columns(csetv1))
    if cset is None or "incident_id" not in cset.columns:
        return pd.DataFrame(columns=["incident_id"])

    cols = cset.columns
    harm_domain_col = "harm_domain" if "harm_domain" in cols else None
    ai_harm_level_col = "ai_harm_level" if "ai_harm_level" in cols else None
    tangible_harm_col = "tangible_harm" if "tangible_harm" in cols else None
    lives_lost_col = "lives_lost" if "lives_lost" in cols else None
    injuries_col = "injuries" if "injuries" in cols else None

    grp = cset.groupby("incident_id", dropna=False)
    out = grp.size().rename("cset_annotations").reset_index()

    if harm_domain_col:
        dom = grp[harm_domain_col].agg(_safe_top_value).rename("harm_domain_top").reset_index()
        out = out.merge(dom, on="incident_id", how="left")
    if ai_harm_level_col:
        lvl = grp[ai_harm_level_col].agg(_safe_top_value).rename("ai_harm_level_top").reset_index()
        out = out.merge(lvl, on="incident_id", how="left")
    if tangible_harm_col:
        th = grp[tangible_harm_col].agg(_safe_top_value).rename("tangible_harm_top").reset_index()
        out = out.merge(th, on="incident_id", how="left")

    if lives_lost_col:
        vals = pd.to_numeric(cset[lives_lost_col], errors="coerce")
        ll = cset.assign(_lives_lost_num=vals).groupby("incident_id")["_lives_lost_num"].max().fillna(0)
        out = out.merge(ll.rename("lives_lost_max").reset_index(), on="incident_id", how="left")
    if injuries_col:
        vals = pd.to_numeric(cset[injuries_col], errors="coerce")
        inj = cset.assign(_injuries_num=vals).groupby("incident_id")["_injuries_num"].max().fillna(0)
        out = out.merge(inj.rename("injuries_max").reset_index(), on="incident_id", how="left")

    return out


def build_incident_overview(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a detailed incident-level overview across core + optional tables."""
    incidents = normalize_columns(tables["incidents"])
    reports = normalize_columns(tables["reports"])
    incidents = parse_dates(incidents)
    incidents = _normalize_incident_id_column(incidents)
    reports = _normalize_incident_id_column(reports)

    # Canonicalize duplicates where mapping is available.
    duplicates = tables.get("duplicates")
    if duplicates is not None and not duplicates.empty:
        dup = normalize_columns(duplicates)
        if {"duplicate_incident_number", "true_incident_number"}.issubset(dup.columns):
            dup_map = (
                dup[["duplicate_incident_number", "true_incident_number"]]
                .dropna()
                .astype(str)
                .set_index("duplicate_incident_number")["true_incident_number"]
                .to_dict()
            )
            if "incident_id" in incidents.columns:
                incidents["incident_id"] = incidents["incident_id"].astype(str).replace(dup_map)
            if reports is not None and "incident_id" in reports.columns:
                reports["incident_id"] = reports["incident_id"].astype(str).replace(dup_map)

    overview = join_reports_to_incidents(incidents, reports)
    overview = _normalize_incident_id_column(overview)

    # Evidence counts from submissions and quickadd.
    if "incident_id" in overview.columns:
        if tables.get("submissions") is not None:
            sub = _normalize_incident_id_column(normalize_columns(tables["submissions"]))
            if sub is not None and "incident_id" in sub.columns:
                sub_counts = sub.groupby("incident_id").size().rename("submission_count").reset_index()
                overview = overview.merge(sub_counts, on="incident_id", how="left")
        if tables.get("quickadd") is not None:
            qa = _normalize_incident_id_column(normalize_columns(tables["quickadd"]))
            if qa is not None and "incident_id" in qa.columns:
                qa_counts = qa.groupby("incident_id").size().rename("quickadd_count").reset_index()
                overview = overview.merge(qa_counts, on="incident_id", how="left")

    for c in ["submission_count", "quickadd_count"]:
        if c in overview.columns:
            overview[c] = overview[c].fillna(0).astype(int)
        else:
            overview[c] = 0

    if "report_count" not in overview.columns:
        overview["report_count"] = 0
    overview["evidence_total"] = (
        overview["report_count"].fillna(0).astype(int)
        + overview["submission_count"].fillna(0).astype(int)
        + overview["quickadd_count"].fillna(0).astype(int)
    )

    # Harm-oriented aggregates from CSET.
    harm = _aggregate_cset_harm(tables.get("csetv1"))
    if not harm.empty and "incident_id" in overview.columns:
        overview = overview.merge(harm, on="incident_id", how="left")

    # Add boolean helpers for alleged parties from incident core table.
    for src_col, out_col in [
        ("alleged_harmed_or_nearly_harmed_parties", "has_alleged_harmed_party"),
        ("alleged_deployer_of_ai_system", "has_alleged_deployer"),
        ("alleged_developer_of_ai_system", "has_alleged_developer"),
    ]:
        if src_col in overview.columns:
            vals = overview[src_col].astype(str).str.strip().str.lower()
            overview[out_col] = (vals != "") & (vals != "nan") & (vals != "[]")
        else:
            overview[out_col] = False

    return overview
