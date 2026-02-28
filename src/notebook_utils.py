from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_FIGSIZE = (8, 4.5)
DEFAULT_DPI = 200


def ensure_output_dir(output_path: str | Path) -> Path:
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def normalize_columns(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    normalized = df.copy()
    normalized.columns = [str(c).strip().lower().replace(" ", "_") for c in normalized.columns]
    return normalized


def normalize_incident_id(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    normalized = normalize_columns(df)
    if normalized is None:
        return None

    rename_map = {}
    for column_name in normalized.columns:
        if column_name in {"incident_id", "incidentid", "incident_id_"}:
            rename_map[column_name] = "incident_id"
        if column_name == "incident id":
            rename_map[column_name] = "incident_id"

    normalized = normalized.rename(columns=rename_map)
    if "incident_id" in normalized.columns:
        incident_series = normalized["incident_id"]
        if isinstance(incident_series, pd.DataFrame):
            incident_series = incident_series.bfill(axis=1).iloc[:, 0]
            normalized = normalized.loc[:, ~normalized.columns.duplicated()]
        normalized["incident_id"] = incident_series.astype(str).str.strip()
    return normalized


def load_data(
    data_path: str | Path,
    tables: Iterable[str],
    reports_usecols: Iterable[str] | None = None,
) -> dict[str, pd.DataFrame | None]:
    data_dir = Path(data_path)
    loaded: dict[str, pd.DataFrame | None] = {}
    reports_usecols_set = {c.strip().lower() for c in (reports_usecols or [])}
    file_map = {
        "incidents": "incidents.csv",
        "reports": "reports.csv",
        "submissions": "submissions.csv",
        "quickadd": "quickadd.csv",
        "duplicates": "duplicates.csv",
        "mit": "classifications_MIT.csv",
        "gmf": "classifications_GMF.csv",
        "cset": "classifications_CSETv1.csv",
        "csetv1": "classifications_CSETv1.csv",
        "csetv0": "classifications_CSETv0.csv",
    }

    for table_name in tables:
        csv_name = file_map.get(table_name, f"{table_name}.csv")
        csv_path = data_dir / csv_name
        if not csv_path.exists():
            loaded[table_name] = None
            continue

        if table_name == "reports" and reports_usecols_set:
            table_df = pd.read_csv(
                csv_path,
                usecols=lambda c: c.strip().lower() in reports_usecols_set,
            )
        else:
            table_df = pd.read_csv(csv_path)

        if table_name in {"mit", "gmf", "cset", "incidents", "reports", "submissions", "quickadd"}:
            table_df = normalize_incident_id(table_df)
        else:
            table_df = normalize_columns(table_df)
        loaded[table_name] = table_df

    return loaded


def pick_column_by_keywords(
    df: pd.DataFrame | None,
    must_have_keywords: Iterable[str],
    nice_to_have: Iterable[str] | None = None,
) -> str | None:
    if df is None:
        return None
    required = [k.lower() for k in must_have_keywords]
    bonus = [k.lower() for k in (nice_to_have or [])]
    candidates: list[tuple[int, int, str]] = []

    for column_name in df.columns:
        normalized_name = column_name.lower()
        if all(keyword in normalized_name for keyword in required):
            bonus_score = sum(1 for keyword in bonus if keyword in normalized_name)
            non_null_count = int(df[column_name].notna().sum())
            candidates.append((bonus_score, non_null_count, column_name))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][2]


def parse_datetime_column(df: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name not in df.columns:
        return pd.Series([], dtype="datetime64[ns]")
    parsed = pd.to_datetime(df[column_name], errors="coerce", utc=True)
    return parsed.dt.tz_convert(None)


def pick_column_by_candidates(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for column_name in candidates:
        if column_name in df.columns:
            parsed = parse_datetime_column(df, column_name)
            if parsed.notna().any():
                return column_name
    return None


def plot_barh_top(
    series: pd.Series,
    title: str,
    xlabel: str,
    output_file: str | Path,
    top_n: int = 15,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    dpi: int = DEFAULT_DPI,
) -> pd.Series:
    counts = series.dropna().astype(str).str.strip()
    counts = counts[counts != ""].value_counts().head(top_n).sort_values()

    fig, ax = plt.subplots(figsize=figsize)
    counts.plot(kind="barh", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.savefig(Path(output_file), bbox_inches="tight", dpi=dpi)
    plt.show()
    return counts


def plot_percent_barh(
    series: pd.Series,
    title: str,
    output_file: str | Path,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    dpi: int = DEFAULT_DPI,
) -> pd.Series:
    percentages = (
        series.dropna().astype(str).str.strip()
        .pipe(lambda s: s[s != ""])
        .value_counts(normalize=True)
        .mul(100)
        .round(1)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=figsize)
    percentages.plot(kind="barh", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Percent")
    fig.savefig(Path(output_file), bbox_inches="tight", dpi=dpi)
    plt.show()
    return percentages
