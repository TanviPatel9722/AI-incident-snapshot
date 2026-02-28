from __future__ import annotations
from pathlib import Path
import tarfile
import json
import pandas as pd

RAW_DIRNAME = "raw_snapshot"

def extract_snapshot(snapshot_path: str | Path, out_dir: str | Path) -> Path:
    """Extract .tar.bz2 snapshot into out_dir/raw_snapshot/<snapshot_stem>/"""
    snapshot_path = Path(snapshot_path)
    out_dir = Path(out_dir)
    target_root = out_dir / RAW_DIRNAME / snapshot_path.stem
    target_root.mkdir(parents=True, exist_ok=True)

    # Extract
    with tarfile.open(snapshot_path, "r:bz2") as tf:
        tf.extractall(path=target_root)

    # Inside archive there's typically a single folder (e.g., mongodump_full_snapshot)
    # We'll return the extracted root for downstream discovery.
    return target_root

def find_csv_root(extracted_root: Path) -> Path:
    """Find folder containing incidents.csv and reports.csv."""
    # Search down a couple levels
    for p in extracted_root.rglob("incidents.csv"):
        return p.parent
    raise FileNotFoundError("Could not find incidents.csv inside extracted snapshot.")

def load_tables(csv_root: Path) -> dict[str, pd.DataFrame]:
    """Load the core CSV tables we use."""
    tables = {}
    def read(name: str) -> pd.DataFrame:
        path = csv_root / name
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        return pd.read_csv(path)

    tables["incidents"] = read("incidents.csv")
    tables["reports"] = read("reports.csv")

    # Optional tables for richer analysis.
    for fname, key in [
        ("classifications_MIT.csv", "mit"),
        ("classifications_GMF.csv", "gmf"),
        ("classifications_CSETv1.csv", "csetv1"),
        ("classifications_CSETv0.csv", "csetv0"),
        ("submissions.csv", "submissions"),
        ("quickadd.csv", "quickadd"),
        ("duplicates.csv", "duplicates"),
        ("entities.csv", "entities"),
        ("entity_relationships.csv", "entity_relationships"),
        ("incident_links.csv", "incident_links"),
    ]:
        path = csv_root / fname
        if path.exists():
            tables[key] = pd.read_csv(path)

    # Generic fallback: include any CSV not already loaded.
    known = {
        "incidents.csv",
        "reports.csv",
        "classifications_MIT.csv",
        "classifications_GMF.csv",
        "classifications_CSETv1.csv",
        "classifications_CSETv0.csv",
        "submissions.csv",
        "quickadd.csv",
        "duplicates.csv",
        "entities.csv",
        "entity_relationships.csv",
        "incident_links.csv",
    }
    for path in csv_root.glob("*.csv"):
        if path.name in known:
            continue
        key = path.stem.strip().lower().replace(" ", "_")
        if key not in tables:
            tables[key] = pd.read_csv(path)
    return tables

def save_run_metadata(out_path: Path, snapshot_path: Path, versions: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "snapshot_file": str(snapshot_path),
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "package_versions": versions,
    }
    out_path.write_text(json.dumps(payload, indent=2))
