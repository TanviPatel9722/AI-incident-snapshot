from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

def save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=200)

def bar_counts(series: pd.Series, title: str, out_path: Path, top_n: int = 15):
    counts = series.dropna().astype(str).value_counts().head(top_n)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    counts.sort_values().plot(kind="barh", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Count")
    save_fig(fig, out_path)
    plt.close(fig)

def incidents_over_time(incidents: pd.DataFrame, date_col: str, out_path: Path, freq: str = "M"):
    df = incidents.copy()
    df = df.dropna(subset=[date_col])
    ts = df.set_index(date_col).resample(freq).size()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ts.plot(ax=ax)
    ax.set_title(f"Incidents over time ({freq})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    save_fig(fig, out_path)
    plt.close(fig)

def cooccurrence_network(df: pd.DataFrame, cols: list[str], out_path: Path, max_nodes: int = 40):
    # Build edges for co-occurrence within each row
    g = nx.Graph()
    for _, row in df[cols].dropna(how="all").iterrows():
        items = [str(row[c]) for c in cols if pd.notna(row.get(c))]
        items = [i for i in items if i and i.lower() not in ("nan", "none", "all others")]
        items = list(dict.fromkeys(items))[:10]
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                a, b = items[i], items[j]
                if g.has_edge(a,b):
                    g[a][b]["weight"] += 1
                else:
                    g.add_edge(a,b, weight=1)

    # Keep most connected nodes
    if g.number_of_nodes() > max_nodes:
        nodes_sorted = sorted(g.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
        keep = {n for n,_ in nodes_sorted}
        g = g.subgraph(keep).copy()

    pos = nx.spring_layout(g, seed=42, k=0.7)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Risk Co-occurrence Network (subset)")
    # edge widths by weight
    weights = [g[u][v]["weight"] for u,v in g.edges]
    nx.draw_networkx_edges(g, pos, ax=ax, width=[max(1, w/2) for w in weights], alpha=0.4)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=200)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=7)
    ax.axis("off")
    save_fig(fig, out_path)
    plt.close(fig)
