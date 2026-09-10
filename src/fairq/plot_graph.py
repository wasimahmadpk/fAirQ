"""Draw the CDMI graph: interactive HTML and an optional NetworkX PNG."""

from __future__ import annotations

import json
import os

from fairq.cdmi import latest_graph
from fairq.db import connect

LABELS = {
    "no2_street": "Street NO2",
    "no2_bg": "Forest NO2",
    "temp": "Temp",
    "wind": "Wind",
    "humidity": "Humidity",
    "traffic": "Traffic",
}
COLORS = {
    "no2_street": "#ff6b4a",
    "no2_bg": "#ff8a65",
    "temp": "#4fc3f7",
    "wind": "#80deea",
    "humidity": "#81d4fa",
    "traffic": "#ffd54f",
}
# Hand-placed so weather sits above, traffic in the middle, NO2 below.
LAYOUT = {
    "temp": (-0.85, 0.82),
    "humidity": (0.0, 1.0),
    "wind": (0.85, 0.82),
    "traffic": (0.0, 0.08),
    "no2_street": (-0.72, -0.85),
    "no2_bg": (0.72, -0.85),
}
OUT_DIR = os.environ.get("FAIRQ_PLOT_DIR", "/app/models")


def accepted_edges(graph: dict) -> list[dict]:
    return [edge for edge in graph.get("edges", []) if edge.get("accepted")]


def graph_html(graph: dict) -> str:
    edges = accepted_edges(graph)
    nodes = []
    seen = set()
    for edge in edges:
        for key in ("cause", "effect"):
            name = edge[key]
            if name in seen:
                continue
            seen.add(name)
            x, y = LAYOUT.get(name, (0.0, 0.0))
            nodes.append(
                {
                    "id": name,
                    "label": LABELS.get(name, name),
                    "x": x * 280,
                    "y": -y * 220,
                    "color": {
                        "background": COLORS.get(name, "#90a4ae"),
                        "border": "#eceff1",
                        "highlight": {"background": "#fff8e1", "border": "#ffd54f"},
                    },
                    "font": {"color": "#0d1117", "face": "IBM Plex Sans, sans-serif", "size": 17, "bold": True},
                }
            )
    vis_edges = []
    for edge in edges:
        rel = max(float(edge.get("rel_mae") or 0), 0)
        vis_edges.append(
            {
                "from": edge["cause"],
                "to": edge["effect"],
                "arrows": "to",
                "width": 1.4 + 10 * rel,
                "label": f"{rel:.2f}",
                "color": {"color": "#80cbc4", "highlight": "#ffd54f"},
                "title": (
                    f"KS={edge.get('ks_stat')}  shape={edge.get('ks_shape')}  "
                    f"ΔMAE μ+ε={edge.get('rel_mae')}  U={edge.get('rel_mae_uniform')}  "
                    f"G={edge.get('rel_mae_gaussian')}"
                ),
                "smooth": {"type": "curvedCW", "roundness": 0.18},
            }
        )
    payload = {"nodes": nodes, "edges": vis_edges, "version": graph.get("model_version", "")}
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>fAirQ causal graph</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    html, body {{ margin: 0; height: 100%; background: #0d1117; color: #e6edf3;
      font-family: "IBM Plex Sans", ui-sans-serif, system-ui; }}
    #banner {{ padding: 16px 22px; border-bottom: 1px solid #30363d; }}
    h1 {{ margin: 0 0 4px; font-size: 20px; letter-spacing: 0.04em; }}
    p {{ margin: 0; color: #8b949e; font-size: 13px; }}
    #net {{ height: calc(100% - 72px); }}
  </style>
</head>
<body>
  <div id="banner">
    <h1>fAirQ · light CDMI</h1>
    <p>Accepted edges · {payload["version"]} · thicker arrow = larger ΔMAE · drag nodes · hover for KS</p>
  </div>
  <div id="net"></div>
  <script>
    const data = {json.dumps(payload)};
    const net = new vis.Network(document.getElementById("net"), {{
      nodes: new vis.DataSet(data.nodes),
      edges: new vis.DataSet(data.edges)
    }}, {{
      nodes: {{
        shape: "box",
        margin: 12,
        borderWidth: 2,
        shadow: {{ enabled: true, color: "rgba(128,203,196,0.35)", size: 18 }},
        shapeProperties: {{ borderRadius: 10 }}
      }},
      edges: {{
        font: {{ color: "#c9d1d9", size: 11, strokeWidth: 0, align: "middle" }},
        arrows: {{ to: {{ enabled: true, scaleFactor: 0.85 }} }},
        shadow: true
      }},
      physics: {{ enabled: false }},
      interaction: {{ hover: true, tooltipDelay: 80, dragNodes: true }}
    }});
  </script>
</body>
</html>
"""


def write_png(graph: dict, path: str) -> None:
    import matplotlib.pyplot as plt
    import networkx as nx

    edges = accepted_edges(graph)
    drawn = nx.DiGraph()
    for edge in edges:
        drawn.add_edge(
            LABELS.get(edge["cause"], edge["cause"]),
            LABELS.get(edge["effect"], edge["effect"]),
            weight=max(float(edge.get("rel_mae") or 0), 0.005),
            ks=edge.get("ks_stat"),
        )
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 8.5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    layout = {}
    for node in drawn.nodes:
        raw = next((key for key, label in LABELS.items() if label == node), node)
        layout[node] = LAYOUT.get(raw, (0.0, 0.0))
    node_color = []
    for node in drawn.nodes:
        raw = next((key for key, label in LABELS.items() if label == node), node)
        node_color.append(COLORS.get(raw, "#90a4ae"))
    widths = [1.4 + 8 * drawn[u][v]["weight"] for u, v in drawn.edges]
    nx.draw_networkx_edges(
        drawn,
        layout,
        ax=ax,
        width=widths,
        edge_color="#80cbc4",
        arrows=True,
        arrowsize=22,
        connectionstyle="arc3,rad=0.14",
        min_source_margin=28,
        min_target_margin=28,
        alpha=0.9,
    )
    labels = {node: node for node in drawn.nodes}
    for node, (x, y), color in zip(drawn.nodes, [layout[n] for n in drawn.nodes], node_color):
        ax.text(
            x,
            y,
            labels[node],
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#0d1117",
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": color,
                "edgecolor": "#eceff1",
                "linewidth": 1.4,
            },
            zorder=3,
        )
    edge_labels = {(u, v): f"{drawn[u][v]['weight']:.2f}" for u, v in drawn.edges}
    nx.draw_networkx_edge_labels(
        drawn,
        layout,
        edge_labels=edge_labels,
        ax=ax,
        font_size=8,
        font_color="#c9d1d9",
        bbox={"facecolor": "#0d1117", "edgecolor": "none", "alpha": 0.75, "pad": 1},
    )
    ax.set_title("fAirQ light CDMI · accepted links  (edge label = ΔMAE)", color="#e6edf3", pad=18)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.25, 1.35)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def run() -> None:
    db = connect()
    graph = latest_graph(db)
    if graph is None:
        raise RuntimeError("no causal graph; run python -m fairq.cdmi first")
    os.makedirs(OUT_DIR, exist_ok=True)
    html_path = os.path.join(OUT_DIR, "causal_graph.html")
    png_path = os.path.join(OUT_DIR, "causal_graph.png")
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(graph_html(graph))
    print(f"wrote {html_path}")
    try:
        write_png(graph, png_path)
        print(f"wrote {png_path}")
    except ImportError:
        print("skip PNG (install networkx matplotlib)")


if __name__ == "__main__":
    run()
