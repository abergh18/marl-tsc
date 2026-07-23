"""
visualise_het_network.py — v3

Two visualisations of the Lancaster heterogeneous graph:

1. NetworkX graph (Kamada-Kawai layout) — coloured by node type,
   with annotation boxes showing key attributes per node.

2. Geographic layout — real SUMO (x, y) coordinates with road edges
   drawn as background.

Usage (Colab)
-------------
    from visualise_het_network import visualise_het_network
    visualise_het_network(network_file, output_dir)
"""

from __future__ import annotations
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from sumolib.net import readNet


# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    "intersection": "#1565C0",
    "primary":      "#C62828",
    "tertiary":     "#E65100",
    "unclassified": "#2E7D32",
    "residential":  "#6A1B9A",
    "edge":         "#90A4AE",
}

PRIORITY_COLOUR = {12: C["primary"], 10: C["tertiary"],
                   4: C["unclassified"], 3: C["residential"]}
ROAD_TYPE_LABEL = {12: "primary", 10: "tertiary",
                   4: "unclassified", 3: "residential"}


def _parse_tls_from_xml(network_file: str) -> dict:
    """
    Parse TLS phase info directly from the .net.xml file.
    Returns dict: tls_id -> {"num_phases": int, "action_dim": int}
    sumolib's getPrograms() often returns empty when phases aren't
    loaded into memory — reading the XML directly is more reliable.
    """
    tree = ET.parse(network_file)
    root = tree.getroot()

    tls_info = {}
    for tl_logic in root.findall(".//tlLogic"):
        tls_id = tl_logic.get("id")
        phases  = tl_logic.findall("phase")
        num_phases = len(phases)
        # count green phases (contain G or g, exclude pure yellow/red)
        action_dim = sum(
            1 for p in phases
            if any(c in ("G", "g") for c in p.get("state", ""))
            and "y" not in p.get("state", "")
        )
        # joinedS TLS cover multiple junctions — record for each
        if tls_id not in tls_info:
            tls_info[tls_id] = {"num_phases": num_phases,
                                 "action_dim": action_dim}

    # Also handle joined TLS: the junction may appear under a different id
    # Map each junction id that appears in the net to its TLS program
    junction_to_tls = {}
    for junction in root.findall(".//junction"):
        jid = junction.get("id")
        # junctions reference their TLS via incLanes/type but the TLS id
        # often matches the junction id or a cluster_ prefix
        if jid in tls_info:
            junction_to_tls[jid] = tls_info[jid]

    return tls_info


def _short(s, n=13):
    return s[:n] + "…" if len(s) > n else s

def _get_tls_coord(net, tls_id):
    if net.hasNode(tls_id):
        return net.getNode(tls_id).getCoord()
    parts = tls_id.replace("joinedS_", "").split("_cluster_")
    for part in parts:
        for candidate in [part] + part.split("_"):
            if net.hasNode(candidate):
                return net.getNode(candidate).getCoord()
    return (0.0, 0.0)

def _find_multihop_connections(net, tls_ids, max_hops=3):
    node_index = set(tls_ids)
    connections = []
    seen_pairs = set()

    for start_tls_id in tls_ids:
        start_node = None
        if net.hasNode(start_tls_id):
            start_node = net.getNode(start_tls_id)
        else:
            parts = start_tls_id.replace("joinedS_", "").split("_cluster_")
            for part in parts:
                for candidate in [part] + part.split("_"):
                    if net.hasNode(candidate):
                        start_node = net.getNode(candidate)
                        break
                if start_node:
                    break
        if start_node is None:
            continue

        queue   = [(start_node, 0, None)]
        visited = {start_node.getID()}

        while queue:
            current_node, hops, first_edge = queue.pop(0)
            if hops >= max_hops:
                continue

            for out_edge in current_node.getOutgoing():
                to_node    = out_edge.getToNode()
                to_node_id = to_node.getID()

                if to_node_id in visited:
                    continue
                visited.add(to_node_id)

                if first_edge is None:
                    shape = out_edge.getShape()
                    edge_attrs = {
                        "priority":    out_edge.getPriority(),
                        "road_type":   ROAD_TYPE_LABEL.get(out_edge.getPriority(), "other"),
                        "num_lanes":   out_edge.getLaneNumber(),
                        "length":      round(out_edge.getLength(), 1),
                        "speed_mph":   round(out_edge.getSpeed() * 2.23694, 1),
                        "edge_id":     out_edge.getID(),
                        "shape_mid_x": float(np.mean([p[0] for p in shape])) if shape else None,
                        "shape_mid_y": float(np.mean([p[1] for p in shape])) if shape else None,
                    }
                else:
                    edge_attrs = first_edge

                to_tls_id = to_node.getTLSID()
                if to_tls_id and to_tls_id in node_index and to_tls_id != start_tls_id:
                    pair = (start_tls_id, to_tls_id)
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        connections.append({
                            "from_tls": start_tls_id,
                            "to_tls":   to_tls_id,
                            "hops":     hops + 1,
                            **edge_attrs,
                        })
                else:
                    queue.append((to_node, hops + 1, edge_attrs))

    return connections


def _build_het_graph(net, tls_info: dict, max_hops=3):
    G = nx.Graph()

    tls_list   = list(net.getTrafficLights())
    tls_ids    = sorted(tls.getID() for tls in tls_list)

    # ── Intersection nodes ────────────────────────────────────────────────
    for tls_id in tls_ids:
        x, y = _get_tls_coord(net, tls_id)
        info = tls_info.get(tls_id, {})
        G.add_node(tls_id, node_type="intersection",
                   num_phases=info.get("num_phases", "?"),
                   action_dim=info.get("action_dim", "?"),
                   x=x, y=y)

    # ── Connection nodes — multi-hop BFS ──────────────────────────────────
    connections = _find_multihop_connections(net, tls_ids, max_hops=max_hops)

    for conn in connections:
        from_tls_id = conn["from_tls"]
        to_tls_id   = conn["to_tls"]

        # Use actual edge shape midpoint if available, else TLS midpoint
        if conn.get("shape_mid_x") is not None:
            cx, cy = conn["shape_mid_x"], conn["shape_mid_y"]
        else:
            fx, fy = _get_tls_coord(net, from_tls_id)
            tx, ty = _get_tls_coord(net, to_tls_id)
            cx, cy = (fx + tx) / 2, (fy + ty) / 2

        conn_id = f"conn::{conn['edge_id']}"
        G.add_node(conn_id, node_type="connection",
                   from_tls=from_tls_id, to_tls=to_tls_id,
                   edge_id=conn["edge_id"],
                   priority=conn["priority"],
                   road_type=conn["road_type"],
                   num_lanes=conn["num_lanes"],
                   length=conn["length"],
                   speed_mph=conn["speed_mph"],
                   is_signalised=False,
                   hops=conn["hops"],
                   x=cx, y=cy)
        G.add_edge(from_tls_id, conn_id, edge_type="to_conn")
        G.add_edge(conn_id,     to_tls_id, edge_type="from_conn")

    return G, tls_ids


def plot_networkx(G, tls_ids, ax):
    # Revert to Kamada-Kawai — better spacing for sparse graphs
    pos = nx.kamada_kawai_layout(G, scale=3.5)

    i_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "intersection"]
    c_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "connection"]
    c_cols  = [PRIORITY_COLOUR.get(G.nodes[n]["priority"], "#9E9E9E") for n in c_nodes]

    # Edges
    nx.draw_networkx_edges(G, pos, ax=ax,
        edge_color=C["edge"], arrows=True,
        arrowsize=16, width=2.0, alpha=0.65,
        min_source_margin=22, min_target_margin=22,
        connectionstyle="arc3,rad=0.07")

    # Connection nodes
    nx.draw_networkx_nodes(G, pos, ax=ax,
        nodelist=c_nodes, node_color=c_cols,
        node_size=700, alpha=0.92,
        linewidths=1.5, edgecolors="white")

    # Intersection nodes
    nx.draw_networkx_nodes(G, pos, ax=ax,
        nodelist=i_nodes, node_color=C["intersection"],
        node_size=2400, alpha=0.95,
        linewidths=2.0, edgecolors="white")

    # Intersection labels
    nx.draw_networkx_labels(G, pos,
        labels={n: _short(n, 11) for n in i_nodes}, ax=ax,
        font_size=7, font_color="white", font_weight="bold")

    # Intersection annotation boxes
    for n in i_nodes:
        d = G.nodes[n]
        x, y = pos[n]
        txt = f"phases: {d['num_phases']}\nactions: {d['action_dim']}"
        ax.annotate(txt, xy=(x, y), xytext=(0, 36),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.35",
                              fc="#E3F2FD", ec="#1565C0",
                              alpha=0.95, linewidth=1.2),
                    zorder=10)

    # Connection node annotation boxes
    for n in c_nodes:
        d = G.nodes[n]
        x, y = pos[n]
        col = PRIORITY_COLOUR.get(d["priority"], "#9E9E9E")
        txt = (f"{d['road_type']}\n"
               f"{d['num_lanes']} lane{'s' if d['num_lanes']!=1 else ''}"
               f"  {d['length']}m\n"
               f"{d['speed_mph']} mph")
        ax.annotate(txt, xy=(x, y), xytext=(0, -44),
                    textcoords="offset points",
                    ha="center", va="top", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc="#FAFAFA", ec=col,
                              alpha=0.92, linewidth=1.0),
                    zorder=10)

    ax.set_title(
        "Heterogeneous Graph — Kamada-Kawai Layout\n"
        "Blue = intersection (agent)   |   Coloured = connection node by road type",
        fontsize=10, pad=12)
    ax.axis("off")

    ax.legend(handles=[
        mpatches.Patch(color=C["intersection"], label="Intersection (agent)"),
        mpatches.Patch(color=C["primary"],      label="Connection: primary"),
        mpatches.Patch(color=C["tertiary"],     label="Connection: tertiary"),
        mpatches.Patch(color=C["unclassified"], label="Connection: unclassified"),
        mpatches.Patch(color=C["residential"],  label="Connection: residential"),
    ], loc="lower left", fontsize=8, framealpha=0.9, edgecolor="#90A4AE")

def plot_geographic(G, net, tls_ids, ax):
    # Road edges background
    for edge in net.getEdges():
        shape = edge.getShape()
        if len(shape) < 2:
            continue
        xs = [p[0] for p in shape]
        ys = [p[1] for p in shape]
        p  = edge.getPriority()
        col = PRIORITY_COLOUR.get(p, "#B0BEC5")
        lw  = 2.5 if p >= 12 else (1.5 if p >= 10 else 0.7)
        ax.plot(xs, ys, color=col, linewidth=lw, alpha=0.45, zorder=1)

    # Connection nodes
    for nid, d in G.nodes(data=True):
        if d["node_type"] != "connection":
            continue
        col = PRIORITY_COLOUR.get(d["priority"], "#9E9E9E")
        ax.scatter(d["x"], d["y"], c=col, s=70, zorder=3,
                   alpha=0.82, edgecolors="white", linewidths=0.6)

    # Intersection nodes + labels
    for tls_id in tls_ids:
        d = G.nodes[tls_id]
        ax.scatter(d["x"], d["y"], c=C["intersection"],
                   s=280, zorder=5, edgecolors="white", linewidths=1.5)
        short = _short(tls_id, 13)
        txt   = f"{short}\np:{d['num_phases']}  a:{d['action_dim']}"
        ax.annotate(txt, xy=(d["x"], d["y"]),
                    xytext=(7, 7), textcoords="offset points",
                    fontsize=7, color="#0D47A1", fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc="#E3F2FD", ec="#1565C0",
                              alpha=0.92, linewidth=0.9))

    ax.set_title(
        "Geographic Layout — Real SUMO Coordinates\n"
        "p = signal phases   a = discrete actions available",
        fontsize=10, pad=12)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.legend(handles=[
        mpatches.Patch(color=C["intersection"], label="Intersection (agent)"),
        mpatches.Patch(color=C["primary"],      label="Primary"),
        mpatches.Patch(color=C["tertiary"],     label="Tertiary"),
        mpatches.Patch(color=C["unclassified"], label="Unclassified"),
        mpatches.Patch(color=C["residential"],  label="Residential"),
    ], loc="lower left", fontsize=8, framealpha=0.9, edgecolor="#90A4AE")


def print_node_summary(G, tls_ids):
    print("\n" + "=" * 70)
    print("INTERSECTION NODES (agents)")
    print("=" * 70)
    for tls_id in tls_ids:
        d = G.nodes[tls_id]
        print(f"  {tls_id}")
        print(f"    phases={d['num_phases']}  actions={d['action_dim']}  "
              f"degree={G.degree(tls_id)}  pos=({d['x']:.1f}, {d['y']:.1f})")

    conn_nodes = [(n, d) for n, d in G.nodes(data=True)
                  if d.get("node_type") == "connection"]
    print("\n" + "=" * 70)
    print("CONNECTION NODES")
    print("=" * 70)
    for _, d in sorted(conn_nodes, key=lambda x: x[1]["priority"], reverse=True):
        print(f"  {_short(d['from_tls'],22)} -> {_short(d['to_tls'],22)}")
        print(f"    type={d['road_type']}  lanes={d['num_lanes']}  "
              f"length={d['length']}m  speed={d['speed_mph']}mph")

    road_types = {}
    for _, d in conn_nodes:
        road_types[d["road_type"]] = road_types.get(d["road_type"], 0) + 1
    print("\n" + "=" * 70)
    print(f"  Intersections : {len(tls_ids)}")
    print(f"  Connections   : {len(conn_nodes)}")
    print(f"  Total nodes   : {len(tls_ids) + len(conn_nodes)}")
    print(f"  Road types    : {road_types}")
    print("=" * 70 + "\n")


def visualise_het_network(
      network_file: str, 
      output_dir: str = ".", 
      max_hops=3,
      city_name: str = None
  ):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading network: {network_file}")
    net = readNet(str(network_file))

    print("Parsing TLS phase info from XML...")
    tls_info = _parse_tls_from_xml(str(network_file))

    print("Building heterogeneous graph...")
    G, tls_ids = _build_het_graph(net, tls_info, max_hops=max_hops)

    print_node_summary(G, tls_ids)

    fig, axes = plt.subplots(1, 2, figsize=(24, 11))
    fig.patch.set_facecolor("#F5F5F5")

    plot_networkx(G, tls_ids, axes[0])
    plot_geographic(G, net, tls_ids, axes[1])

    fig.suptitle(f"{city_name} — Heterogeneous Graph Visualisation",
                 fontsize=15, fontweight="bold", y=1.01)

    plt.tight_layout()
    out_path = output_dir / "het_network_visualisation.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#F5F5F5")
    plt.show()
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    import os
    visualise_het_network(
        network_file=os.environ.get("LANCASTER_NET",
            "/content/drive/MyDrive/Uni-Masters/Group Project/outputs/lancasterv2.net.xml"),
        output_dir=os.environ.get("OUTPUT_DIR",
            "/content/drive/MyDrive/Uni-Masters/Group Project/outputs"),
    max_hops=3,
    city_name= 'Lancaster, UK',
    )