import numpy as np
import networkx as nx
from math import dist
import argparse
import pandas as pd
import os


# =============================================
# source, sink を直径上から決定
# =============================================
def choose_source_sink_on_diameter(pos):

    node_ids = list(pos.keys())
    coords = np.array([pos[i] for i in node_ids])

    xs = coords[:, 0]
    ys = coords[:, 1]

    cx = xs.mean()
    cy = ys.mean()

    R = np.sqrt((xs-cx)**2 + (ys-cy)**2).max()

    target_source = np.array([cx-R, cy])
    target_sink   = np.array([cx+R, cy])

    dL = np.sum((coords-target_source)**2, axis=1)
    dR = np.sum((coords-target_sink)**2, axis=1)

    source = node_ids[np.argmin(dL)]
    sink   = node_ids[np.argmin(dR)]

    return source, sink


# =============================================
parser = argparse.ArgumentParser()
parser.add_argument("--h", type=float, default=2.7)
args = parser.parse_args()

h = args.h
L = 300.0
radius = 45.0

prefix = f"circle_r{radius}_L{L}_h{h}"

nodes_csv = f"networks_for_simulation/{prefix}_nodes.csv"
edges_csv = f"networks_for_simulation/{prefix}_edges.csv"


# =============================================
# ネットワーク読込
# =============================================

nodes_arr = np.loadtxt(nodes_csv, delimiter=",", skiprows=1)

if nodes_arr.ndim == 1:
    nodes_arr = nodes_arr.reshape(1, -1)

node_ids = nodes_arr[:,0].astype(int)
coords = nodes_arr[:,1:3]

pos = {
    int(i):(float(x), float(y))
    for i,(x,y) in zip(node_ids,coords)
}

edges_arr = np.loadtxt(edges_csv,
                       delimiter=",",
                       skiprows=1,
                       dtype=int)

if edges_arr.ndim == 1:
    edges_arr = edges_arr.reshape(1,-1)

G = nx.Graph()

for n in node_ids:
    G.add_node(int(n))

for u,v in edges_arr:

    u=int(u)
    v=int(v)

    length = dist(pos[u],pos[v])

    G.add_edge(u,v,weight=length)


# =============================================
# source sink
# =============================================

source,sink = choose_source_sink_on_diameter(pos)

print("source =",source)
print("sink   =",sink)


# =============================================
# Dijkstra
# =============================================

shortest_path = nx.shortest_path(
    G,
    source=source,
    target=sink,
    weight="weight"
)

shortest_length = nx.shortest_path_length(
    G,
    source=source,
    target=sink,
    weight="weight"
)

print("Shortest length =",shortest_length)
print("Number of nodes =",len(shortest_path))


# =============================================
# 保存
# =============================================

df = pd.DataFrame({
    "h":[h],
    "source":[source],
    "sink":[sink],
    "shortest_length":[shortest_length],
    "path":[",".join(map(str,shortest_path))]
})

save_dir = "correct_shortest_paths"
os.makedirs(save_dir, exist_ok=True)

df.to_csv(
    f"{save_dir}/correct_shortest_path_h{h}.csv",
    index=False
)

print("saved.")