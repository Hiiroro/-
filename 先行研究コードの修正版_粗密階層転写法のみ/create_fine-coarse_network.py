import pandas as pd
from shapely.geometry import Polygon, Point
import networkx as nx
import numpy as np
from math import dist 
import matplotlib.pyplot as plt

from shapely.geometry import Point
from shapely.ops import unary_union
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial import Delaunay
import os


def get_nodes_inside_cells(polys, pos):
    inside_nodes = set()
    # for node, (x, y) in pos.items():
    #     p = Point(x, y)

    #     for cell_id, poly in polys.items():
    #         if poly.contains(p):
    #             inside_nodes.add(node)
    #             break

    for n, (x, y) in pos.items():
        p = Point(x, y)
        if any(poly.contains(p) for poly in selected_polys.values()):
            inside_nodes.add(n)

    return inside_nodes

def merge_network_nodes(N1, N2, inside_nodes, pos):

    G_new = nx.Graph()

    inside_nodes = set(inside_nodes)

    # ===== 内側は細かいネットワーク =====
    for n in inside_nodes:
        G_new.add_node(n)

    for u, v, data in N1.edges(data=True):
        if u in inside_nodes and v in inside_nodes:
            G_new.add_edge(u, v, **data)

    # ===== 外側は粗いネットワーク =====
    for u, v, data in N2.edges(data=True):

        if (u not in inside_nodes) and (v not in inside_nodes):
            G_new.add_edge(u, v, **data)

    return G_new

def save_graph_as_mesh(G_merged, pos, prefix="mesh", h=None):

    node_list = list(G_merged.nodes())

    points = np.array([pos[n] for n in node_list])

    N = len(node_list)
    # メッシュ組み直し
    tri = Delaunay(points)
    triangles = tri.simplices  # (E,3)

    for simplex in tri.simplices:
        for a, b in [(simplex[0], simplex[1]),
                    (simplex[1], simplex[2]),
                    (simplex[2], simplex[0])]:

            u = node_list[a]
            v = node_list[b]

            x1, y1 = pos[u]
            x2, y2 = pos[v]
            dist = ((x1-x2)**2 + (y1-y2)**2)**0.5

            G_merged.add_edge(u, v, weight=dist)

    # 孤立しているノードは削除する
    isolated = list(nx.isolates(G_merged))
    G_merged.remove_nodes_from(isolated)
    pos_merged = {n: pos[n] for n in G_merged.nodes()}
    
    node_list = list(G_merged.nodes())
    points = np.array([pos[n] for n in node_list])
    N = len(node_list)

    # --- h をファイル名に追加 ---
    if h is not None:
        fname_prefix = f"{prefix}_h{h}"
    else:
        fname_prefix = prefix

    # --- ノード一覧 CSV ---
    nodes_out = np.array([
        [n, pos[n][0], pos[n][1]]
        for n in node_list
    ])
    
    save_dir = "coarse-fine_networks"
    os.makedirs(save_dir, exist_ok=True)

    np.savetxt(
        f"{save_dir}/coarse-fine_h{h}_r{r}_sc{source}_sk{sink}_Q{Q}_nodes.csv",
        nodes_out,
        delimiter=",",
        header="id,x,y",
        comments=""
    )

    # --- 無向エッジ集合 ---
    edges_out = np.array(list(G_merged.edges()), dtype=int)

    np.savetxt(
        f"{save_dir}/coarse-fine_h{h}_r{r}_sc{source}_sk{sink}_Q{Q}_edges.csv",
        edges_out,
        delimiter=",",
        header="src,dst",
        comments="",
        fmt="%d"
    )
    
    print("node:", G_merged.number_of_nodes())
    print("edge:", G_merged.number_of_edges())

    print(f"saved: {fname_prefix}_nodes.csv, {fname_prefix}_edges.csv")
    # ===========================================
    # グラフ描画
    plt.figure(figsize=(6,6))
    nx.draw(G_merged, pos_merged, node_size=5,node_color='black', with_labels= False, edge_color="black")
    nx.draw_networkx_nodes(G_merged,pos_merged,nodelist=[source],node_size=200,node_color="blue",linewidths=1.0, edgecolors="black")
    nx.draw_networkx_nodes(G_merged,pos_merged,nodelist=[sink],node_size=200,node_color="red",linewidths=1.0, edgecolors="black")


    for cid, poly in selected_polys.items():
        x, y = poly.exterior.xy
        plt.fill(x, y, alpha=0.4, color="lightgray")
        plt.plot(x, y, "k-", linewidth=1)
        

    plt.xlim(100,200)
    plt.ylim(100,200)
    plt.gca().set_aspect("equal")
    plt.savefig(f"{save_dir}/coarse-fine_h{h}_r{r}_sc{source}_sk{sink}_Q{Q}.png", format="png", dpi=300)
    plt.close() 

    return points, triangles, node_list

# h=2.7, 3.0, 4.0, 6.0
# r=0.1, 0.3, 0.5, 0.7, 0.9
# Q=1.0, 2.0, 4.0, 6.0

# voronoi_cell_num/の条件を見ながら手動で作成
h = 2.7
r = 0.6
Q = 6.0
source, sink = 373, 439 # --h 2.7 
#source, sink = 369, 368 # --h 3.0
# source, sink = 81, 95 # --h 6.0

polys = f"voronoi_cell_num/voronoi_coarse_h{h}_r{r}_sc{source}_sk{sink}_Q{Q}.csv"
base_nodes_csv   = f"networks_for_simulation/circle_r45.0_L300.0_h{h}_nodes.csv"
base_edges_csv   = f"networks_for_simulation/circle_r45.0_L300.0_h{h}_edges.csv"
coarse_nodes_csv = f"coarse_networks/coarse_h{h}_r{r}_sc{source}_sk{sink}_nodes.csv"
coarse_edges_csv = f"coarse_networks/coarse_h{h}_r{r}_sc{source}_sk{sink}_edges.csv"

# G1 base =====================================
nodes_arr = np.loadtxt(base_nodes_csv, delimiter=",", skiprows=1)
if nodes_arr.ndim == 1:
    nodes_arr = nodes_arr.reshape(1, -1)

node_ids = nodes_arr[:, 0].astype(int)
coords   = nodes_arr[:, 1:3]
nodes = list(node_ids)
base_pos   = {int(i): (float(x), float(y)) for i, (x, y) in zip(node_ids, coords)}

edges_arr = np.loadtxt(base_edges_csv, delimiter=",", skiprows=1, dtype=int)
if edges_arr.ndim == 1:
    edges_arr = edges_arr.reshape(1, -1)
edges = [(int(i), int(j)) for i, j in edges_arr]

G1 = nx.Graph()
G1.add_nodes_from(nodes)

for u, v in edges:
    p1, p2 = base_pos[u], base_pos[v]
    weight = dist(p1, p2)
    G1.add_edge(u, v, weight=weight)

# ===========================================
# G2 coarse =====================================
nodes_arr = np.loadtxt(coarse_nodes_csv, delimiter=",", skiprows=1)
if nodes_arr.ndim == 1:
    nodes_arr = nodes_arr.reshape(1, -1)

node_ids = nodes_arr[:, 0].astype(int)
coords   = nodes_arr[:, 1:3]
nodes = list(node_ids)
coarse_pos   = {int(i): (float(x), float(y)) for i, (x, y) in zip(node_ids, coords)}

edges_arr = np.loadtxt(coarse_edges_csv, delimiter=",", skiprows=1, dtype=int)
if edges_arr.ndim == 1:
    edges_arr = edges_arr.reshape(1, -1)
edges = [(int(i), int(j)) for i, j in edges_arr]

G2 = nx.Graph()
G2.add_nodes_from(nodes)

for u, v in edges:
    p1, p2 = coarse_pos[u], coarse_pos[v]
    weight = dist(p1, p2)
    G2.add_edge(u, v, weight=weight)
# ===========================================

df = pd.read_csv(polys)
# verticesをリストに変換
cell_polygons = {}
cell_values = {}
for _, row in df.iterrows():
    cell_id = row["cell_id"]
    verts_str = row["vertices"].split(";")
    verts = [list(map(float, v.split(":"))) for v in verts_str]
    cell_polygons[cell_id] = verts
    cell_values[cell_id] = row["edge_sum"]

# ===========================================
threshold = max(cell_values.values())/2
print(threshold)
selected_cell_ids = [cid for cid, val in cell_values.items() if val > threshold]
selected_polys = {cid: Polygon(cell_polygons[cid]) for cid in selected_cell_ids}
# ===========================================

inside_nodes = get_nodes_inside_cells(selected_polys, base_pos)
G_merged = merge_network_nodes(G1, G2, inside_nodes, base_pos)

save_graph_as_mesh(
    G_merged,
    base_pos,
    prefix="merged_mesh",
    h=h
)

print(G_merged.number_of_nodes())