import pickle
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial import Voronoi
from collections import defaultdict
import pandas as pd
from math import dist 
import re 
from shapely.geometry import MultiPoint, LineString, box
from shapely.ops import voronoi_diagram
from shapely.strtree import STRtree
import argparse
import matplotlib.cm as cm
import matplotlib.colors as colors
import os

def parse_filename_params(f):
    h  = float(re.search(r"h([0-9.]+)", f).group(1))
    r  = float(re.search(r"r([0-9.]+)", f).group(1))
    sc = int(re.search(r"sc(\d+)", f).group(1))
    sk = int(re.search(r"sk(\d+)", f).group(1))
    return h, r, sc, sk

def extract_pickup_nodes(txt_path, target_h, target_r, target_sc, target_sk):
    with open(txt_path, "r") as f:
        lines = f.readlines()

    for i in range(len(lines)):
        line = lines[i].strip()

        if line.startswith("Running:"):
            m = re.search(r"h=([0-9.]+)\s+r=([0-9.]+)\s+sc=(\d+)\s+sk=(\d+)", line)
            if not m:
                continue

            h  = float(m.group(1))
            r  = float(m.group(2))
            sc = int(m.group(3))
            sk = int(m.group(4))

            if (h == target_h) and (r == target_r) and (sc == target_sc) and (sk == target_sk):
                pickup_line = lines[i+1].strip()

                # pickup_nodes= [ ... ] を取る
                arr_str = pickup_line.split("=", 1)[1].strip()

                # np.int64(123) を 123 に置換
                arr_str = re.sub(r"np\.int64\((\d+)\)", r"\1", arr_str)

                # [] の中身を数字として抽出
                nums = list(map(int, re.findall(r"\d+", arr_str)))

                return nums

    raise ValueError("対応する Running 条件が txt 内に見つかりません")

def compute_voronoi_cells(nodes, pos, xmin, xmax, ymin, ymax):
    boundary = box(xmin, ymin, xmax, ymax)

    pts = [pos[n] for n in nodes]
    mp = MultiPoint(pts)

    vd = voronoi_diagram(mp, envelope=boundary)
    cells = list(vd.geoms)  # Polygonのリスト

    # 各セルを代表点で最近傍ノードに割り当て
    node_points = np.array([pos[n] for n in nodes])

    def nearest_node(x, y):
        d2 = ((node_points[:, 0] - x) ** 2 + (node_points[:, 1] - y) ** 2)
        return nodes[int(np.argmin(d2))]

    node_to_poly = {}
    for poly in cells:
        rp = poly.representative_point()
        node = nearest_node(rp.x, rp.y)
        node_to_poly[node] = poly

    return node_to_poly

def distribute_edge_flow_to_cells(G, pos, weights, node_to_poly):
    # polygonリスト
    polys = list(node_to_poly.values())

    # polygon -> node の逆引き
    poly_to_node = {poly: node for node, poly in node_to_poly.items()}

    # 空間インデックス作成
    tree = STRtree(polys)

    cell_edge_sum = defaultdict(float)

    for (u, v), flow in zip(G.edges(), weights):
        line = LineString([pos[u], pos[v]])
        total_len = line.length
        if total_len == 0:
            continue

        # lineとbboxが交差するpolygon候補だけ取る
        idxs = tree.query(line)

        # for poly in candidates:
        for idx in idxs:
            poly = polys[idx]   # ← idx は整数、polyはPolygonになる

            inter = line.intersection(poly)
            inter = line.intersection(poly)
            if inter.is_empty:
                continue

            inter_len = inter.length
            if inter_len > 0:
                node = poly_to_node[poly]
                # 代表点のnode番号が領域の番号になる

                eps = 1e-24

                # flow = round(flow, 24)
                if flow <= eps:  
                    flow_R = 0.0  

                else:  
                    flow_R = flow  
                
                cell_edge_sum[node] += flow_R * (inter_len / total_len)

    return cell_edge_sum


# 引数設定する
parser = argparse.ArgumentParser()
parser.add_argument("--f", type=str, default="coarse_h4.0_r0.1_sc178_sk199_Q2.0")
parser.add_argument("--pick_file", type=str, default="pickup_random_nodes/result_auto_pickup_random_node.txt")
args = parser.parse_args()

h, r, sc, sk = parse_filename_params(args.f)
pickup_nodes = extract_pickup_nodes(args.pick_file, h, r, sc, sk)
print("pickup_nodes =", pickup_nodes)

# バイナリモードでファイルを読み込む
save_file = "voronoi_cell_num/"
os.makedirs(save_file, exist_ok=True)
file_name = args.f
lst = file_name.split("_")[:-1]
network_file = "_".join(lst)
source = int(re.search(r"\d+", file_name.split("_")[-3]).group())
sink   = int(re.search(r"\d+", file_name.split("_")[-2]).group())
dir = "coarse_networks/"
with open(f'result_coarse_networks/{file_name}.pkl', 'rb') as f:
    D = pickle.load(f)
    Q = pickle.load(f)

top5 = np.sort(np.unique(D.flatten()))[::-1][:5]
print(top5)
nodes_csv = dir + f"{network_file}_nodes.csv"   # id,x,y
edges_csv = dir + f"{network_file}_edges.csv"   # src,dst
# pickup_nodes= args.pick

# ノード、最短経路描画の色
edge_colors  = "black"
source_color = "blue"
sink_color   = "red"
pickup_color = "green"
voronoi_line = "black"

# ====================================================
# --- 読み込み ---
nodes_arr = np.loadtxt(nodes_csv, delimiter=",", skiprows=1)
if nodes_arr.ndim == 1:
    nodes_arr = nodes_arr.reshape(1, -1)

node_ids = nodes_arr[:, 0].astype(int)
coords   = nodes_arr[:, 1:3]
nodes = list(node_ids)
pos   = {int(i): (float(x), float(y)) for i, (x, y) in zip(node_ids, coords)}

# エッジ読み込み: src,dst
edges_arr = np.loadtxt(edges_csv, delimiter=",", skiprows=1, dtype=int)
if edges_arr.ndim == 1:
    edges_arr = edges_arr.reshape(1, -1)
edges = [(int(i), int(j)) for i, j in edges_arr]

# =============================================
# 読み込んだ情報をNetworkXデータとして反映する
# =============================================
G = nx.Graph()
G.add_nodes_from(nodes)

for u, v in edges:
    p1, p2 = pos[u], pos[v]
    weight = dist(p1, p2)
    G.add_edge(u, v, weight=weight)

mapping = {old:i for i, old in enumerate(nodes)}

# ====================================================
# PSで求めた経路を計算する
weights = []
draw_weights = [] # 描画用のweights
for u, v in G.edges():
    u = mapping[u]
    v = mapping[v]
    weight = max(D[u, v], D[v, u])
    weights.append(weight)
    draw_weights.append(weight * 5)


# ボロノイ境界
# ランダムでピックアップされたノードを選択する
points = np.array([pos[n] for n in pickup_nodes])

nodes = list(G.nodes())
# 無限平面上で作成されるのでクリッピングする
node_to_poly  = compute_voronoi_cells(nodes, pos, xmin=100, xmax=200, ymin=100, ymax=200)
cell_edge_sum = distribute_edge_flow_to_cells(G, pos, weights, node_to_poly)

for cell_id, total in cell_edge_sum.items():
    print(f"セル {cell_id} 内のエッジ合計: {total}")

# ====================================================
# 描画
fig, ax = plt.subplots(figsize=(6,6))
nx.draw(G, pos=pos, node_size=5, node_color="black", edge_color="lightgray")
nx.draw(G, pos=pos, width=weights, node_size=5, node_color="black", edge_color="black")
nx.draw_networkx_nodes(G,pos,nodelist=[source],node_size=200,node_color=source_color, ax=ax,linewidths=1.0, edgecolors="black")
nx.draw_networkx_nodes(G,pos,nodelist=[sink],node_size=200,node_color=sink_color, ax=ax,linewidths=1.0, edgecolors="black")

# 値の範囲を作る
vals = list(cell_edge_sum.values())
norm = colors.Normalize(vmin=min(vals), vmax=max(vals))
cmap = cm.Oranges   # 好きなカラーマップに変更可

# Voronoiセル塗りつぶし
for node, poly in node_to_poly.items():
    value = cell_edge_sum.get(node, 0)
    x, y = poly.exterior.xy
    ax.fill(x, y, color=cmap(norm(value)), alpha=0.2)
    ax.plot(x, y, "k-", alpha=0.3)

# 具体的な領域数値を描画する
# for node, value in cell_edge_sum.items():
#     x, y = pos[node]
#     ax.text(x, y+3, f"{node}", color="blue", fontsize=12, ha="right")
#     ax.text(x+1, y, f"{value:.3f}", color="black", fontsize=12, ha="left")

# カラーバー
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([]) 
fig.colorbar(sm, ax=ax, label="cell value")

plt.xlim(100,200)
plt.ylim(100,200)
plt.gca().set_aspect("equal")
plt.savefig(f"{save_file}{file_name}.png", format="png", dpi=300)
plt.close() 

# ====================================================
# 計算した領域ごとの値をファイルとして書き出す
data = []

for node, poly in node_to_poly.items():
    # polygonの頂点座標
    vertices = list(poly.exterior.coords)
    total = cell_edge_sum.get(node, 0)
    data.append({
        "cell_id": node,
        "vertices": vertices,
        "edge_sum": total
    })

df = pd.DataFrame(data)

# verticesを文字列化
df["vertices"] = df["vertices"].apply(
    lambda vs: ";".join([f"{x}:{y}" for x, y in vs])
)

df.to_csv(f"{save_file}voronoi_{file_name}.csv", index=False)