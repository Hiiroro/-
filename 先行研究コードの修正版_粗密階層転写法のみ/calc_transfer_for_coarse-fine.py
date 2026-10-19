import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
from collections import defaultdict
from math import dist 
import re 
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import argparse
import os 
import pickle
from shapely.geometry import LineString, Polygon as ShapelyPolygon

def choose_source_sink_on_diameter(pos):
    """
    円形メッシュを想定して、
      - ノード全体の重心を円の中心 (cx, cy) とみなす
      - そこからの最大距離を半径 R とみなす
      - 左側の直径端 (cx-R, cy) に最も近いノードを source（湧き出し；x<0 側）
      - 右側の直径端 (cx+R, cy) に最も近いノードを sink  （吸い込み；x>0 側）
    とする。
    """
    node_ids = list(pos.keys())
    coords   = np.array([pos[i] for i in node_ids])

    xs = coords[:, 0]
    ys = coords[:, 1]

    # 円の中心を「全ノードの重心」とみなす
    cx = xs.mean()
    cy = ys.mean()

    # 最大距離を半径とみなす
    r = np.sqrt((xs - cx)**2 + (ys - cy)**2).max()

    # 左右の理想的な直径の端点
    target_source = np.array([cx - r, cy])  # 左端
    target_sink   = np.array([cx + r, cy])  # 右端

    # 各ノードからターゲット点への距離（二乗）
    dL = np.sum((coords - target_source)**2, axis=1)
    dR = np.sum((coords - target_sink)**2, axis=1)

    source_idx = int(np.argmin(dL))
    sink_idx   = int(np.argmin(dR))

    source = node_ids[source_idx]
    sink   = node_ids[sink_idx]

    return source, sink, (cx, cy), r

def split_length(u, v, cell_u, cell_v, pos_new, cell_polygons):

    line = LineString([pos_new[u], pos_new[v]])

    poly_u = ShapelyPolygon(cell_polygons[cell_u])
    poly_v = ShapelyPolygon(cell_polygons[cell_v])

    seg_u = line.intersection(poly_u)
    seg_v = line.intersection(poly_v)

    Lu = seg_u.length if not seg_u.is_empty else 0.0
    Lv = seg_v.length if not seg_v.is_empty else 0.0

    return Lu, Lv



# 引数設定する
parser = argparse.ArgumentParser()
parser.add_argument("--f", type=str, default="voronoi_cell_num/voronoi_coarse_h2.7_r0.6_sc373_sk439_Q6.0.csv")
# parser.add_argument("--v", type=str, default="coarse-fine_networks/coarse-fine_h2.7_r0.1_sc373_sk439_Q1.0.pkl")
args = parser.parse_args()

save_dir = "transfer_voronoi_coarse-fine/"
os.makedirs(save_dir, exist_ok=True)

# ボロノイ領域と領域内計算したファイル
voronoi_file = os.path.splitext(os.path.basename(args.f))[0]
file = args.f
df = pd.read_csv(f"{file}")
# h = float(re.search(r"\d+\.?\d*", file.split("_")[-5]).group())
# source = int(re.search(r"\d+", file.split("_")[-2]).group())
# sink   = int(re.search(r"\d+", file.split("_")[-3]).group())

# coarse_fine_file = os.path.splitext(os.path.basename(args.v))[0]

print("セル数 =", len(df))
print(sorted(df["cell_id"].tolist())[:20])
print(sorted(df["cell_id"].tolist())[-20:])

h  = float(re.search(r"h([0-9.]+)", file).group(1))
r  = float(re.search(r"r([0-9.]+)", file).group(1))
source = int(re.search(r"sc(\d+)", file).group(1))
sink = int(re.search(r"sk(\d+)", file).group(1))
Q0  = float(re.search(r"Q([0-9]+)", file).group(1))

# 元のノード番号を保存
source_old = source
sink_old = sink

# ベースネットワークのedges,nodes情報が書かれているファイル名
# network_file = "coarse-fine_networks"

# ノード、最短経路描画の色
edge_colors  = "black"
source_color = "blue"
sink_color   = "red"
pickup_color = "green"
voronoi_line = "black"

# verticesをリストに変換
cell_polygons = {}

for _, row in df.iterrows():
    cell_id = row["cell_id"]
    verts_str = row["vertices"].split(";")
    verts = [list(map(float, v.split(":"))) for v in verts_str]

    cell_polygons[cell_id] = verts

# ====================================================
dir = "result_coarse_networks/"

with open(
    f"{dir}coarse_h{h}_r{r}_sc{source}_sk{sink}_Q{Q0}.pkl",
    "rb"
) as f:
    D = pickle.load(f)
    Q = pickle.load(f)
    
print("D.shape =", D.shape)    
    
# ここに追加！
print("対称行列？", np.allclose(D, D.T))
print("最大差 =", np.max(np.abs(D - D.T)))    

nodes_csv = f"coarse-fine_networks/coarse-fine_h{h}_r{r}_sc{source}_sk{sink}_Q{Q0}_nodes.csv"
edges_csv = f"coarse-fine_networks/coarse-fine_h{h}_r{r}_sc{source}_sk{sink}_Q{Q0}_edges.csv"

low_nodes_csv = f"coarse_networks/coarse_h{h}_r{r}_sc{source}_sk{sink}_nodes.csv"

low_nodes_arr = np.loadtxt(
    low_nodes_csv,
    delimiter=",",
    skiprows=1
)

low_node_ids = low_nodes_arr[:,0].astype(int)

mapping_low = {
    old:i
    for i,old in enumerate(low_node_ids)
}


nodes_arr = np.loadtxt(nodes_csv,
                       delimiter=",",
                       skiprows=1)

node_ids = nodes_arr[:,0].astype(int)
coords = nodes_arr[:,1:3]

print(node_ids[:10])       # ←追加
print(node_ids.min())      # ←追加
print(node_ids.max())      # ←追加

nodes = list(node_ids)

pos = {
    int(i):(float(x),float(y))
    for i,(x,y) in zip(node_ids,coords)
}

edges_arr = np.loadtxt(edges_csv,
                       delimiter=",",
                       skiprows=1,
                       dtype=int)

edges = [
    (int(i),int(j))
    for i,j in edges_arr
]

# ====================================================
G = nx.Graph()
G.add_nodes_from(nodes)

for u, v in edges:
    p1, p2 = pos[u], pos[v]
    weight = dist(p1, p2)
    G.add_edge(u, v, weight=weight)

# ====================================================
# 元ノード一覧（飛び飛び）
nodes_list = list(G.nodes)

# old -> new の対応表
old2new = {old: i for i, old in enumerate(nodes_list)}
new2old = {i: old for old, i in old2new.items()}



# pos を作り直す (new id -> (x,y))
pos_new = {old2new[n]: pos[n] for n in nodes_list}

# 新しいグラフ作成
G_new = nx.Graph()
G_new.add_nodes_from(range(len(nodes_list)))

# edge を張り直す
for u, v, d in G.edges(data=True):
    u2 = old2new[u]
    v2 = old2new[v]
    G_new.add_edge(u2, v2, **d)   # weightなど属性もそのままコピー
    
print("G_newのノード数 =", G_new.number_of_nodes())
print("D.shape =", D.shape)    

print(f"source:{source}->{old2new[source]}, sink:{sink}->{old2new[sink]}")
source, sink = old2new[source], old2new[sink]
# plt.figure(figsize=(6,6))
# nx.draw(G_new, pos_new, node_size=5,node_color='black', with_labels= False, edge_color="black")
# nx.draw_networkx_nodes(G_new,pos_new,nodelist=[old2new[source]],node_size=200,node_color="blue",linewidths=1.0, edgecolors="black")
# nx.draw_networkx_nodes(G_new,pos_new,nodelist=[old2new[sink]],node_size=200,node_color="red",linewidths=1.0, edgecolors="black")
# plt.xlim(100,200)
# plt.ylim(100,200)
# plt.gca().set_aspect("equal")
# plt.show()
# exit()
# mapping = {old:i for i, old in enumerate(nodes)}
# ====================================================
# --- 3. ノードがどのセルに入るか判定 ---
def point_in_polygon(point, polygon):
    # ray casting法
    x, y = point
    poly = np.array(polygon)
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj-xi)*(y-yi)/(yj-yi)+xi):
            inside = not inside
        j = i
    return inside

node_to_cell = {}
for node, coord in pos_new.items():
    for cell_id, polygon in cell_polygons.items():
        if point_in_polygon(coord, polygon):
            node_to_cell[node] = cell_id
            break
        
        
cell_values = defaultdict(float)

for u, v in G_new.edges():

    cell_u = node_to_cell.get(u)
    cell_v = node_to_cell.get(v)

    if cell_u is None or cell_v is None:
        continue

    # cell_id → Dの添字へ変換
    if cell_u not in mapping_low or cell_v not in mapping_low:
        continue

    low_u = mapping_low[cell_u]
    low_v = mapping_low[cell_v]

    Duv = D[low_u, low_v]

    if cell_u == cell_v:
        cell_values[cell_u] += Duv

    else:

        Lu, Lv = split_length(
            u, v,
            cell_u, cell_v,
            pos_new,
            cell_polygons
        )

        L = Lu + Lv

        if L > 0:
            cell_values[cell_u] += Duv * Lu / L
            cell_values[cell_v] += Duv * Lv / L
        
print("sum(cell_values) =", sum(cell_values.values()))
print("sum(D) =", np.sum(D))        
        
print("セル数 =", len(cell_values))
print("cell_valuesの先頭5件")
for i, (k, v) in enumerate(cell_values.items()):
    print(k, v)
    if i == 4:
        break           
# ====================================================

##################################################################
# 
##################################################################

nnodes = len(G_new.nodes)
adj_array = np.zeros((nnodes, nnodes))
cell_edges = defaultdict(list)

for u, v in G_new.edges():

    cell_u = node_to_cell.get(u)
    cell_v = node_to_cell.get(v)

    if cell_u is None or cell_v is None:
        continue

    # 粗密ネットワーク上のエッジ長
    if cell_u == cell_v:

        length = G_new[u][v]["weight"]
        cell_edges[cell_u].append((u, v, length))

    else:

        Lu, Lv = split_length(
            u, v,
            cell_u,
            cell_v,
            pos_new,
            cell_polygons
        )

        cell_edges[cell_u].append((u, v, Lu))
        cell_edges[cell_v].append((u, v, Lv))

# adj_arrayにセル合計値を均等割り
for cell_id, edges_list in cell_edges.items():
    total = cell_values.get(cell_id, 0)
    # edgeの分配比率を考慮
    weight_sum = sum(w for _,_,w in edges_list)
    if weight_sum == 0:
        continue
    for u, v, w in edges_list:
        adj_array[u, v] += total * (w / weight_sum)
        adj_array[v, u] += total * (w / weight_sum)  # 無向グラフ

weights = []
unique_nodes = []
for u, v in G_new.edges():
    weight = max(adj_array[u, v], adj_array[v, u])
    if (weight>0.01):
        unique_nodes.append(u)
        unique_nodes.append(v)
    weights.append(weight*10)

fig, ax = plt.subplots(figsize=(6,6))
patches = []
colors = []

# ノード・エッジ描画
nx.draw(G_new, pos=pos_new, width=weights, node_size=3, node_color="lightgray", edge_color="black")

nx.draw_networkx_nodes(G_new,pos_new,nodelist=unique_nodes,node_size=5,node_color="black",ax=ax,linewidths=1.0, )
nx.draw_networkx_nodes(G_new,pos_new,nodelist=[source],node_size=200,node_color=source_color, ax=ax,linewidths=1.0, edgecolors="black")
nx.draw_networkx_nodes(G_new,pos_new,nodelist=[sink],node_size=200,node_color=sink_color, ax=ax,linewidths=1.0, edgecolors="black")

for cell_id, polygon in cell_polygons.items():
    poly_patch = Polygon(polygon, closed=True, edgecolor='black', alpha=0.3)
    patches.append(poly_patch)
    colors.append(cell_values[cell_id])

p = PatchCollection(patches, array=np.array(colors), cmap=cm.Oranges, alpha=0.3, edgecolor='black')
ax.add_collection(p)
plt.colorbar(p, ax=ax, label="Edge sum")
plt.xlim(100,200)
plt.ylim(100,200)
plt.gca().set_aspect("equal")
plt.savefig(f"{save_dir}transfer_coarse_h{h}_r{r}_sc{source_old}_sk{sink_old}_Q{Q0}.png", format="png", dpi=300)
plt.close() 

np.save(f"{save_dir}transfer_coarse_h{h}_r{r}_sc{source_old}_sk{sink_old}_Q{Q0}.npy", adj_array)
