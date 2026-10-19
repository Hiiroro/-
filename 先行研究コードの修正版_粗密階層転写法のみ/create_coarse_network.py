import numpy as np
import networkx as nx
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import re
import ast
import os


def parse_log_file(filename):
    results = []

    with open(filename, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Running行を探す
        if line.startswith("Running:"):
            m = re.search(r"h=([0-9.]+)\s+r=([0-9.]+)\s+sc=([0-9]+)\s+sk=([0-9]+)", line)
            if m:
                h = float(m.group(1))
                r = float(m.group(2))
                sc = int(m.group(3))
                sk = int(m.group(4))

                # 次の行がpickup_nodes行
                pickup_line = lines[i+1].strip()

                # pickup_nodes= を除去
                pickup_str = pickup_line.split("=", 1)[1].strip()

                # np.int64(123) を 123 に変換
                pickup_str = re.sub(r"np\.int64\((\d+)\)", r"\1", pickup_str)

                pickup_nodes = ast.literal_eval(pickup_str)

                results.append((h, r, sc, sk, pickup_nodes))

                i += 2
                continue

        i += 1

    return results

def export_undirected_network_with_figure(points, pickup, edges, prefix="network", save_png=True):
    """
    points   : (N, 2) ノード座標 (x, y)
    triangles: (E, 3) 三角形要素
    prefix   : 出力ファイル名の接頭辞
    h        : メッシュサイズ (ファイル名に含めたい値)
    """
    N = points.shape[0]

    # --- ノード一覧 CSV ---
    # node_ids = np.arange(N, dtype=int)
    node_ids  = pickup
    nodes_out = np.column_stack([node_ids, points])

    save_dir = "coarse_networks"
    os.makedirs(save_dir, exist_ok=True)
    
    np.savetxt(

        
        f"{save_dir}/{prefix}_nodes.csv",
        nodes_out,
        delimiter=",",
        header="id,x,y",
        comments=""
    )

    # --- 無向エッジ集合 ---
    edges = np.array(sorted(edges), dtype=int)

    np.savetxt(
        f"coarse_networks/{prefix}_edges.csv",
        edges,
        delimiter=",",
        header="src,dst",
        comments="",
        fmt="%d"
    )

    print(f"Saved: {prefix}_nodes.csv, {prefix}_edges.csv")

# ** ランダムピックアップノード配列が書いているファイルを選択する **
data = parse_log_file("pickup_random_nodes/result_auto_pickup_random_node.txt")

for h, r, sc, sk, pickup_nodes in data:
    # ** 保存先 **
    save_filename=f"coarse_h{h}_r{r}_sc{sc}_sk{sk}" 

    print("条件:", h, r, sc, sk)
    print("pickup_nodes数:", len(pickup_nodes))

    L      = 300.0 # 固定
    radius = 45.0  # 固定
    dir    = "networks_for_simulation/"
    prefix = f"circle_r{radius}_L{L}_h{h}"   # メッシュ生成側と揃える
    nodes_csv = dir + f"{prefix}_nodes.csv"   # id,x,y
    edges_csv = dir + f"{prefix}_edges.csv"   # src,dst

    nodes_arr = np.loadtxt(nodes_csv, delimiter=",", skiprows=1)
    if nodes_arr.ndim == 1:
        nodes_arr = nodes_arr.reshape(1, -1)
    node_ids = nodes_arr[:, 0].astype(int)
    coords   = nodes_arr[:, 1:3]
    nodes = list(node_ids)
    pos   = {int(i): (float(x), float(y)) for i, (x, y) in zip(node_ids, coords)}

    # pickup_nodes: 選ばれたノード集合
    points = np.array([pos[n] for n in pickup_nodes])
    tri = Delaunay(points)
    G_mesh = nx.Graph()

    # ノード追加
    for i, n in enumerate(pickup_nodes):
        G_mesh.add_node(n, pos=pos[n])

    # 三角形の辺を全部追加
    for simplex in tri.simplices:
        for a, b in [(simplex[0], simplex[1]),
                    (simplex[1], simplex[2]),
                    (simplex[2], simplex[0])]:

            n1 = pickup_nodes[a]
            n2 = pickup_nodes[b]

            # 距離を重みとして追加
            x1, y1 = pos[n1]
            x2, y2 = pos[n2]
            dist = ((x1-x2)**2 + (y1-y2)**2)**0.5

            G_mesh.add_edge(n1, n2, weight=dist)

    points = np.array([pos[n] for n in pickup_nodes])
    export_undirected_network_with_figure(
        points,
        pickup_nodes,
        G_mesh.edges,
        prefix=save_filename,
        save_png=True
    )

    plt.figure(figsize=(6,6))
    nx.draw(G_mesh, pos, node_size=50, edge_color="gray", with_labels=False)
    plt.gca().set_aspect("equal")
    plt.savefig(f"coarse_networks/{save_filename}.png", format="png", dpi=300)
    plt.close() 