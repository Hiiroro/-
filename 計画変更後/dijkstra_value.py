import networkx as nx
import numpy as np
import csv
import os
from math import dist
import argparse
import matplotlib.pyplot as plt


# ============================================================
# source / sink を選択
# ============================================================

def choose_source_sink_on_diameter(pos):

    node_ids = list(pos.keys())

    coords = np.array(
        [pos[i] for i in node_ids]
    )

    xs = coords[:, 0]
    ys = coords[:, 1]

    cx = xs.mean()
    cy = ys.mean()

    r = np.sqrt(
        (xs - cx) ** 2
        + (ys - cy) ** 2
    ).max()

    target_source = np.array(
        [cx - r, cy]
    )

    target_sink = np.array(
        [cx + r, cy]
    )

    dL = np.sum(
        (coords - target_source) ** 2,
        axis=1
    )

    dR = np.sum(
        (coords - target_sink) ** 2,
        axis=1
    )

    source_idx = int(
        np.argmin(dL)
    )

    sink_idx = int(
        np.argmin(dR)
    )

    source = node_ids[source_idx]
    sink = node_ids[sink_idx]

    return source, sink, (cx, cy), r


# ============================================================
# Dijkstraで最短経路を求める
# ============================================================

def save_dijkstra_graph(
    G,
    pos,
    shortest_path,
    source,
    sink,
    obstacle,
    output_path
):

    plt.figure(figsize=(8, 8))

    # 全エッジ
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="lightgray",
        width=0.5
    )

    # 最短経路
    path_edges = list(
        zip(
            shortest_path[:-1],
            shortest_path[1:]
        )
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=path_edges,
        edge_color="red",
        width=2.5
    )

    # source
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[source],
        node_color="blue",
        node_size=50
    )

    # sink
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[sink],
        node_color="red",
        node_size=50
    )

    # 障害物
    xmin, xmax, ymin, ymax = obstacle

    rect = plt.Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        facecolor="gray",
        alpha=0.7
    )

    plt.gca().add_patch(rect)

    plt.gca().set_aspect("equal")

    plt.xlim(100, 200)
    plt.ylim(100, 200)

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.title("Dijkstra Shortest Path")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Dijkstra経路画像を保存:",
        output_path
    )

def calculate_dijkstra_value(
    G,
    source,
    sink
):
    """
    Dijkstraでsourceからsinkまでの
    最短経路を求める。

    最短経路上のエッジ：
        value = 1

    それ以外：
        value = 0
    """

    # --------------------------------------------------------
    # 1. 最短経路を求める
    # --------------------------------------------------------

    shortest_path = nx.shortest_path(
        G,
        source=source,
        target=sink,
        weight="weight"
    )
    
    # ★ここで画像を保存
    obstacle = (144, 156, 127.5, 172.5)
    
    save_dijkstra_graph(
    G,
    pos,
    shortest_path,
    source,
    sink,
    obstacle,
    "value_data/dijkstra_shortest_path.png"
    )

    # --------------------------------------------------------
    # 2. 最短経路上のエッジを取得
    # --------------------------------------------------------

    shortest_edges = set()

    for i in range(
        len(shortest_path) - 1
    ):

        u = shortest_path[i]
        v = shortest_path[i + 1]

        # 無向グラフなので、
        # (u,v) と (v,u) を同じものとして扱う
        edge = tuple(
            sorted((u, v))
        )

        shortest_edges.add(edge)

    # --------------------------------------------------------
    # 3. 全エッジについてvalueを作る
    # --------------------------------------------------------

    value_data = []

    for u, v in G.edges():

        edge = tuple(
            sorted((u, v))
        )

        if edge in shortest_edges:

            value = 1.0

        else:

            value = 0.0

        value_data.append(
            (u, v, value)
        )

    return shortest_path, value_data


# ============================================================
# valueをCSVに保存
# ============================================================

def save_value_to_csv(
    value_data,
    output_path
):

    # --------------------------------------------------------
    # 保存先フォルダを作成
    # --------------------------------------------------------

    output_dir = os.path.dirname(
        output_path
    )

    if output_dir:

        os.makedirs(
            output_dir,
            exist_ok=True
        )

    # --------------------------------------------------------
    # CSV保存
    # --------------------------------------------------------

    with open(
        output_path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        # ヘッダー
        writer.writerow(
            [
                "src",
                "dst",
                "value"
            ]
        )

        # データ
        for u, v, value in value_data:

            writer.writerow(
                [
                    u,
                    v,
                    value
                ]
            )

    print()
    print("========================================")
    print("Dijkstraの価値情報を保存しました")
    print("========================================")

    print(
        "保存先:",
        output_path
    )

    print(
        "エッジ数:",
        len(value_data)
    )

    print(
        "value = 1 のエッジ数:",
        sum(
            value == 1.0
            for _, _, value
            in value_data
        )
    )

    print(
        "value = 0 のエッジ数:",
        sum(
            value == 0.0
            for _, _, value
            in value_data
        )
    )

    print("========================================")
    print()


# ============================================================
# argparse
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--h",
    type=float,
    default=2.7
)

args = parser.parse_args()


# ============================================================
# パラメータ
# ============================================================

h = args.h

L = 300.0

radius = 45.0

network_dir = (
    "networks_for_simulation2/"
)

prefix = (
    f"circle_r{radius}_L{L}_h{h}_obstacle"
)

nodes_csv = (
    network_dir
    + f"{prefix}_nodes.csv"
)

edges_csv = (
    network_dir
    + f"{prefix}_edges.csv"
)


print(
    f"########### "
    f"Dijkstra実行: {prefix} "
    f"###########"
)


# ============================================================
# ノード読み込み
# ============================================================

nodes_arr = np.loadtxt(
    nodes_csv,
    delimiter=",",
    skiprows=1
)


if nodes_arr.ndim == 1:

    nodes_arr = nodes_arr.reshape(
        1,
        -1
    )


node_ids = nodes_arr[
    :, 0
].astype(int)

coords = nodes_arr[
    :, 1:3
]


nodes = list(
    node_ids
)


pos = {
    int(i): (
        float(x),
        float(y)
    )
    for i, (x, y)
    in zip(
        node_ids,
        coords
    )
}


# ============================================================
# エッジ読み込み
# ============================================================

edges_arr = np.loadtxt(
    edges_csv,
    delimiter=",",
    skiprows=1,
    dtype=int
)


if edges_arr.ndim == 1:

    edges_arr = edges_arr.reshape(
        1,
        -1
    )


edges = [
    (int(i), int(j))
    for i, j in edges_arr
]


# ============================================================
# source / sink
# ============================================================

source, sink, center, radius = (
    choose_source_sink_on_diameter(
        pos
    )
)


print(
    "中心座標:",
    center
)

print(
    "source:",
    source,
    "座標:",
    pos[source]
)

print(
    "sink:",
    sink,
    "座標:",
    pos[sink]
)


# ============================================================
# NetworkXグラフを作成
# ============================================================

G = nx.Graph()

G.add_nodes_from(
    nodes
)


for u, v in edges:

    p1 = pos[u]
    p2 = pos[v]

    # ノード間の実際の距離
    weight = dist(
        p1,
        p2
    )

    G.add_edge(
        u,
        v,
        weight=weight
    )


# ============================================================
# Dijkstra実行
# ============================================================

print()
print(
    "Dijkstraで最短経路を計算しています..."
)
print()


shortest_path, value_data = (
    calculate_dijkstra_value(
        G,
        source,
        sink
    )
)


# ============================================================
# 最短経路を表示
# ============================================================

print()
print("最短経路:")
print(shortest_path)

print(
    "最短経路のノード数:",
    len(shortest_path)
)


# ============================================================
# 最短経路の長さを計算
# ============================================================

shortest_path_length = 0.0

for i in range(
    len(shortest_path) - 1
):

    u = shortest_path[i]
    v = shortest_path[i + 1]

    shortest_path_length += G[u][v]["weight"]


print(
    "最短経路の距離:",
    shortest_path_length
)


# ============================================================
# CSV保存
# ============================================================

output_path = (
    "value_data/"
    f"{prefix}_dijkstra_value.csv"
)


save_value_to_csv(
    value_data,
    output_path
)