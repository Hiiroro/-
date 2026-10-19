import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from math import dist 
import matplotlib.colors as mcolors
import os 
import argparse
import pickle
import csv


# ============================================================
# source / sink を選択
# ============================================================

def choose_source_sink_on_diameter(pos):
    node_ids = list(pos.keys())
    coords   = np.array([pos[i] for i in node_ids])

    xs = coords[:, 0]
    ys = coords[:, 1]

    cx = xs.mean()
    cy = ys.mean()

    r = np.sqrt((xs - cx)**2 + (ys - cy)**2).max()

    target_source = np.array([cx - r, cy])
    target_sink   = np.array([cx + r, cy])

    dL = np.sum((coords - target_source)**2, axis=1)
    dR = np.sum((coords - target_sink)**2, axis=1)

    source_idx = int(np.argmin(dL))
    sink_idx   = int(np.argmin(dR))

    source = node_ids[source_idx]
    sink   = node_ids[sink_idx]

    return source, sink, (cx, cy), r


# ============================================================
# グラフ描画
# ============================================================

def graph(G, i, Q):

    Q_i = Q[i]

    weights = []

    for u, v in G.edges():
        weight = max(Q_i[u, v], Q_i[v, u])
        weights.append(weight)

    fig, ax = plt.subplots(figsize=(17, 8))

    source_node_color = 'blue'
    sink_node_color   = 'red'

    norm = mcolors.Normalize(vmin=0, vmax=I0)

    cmap = plt.cm.binary

    edge_colors = cmap(norm(weights))

    nx.draw(
        G,
        pos,
        width=weights,
        node_size=5,
        node_color='black',
        with_labels=False,
        edge_color=edge_colors
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[source],
        node_size=200,
        node_color=source_node_color,
        ax=ax,
        linewidths=1.0,
        edgecolors="black"
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[sink],
        node_size=200,
        node_color=sink_node_color,
        ax=ax,
        linewidths=1.0,
        edgecolors="black"
    )

    fig.tight_layout()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    colorbar = fig.colorbar(sm, ax=ax)
    colorbar.ax.tick_params(labelsize=20)

    plt.xlim(100, 200)
    plt.ylim(100, 200)

    plt.axis("equal")

    save_root = "result_high_density_networks"

    save_fig_dir = os.path.join(
        save_root,
        f"{prefix}_Q{I0}"
    )

    os.makedirs(save_fig_dir, exist_ok=True)

    plt.savefig(
        f"{save_fig_dir}/" + str(i) + ".png",
        format="png",
        dpi=300
    )

    plt.close()


# ============================================================
# conductivity / length
# ============================================================

def divide_non_zero_element(D, L, num_nodes, list_edges):

    X = np.zeros((num_nodes, num_nodes))

    for (i, j) in list_edges:

        i = int(i)
        j = int(j)

        X[i, j] = D[i, j] / L[i, j]
        X[j, i] = X[i, j]

    return X


# ============================================================
# pressureを求める
# ============================================================

def deduce_p(D, L, F, num_nodes, list_edges, eps):

    Y = divide_non_zero_element(
        D,
        L,
        num_nodes,
        list_edges
    )

    A = np.diag(
        np.sum(Y, axis=1) + eps
    ) - Y

    p = np.linalg.solve(A, F)

    return p


# ============================================================
# Dの更新式
# ============================================================

def f(x, gamma):

    powered = x ** gamma

    return powered


# ============================================================
# 流量QとDの変化量を計算
# ============================================================

def dD(
    D,
    L,
    P,
    nnodes,
    edge,
    tick,
    mu
):

    X = divide_non_zero_element(
        D,
        L,
        nnodes,
        edge
    )

    Q = np.multiply(
        X,
        np.expand_dims(P, axis=1) - P
    )

    a = 1.0

    ans = f(
        np.abs(Q),
        mu
    ) - a * D

    return ans, Q


# ============================================================
# Runge-Kutta法
# ============================================================

def RK_Dt(
    nnodes,
    eps,
    t,
    length,
    edge_list
):

    for i in range(0, len(t) - 1):

        # ----------------------------------------------------
        # k1
        # ----------------------------------------------------

        k1, Q1 = dD(
            conductivity[i],
            length,
            pressure[i],
            nnodes,
            edge_list,
            t[i],
            gamma
        )

        k1 = k1 * dt
        Q1 = Q1 * dt

        l1 = dt * deduce_p(
            conductivity[i],
            length,
            flux,
            nnodes,
            edge_list,
            t[i]
        )


        # ----------------------------------------------------
        # k2
        # ----------------------------------------------------

        k2, Q2 = dD(
            conductivity[i] + (k1 / 2),
            length,
            pressure[i] + (l1 / 2),
            nnodes,
            edge_list,
            t[i] + (dt / 2),
            gamma
        )

        k2 = k2 * dt
        Q2 = Q2 * dt

        l2 = dt * deduce_p(
            conductivity[i] + (k1 / 2),
            length,
            flux,
            nnodes,
            edge_list,
            t[i] + (dt / 2)
        )


        # ----------------------------------------------------
        # k3
        # ----------------------------------------------------

        k3, Q3 = dD(
            conductivity[i] + (k2 / 2),
            length,
            pressure[i] + (l2 / 2),
            nnodes,
            edge_list,
            t[i] + (dt / 2),
            gamma
        )

        k3 = k3 * dt
        Q3 = Q3 * dt

        l3 = dt * deduce_p(
            conductivity[i] + (k2 / 2),
            length,
            flux,
            nnodes,
            edge_list,
            t[i] + (dt / 2)
        )


        # ----------------------------------------------------
        # k4
        # ----------------------------------------------------

        k4, Q4 = dD(
            conductivity[i] + k3,
            length,
            pressure[i] + l3,
            nnodes,
            edge_list,
            t[i] + dt,
            gamma
        )

        k4 = k4 * dt
        Q4 = Q4 * dt

        l4 = dt * deduce_p(
            conductivity[i] + k3,
            length,
            flux,
            nnodes,
            edge_list,
            t[i] + dt
        )


        # ----------------------------------------------------
        # Dを更新
        # ----------------------------------------------------

        conductivity[i + 1] = (
            (k1 + 2*k2 + 2*k3 + k4) / 6.0
        ) + conductivity[i]

        conductivity[i + 1] = (
            conductivity[i + 1]
            - np.diag(
                np.diagonal(
                    conductivity[i + 1]
                )
            )
        )


        # ----------------------------------------------------
        # pressureを更新
        # ----------------------------------------------------

        pressure[i + 1] = deduce_p(
            conductivity[i + 1],
            length,
            flux,
            nnodes,
            edge_list,
            eps
        )


        # ----------------------------------------------------
        # Qを更新
        # ----------------------------------------------------

        X = divide_non_zero_element(
            conductivity[i + 1],
            length,
            nnodes,
            edge_list
        )

        Q[i + 1] = np.multiply(
            X,
            np.expand_dims(
                pressure[i + 1],
                axis=1
            ) - pressure[i + 1]
        )


        # ----------------------------------------------------
        # 100ステップごとに画像保存
        # ----------------------------------------------------

        if (i % 100 == 0):

            graph(
                G,
                i,
                Q
            )

    return conductivity, pressure, Q


# ============================================================
# D → value → CSV
# ============================================================

def save_value_from_D(
    D_final,
    edge_list,
    output_path
):
    """
    Physarum Solverの最終時刻のDから
    0〜1に正規化したvalueを作成し、
    CSVに保存する。
    """

    # --------------------------------------------------------
    # 1. 各エッジのDを取得
    # --------------------------------------------------------

    values = []

    for u, v in edge_list:

        d_value = D_final[u, v]

        values.append(d_value)


    values = np.array(
        values,
        dtype=float
    )


    # --------------------------------------------------------
    # 2. Dを0〜1に正規化
    # --------------------------------------------------------

    d_min = values.min()
    d_max = values.max()


    if d_max == d_min:

        # 全て同じDだった場合
        normalized_values = np.zeros_like(
            values
        )

    else:

        normalized_values = (
            values - d_min
        ) / (
            d_max - d_min
        )


    # --------------------------------------------------------
    # 3. 保存先フォルダを作成
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
    # 4. CSVへ保存
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
        for (u, v), value in zip(
            edge_list,
            normalized_values
        ):

            writer.writerow(
                [
                    u,
                    v,
                    value
                ]
            )


    # --------------------------------------------------------
    # 5. 結果を表示
    # --------------------------------------------------------

    print()
    print("========================================")
    print("価値情報の保存が完了しました")
    print("========================================")

    print("Dの最小値:", d_min)
    print("Dの最大値:", d_max)

    print("valueの最小値:", normalized_values.min())
    print("valueの最大値:", normalized_values.max())

    print("保存先:", output_path)

    print("========================================")
    print()


# ============================================================
# argparse
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--h",
    type=float,
    default=3.0
)

parser.add_argument(
    "--Q0",
    type=int,
    default=10
)

args = parser.parse_args()


# ============================================================
# パラメータ
# ============================================================

h = args.h

L = 300.0

radius = 45.0

dir = "networks_for_simulation/"

prefix = (
    f"circle_r{radius}_L{L}_h{h}"
)

nodes_csv = (
    dir
    + f"{prefix}_nodes.csv"
)

edges_csv = (
    dir
    + f"{prefix}_edges.csv"
)


print(
    f"########### "
    f"実行:{prefix} "
    f"流量:{args.Q0} "
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

cx, cy = center


print(
    "中心座標 (cx, cy):",
    center
)

print(
    "推定半径 R:",
    radius
)

print(
    "選ばれた source (湧き出し):",
    source,
    "座標:",
    pos[source]
)

print(
    "選ばれた sink   (吸い込み):",
    sink,
    "座標:",
    pos[sink]
)


# ============================================================
# NetworkXグラフ作成
# ============================================================

G = nx.Graph()

G.add_nodes_from(
    nodes
)


for u, v in edges:

    p1 = pos[u]
    p2 = pos[v]

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
# ネットワーク情報
# ============================================================

nnodes = nx.number_of_nodes(
    G
)

edge_list = list(
    G.edges()
)


nbr_mtx = nx.to_numpy_array(
    G,
    weight="weight",
    dtype=float
)


L = nbr_mtx.copy()


edge_lengths = [
    d.get("weight")
    for u, v, d
    in G.edges(data=True)
]


avg_edge_length = (
    sum(edge_lengths)
    / len(edge_lengths)
)


print(
    f"座標ベースで取ってきた平均距離 "
    f"h={h}のとき:",
    avg_edge_length
)


# ============================================================
# Physarum Solver パラメータ
# ============================================================

nt = 301

dt = 0.1

tick = np.arange(
    0,
    nt,
    dt
)

eps = 0.001

I0 = args.Q0

gamma = 1


# ============================================================
# 配列を確保
# ============================================================

edge_list = list(
    G.edges
)


flux = np.zeros(
    nnodes,
    dtype=np.float32
)


length = np.zeros(
    (nnodes, nnodes),
    dtype=np.float32
)


pressure = np.zeros(
    (len(tick), nnodes),
    dtype=np.float32
)


conductivity = np.zeros(
    (
        len(tick),
        nnodes,
        nnodes
    ),
    dtype=np.float32
)


Q = np.zeros(
    (
        len(tick),
        nnodes,
        nnodes
    ),
    dtype=np.float32
)


# ============================================================
# 初期値
# ============================================================

length = L


# sourceから流入
flux[source] = I0

# sinkへ流出
flux[sink] = -I0


# 初期conductivity
conductivity[0] = nbr_mtx


np.divide(
    1,
    nbr_mtx,
    out=conductivity[0],
    where=nbr_mtx != 0
)


# ============================================================
# 初期pressure
# ============================================================

pressure[0] = deduce_p(
    conductivity[0],
    length,
    flux,
    nnodes,
    edge_list,
    eps
)


# ============================================================
# 初期Q
# ============================================================

X = divide_non_zero_element(
    conductivity[0],
    length,
    nnodes,
    edge_list
)


Q[0] = np.multiply(
    X,
    np.expand_dims(
        pressure[0],
        axis=1
    ) - pressure[0]
)


# ============================================================
# Physarum Solver 実行
# ============================================================

print()
print("Physarum Solverを実行しています...")
print()


D, P, Q = RK_Dt(
    nnodes,
    eps,
    tick,
    length,
    edge_list
)


print()
print("Physarum Solverが終了しました")
print()


# ============================================================
# 最終時刻のDを取得
# ============================================================

D_final = D[-1]


print(
    "最終時刻のDを取得しました"
)


# ============================================================
# D → value → CSV
# ============================================================

output_path = (
    "value_data/"
    f"{prefix}_Q{I0}_physarum_value.csv"
)


save_value_from_D(
    D_final,
    edge_list,
    output_path
)