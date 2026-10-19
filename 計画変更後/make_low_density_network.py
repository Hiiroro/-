import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import os


# ============================================================
# 高密度ネットワークを読み込む
# ============================================================
def load_high_density_network(nodes_file, edges_file):

    nodes = np.loadtxt(
        nodes_file,
        delimiter=",",
        skiprows=1
    )

    edges = np.loadtxt(
        edges_file,
        delimiter=",",
        skiprows=1,
        dtype=int
    )

    # nodes.csv : id, x, y
    node_ids = nodes[:, 0].astype(int)
    points = nodes[:, 1:3]

    pos = {
        int(node_id): (float(x), float(y))
        for node_id, (x, y) in zip(node_ids, points)
    }

    G = nx.Graph()

    for node_id in node_ids:
        G.add_node(int(node_id))

    for u, v in edges:
        G.add_edge(int(u), int(v))

    return G, pos


# ============================================================
# 代表ノードを選ぶ
#
# spacing以上離れたノードを代表ノードとして選択する
# ============================================================
def select_representative_nodes(G, pos, spacing=5.4, seed=123):

    rng = np.random.default_rng(seed)

    nodes = list(G.nodes())
    rng.shuffle(nodes)

    representatives = []

    for node in nodes:

        x, y = pos[node]

        # 既に選んだ代表ノードとの距離を確認
        too_close = False

        for rep in representatives:

            rx, ry = pos[rep]

            distance = np.sqrt(
                (x - rx) ** 2 +
                (y - ry) ** 2
            )

            if distance < spacing:
                too_close = True
                break

        if not too_close:
            representatives.append(node)

    return representatives


# ============================================================
# 各高密度ノードを、一番近い代表ノードに割り当てる
#
# 例：
#
# 高密度ノード
#   A B C
#   D E F
#
#       ↓
#
# 代表ノード
#   A       F
#
# A,B,C,D,E → A
# F         → F
# ============================================================
def assign_nodes_to_regions(G, pos, representatives):

    node_to_region = {}

    for node in G.nodes():

        x, y = pos[node]

        nearest_rep = None
        nearest_distance = float("inf")

        for rep in representatives:

            rx, ry = pos[rep]

            distance = np.sqrt(
                (x - rx) ** 2 +
                (y - ry) ** 2
            )

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_rep = rep

        node_to_region[node] = nearest_rep

    return node_to_region


# ============================================================
# 高密度エッジから低密度エッジを作る
#
# 高密度側：
#
# A ─ B ─ C
#
# A,B → 代表A
# C   → 代表C
#
# B-C が領域をまたぐので
#
# 低密度：
#
# A ─ C
#
# とする。
#
# 同時に
#
# low_edge (A,C)
#       ↓
# high_edge (B,C)
#
# という対応関係も保存する。
# ============================================================
def build_low_density_network(G, node_to_region):

    low_graph = nx.Graph()

    # 代表ノードを低密度ノードとして追加
    representatives = sorted(set(node_to_region.values()))

    for rep in representatives:
        low_graph.add_node(rep)

    # 低密度エッジ → 高密度エッジの対応
    low_edge_to_high_edges = {}

    for u, v in G.edges():

        region_u = node_to_region[u]
        region_v = node_to_region[v]

        # 同じ領域内なら低密度エッジは作らない
        if region_u == region_v:
            continue

        low_edge = tuple(sorted((region_u, region_v)))
        high_edge = tuple(sorted((u, v)))

        # 低密度エッジを追加
        low_graph.add_edge(*low_edge)

        # 対応する高密度エッジを保存
        if low_edge not in low_edge_to_high_edges:
            low_edge_to_high_edges[low_edge] = []

        low_edge_to_high_edges[low_edge].append(high_edge)

    return low_graph, low_edge_to_high_edges


# ============================================================
# CSV保存
# ============================================================
def save_low_density_network(
    low_graph,
    pos,
    low_edge_to_high_edges,
    node_to_region,
    output_dir
):

    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------
    # 低密度ノード
    # --------------------------------------------------------
    node_file = os.path.join(
        output_dir,
        "low_density_nodes.csv"
    )

    with open(node_file, "w") as f:

        f.write("low_id,x,y,representative_high_id\n")

        for low_id in sorted(low_graph.nodes()):

            x, y = pos[low_id]

            f.write(
                f"{low_id},{x},{y},{low_id}\n"
            )

    # --------------------------------------------------------
    # 低密度エッジ
    # --------------------------------------------------------
    edge_file = os.path.join(
        output_dir,
        "low_density_edges.csv"
    )

    with open(edge_file, "w") as f:

        f.write("src,dst\n")

        for u, v in sorted(low_graph.edges()):

            f.write(f"{u},{v}\n")

    # --------------------------------------------------------
    # 高密度エッジとの対応
    # --------------------------------------------------------
    mapping_file = os.path.join(
        output_dir,
        "low_edge_to_high_edges.csv"
    )

    with open(mapping_file, "w") as f:

        f.write(
            "low_src,low_dst,high_src,high_dst\n"
        )

        for low_edge in sorted(low_edge_to_high_edges):

            low_u, low_v = low_edge

            for high_edge in low_edge_to_high_edges[low_edge]:

                high_u, high_v = high_edge

                f.write(
                    f"{low_u},{low_v},"
                    f"{high_u},{high_v}\n"
                )

    # --------------------------------------------------------
    # 各高密度ノードがどの低密度領域に属するか
    # --------------------------------------------------------
    region_file = os.path.join(
        output_dir,
        "high_node_to_low_region.csv"
    )

    with open(region_file, "w") as f:

        f.write("high_node,low_region\n")

        for high_node in sorted(node_to_region):

            low_region = node_to_region[high_node]

            f.write(
                f"{high_node},{low_region}\n"
            )

    print()
    print("保存完了")
    print(f"  {node_file}")
    print(f"  {edge_file}")
    print(f"  {mapping_file}")
    print(f"  {region_file}")


# ============================================================
# ネットワークを図として保存
# ============================================================
def save_figure(
    G,
    low_graph,
    pos,
    representatives,
    output_dir
):

    fig, ax = plt.subplots(figsize=(7, 7))

    # 高密度ネットワーク
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color="lightgray",
        width=0.5
    )

    # 低密度ネットワーク
    nx.draw_networkx_edges(
        low_graph,
        pos,
        ax=ax,
        edge_color="red",
        width=2.0
    )

    # 代表ノード
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=representatives,
        ax=ax,
        node_size=25
    )

    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(
        "High-density and Low-density Network"
    )

    ax.set_xlim(100, 200)
    ax.set_ylim(100, 200)

    fig.tight_layout()

    figure_file = os.path.join(
        output_dir,
        "low_density_network.png"
    )

    fig.savefig(
        figure_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"  {figure_file}")


# ============================================================
# main
# ============================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--h",
        type=float,
        default=2.7
    )

    parser.add_argument(
        "--low_spacing",
        type=float,
        default=5.4
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # 高密度ネットワーク
    # --------------------------------------------------------
    L = 300.0
    radius = 45.0

    network_dir = "networks_for_simulation"

    nodes_file = (
        f"{network_dir}/"
        f"circle_r{radius}_L{L}_h{args.h}_nodes.csv"
    )

    edges_file = (
        f"{network_dir}/"
        f"circle_r{radius}_L{L}_h{args.h}_edges.csv"
    )

    # --------------------------------------------------------
    # 読み込み
    # --------------------------------------------------------
    print("高密度ネットワークを読み込みます")

    G, pos = load_high_density_network(
        nodes_file,
        edges_file
    )

    print(f"高密度ノード数: {G.number_of_nodes()}")
    print(f"高密度エッジ数: {G.number_of_edges()}")

    # --------------------------------------------------------
    # 代表ノードを選ぶ
    # --------------------------------------------------------
    print()
    print("代表ノードを選択します")

    representatives = select_representative_nodes(
        G,
        pos,
        spacing=args.low_spacing,
        seed=args.seed
    )

    print(
        f"代表ノード数: {len(representatives)}"
    )

    # --------------------------------------------------------
    # 高密度ノードを領域に割り当てる
    # --------------------------------------------------------
    print()
    print("高密度ノードを領域に割り当てます")

    node_to_region = assign_nodes_to_regions(
        G,
        pos,
        representatives
    )

    # --------------------------------------------------------
    # 低密度ネットワーク作成
    # --------------------------------------------------------
    print()
    print("低密度ネットワークを作成します")

    low_graph, low_edge_to_high_edges = (
        build_low_density_network(
            G,
            node_to_region
        )
    )

    print(
        f"低密度ノード数: "
        f"{low_graph.number_of_nodes()}"
    )

    print(
        f"低密度エッジ数: "
        f"{low_graph.number_of_edges()}"
    )

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------
    output_dir = (
        f"low_density_h{args.h}"
        f"_spacing{args.low_spacing}"
    )

    save_low_density_network(
        low_graph,
        pos,
        low_edge_to_high_edges,
        node_to_region,
        output_dir
    )

    # --------------------------------------------------------
    # 図
    # --------------------------------------------------------
    save_figure(
        G,
        low_graph,
        pos,
        representatives,
        output_dir
    )

    print()
    print("===== 完了 =====")