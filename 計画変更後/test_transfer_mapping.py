import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from math import dist


# =========================================================
# 1. ネットワーク生成
# =========================================================

def generate_jittered_grid_mesh(
    L=300.0,
    h=5.0,
    jitter_ratio=0.4,
    seed=123
):
    rng = np.random.default_rng(seed)

    n = int(round(L / h))

    xs = np.linspace(0.0, L, n + 1)
    ys = np.linspace(0.0, L, n + 1)

    X, Y = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack([X.ravel(), Y.ravel()])

    # 境界上の点
    on_left = np.isclose(points[:, 0], 0.0)
    on_right = np.isclose(points[:, 0], L)
    on_bottom = np.isclose(points[:, 1], 0.0)
    on_top = np.isclose(points[:, 1], L)

    is_boundary = (
        on_left |
        on_right |
        on_bottom |
        on_top
    )

    # 内部点だけ揺らす
    jitter = jitter_ratio * h

    noise = rng.uniform(
        -jitter,
        jitter,
        size=points.shape
    )

    points_jittered = points.copy()

    points_jittered[~is_boundary] += noise[~is_boundary]

    # 範囲外に出ないようにする
    points_jittered[:, 0] = np.clip(
        points_jittered[:, 0],
        0.0,
        L
    )

    points_jittered[:, 1] = np.clip(
        points_jittered[:, 1],
        0.0,
        L
    )

    # Delaunay三角形分割
    tri = Delaunay(points_jittered)

    triangles = tri.simplices

    return points_jittered, triangles


# =========================================================
# 2. 円形領域だけ取り出す
# =========================================================

def extract_circular_submesh(
    points,
    triangles,
    center,
    radius
):
    cx, cy = center

    r2 = (
        (points[:, 0] - cx) ** 2
        +
        (points[:, 1] - cy) ** 2
    )

    mask = r2 <= radius ** 2

    inside_idx = np.where(mask)[0]

    old_to_new = -np.ones(
        len(points),
        dtype=int
    )

    old_to_new[inside_idx] = np.arange(
        len(inside_idx)
    )

    # 三角形の3頂点が全部円の内部にあるもの
    tri_mask = mask[triangles].all(axis=1)

    triangles_in = triangles[tri_mask]

    triangles_sub = old_to_new[triangles_in]

    points_sub = points[inside_idx]

    return points_sub, triangles_sub


# =========================================================
# 3. NetworkXグラフを作る
# =========================================================

def build_graph(points, triangles):

    G = nx.Graph()

    pos = {
        i: (
            float(p[0]),
            float(p[1])
        )
        for i, p in enumerate(points)
    }

    G.add_nodes_from(range(len(points)))

    for tri in triangles:

        i, j, k = tri

        G.add_edge(i, j)
        G.add_edge(j, k)
        G.add_edge(k, i)

    return G, pos


# =========================================================
# 4. エッジ情報を取得
# =========================================================

def get_edge_data(G, pos):

    edge_data = []

    for u, v in G.edges():

        p1 = np.array(pos[u])
        p2 = np.array(pos[v])

        length = np.linalg.norm(p2 - p1)

        midpoint = (p1 + p2) / 2.0

        edge_data.append({
            "u": u,
            "v": v,
            "p1": p1,
            "p2": p2,
            "length": length,
            "midpoint": midpoint
        })

    return edge_data


# =========================================================
# 5. 点と線分の距離
# =========================================================

def point_to_segment_distance(
    point,
    seg_a,
    seg_b
):

    ab = seg_b - seg_a

    ab_length_squared = np.dot(ab, ab)

    if ab_length_squared == 0:
        return np.linalg.norm(point - seg_a)

    t = np.dot(
        point - seg_a,
        ab
    ) / ab_length_squared

    t = np.clip(t, 0.0, 1.0)

    closest = seg_a + t * ab

    return np.linalg.norm(
        point - closest
    )


# =========================================================
# 6. 高密度エッジを低密度エッジに対応付ける
# =========================================================

def map_high_to_low_edges(
    high_edges,
    low_edges,
    distance_threshold=2.0
):

    mapping = {}

    for h_idx, h_edge in enumerate(high_edges):

        h_mid = h_edge["midpoint"]

        best_low_idx = None
        best_distance = float("inf")

        for l_idx, l_edge in enumerate(low_edges):

            distance = point_to_segment_distance(
                h_mid,
                l_edge["p1"],
                l_edge["p2"]
            )

            if distance < best_distance:

                best_distance = distance
                best_low_idx = l_idx

        # 閾値以内なら対応付ける
        if best_distance <= distance_threshold:

            mapping[h_idx] = best_low_idx

    return mapping


# =========================================================
# 7. メイン
# =========================================================

if __name__ == "__main__":

    L = 300.0
    radius = 45.0

    center = (
        L / 2.0,
        L / 2.0
    )

    # -----------------------------------------
    # 高密度ネットワーク
    # -----------------------------------------

    h_high = 2.7

    points_high, triangles_high = generate_jittered_grid_mesh(
        L=L,
        h=h_high,
        jitter_ratio=0.4,
        seed=123
    )

    points_high, triangles_high = extract_circular_submesh(
        points_high,
        triangles_high,
        center,
        radius
    )

    G_high, pos_high = build_graph(
        points_high,
        triangles_high
    )

    high_edges = get_edge_data(
        G_high,
        pos_high
    )


    # -----------------------------------------
    # 低密度ネットワーク
    # -----------------------------------------

    h_low = 5.4

    points_low, triangles_low = generate_jittered_grid_mesh(
        L=L,
        h=h_low,
        jitter_ratio=0.4,
        seed=123
    )

    points_low, triangles_low = extract_circular_submesh(
        points_low,
        triangles_low,
        center,
        radius
    )

    G_low, pos_low = build_graph(
        points_low,
        triangles_low
    )

    low_edges = get_edge_data(
        G_low,
        pos_low
    )


    # -----------------------------------------
    # ネットワーク情報表示
    # -----------------------------------------

    print()
    print("========== ネットワーク情報 ==========")

    print(
        f"高密度: "
        f"nodes={len(G_high.nodes)}, "
        f"edges={len(G_high.edges)}"
    )

    print(
        f"低密度: "
        f"nodes={len(G_low.nodes)}, "
        f"edges={len(G_low.edges)}"
    )


    # -----------------------------------------
    # エッジ対応付け
    # -----------------------------------------

    mapping = map_high_to_low_edges(
        high_edges,
        low_edges,
        distance_threshold=2.0
    )

    print()
    print("========== 対応付け結果 ==========")

    print(
        "高密度エッジ数:",
        len(high_edges)
    )

    print(
        "対応付けできた高密度エッジ数:",
        len(mapping)
    )

    print(
        "対応率:",
        len(mapping) / len(high_edges) * 100,
        "%"
    )


    # -----------------------------------------
    # 可視化
    # -----------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    # 高密度ネットワーク
    for edge in high_edges:

        p1 = edge["p1"]
        p2 = edge["p2"]

        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color="lightgray",
            linewidth=0.5
        )


    # 対応付けできた高密度エッジ
    for h_idx in mapping:

        edge = high_edges[h_idx]

        p1 = edge["p1"]
        p2 = edge["p2"]

        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color="red",
            linewidth=1.5
        )


    # 低密度エッジ
    for edge in low_edges:

        p1 = edge["p1"]
        p2 = edge["p2"]

        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color="blue",
            linewidth=1.0,
            alpha=0.5
        )


    ax.set_aspect("equal")

    ax.set_xlim(
        center[0] - radius - 5,
        center[0] + radius + 5
    )

    ax.set_ylim(
        center[1] - radius - 5,
        center[1] + radius + 5
    )

    ax.set_title(
        "Low-density to High-density Edge Mapping"
    )

    plt.tight_layout()
    plt.show()