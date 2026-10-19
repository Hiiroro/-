import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import os


# =========================================================
# 1. ジッター付き格子を生成
# =========================================================
def generate_jittered_grid_mesh(
    L=300.0,
    h=2.7,
    jitter_ratio=0.4,
    seed=123
):
    rng = np.random.default_rng(seed)

    n = int(round(L / h))

    xs = np.linspace(0.0, L, n + 1)
    ys = np.linspace(0.0, L, n + 1)

    X, Y = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack([X.ravel(), Y.ravel()])

    # 境界上の点は動かさない
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

    jitter = jitter_ratio * h

    noise = rng.uniform(
        -jitter,
        jitter,
        size=points.shape
    )

    points_jittered = points.copy()

    points_jittered[~is_boundary] += noise[~is_boundary]

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

    tri = Delaunay(points_jittered)

    return points_jittered, tri.simplices


# =========================================================
# 2. 円形領域を切り出す
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

    tri_mask = mask[triangles].all(axis=1)

    triangles_in = triangles[tri_mask]

    triangles_sub = old_to_new[triangles_in]

    points_sub = points[inside_idx]

    return points_sub, triangles_sub


# =========================================================
# 3. 線分と長方形障害物が交差するか判定
# =========================================================
def segment_intersects_rectangle(
    p1,
    p2,
    xmin,
    xmax,
    ymin,
    ymax
):
    """
    線分 p1-p2 が長方形に交差するか判定する。

    障害物の内部または境界に入る場合は True。
    """

    x1, y1 = p1
    x2, y2 = p2

    dx = x2 - x1
    dy = y2 - y1

    # Liang-Barsky法
    p = [-dx, dx, -dy, dy]
    q = [
        x1 - xmin,
        xmax - x1,
        y1 - ymin,
        ymax - y1
    ]

    t_min = 0.0
    t_max = 1.0

    for pi, qi in zip(p, q):

        if np.isclose(pi, 0.0):

            # 平行で、長方形の外側
            if qi < 0:
                return False

        else:

            t = qi / pi

            if pi < 0:
                t_min = max(t_min, t)
            else:
                t_max = min(t_max, t)

            if t_min > t_max:
                return False

    return True


# =========================================================
# 4. 障害物を考慮して三角形を削除
# =========================================================
def remove_obstacle_triangles(
    points,
    triangles,
    obstacle
):
    xmin, xmax, ymin, ymax = obstacle

    valid_triangles = []

    removed_triangles = 0

    for tri in triangles:

        i, j, k = tri

        p1 = points[i]
        p2 = points[j]
        p3 = points[k]

        # 3辺について障害物との交差を確認
        edge_intersects = (
            segment_intersects_rectangle(
                p1, p2,
                xmin, xmax, ymin, ymax
            )
            or
            segment_intersects_rectangle(
                p2, p3,
                xmin, xmax, ymin, ymax
            )
            or
            segment_intersects_rectangle(
                p3, p1,
                xmin, xmax, ymin, ymax
            )
        )

        if edge_intersects:
            removed_triangles += 1
            continue

        valid_triangles.append(tri)

    valid_triangles = np.array(
        valid_triangles,
        dtype=int
    )

    print(
        "障害物によって削除した三角形:",
        removed_triangles
    )

    return valid_triangles


# =========================================================
# 5. 三角形からグラフを作成
# =========================================================
def build_graph_from_triangles(
    points,
    triangles
):
    G = nx.Graph()

    pos = {
        i: (
            float(p[0]),
            float(p[1])
        )
        for i, p in enumerate(points)
    }

    for i in range(len(points)):
        G.add_node(i)

    for tri in triangles:

        i, j, k = tri

        G.add_edge(i, j)
        G.add_edge(j, k)
        G.add_edge(k, i)

    return G, pos


# =========================================================
# 6. 孤立ノードを削除してIDを振り直す
# =========================================================
def remove_isolated_nodes(
    points,
    triangles,
    G
):
    isolated = list(nx.isolates(G))

    print("孤立ノード:", isolated)

    if len(isolated) == 0:
        return points, triangles

    # 孤立していないノード
    keep_nodes = [
        i for i in range(len(points))
        if i not in isolated
    ]

    old_to_new = {
        old_id: new_id
        for new_id, old_id
        in enumerate(keep_nodes)
    }

    new_points = points[keep_nodes]

    new_triangles = []

    for tri in triangles:

        # 三角形の3点がすべて残っている場合だけ残す
        if all(int(v) in old_to_new for v in tri):

            new_triangles.append([
                old_to_new[int(tri[0])],
                old_to_new[int(tri[1])],
                old_to_new[int(tri[2])]
            ])

    new_triangles = np.array(
        new_triangles,
        dtype=int
    )

    return new_points, new_triangles


# =========================================================
# 7. CSV保存 + 図の保存
# =========================================================
def export_network(
    points,
    triangles,
    prefix,
    obstacle,
    center,
    radius
):

    save_dir = "networks_for_simulation2"

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    # -----------------------------
    # Graph
    # -----------------------------
    G, pos = build_graph_from_triangles(
        points,
        triangles
    )

    # -----------------------------
    # ノードCSV
    # -----------------------------
    node_ids = np.arange(
        len(points)
    )

    nodes_out = np.column_stack([
        node_ids,
        points
    ])

    np.savetxt(
        f"{save_dir}/{prefix}_nodes.csv",
        nodes_out,
        delimiter=",",
        header="id,x,y",
        comments=""
    )

    # -----------------------------
    # エッジCSV
    # -----------------------------
    edges = set()

    for tri in triangles:

        i, j, k = map(int, tri)

        edges.add(
            tuple(sorted((i, j)))
        )

        edges.add(
            tuple(sorted((j, k)))
        )

        edges.add(
            tuple(sorted((k, i)))
        )

    edges = np.array(
        sorted(edges),
        dtype=int
    )

    np.savetxt(
        f"{save_dir}/{prefix}_edges.csv",
        edges,
        delimiter=",",
        header="src,dst",
        comments="",
        fmt="%d"
    )

    print()
    print("========== 保存結果 ==========")
    print("ノード数:", len(points))
    print("エッジ数:", len(edges))
    print(
        "連結成分数:",
        nx.number_connected_components(G)
    )

    # -----------------------------
    # 図
    # -----------------------------
    xmin, xmax, ymin, ymax = obstacle

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    ax.triplot(
        points[:, 0],
        points[:, 1],
        triangles,
        linewidth=0.4
    )

    # 障害物
    rect = plt.Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        facecolor="gray",
        alpha=0.7
    )

    ax.add_patch(rect)

    # 円
    circle = plt.Circle(
        center,
        radius,
        fill=False,
        linestyle="--",
        linewidth=2
    )

    ax.add_patch(circle)

    ax.set_aspect("equal")

    ax.set_xlim(100, 200)
    ax.set_ylim(100, 200)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    ax.set_title(
        f"High-density Network with Obstacle\n"
        f"N={len(points)}, E={len(edges)}"
    )

    ax.grid(
        True,
        linestyle=":",
        alpha=0.5
    )

    fig.tight_layout()

    fig.savefig(
        f"{save_dir}/{prefix}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Saved: {save_dir}/{prefix}.png"
    )


# =========================================================
# メイン
# =========================================================
if __name__ == "__main__":

    # -----------------------------
    # ネットワーク条件
    # -----------------------------
    L = 300.0
    h = 2.7

    center = (
        L / 2.0,
        L / 2.0
    )

    radius = 45.0

    # -----------------------------
    # 障害物
    # -----------------------------
    obstacle_width = 12.0
    obstacle_height = 45.0

    cx, cy = center

    xmin = cx - obstacle_width / 2
    xmax = cx + obstacle_width / 2

    ymin = cy - obstacle_height / 2
    ymax = cy + obstacle_height / 2

    obstacle = (
        xmin,
        xmax,
        ymin,
        ymax
    )

    print("障害物:")
    print(
        f"x = {xmin} ～ {xmax}"
    )
    print(
        f"y = {ymin} ～ {ymax}"
    )

    # -----------------------------
    # ① 元ネットワーク生成
    # -----------------------------
    points, triangles = generate_jittered_grid_mesh(
        L=L,
        h=h,
        jitter_ratio=0.4,
        seed=123
    )

    # -----------------------------
    # ② 円形領域を切り出す
    # -----------------------------
    points_sub, triangles_sub = extract_circular_submesh(
        points,
        triangles,
        center,
        radius
    )

    print(
        "円形領域のノード数:",
        len(points_sub)
    )

    print(
        "円形領域の三角形数:",
        len(triangles_sub)
    )

    # -----------------------------
    # ③ 障害物内のノードを除外
    # -----------------------------
    x = points_sub[:, 0]
    y = points_sub[:, 1]

    inside_obstacle = (
        (x >= xmin) &
        (x <= xmax) &
        (y >= ymin) &
        (y <= ymax)
    )

    print(
        "障害物内のノード数:",
        np.sum(inside_obstacle)
    )

    keep_node = ~inside_obstacle

    old_to_new = -np.ones(
        len(points_sub),
        dtype=int
    )

    inside_idx = np.where(keep_node)[0]

    old_to_new[inside_idx] = np.arange(
        len(inside_idx)
    )

    points_obstacle = points_sub[
        inside_idx
    ]

    tri_mask = keep_node[
        triangles_sub
    ].all(axis=1)

    triangles_obstacle = (
        old_to_new[
            triangles_sub[tri_mask]
        ]
    )

    # -----------------------------
    # ④ 障害物を横切る三角形を削除
    # -----------------------------
    triangles_obstacle = remove_obstacle_triangles(
        points_obstacle,
        triangles_obstacle,
        obstacle
    )

    # -----------------------------
    # ⑤ グラフ化
    # -----------------------------
    G, pos = build_graph_from_triangles(
        points_obstacle,
        triangles_obstacle
    )

    print()
    print("障害物処理後")
    print(
        "ノード数:",
        G.number_of_nodes()
    )
    print(
        "エッジ数:",
        G.number_of_edges()
    )
    print(
        "連結成分数:",
        nx.number_connected_components(G)
    )

    # -----------------------------
    # ⑥ 孤立ノードを削除
    # -----------------------------
    points_final, triangles_final = remove_isolated_nodes(
        points_obstacle,
        triangles_obstacle,
        G
    )

    # -----------------------------
    # ⑦ 最終ネットワーク確認
    # -----------------------------
    G_final, pos_final = build_graph_from_triangles(
        points_final,
        triangles_final
    )

    print()
    print("========== 最終確認 ==========")
    print(
        "ノード数:",
        G_final.number_of_nodes()
    )
    print(
        "エッジ数:",
        G_final.number_of_edges()
    )
    print(
        "連結成分数:",
        nx.number_connected_components(G_final)
    )

    # 連結成分が1つでなければ停止
    if nx.number_connected_components(G_final) != 1:
        raise RuntimeError(
            "ネットワークが1つにつながっていません。"
            "障害物サイズやメッシュ条件を確認してください。"
        )

    # -----------------------------
    # ⑧ 保存
    # -----------------------------
    prefix = (
        f"circle_r{radius}_L{L}_h{h}_obstacle"
    )

    export_network(
        points_final,
        triangles_final,
        prefix,
        obstacle,
        center,
        radius
    )