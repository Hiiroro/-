import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import os


# ============================================================
# 1. ジッター付き格子を作る
# ============================================================
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

    points = np.column_stack([
        X.ravel(),
        Y.ravel()
    ])

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

    points_jittered[~is_boundary] += (
        noise[~is_boundary]
    )

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


# ============================================================
# 2. 円形領域 + 障害物を作る
# ============================================================
def extract_obstacle_submesh(
    points,
    triangles,
    center,
    radius,
    obstacle_xmin,
    obstacle_xmax,
    obstacle_ymin,
    obstacle_ymax
):

    cx, cy = center

    # --------------------------------------------------------
    # 円の中にあるノード
    # --------------------------------------------------------
    distance2 = (
        (points[:, 0] - cx) ** 2 +
        (points[:, 1] - cy) ** 2
    )

    inside_circle = (
        distance2 <= radius ** 2
    )

    # --------------------------------------------------------
    # 障害物の中にあるノード
    # --------------------------------------------------------
    inside_obstacle = (
        (points[:, 0] >= obstacle_xmin) &
        (points[:, 0] <= obstacle_xmax) &
        (points[:, 1] >= obstacle_ymin) &
        (points[:, 1] <= obstacle_ymax)
    )

    # --------------------------------------------------------
    # 円の中 かつ 障害物の外
    # --------------------------------------------------------
    keep_node = (
        inside_circle &
        (~inside_obstacle)
    )

    inside_idx = np.where(keep_node)[0]

    # 元ID → 新ID
    old_to_new = -np.ones(
        len(points),
        dtype=int
    )

    old_to_new[inside_idx] = np.arange(
        len(inside_idx)
    )

    # --------------------------------------------------------
    # 三角形も障害物を跨がないようにする
    #
    # 三角形の3頂点が全部残っているものだけ使用
    # --------------------------------------------------------
    tri_mask = keep_node[triangles].all(axis=1)

    triangles_in = triangles[tri_mask]

    triangles_sub = old_to_new[
        triangles_in
    ]

    points_sub = points[
        inside_idx
    ]

    return points_sub, triangles_sub


# ============================================================
# 3. グラフを作る
# ============================================================
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


# ============================================================
# 4. CSV保存
# ============================================================
def save_network(
    points,
    triangles,
    prefix,
    save_dir
):

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # nodes
    # --------------------------------------------------------
    node_ids = np.arange(
        len(points)
    )

    nodes_out = np.column_stack([
        node_ids,
        points
    ])

    np.savetxt(
        os.path.join(
            save_dir,
            f"{prefix}_nodes.csv"
        ),
        nodes_out,
        delimiter=",",
        header="id,x,y",
        comments=""
    )

    # --------------------------------------------------------
    # edges
    # --------------------------------------------------------
    edges = set()

    for tri in triangles:

        i, j, k = map(
            int,
            tri
        )

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
        os.path.join(
            save_dir,
            f"{prefix}_edges.csv"
        ),
        edges,
        delimiter=",",
        header="src,dst",
        comments="",
        fmt="%d"
    )

    print(
        f"Saved: {prefix}"
    )

    print(
        f"  Nodes: {len(points)}"
    )

    print(
        f"  Edges: {len(edges)}"
    )


# ============================================================
# 5. ネットワークを描画
# ============================================================
def save_figure(
    points,
    triangles,
    center,
    radius,
    obstacle,
    save_path
):

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    ax.triplot(
        points[:, 0],
        points[:, 1],
        triangles,
        linewidth=0.4
    )

    # 円
    circle = plt.Circle(
        center,
        radius,
        fill=False,
        linestyle="--",
        linewidth=1.5
    )

    ax.add_patch(circle)

    # 障害物
    xmin, xmax, ymin, ymax = obstacle

    obstacle_rect = plt.Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        fill=True,
        alpha=0.8
    )

    ax.add_patch(
        obstacle_rect
    )

    ax.set_aspect(
        "equal"
    )

    ax.set_xlim(
        center[0] - radius - 5,
        center[0] + radius + 5
    )

    ax.set_ylim(
        center[1] - radius - 5,
        center[1] + radius + 5
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    ax.set_title(
        "High-density Network with Obstacle"
    )

    ax.grid(
        True,
        linestyle=":",
        alpha=0.5
    )

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# main
# ============================================================
if __name__ == "__main__":

    # --------------------------------------------------------
    # 基本パラメータ
    # --------------------------------------------------------
    L = 300.0
    h = 2.7

    center = (
        L / 2.0,
        L / 2.0
    )

    radius = 45.0

    # --------------------------------------------------------
    # 障害物
    #
    # 中央に縦長の障害物
    # --------------------------------------------------------
    obstacle_width = 12.0
    obstacle_height = 45.0

    cx, cy = center

    obstacle_xmin = (
        cx - obstacle_width / 2
    )

    obstacle_xmax = (
        cx + obstacle_width / 2
    )

    obstacle_ymin = (
        cy - obstacle_height / 2
    )

    obstacle_ymax = (
        cy + obstacle_height / 2
    )

    obstacle = (
        obstacle_xmin,
        obstacle_xmax,
        obstacle_ymin,
        obstacle_ymax
    )

    print(
        "Obstacle:"
    )

    print(
        f"X = {obstacle_xmin:.2f}"
        f" ～ {obstacle_xmax:.2f}"
    )

    print(
        f"Y = {obstacle_ymin:.2f}"
        f" ～ {obstacle_ymax:.2f}"
    )

    # --------------------------------------------------------
    # 高密度メッシュ生成
    # --------------------------------------------------------
    points, triangles = (
        generate_jittered_grid_mesh(
            L=L,
            h=h,
            jitter_ratio=0.4,
            seed=123
        )
    )

    # --------------------------------------------------------
    # 円形領域 + 障害物
    # --------------------------------------------------------
    points_sub, triangles_sub = (
        extract_obstacle_submesh(
            points,
            triangles,
            center,
            radius,
            obstacle_xmin,
            obstacle_xmax,
            obstacle_ymin,
            obstacle_ymax
        )
    )

    print()
    print(
        "High-density network"
    )

    print(
        f"Nodes: {len(points_sub)}"
    )

    print(
        f"Triangles: {len(triangles_sub)}"
    )

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------
    save_dir = (
        "networks_for_simulation"
    )

    prefix = (
        f"circle_r{radius}_"
        f"L{L}_h{h}_obstacle"
    )

    save_network(
        points_sub,
        triangles_sub,
        prefix,
        save_dir
    )

    # --------------------------------------------------------
    # 図
    # --------------------------------------------------------
    save_figure(
        points_sub,
        triangles_sub,
        center,
        radius,
        obstacle,
        os.path.join(
            save_dir,
            f"{prefix}.png"
        )
    )

    # --------------------------------------------------------
    # グラフ確認
    # --------------------------------------------------------
    G, pos = (
        build_graph_from_triangles(
            points_sub,
            triangles_sub
        )
    )

    print(
        f"Graph nodes: "
        f"{G.number_of_nodes()}"
    )

    print(
        f"Graph edges: "
        f"{G.number_of_edges()}"
    )

    print()
    print("完了")
    
    # source, sink が連結しているか確認
    print()
    print("===== 連結性チェック =====")

    print(
        "連結成分数:",
        nx.number_connected_components(G)
    )

    print(
        "sourceとsinkが連結しているか:",
        nx.has_path(G, source, sink)
    )

    if nx.has_path(G, source, sink):
        path = nx.shortest_path(G, source, sink)

        print(
            "source-sink間の最短経路長:",
            len(path)
        )
    else:
        print(
            "sourceとsinkの間に経路がありません"
        )