
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from scipy.spatial import Delaunay
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D


# ============================================================
# 1. ジッター付き格子メッシュを生成
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
    points = np.column_stack([X.ravel(), Y.ravel()])

    # 外周の点は固定し、内部の点だけをランダムにずらす
    on_boundary = (
        np.isclose(points[:, 0], 0.0)
        | np.isclose(points[:, 0], L)
        | np.isclose(points[:, 1], 0.0)
        | np.isclose(points[:, 1], L)
    )

    noise = rng.uniform(
        -jitter_ratio * h,
        jitter_ratio * h,
        size=points.shape
    )

    points_jittered = points.copy()
    points_jittered[~on_boundary] += noise[~on_boundary]

    points_jittered[:, 0] = np.clip(
        points_jittered[:, 0], 0.0, L
    )
    points_jittered[:, 1] = np.clip(
        points_jittered[:, 1], 0.0, L
    )

    triangles = Delaunay(points_jittered).simplices

    return points_jittered, triangles


# ============================================================
# 2. 円の内部にあるノード・三角形を抽出
# ============================================================
def extract_circular_submesh(
    points,
    triangles,
    center,
    radius
):
    cx, cy = center

    distance_squared = (
        (points[:, 0] - cx) ** 2
        + (points[:, 1] - cy) ** 2
    )

    node_mask = distance_squared <= radius ** 2

    inside_indices = np.where(node_mask)[0]

    old_to_new = -np.ones(len(points), dtype=int)
    old_to_new[inside_indices] = np.arange(
        len(inside_indices)
    )

    # 3頂点すべてが円内にある三角形だけを残す
    triangle_mask = node_mask[triangles].all(axis=1)

    triangles_inside = triangles[triangle_mask]
    triangles_sub = old_to_new[triangles_inside]

    points_sub = points[inside_indices]

    return points_sub, triangles_sub


# ============================================================
# 3. 三角形からエッジ一覧を作る
# ============================================================
def extract_edges_from_triangles(triangles):
    edges = set()

    for tri in triangles:
        i, j, k = map(int, tri)

        edges.add(tuple(sorted((i, j))))
        edges.add(tuple(sorted((j, k))))
        edges.add(tuple(sorted((k, i))))

    return np.array(sorted(edges), dtype=int)


# ============================================================
# 4. 回転した長方形の内部にあるエッジを判定し、コストを計算
# ============================================================
def calculate_edge_costs(
    points,
    edges,
    center=(153.0, 150.0),
    width=12.0,
    height=45.0,
    angle_deg=30.0,
    high_cost=3.0,
    n_samples=21
):
    if width <= 0 or height <= 0:
        raise ValueError("width と height は正の値にしてください。")

    if high_cost < 1.0:
        raise ValueError("high_cost は1.0以上にしてください。")

    if n_samples < 2:
        raise ValueError("n_samples は2以上にしてください。")

    cx, cy = center
    theta = np.radians(angle_deg)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    t = np.linspace(0.0, 1.0, n_samples)
    results = []

    for src, dst in edges:
        p1 = points[src]
        p2 = points[dst]

        # エッジ上のサンプル点
        sample_points = (
            p1[None, :]
            + t[:, None] * (p2 - p1)[None, :]
        )

        # 長方形の中心を原点に移動
        dx = sample_points[:, 0] - cx
        dy = sample_points[:, 1] - cy

        # 回転した長方形のローカル座標へ変換
        local_x = (
            dx * cos_theta
            + dy * sin_theta
        )

        local_y = (
            -dx * sin_theta
            + dy * cos_theta
        )

        # ローカル座標で長方形の内部か判定
        inside = (
            (np.abs(local_x) <= width / 2.0)
            & (np.abs(local_y) <= height / 2.0)
        )

        # エッジ上のサンプル点のうち、領域内にある割合
        fraction = float(inside.mean())

        # 領域外は1、領域内はhigh_costとなるように設定
        cost = 1.0 + (high_cost - 1.0) * fraction

        results.append([
            int(src),
            int(dst),
            cost,
            fraction
        ])

    return np.asarray(results, dtype=float)


# ============================================================
# 5. 回転した長方形を描画
# ============================================================
def add_rotated_rectangle(
    ax,
    center,
    width,
    height,
    angle_deg,
    **kwargs
):
    cx, cy = center

    # Rectangle 自体は軸に平行な形で作り、
    # Affine2D によって中心の周りに回転する
    rect = Rectangle(
        (cx - width / 2.0, cy - height / 2.0),
        width,
        height,
        **kwargs
    )

    transform = (
        Affine2D()
        .rotate_deg_around(cx, cy, angle_deg)
        + ax.transData
    )

    rect.set_transform(transform)
    ax.add_patch(rect)

    return rect


# ============================================================
# 6. CSV とネットワーク画像を保存
# ============================================================
def export_network_with_gradient(
    points,
    triangles,
    prefix="network",
    h=None,
    save_png=True,
    cost_center=(153.0, 150.0),
    cost_width=12.0,
    cost_height=45.0,
    angle_deg=30.0,
    high_cost=3.0,
    circle_center=(150.0, 150.0),
    circle_radius=45.0
):
    save_dir = "networks_for_simulation"
    os.makedirs(save_dir, exist_ok=True)

    fname_prefix = (
        f"{prefix}_h{h}"
        if h is not None
        else prefix
    )

    # --------------------------------------------------------
    # ノード CSV
    # --------------------------------------------------------
    node_ids = np.arange(len(points), dtype=int)

    nodes_out = np.column_stack([
        node_ids,
        points
    ])

    nodes_path = os.path.join(
        save_dir,
        f"{fname_prefix}_nodes.csv"
    )

    np.savetxt(
        nodes_path,
        nodes_out,
        delimiter=",",
        header="id,x,y",
        comments="",
        fmt=["%d", "%.8f", "%.8f"]
    )

    # --------------------------------------------------------
    # エッジ CSV
    # --------------------------------------------------------
    edges = extract_edges_from_triangles(triangles)

    edges_path = os.path.join(
        save_dir,
        f"{fname_prefix}_edges.csv"
    )

    np.savetxt(
        edges_path,
        edges,
        delimiter=",",
        header="src,dst",
        comments="",
        fmt="%d"
    )

    # --------------------------------------------------------
    # エッジコスト CSV
    # --------------------------------------------------------
    edge_costs = calculate_edge_costs(
        points=points,
        edges=edges,
        center=cost_center,
        width=cost_width,
        height=cost_height,
        angle_deg=angle_deg,
        high_cost=high_cost
    )

    costs_path = os.path.join(
        save_dir,
        f"{fname_prefix}_edge_costs.csv"
    )

    np.savetxt(
        costs_path,
        edge_costs,
        delimiter=",",
        header="src,dst,cost,high_cost_fraction",
        comments="",
        fmt=["%d", "%d", "%.6f", "%.6f"]
    )

    # --------------------------------------------------------
    # グラフ構築と連結成分数の確認
    # --------------------------------------------------------
    G = nx.Graph()

    pos = {
        i: (float(point[0]), float(point[1]))
        for i, point in enumerate(points)
    }

    G.add_nodes_from(range(len(points)))
    G.add_edges_from(
        (int(u), int(v))
        for u, v in edges
    )

    component_count = nx.number_connected_components(G)

    # --------------------------------------------------------
    # 実行結果
    # --------------------------------------------------------
    print("\n===== ネットワーク生成完了 =====")
    print(f"ノード数: {G.number_of_nodes()}")
    print(f"エッジ数: {G.number_of_edges()}")
    print(f"連結成分数: {component_count}")

    print(f"高コスト領域の中心: {cost_center}")
    print(f"高コスト領域の幅: {cost_width}")
    print(f"高コスト領域の高さ: {cost_height}")
    print(f"高コスト領域の回転角: {angle_deg}度")
    print(f"通常領域のコスト: 1.0")
    print(f"高コスト領域のコスト: {high_cost}")

    print("\n保存先:")
    print(nodes_path)
    print(edges_path)
    print(costs_path)

    # --------------------------------------------------------
    # ネットワーク画像
    # --------------------------------------------------------
    if save_png:
        fig, ax = plt.subplots(figsize=(9, 9))

        # 高コスト領域を回転させて描画
        add_rotated_rectangle(
            ax,
            center=cost_center,
            width=cost_width,
            height=cost_height,
            angle_deg=angle_deg,
            facecolor="red",
            edgecolor="red",
            alpha=0.20,
            linewidth=1.5,
            zorder=1,
            label="High-cost region"
        )

        # 三角形メッシュ
        ax.triplot(
            points[:, 0],
            points[:, 1],
            triangles,
            color="gray",
            linewidth=0.4,
            zorder=2
        )

        # 円の境界
        circle = plt.Circle(
            circle_center,
            circle_radius,
            fill=False,
            linestyle="--",
            color="black",
            linewidth=1.2,
            zorder=3
        )
        ax.add_patch(circle)

        ax.set_aspect("equal")

        margin = circle_radius + 5
        ax.set_xlim(
            circle_center[0] - margin,
            circle_center[0] + margin
        )
        ax.set_ylim(
            circle_center[1] - margin,
            circle_center[1] + margin
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(
            f"{fname_prefix}\n"
            f"N={len(points)}, E={len(edges)}, "
            f"angle={angle_deg} deg"
        )

        ax.grid(True, linestyle=":", alpha=0.3)
        ax.legend()
        fig.tight_layout()

        png_path = os.path.join(
            save_dir,
            f"{fname_prefix}.png"
        )

        fig.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight"
        )
        plt.close(fig)

        print("ネットワーク画像:", png_path)

    return G, pos


# ============================================================
# 7. メイン処理
# ============================================================
if __name__ == "__main__":

    # --------------------------------------------------------
    # ネットワーク設定
    # --------------------------------------------------------
    L = 300.0
    h = 2.7
    jitter_ratio = 0.4
    seed = 123

    circle_center = (150.0, 150.0)
    circle_radius = 45.0

    # --------------------------------------------------------
    # 高コスト領域の設定
    # ここを変更すると位置・大きさ・角度を変えられる
    # --------------------------------------------------------
    cost_center = (180.0, 160.0)
    cost_width = 12.0
    cost_height = 60.0
    angle_deg = 60.0
    high_cost = 3.0

    # --------------------------------------------------------
    # メッシュ生成
    # --------------------------------------------------------
    points, triangles = generate_jittered_grid_mesh(
        L=L,
        h=h,
        jitter_ratio=jitter_ratio,
        seed=seed
    )

    # --------------------------------------------------------
    # 円形領域を抽出
    # --------------------------------------------------------
    points_sub, triangles_sub = extract_circular_submesh(
        points,
        triangles,
        center=circle_center,
        radius=circle_radius
    )

    print("Submesh points:", points_sub.shape)
    print("Submesh triangles:", triangles_sub.shape)

    # --------------------------------------------------------
    # ネットワークを保存
    # ソルバー側のファイル名と合わせる
    # --------------------------------------------------------
    prefix = f"circle_r{circle_radius}_L{L}_gradient"

    G, pos = export_network_with_gradient(
        points_sub,
        triangles_sub,
        prefix=prefix,
        h=h,
        save_png=True,
        cost_center=cost_center,
        cost_width=cost_width,
        cost_height=cost_height,
        angle_deg=angle_deg,
        high_cost=high_cost,
        circle_center=circle_center,
        circle_radius=circle_radius
    )

    # --------------------------------------------------------
    # グラフ表示
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 9))

    add_rotated_rectangle(
        ax,
        center=cost_center,
        width=cost_width,
        height=cost_height,
        angle_deg=angle_deg,
        facecolor="red",
        edgecolor="red",
        alpha=0.20,
        linewidth=1.5,
        zorder=1,
        label="High-cost region"
    )

    nx.draw_networkx_edges(
        G, pos,
        edge_color="gray",
        width=0.4,
        ax=ax
    )

    nx.draw_networkx_nodes(
        G, pos,
        node_size=5,
        node_color="black",
        ax=ax
    )

    circle = plt.Circle(
        circle_center,
        circle_radius,
        fill=False,
        linestyle="--",
        color="black",
        linewidth=1.2
    )
    ax.add_patch(circle)

    margin = circle_radius + 5
    ax.set_xlim(
        circle_center[0] - margin,
        circle_center[0] + margin
    )
    ax.set_ylim(
        circle_center[1] - margin,
        circle_center[1] + margin
    )

    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Circular network with rotated high-cost region")
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()
