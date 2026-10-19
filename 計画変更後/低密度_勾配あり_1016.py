import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import Delaunay
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

# ============================================================
# 実験設定：傾斜（高コスト領域）を変更するときは基本的にここだけ編集
# ============================================================
EXPERIMENT_NAME = "gradient_low_density"  # 低密度ネットワーク）

DOMAIN_L = 300.0
CIRCLE_CENTER = (150.0, 150.0)
CIRCLE_RADIUS = 45.0
GRID_SPACING = 5.4
JITTER = 0.4
RANDOM_SEED = 456  # 高密度側と異なるノード配置にする

# 傾斜領域（回転した長方形）の設定
COST_CENTER = (180.0, 160.0)  # (x, y)
COST_WIDTH = 12.0
COST_HEIGHT = 60.0
COST_ANGLE_DEG = 60.0
HIGH_COST = 3.0
N_EDGE_SAMPLES = 21

# 出力先。ソルバー側がこの名前を読んでいるため、prefixは互換性を維持
OUTPUT_ROOT = "networks_for_simulation"
SAVE_PNG = True
# ============================================================


def generate_jittered_grid(L, h, jitter, seed):
    """正方格子を作り、内部点に小さなランダム変位を加える。"""
    rng = np.random.default_rng(seed)
    xs = np.arange(0.0, L + h * 0.5, h)
    ys = np.arange(0.0, L + h * 0.5, h)
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    # 領域の外周にある点は固定し、内部点だけ揺らす
    is_boundary = (
        np.isclose(points[:, 0], 0.0)
        | np.isclose(points[:, 0], L)
        | np.isclose(points[:, 1], 0.0)
        | np.isclose(points[:, 1], L)
    )
    offsets = rng.uniform(-jitter * h, jitter * h, size=points.shape)
    points[~is_boundary] += offsets[~is_boundary]
    return points


def make_circular_mesh(points, circle_center, circle_radius):
    """円内の点を抽出してDelaunay三角形分割を行う。"""
    center = np.asarray(circle_center, dtype=float)
    distances = np.linalg.norm(points - center, axis=1)
    inside_points = points[distances <= circle_radius]

    if len(inside_points) < 3:
        raise ValueError("円内の点が3点未満です。GRID_SPACINGやCIRCLE_RADIUSを確認してください。")

    triangulation = Delaunay(inside_points)
    return inside_points, triangulation.simplices


def triangles_to_edges(triangles):
    """三角形リストから重複を除いた無向エッジを作る。"""
    edge_set = set()
    for a, b, c in triangles:
        for u, v in ((a, b), (b, c), (c, a)):
            edge_set.add(tuple(sorted((int(u), int(v)))))
    return np.asarray(sorted(edge_set), dtype=int)


def points_in_rotated_rectangle(points, center, width, height, angle_deg):
    """座標群が回転長方形の内側にあるか判定する。"""
    pts = np.asarray(points, dtype=float)
    center = np.asarray(center, dtype=float)
    theta = np.deg2rad(angle_deg)
    dx = pts[:, 0] - center[0]
    dy = pts[:, 1] - center[1]

    # 長方形を逆回転して、軸に平行な長方形として判定
    local_x = dx * np.cos(theta) + dy * np.sin(theta)
    local_y = -dx * np.sin(theta) + dy * np.cos(theta)
    return (np.abs(local_x) <= width / 2.0) & (np.abs(local_y) <= height / 2.0)


def calculate_edge_costs(
    points,
    edges,
    center=(153.0, 150.0),
    width=12.0,
    height=45.0,
    angle_deg=30.0,
    high_cost=3.0,
    n_samples=21,
):
    """各エッジに沿ってサンプル点を取り、高コスト領域内の割合から平均コストを計算。"""
    points = np.asarray(points, dtype=float)
    edges = np.asarray(edges, dtype=int)
    if n_samples < 2:
        raise ValueError("n_samplesは2以上にしてください。")
    if high_cost < 1.0:
        raise ValueError("high_costは1.0以上を指定してください。")

    fractions = np.linspace(0.0, 1.0, n_samples)
    costs = np.ones(len(edges), dtype=float)

    for idx, (u, v) in enumerate(edges):
        p0 = points[u]
        p1 = points[v]
        sample_points = p0[None, :] + fractions[:, None] * (p1 - p0)[None, :]
        inside = points_in_rotated_rectangle(
            sample_points, center=center, width=width, height=height, angle_deg=angle_deg
        )
        # エッジ上のサンプル点のうち領域内にある割合に応じた平均コスト
        costs[idx] = 1.0 + (high_cost - 1.0) * np.mean(inside)

    return costs


def add_rotated_rectangle(ax, center, width, height, angle_deg, **kwargs):
    """図に回転長方形を描画する。"""
    cx, cy = center
    patch = Rectangle(
        (cx - width / 2.0, cy - height / 2.0),
        width,
        height,
        angle=0,
        **kwargs,
    )
    transform = Affine2D().rotate_deg_around(cx, cy, angle_deg) + ax.transData
    patch.set_transform(transform)
    ax.add_patch(patch)
    return patch


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
    circle_radius=45.0,
):
    """ネットワーク、エッジコスト、図をCSV/PNGに出力する。"""
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=int)
    edges = triangles_to_edges(triangles)

    edge_costs = calculate_edge_costs(
        points,
        edges,
        center=cost_center,
        width=cost_width,
        height=cost_height,
        angle_deg=angle_deg,
        high_cost=high_cost,
        n_samples=N_EDGE_SAMPLES,
    )

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    nodes_path = os.path.join(OUTPUT_ROOT, f"{prefix}_nodes.csv")
    edges_path = os.path.join(OUTPUT_ROOT, f"{prefix}_edges.csv")
    costs_path = os.path.join(OUTPUT_ROOT, f"{prefix}_edge_costs.csv")
    png_path = os.path.join(OUTPUT_ROOT, f"{prefix}.png")

    node_df = pd.DataFrame({
        "node": np.arange(len(points), dtype=int),
        "x": points[:, 0],
        "y": points[:, 1],
    })
    # ソルバーのload_network()が期待するCSV形式に合わせる
    edge_df = pd.DataFrame({
        "u": edges[:, 0],
        "v": edges[:, 1],
    })
    cost_df = pd.DataFrame({
        "u": edges[:, 0],
        "v": edges[:, 1],
        "cost": edge_costs,
    })

    node_df.to_csv(nodes_path, index=False)
    edge_df.to_csv(edges_path, index=False)
    cost_df.to_csv(costs_path, index=False)

    graph = nx.Graph()
    graph.add_nodes_from(range(len(points)))
    graph.add_edges_from(edges.tolist())
    components = nx.number_connected_components(graph)

    if save_png:
        fig, ax = plt.subplots(figsize=(9, 9))
        # 通常領域のネットワークは薄い灰色
        for u, v in edges:
            ax.plot(
                [points[u, 0], points[v, 0]],
                [points[u, 1], points[v, 1]],
                color="#c9c9c9",
                linewidth=0.55,
                zorder=1,
            )

        add_rotated_rectangle(
            ax,
            cost_center,
            cost_width,
            cost_height,
            angle_deg,
            facecolor="#f4c542",
            edgecolor="#c28b00",
            alpha=0.30,
            linewidth=1.5,
            zorder=2,
        )
        ax.scatter(points[:, 0], points[:, 1], s=5, color="#555555", zorder=3)
        circle = plt.Circle(circle_center, circle_radius, fill=False, color="black", linewidth=1.2)
        ax.add_patch(circle)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(circle_center[0] - circle_radius - 5, circle_center[0] + circle_radius + 5)
        ax.set_ylim(circle_center[1] - circle_radius - 5, circle_center[1] + circle_radius + 5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        title_h = f", h={h}" if h is not None else ""
        ax.set_title(
            f"Gradient network: {EXPERIMENT_NAME}{title_h}\n"
            f"center={cost_center}, size=({cost_width}, {cost_height}), "
            f"angle={angle_deg}°, high_cost={high_cost}"
        )
        ax.grid(False)
        fig.tight_layout()
        fig.savefig(png_path, dpi=200)
        plt.close(fig)

    print("\n===== Network generation complete =====")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Nodes: {len(points)}")
    print(f"Triangles: {len(triangles)}")
    print(f"Edges: {len(edges)}")
    print(f"Connected components: {components}")
    print(f"Saved nodes: {nodes_path}")
    print(f"Saved edges: {edges_path}")
    print(f"Saved costs: {costs_path}")
    if save_png:
        print(f"Saved image: {png_path}")
    return nodes_path, edges_path, costs_path, png_path


def main():
    # 低密度側のprefix：circle_r45.0_L300.0_gradient_h5.4
    prefix = f"circle_r{CIRCLE_RADIUS}_L{DOMAIN_L}_gradient_h{GRID_SPACING}"

    all_points = generate_jittered_grid(
        L=DOMAIN_L,
        h=GRID_SPACING,
        jitter=JITTER,
        seed=RANDOM_SEED,
    )
    points, triangles = make_circular_mesh(
        all_points,
        circle_center=CIRCLE_CENTER,
        circle_radius=CIRCLE_RADIUS,
    )

    export_network_with_gradient(
        points,
        triangles,
        prefix=prefix,
        h=GRID_SPACING,
        save_png=SAVE_PNG,
        cost_center=COST_CENTER,
        cost_width=COST_WIDTH,
        cost_height=COST_HEIGHT,
        angle_deg=COST_ANGLE_DEG,
        high_cost=HIGH_COST,
        circle_center=CIRCLE_CENTER,
        circle_radius=CIRCLE_RADIUS,
    )


if __name__ == "__main__":
    main()
