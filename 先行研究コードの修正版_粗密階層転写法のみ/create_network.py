import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import os


def generate_jittered_grid_mesh(L=300.0, h=5.0, jitter_ratio=0.4, seed=0):
    rng = np.random.default_rng(seed)

    n = int(round(L / h))
    xs = np.linspace(0.0, L, n + 1)
    ys = np.linspace(0.0, L, n + 1)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack([X.ravel(), Y.ravel()])

    on_left   = np.isclose(points[:, 0], 0.0)
    on_right  = np.isclose(points[:, 0], L)
    on_bottom = np.isclose(points[:, 1], 0.0)
    on_top    = np.isclose(points[:, 1], L)
    is_boundary = on_left | on_right | on_bottom | on_top

    jitter = jitter_ratio * h
    noise = rng.uniform(-jitter, jitter, size=points.shape)

    points_jittered = points.copy()
    points_jittered[~is_boundary] += noise[~is_boundary]
    points_jittered[:, 0] = np.clip(points_jittered[:, 0], 0.0, L)
    points_jittered[:, 1] = np.clip(points_jittered[:, 1], 0.0, L)

    tri = Delaunay(points_jittered)
    triangles = tri.simplices

    return points_jittered, triangles


def extract_circular_submesh(points, triangles, center, radius):
    """
    center=(cx, cy), radius の円に含まれるノードだけからなる
    サブメッシュを返す（インデックスは 0 始まりで再番号付け済）。

    戻り値:
        points_sub    : (N_sub, 2)
        triangles_sub : (E_sub, 3)
    """
    cx, cy = center
    r2 = (points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2
    mask = r2 <= radius ** 2  # 円内にあるかどうか

    inside_idx = np.where(mask)[0]  # 円内ノードの元インデックス

    # 元 -> 新インデックスの対応表（円外は -1）
    old_to_new = -np.ones(len(points), dtype=int)
    old_to_new[inside_idx] = np.arange(len(inside_idx))

    # 要素ごとに「3頂点すべて円内」のものだけ残す
    tri_mask = mask[triangles].all(axis=1)
    triangles_in = triangles[tri_mask]

    # 新しいインデックスに書き換え
    triangles_sub = old_to_new[triangles_in]

    # サブメッシュの座標
    points_sub = points[inside_idx]

    return points_sub, triangles_sub


def build_graph_from_triangles(points, triangles):
    G = nx.Graph()
    pos = {i: (float(p[0]), float(p[1])) for i, p in enumerate(points)}

    for i in range(len(points)):
        G.add_node(i)

    for tri in triangles:
        i, j, k = tri
        G.add_edge(i, j)
        G.add_edge(j, k)
        G.add_edge(k, i)

    return G, pos

def export_undirected_network_with_figure(points, triangles, prefix="network",
                                          h=None, save_png=True):
    """
    points   : (N, 2) ノード座標 (x, y)
    triangles: (E, 3) 三角形要素
    prefix   : 出力ファイル名の接頭辞
    h        : メッシュサイズ (ファイル名に含めたい値)
    """
    N = points.shape[0]

    # --- h をファイル名に追加 ---
    if h is not None:
        fname_prefix = f"{prefix}_h{h}"
    else:
        fname_prefix = prefix

    # --- ノード一覧 CSV ---
    node_ids = np.arange(N)
    nodes_out = np.column_stack([node_ids, points])
    
    save_dir = "networks_for_simulation"
    os.makedirs(save_dir, exist_ok=True)

    np.savetxt(
        f"{save_dir}/{fname_prefix}_nodes.csv",
        nodes_out,
        delimiter=",",
        header="id,x,y",
        comments=""
    )

    # --- 無向エッジ集合 ---
    edges = set()
    for tri in triangles:
        i, j, k = map(int, tri)
        edges.add(tuple(sorted((i, j))))
        edges.add(tuple(sorted((j, k))))
        edges.add(tuple(sorted((k, i))))

    edges = np.array(sorted(edges), dtype=int)
    E = len(edges)

    np.savetxt(
        f"{save_dir}/{fname_prefix}_edges.csv",
        edges,
        delimiter=",",
        header="src,dst",
        comments="",
        fmt="%d"
    )

    print(f"Saved: {fname_prefix}_nodes.csv, {fname_prefix}_edges.csv")

    # --- PNG 画像 ---
    if save_png:
        fig, ax = plt.subplots(figsize=(6, 6))
        print(np.min(points[:, 1]))
        ax.triplot(points[:, 0], points[:, 1], triangles, linewidth=0.4)
        ax.set_aspect("equal")
        ax.set_title(f"{prefix} mesh (N={N}, E={E}, h={h})")
        ax.set_xlim(100, 200)
        ax.set_ylim(100, 200)
        fig.tight_layout()
        fig.savefig(f"{save_dir}/{fname_prefix}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {fname_prefix}.png")


if __name__ == "__main__":
    L = 300.0
    h = 2.7

    # 全体メッシュ（ランダム寄りにしたいなら jitter_ratio=0.3〜0.5 程度）
    points, triangles = generate_jittered_grid_mesh(L=L, h=h, jitter_ratio=0.4, seed=123)

    # 中心から半径50の円（ここでは [0,L]×[0,L] の中心を (L/2, L/2) とする）
    center = (L / 2.0, L / 2.0)
    radius = 45.0

    points_sub, triangles_sub = extract_circular_submesh(points, triangles, center, radius)
    
    
    print("Submesh points:", points_sub.shape)
    print("Submesh triangles:", triangles_sub.shape)

    export_undirected_network_with_figure(
        points_sub,
        triangles_sub,
        prefix=f"circle_r{radius}_L{L}",
        h=h,
        save_png=True
    )

    # グラフ化して描画
    G_sub, pos_sub = build_graph_from_triangles(points_sub, triangles_sub)

    plt.figure(figsize=(6, 6))
    nx.draw(G_sub, pos_sub, node_size=10, edge_color="gray", with_labels=False)
    circle = plt.Circle(center, radius, fill=False, linestyle="--")
    plt.gca().add_patch(circle)
    plt.gca().set_aspect("equal")
    # plt.xlim(center[0] - radius - 5, center[0] + radius + 5)
    # plt.ylim(center[1] - radius - 5, center[1] + radius + 5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Nodes and elements inside circle (r=50)")
    plt.grid(True, linestyle=":", alpha=0.7)  # ← grid追加
    plt.tight_layout()
    plt.show()
