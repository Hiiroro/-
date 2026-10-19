import os
import argparse

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import spsolve


# ============================================================
# 1. source / sink を選ぶ
# ============================================================
def choose_source_sink_on_diameter(pos):
    """ネットワークの左右端に最も近いノードを source / sink にする。"""
    node_ids = list(pos.keys())
    coords = np.array([pos[i] for i in node_ids], dtype=float)

    cx, cy = coords.mean(axis=0)
    radius = np.sqrt(
        (coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2
    ).max()

    target_source = np.array([cx - radius, cy])
    target_sink = np.array([cx + radius, cy])

    source_idx = int(np.argmin(np.sum((coords - target_source) ** 2, axis=1)))
    sink_idx = int(np.argmin(np.sum((coords - target_sink) ** 2, axis=1)))

    source = node_ids[source_idx]
    sink = node_ids[sink_idx]

    if source == sink:
        raise ValueError("source と sink が同じノードになっています。")

    return source, sink, (float(cx), float(cy)), float(radius)


# ============================================================
# 2. ネットワークとエッジコストを読み込む
# ============================================================
def load_network(nodes_csv, edges_csv, costs_csv):
    nodes_arr = np.loadtxt(
        nodes_csv, delimiter=",", skiprows=1, ndmin=2
    )
    edges_arr = np.loadtxt(
        edges_csv, delimiter=",", skiprows=1, dtype=int, ndmin=2
    )
    costs_arr = np.loadtxt(
        costs_csv, delimiter=",", skiprows=1, ndmin=2
    )

    node_ids = nodes_arr[:, 0].astype(int)
    ids = sorted(node_ids.tolist())
    if len(ids) != len(set(ids)):
        raise ValueError("nodes.csv に重複したノードIDがあります。")

    id_to_idx = {node_id: i for i, node_id in enumerate(ids)}
    pos = {
        int(row[0]): (float(row[1]), float(row[2]))
        for row in nodes_arr
    }

    # edge_costs.csv の列は「src,dst,cost」を想定
    cost_map = {}
    for row in costs_arr:
        u, v, cost = int(row[0]), int(row[1]), float(row[2])
        cost_map[tuple(sorted((u, v)))] = cost

    edges = []
    lengths = []
    costs = []

    for row in edges_arr:
        u, v = int(row[0]), int(row[1])
        if u not in pos or v not in pos:
            raise ValueError(f"edges.csv のエッジ ({u}, {v}) に未定義ノードがあります。")

        key = tuple(sorted((u, v)))
        if key not in cost_map:
            raise ValueError(
                f"エッジ {key} のコストが edge_costs.csv にありません。"
            )

        length = float(np.linalg.norm(
            np.asarray(pos[u], dtype=float) - np.asarray(pos[v], dtype=float)
        ))
        cost = cost_map[key]

        if length <= 0 or not np.isfinite(length):
            raise ValueError(f"エッジ {key} の長さが不正です: {length}")
        if cost <= 0 or not np.isfinite(cost):
            raise ValueError(f"エッジ {key} のコストが不正です: {cost}")

        edges.append((u, v))
        lengths.append(length)
        costs.append(cost)

    G = nx.Graph()
    G.add_nodes_from(ids)
    for (u, v), length, cost in zip(edges, lengths, costs):
        G.add_edge(u, v, length=length, cost=cost)

    if not nx.is_connected(G):
        raise ValueError(
            "ネットワークが連結ではありません。"
            f"連結成分数: {nx.number_connected_components(G)}"
        )

    edge_u = np.array([id_to_idx[u] for u, v in edges], dtype=int)
    edge_v = np.array([id_to_idx[v] for u, v in edges], dtype=int)

    return (
        G, ids, pos, edges,
        np.asarray(lengths, dtype=float),
        np.asarray(costs, dtype=float),
        edge_u, edge_v
    )


# ============================================================
# 3. 圧力・流量を計算する
# ============================================================
def calculate_pressure_flow(
    D, lengths, costs, edge_u, edge_v, flux, eps
):
    """
    実効コンダクタンス g = D / (length * cost)
    流量 Q = g * (p_u - p_v)
    圧力はグラフのラプラシアンから求める。
    """
    n = len(flux)
    conductance = D / (lengths * costs)

    if not np.all(np.isfinite(conductance)):
        raise FloatingPointError("コンダクタンスが有限値ではありません。")
    if np.any(conductance < 0):
        raise FloatingPointError("負のコンダクタンスが発生しました。")

    # 疎行列でラプラシアン A を構築（密行列よりメモリ・計算量を抑える）
    diagonal_idx = np.concatenate([edge_u, edge_v, edge_u, edge_v])
    diagonal_jdx = np.concatenate([edge_u, edge_v, edge_v, edge_u])
    values = np.concatenate([
        conductance, conductance, -conductance, -conductance
    ])

    A = coo_matrix(
        (values, (diagonal_idx, diagonal_jdx)), shape=(n, n)
    ).tocsr()
    A = A + eps * eye(n, format="csr")

    pressure = spsolve(A, flux)
    flow = conductance * (pressure[edge_u] - pressure[edge_v])

    if not np.all(np.isfinite(pressure)) or not np.all(np.isfinite(flow)):
        raise FloatingPointError("圧力または流量に NaN / inf が発生しました。")

    return pressure, flow


# ============================================================
# 4. 導電率 D の時間変化と RK4
# ============================================================
def calculate_dD(D, flow, gamma):
    """元コードのモデル: dD/dt = |Q|^gamma - D"""
    return np.abs(flow) ** gamma - D


def rk4_step(
    D, dt, lengths, costs, edge_u, edge_v,
    flux, eps, gamma, D_min=1e-12
):
    """導電率 D の時間発展を RK4 で1ステップ進める。"""

    def rhs(D_stage):
        D_stage = np.maximum(D_stage, D_min)
        p_stage, q_stage = calculate_pressure_flow(
            D_stage, lengths, costs, edge_u, edge_v, flux, eps
        )
        return calculate_dD(D_stage, q_stage, gamma), p_stage, q_stage

    k1, _, _ = rhs(D)
    k2, _, _ = rhs(D + 0.5 * dt * k1)
    k3, _, _ = rhs(D + 0.5 * dt * k2)
    k4, _, _ = rhs(D + dt * k3)

    D_next = D + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    D_next = np.maximum(D_next, D_min)

    pressure_next, flow_next = calculate_pressure_flow(
        D_next, lengths, costs, edge_u, edge_v, flux, eps
    )
    return D_next, pressure_next, flow_next


# ============================================================
# 5. ネットワークを描画する
# ============================================================
def save_network_plot(
    G, pos, edges, flow, source, sink, center, radius,
    cost_center, cost_width, cost_height, cost_angle,
    output_path, step, title_prefix
):
    fig, ax = plt.subplots(figsize=(9, 9))

    # 生成コードと同じ高コスト領域を表示（回転角も一致させる）
    cx, cy = cost_center
    rect = Rectangle(
        (cx - cost_width / 2, cy - cost_height / 2),
        cost_width, cost_height,
        facecolor="red", edgecolor="red", alpha=0.20,
        zorder=0, label="High-cost region",
        transform=Affine2D().rotate_deg_around(cx, cy, cost_angle) + ax.transData
    )
    ax.add_patch(rect)

    # まず元のネットワーク全体を薄いグレーで描く
    # これにより、流量が小さいエッジもネットワーク構造として確認できる
    for u, v in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot(
            [x1, x2], [y1, y2],
            color="#d9d9d9", linewidth=0.55,
            alpha=0.8, zorder=1
        )

    # その上に流量を色で重ねる
    abs_flow = np.abs(flow)
    max_flow = max(float(abs_flow.max()), 1e-12)
    norm = mcolors.Normalize(vmin=0, vmax=max_flow)

    # 小さい流量が暗い紫に沈まないよう、明るい色から始まるカラーマップ
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "flow_cmap",
        ["#d8c9f0", "#9675c9", "#5b3f91", "#f4c542"]
    )
    widths = 0.45 + 2.8 * abs_flow / max_flow

    for k, (u, v) in enumerate(edges):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot(
            [x1, x2], [y1, y2],
            color=cmap(norm(abs_flow[k])),
            linewidth=widths[k], alpha=0.98, zorder=2
        )

    nx.draw_networkx_nodes(
        G, pos, nodelist=list(G.nodes()),
        node_size=5, node_color="black", ax=ax
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=[source], node_size=100,
        node_color="blue", edgecolors="black", ax=ax
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=[sink], node_size=100,
        node_color="orange", edgecolors="black", ax=ax
    )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Absolute flow |Q|")

    circle = plt.Circle(
        center, radius, fill=False, linestyle="--",
        color="black", linewidth=1
    )
    ax.add_patch(circle)

    ax.set_aspect("equal")
    ax.set_xlim(center[0] - radius - 5, center[0] + radius + 5)
    ax.set_ylim(center[1] - radius - 5, center[1] + radius + 5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"{title_prefix} | step={step}")
    ax.grid(True, linestyle=":", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ============================================================
# 6. メイン処理
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="高コスト領域を考慮するフィザルムソルバー"
    )
    parser.add_argument("--h", type=float, default=2.7,
                        help="ネットワーク生成時の h")
    parser.add_argument("--Q0", type=float, default=10.0,
                        help="source / sink に設定する流量")
    parser.add_argument("--steps", type=int, default=3000,
                        help="時間発展のステップ数")
    parser.add_argument("--dt", type=float, default=0.1,
                        help="時間刻み幅")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="導電率更新式の指数 gamma")
    parser.add_argument("--eps", type=float, default=1e-8,
                        help="圧力行列の正則化係数")
    parser.add_argument("--save_every", type=int, default=100,
                        help="何ステップごとに画像を保存するか")
    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError("--steps は1以上にしてください。")
    if args.dt <= 0 or args.eps <= 0 or args.save_every < 1:
        raise ValueError("--dt、--eps、--save_every の値を確認してください。")

    # ネットワーク生成コードと同じ設定
    L_domain = 300.0
    radius = 45.0
    network_prefix = f"circle_r{radius}_L{L_domain}_gradient_h{args.h}"
    data_dir = "networks_for_simulation"

    nodes_csv = os.path.join(data_dir, f"{network_prefix}_nodes.csv")
    edges_csv = os.path.join(data_dir, f"{network_prefix}_edges.csv")
    costs_csv = os.path.join(data_dir, f"{network_prefix}_edge_costs.csv")

    for path in (nodes_csv, edges_csv, costs_csv):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"ファイルがありません: {path}\n"
                "ネットワーク生成コードを先に実行してください。"
            )

    (
        G, ids, pos, edges, lengths, costs,
        edge_u, edge_v
    ) = load_network(nodes_csv, edges_csv, costs_csv)

    id_to_idx = {node_id: i for i, node_id in enumerate(ids)}
    source, sink, center, inferred_radius = choose_source_sink_on_diameter(pos)

    if not nx.has_path(G, source, sink):
        raise ValueError("source と sink の間に経路がありません。")

    source_idx = id_to_idx[source]
    sink_idx = id_to_idx[sink]
    n_nodes = len(ids)
    n_edges = len(edges)

    print("=" * 60)
    print("フィザルムソルバー：高コスト領域あり")
    print("ネットワーク:", network_prefix)
    print("ノード数:", n_nodes)
    print("エッジ数:", n_edges)
    print("連結成分数:", nx.number_connected_components(G))
    print("source:", source, pos[source])
    print("sink:", sink, pos[sink])
    print("平均エッジ長:", float(lengths.mean()))
    print("コストの最小値・最大値:", float(costs.min()), float(costs.max()))
    print("ステップ数:", args.steps, " dt:", args.dt)
    print("=" * 60)

    # source から Q0 を注入し、sink で同量を吸収
    flux = np.zeros(n_nodes, dtype=float)
    flux[source_idx] = args.Q0
    flux[sink_idx] = -args.Q0

    # 元コードに近い初期条件
    D = 1.0 / lengths

    pressure, flow = calculate_pressure_flow(
        D, lengths, costs, edge_u, edge_v, flux, args.eps
    )

    output_dir = os.path.join(
        "result_gradient_networks",
        f"{network_prefix}_Q{args.Q0}_gamma{args.gamma}_steps{args.steps}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # ネットワーク生成コードの設定と一致させる
    cost_center = (180.0, 160.0)
    cost_width = 12.0
    cost_height = 60.0
    cost_angle = 60.0

    save_network_plot(
        G, pos, edges, flow, source, sink, center, inferred_radius,
        cost_center, cost_width, cost_height, cost_angle,
        os.path.join(output_dir, "step_0000.png"), 0, network_prefix
    )

    history = [[
        0,
        float(np.mean(np.abs(flow))),
        float(np.max(np.abs(flow))),
        float(pressure[source_idx] - pressure[sink_idx]),
        float(np.mean(D)),
    ]]

    for step in range(1, args.steps + 1):
        D, pressure, flow = rk4_step(
            D, args.dt, lengths, costs, edge_u, edge_v,
            flux, args.eps, args.gamma
        )

        history.append([
            step,
            float(np.mean(np.abs(flow))),
            float(np.max(np.abs(flow))),
            float(pressure[source_idx] - pressure[sink_idx]),
            float(np.mean(D)),
        ])

        if step % args.save_every == 0 or step == args.steps:
            print(
                f"step={step:5d} | "
                f"mean|Q|={np.mean(np.abs(flow)):.6g} | "
                f"max|Q|={np.max(np.abs(flow)):.6g} | "
                f"mean D={np.mean(D):.6g} | "
                f"p_source-p_sink={pressure[source_idx] - pressure[sink_idx]:.6g}"
            )
            save_network_plot(
                G, pos, edges, flow, source, sink, center, inferred_radius,
                cost_center, cost_width, cost_height, cost_angle,
                os.path.join(output_dir, f"step_{step:05d}.png"),
                step, network_prefix
            )

    # エッジごとの最終結果
    edge_results = np.column_stack([
        np.asarray(edges, dtype=int),
        lengths, costs, D, flow, np.abs(flow)
    ])
    np.savetxt(
        os.path.join(output_dir, "final_edge_results.csv"),
        edge_results, delimiter=",",
        header="src,dst,length,cost,conductivity_D,flow_signed,flow_abs",
        comments="",
        fmt=["%d", "%d", "%.8f", "%.8f", "%.12e", "%.12e", "%.12e"]
    )

    # ノードごとの最終圧力
    node_results = np.column_stack([
        np.asarray(ids, dtype=int),
        np.array([pos[i][0] for i in ids]),
        np.array([pos[i][1] for i in ids]),
        pressure
    ])
    np.savetxt(
        os.path.join(output_dir, "final_node_results.csv"),
        node_results, delimiter=",",
        header="id,x,y,pressure",
        comments="",
        fmt=["%d", "%.8f", "%.8f", "%.12e"]
    )

    np.savetxt(
        os.path.join(output_dir, "history.csv"),
        np.asarray(history, dtype=float), delimiter=",",
        header="step,mean_abs_flow,max_abs_flow,source_sink_pressure_difference,mean_D",
        comments="",
        fmt=["%d", "%.12e", "%.12e", "%.12e", "%.12e"]
    )

    print("\n計算完了")
    print("出力先:", output_dir)
    print("保存ファイル: final_edge_results.csv, final_node_results.csv, history.csv, step_*.png")


if __name__ == "__main__":
    main()
