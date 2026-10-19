import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from math import dist 
import matplotlib.colors as mcolors
import os 
import argparse
import pickle

# =============================================
# 関数: choose_source_sink_on_diameter
# source, sinkのノード位置を決定する
# =============================================
def choose_source_sink_on_diameter(pos):
    """
    円形メッシュを想定して、
      - ノード全体の重心を円の中心 (cx, cy) とみなす
      - そこからの最大距離を半径 R とみなす
      - 左側の直径端 (cx-R, cy) に最も近いノードを source（湧き出し；x<0 側）
      - 右側の直径端 (cx+R, cy) に最も近いノードを sink  （吸い込み；x>0 側）
    とする。
    """
    node_ids = list(pos.keys())
    coords   = np.array([pos[i] for i in node_ids])

    xs = coords[:, 0]
    ys = coords[:, 1]

    # 円の中心を「全ノードの重心」とみなす
    cx = xs.mean()
    cy = ys.mean()

    # 最大距離を半径とみなす
    r = np.sqrt((xs - cx)**2 + (ys - cy)**2).max()

    # 左右の理想的な直径の端点
    target_source = np.array([cx - r, cy])  # 左端
    target_sink   = np.array([cx + r, cy])  # 右端

    # 各ノードからターゲット点への距離（二乗）
    dL = np.sum((coords - target_source)**2, axis=1)
    dR = np.sum((coords - target_sink)**2, axis=1)

    source_idx = int(np.argmin(dL))
    sink_idx   = int(np.argmin(dR))

    source = node_ids[source_idx]
    sink   = node_ids[sink_idx]

    return source, sink, (cx, cy), r

# =============================================
# 関数: graph
# PS実行過程のグラフを生成する
# =============================================
def graph(G, i, Q):

    # 既存の G.edges() に対応する width を生成（flow が Q_i に基づく）
    Q_i = Q[i]
    weights = []
    for u, v in G.edges():
        weight = max(Q_i[u, v], Q_i[v, u])
        weights.append(weight)

    # グラフ描画
    fig, ax = plt.subplots(figsize=(17, 8))

    source_node_color = 'blue' # 湧き出し点
    sink_node_color   = 'red'  # 吸い込み点
    norm = mcolors.Normalize(vmin=0, vmax=I0)

    cmap = plt.cm.binary  # 使用するカラーマップ, plt.cm.coolwarmもある
    edge_colors = cmap(norm(weights))  # 重みに基づいて色を割り当てる
    nx.draw(G, pos, width=weights,node_size=5,node_color='black', with_labels= False, edge_color=edge_colors)
    nx.draw_networkx_nodes(G,pos,nodelist=[source],node_size=200,node_color=source_node_color, ax=ax,linewidths=1.0, edgecolors="black")
    nx.draw_networkx_nodes(G,pos,nodelist=[sink],node_size=200,node_color=sink_node_color, ax=ax,linewidths=1.0, edgecolors="black")

    fig.tight_layout()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # 空の配列でカラーバーを作成
    colorbar = fig.colorbar(sm, ax=ax)
    colorbar.ax.tick_params(labelsize=20) 
    plt.xlim(100,200)
    plt.ylim(100,200)
    plt.axis("equal")

    save_root = "result_high_density_networks"
    save_fig_dir = os.path.join(save_root, f"{prefix}_Q{I0}")

    os.makedirs(save_fig_dir, exist_ok=True)
    plt.savefig(f"{save_fig_dir}/"+str(i)+".png", format="png", dpi=300)
    plt.close() 

# =============================================
# 関数: PSの計算
# =============================================
#ゼロ除算が発生しないように非ゼロ要素のみで割り算を行う
def divide_non_zero_element(D, L, num_nodes, list_edges):
    X = np.zeros((num_nodes,num_nodes))
    for (i,j) in list_edges:
        i = int(i)
        j = int(j)
        X[i,j]=D[i,j]/L[i,j]
        X[j,i]=X[i,j]
    return X

# 一次連立方程式を解きpを求める
def deduce_p(D, L, F, num_nodes, list_edges, eps):
    Y = divide_non_zero_element(D,L,num_nodes,list_edges) #D/L
    A = np.diag(np.sum(Y,axis=1)+eps)-Y
    p = np.linalg.solve(A,F) # p=q/Aの連立方程式
    return p

def f(x,gamma): # |Qij|^gamma
    powered=x**gamma
    return powered

def dD(D, L, P, nnodes, edge, tick, mu): # Dの時間変化量を求める dDij/dt
    X   = divide_non_zero_element(D,L,nnodes,edge)
    Q   = np.multiply(X, np.expand_dims(P, axis=1) - P)        
    a   = 1.0
    ans = f(np.abs(Q), mu) - a*D # PS

    return ans, Q

def RK_Dt(nnodes, eps, t, length, edge_list): #ルンゲクッタで単位時間あたりのコンダクティビティの変位を求める

    for i in range(0,len(t)-1): #繰り返し計算する
        k1, Q1 = dD( conductivity[i], length, pressure[i], nnodes, edge_list,  t[i], gamma)
        k1, Q1 = k1*dt, Q1*dt
        l1 = dt * deduce_p( conductivity[i], length, flux, nnodes, edge_list, t[i])

        k2, Q2 = dD( conductivity[i]+ (k1/2) ,length, pressure[i]+(l1/2), nnodes, edge_list,  t[i]+(dt/2), gamma)
        k2, Q2 = k2*dt, Q2*dt
        l2 = dt * deduce_p( conductivity[i]+ (k1/2) ,length, flux, nnodes, edge_list, t[i]+(dt/2))
        
        k3, Q3 = dD( conductivity[i]+ (k2/2) ,length, pressure[i]+(l2/2), nnodes, edge_list,  t[i]+(dt/2), gamma)
        k3, Q3 = k3*dt, Q3*dt
        l3 = dt * deduce_p( conductivity[i]+ (k2/2), length, flux, nnodes, edge_list, t[i]+(dt/2))
        
        k4, Q4 = dD( conductivity[i]+ k3     ,length, pressure[i]+l3    , nnodes, edge_list,  t[i]+dt, gamma)
        k4, Q4 = k4*dt, Q4*dt
        l4 = dt * deduce_p( conductivity[i]+ k3, length, flux, nnodes, edge_list, t[i]+dt)

        conductivity[i+1]    = np.divide((k1 + 2*k2 + 2*k3 + k4 ), 6.0) + conductivity[i]
        conductivity[i+1] = conductivity[i+1] - np.diag(np.diagonal(conductivity[i+1]))
        pressure[i+1]     = deduce_p(conductivity[i+1], length, flux, nnodes, edge_list, eps)
        X                 = divide_non_zero_element(conductivity[i+1],length,nnodes,edge_list)
        Q[i+1]            = np.multiply(X,np.expand_dims(pressure[i+1] ,axis=1)-pressure[i+1])

        if (i%100==0):
            graph(G, i, Q) # 毎ステップのグラフ作成する

    return conductivity, pressure, Q

# =============================================
# shellで回すため引数設定できるようにする
# =============================================
parser = argparse.ArgumentParser()
parser.add_argument("--h", type=float, default=3.5)
parser.add_argument("--Q0",type=int, default=10)
args = parser.parse_args()

# =============================================
# 作成したネットワークの情報を読み込む
# =============================================
# --- メッシュファイル ---
h      = args.h   # メッシュ生成側の h と一致させる
L      = 300.0 # 固定
radius = 45.0  # 固定
dir    = "networks_for_simulation/"
prefix = f"circle_r{radius}_L{L}_h{h}"   # メッシュ生成側と揃える
nodes_csv = dir + f"{prefix}_nodes.csv"   # id,x,y
edges_csv = dir + f"{prefix}_edges.csv"   # src,dst
print(f"########### 実行:{prefix} 流量:{args.Q0} ###########")

# --- 読み込み ---
nodes_arr = np.loadtxt(nodes_csv, delimiter=",", skiprows=1)
if nodes_arr.ndim == 1:
    nodes_arr = nodes_arr.reshape(1, -1)

node_ids = nodes_arr[:, 0].astype(int)
coords   = nodes_arr[:, 1:3]

nodes = list(node_ids)
pos   = {int(i): (float(x), float(y)) for i, (x, y) in zip(node_ids, coords)}

# エッジ読み込み: src,dst
edges_arr = np.loadtxt(edges_csv, delimiter=",", skiprows=1, dtype=int)
if edges_arr.ndim == 1:
    edges_arr = edges_arr.reshape(1, -1)
edges = [(int(i), int(j)) for i, j in edges_arr]

# 円の中心と直径上の source/sink を決定
source, sink, center, radius = choose_source_sink_on_diameter(pos)
cx, cy = center

print("中心座標 (cx, cy):", center)
print("推定半径 R:", radius)
print("選ばれた source (湧き出し):", source, "座標:", pos[source])
print("選ばれた sink   (吸い込み):", sink,   "座標:", pos[sink])

# =============================================
# 読み込んだ情報をNetworkXデータとして反映する
# =============================================
G = nx.Graph()
G.add_nodes_from(nodes)

for u, v in edges:
    p1, p2 = pos[u], pos[v]
    weight = dist(p1, p2)
    G.add_edge(u, v, weight=weight)

nnodes    = nx.number_of_nodes(G)
edge_list = list(G.edges())

# 隣接行列（重み付き：エッジ長）
nbr_mtx = nx.to_numpy_array(G, weight='weight', dtype=float)
L = nbr_mtx.copy()   # エッジ長行列

edge_lengths = [d.get('weight') for u, v, d in G.edges(data=True)]
avg_edge_length = sum(edge_lengths) / len(edge_lengths)
print(f"座標ベースで取ってきた平均距離 h={h}のとき:", avg_edge_length)

# =============================================
# PSのパラメータを決定する
# =============================================
# 試行回数
nt     = 301
dt     = 0.1
tick   = np.arange(0,nt,dt) #刻み時刻
eps    = 0.001 #　固定, ランク落ちを防ぐため
I0     = args.Q0 # 全体流量
gamma  = 1 # 多様性をもたせるパラメータ, 論文によってγやμだったりする

edge_list    = list(G.edges)
flux         = np.zeros(nnodes, dtype=np.float32)             #Q
length       = np.zeros((nnodes,nnodes), dtype=np.float32)    #L
pressure     = np.zeros((len(tick),nnodes), dtype=np.float32)        #p
conductivity = np.zeros((len(tick),nnodes,nnodes), dtype=np.float32) #D

Q =  np.zeros((len(tick),nnodes,nnodes), dtype=np.float32)

length       = L
flux[source] = I0
flux[sink]   = -I0

conductivity[0] = nbr_mtx
np.divide(1, nbr_mtx, out=conductivity[0], where=nbr_mtx!=0) # 1/Length、長さに基づいた流れやすさにする
pressure[0] = deduce_p(conductivity[0],length,flux, nnodes, edge_list, eps)
X   = divide_non_zero_element(conductivity[0],length,nnodes,edge_list)
Q[0]  = np.multiply(X,np.expand_dims(pressure[0] ,axis=1)-pressure[0])

D, P, Q = RK_Dt(nnodes, eps, tick, length, edge_list)

# with open(f'{prefix}_Q{I0}.pkl', 'wb') as f:
# # with open(f'selected_edges_a={fusion_alpha}_3.pkl', 'wb') as f:
#     pickle.dump(D, f)
#     pickle.dump(Q, f)