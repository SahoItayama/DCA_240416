import gurobipy as gp
from gurobipy import GRB
import numpy as np  
import time
# import statistics
from tqdm import tqdm

# 劣勾配の計算
# (1) |w|の要素を減少順にソートする
# (2) i=1, ..., Kならw(i)に対応するsの要素のs(i)にsign(w(i))を代入し、それ以外は0を代入する
# def calculate_subgradient(w, K):
#     n = len(w)
#     s = np.zeros(n)
#     w = np.sort(np.abs(w))[::-1] #(1)
#     s[:K] = np.sign(w[:K]) #(2)
#     return s

def process_vector(w, K):
    w = np.asarray(w, dtype=float)
    # print("w", w)
    # wを減少順に並べ替える
    sorted_indices = np.argsort(-np.abs(w))
    sorted_w = np.abs(w[sorted_indices])
    # print("sorted indices", sorted_indices)
    # print("sorted_w", sorted_w)

    # w(i)をK番目まで1、それ以外を0にする
    s = np.zeros_like(w)
    s[sorted_indices[:K]] = np.sign(w[sorted_indices[:K]])
    # print("s", s)

    return s
    

# x = argmin(cx + ρ|z|_1 - zs_z^{t-1})を解く
def solve_argmin_x(C, rho, s_t_minus_1, T, xi, z_t_minus_1, M):
    '''
    K: 供給者数
    J: 需要者数
    n: シナリオ数
    '''

    K, J = C.shape
    n = len(xi)
    # p = int(n * epsilon)

    # zの劣勾配の計算
    # s_t_minus_1 = calculate_subgradient(z_t_minus_1, p)
    
    # モデルの作成
    model = gp.Model("sub_x")
    
    # 変数の定義
    X = model.addMVar((K, J), lb=0.0, vtype=GRB.CONTINUOUS, name="X")

    # 目的関数の定義
    objective = gp.quicksum(C[k, j] * X[k, j] for k in range(K) for j in range(J)) + rho * gp.quicksum(z_t_minus_1[i] for i in range(n))- gp.quicksum(z_t_minus_1[i] * rho * s_t_minus_1[i] for i in range(n))
    model.setObjective(objective, sense=GRB.MINIMIZE)

    # 制約条件の定義
    for i in range(n):
        model.addConstr(T @ X >= (1-z_t_minus_1[i]) * xi[i])
    
    # 供給制約 
    for k in range(K):
        model.addConstr(gp.quicksum(X[k, j] for j in range(J)) <= M[k])
    
    #　時間上限
    model.Params.TimeLimit = 1800

    # 最適化の実行 
    model.optimize()
    
    # 結果の表示
    if model.status == GRB.OPTIMAL:
        optimal_X = np.array([[X[k, j].x for j in range(J)] for k in range(K)]) 
        optimal_obj_value = model.objVal
        return optimal_X, optimal_obj_value
    else:
        print("最適解が見つかりませんでした。")
        return None, None
    
# z = argmin(cx + ρ|z|_1 - zs_z^{t-1})を解く
def solve_argmin_z(C, rho, s_t_minus_1, T, xi, X_t, M):
    '''
    K: 供給者数
    J: 需要者数
    n: シナリオ数
    '''

    K, J = C.shape
    n = len(xi)
    
    # モデルの作成
    model = gp.Model("sub_x")
    
    # 変数の定義
    z = model.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="z")

    # 目的関数の定義
    objective = gp.quicksum(C[k, j] * X_t[k, j] for k in range(K) for j in range(J)) + rho * gp.quicksum(z[i] for i in range(n))- gp.quicksum(z[i] * rho * s_t_minus_1[i] for i in range(n))
    model.setObjective(objective, sense=GRB.MINIMIZE)

    # 制約条件の定義
    for i in range(n):
        for j in range(J):
            model.addConstr(gp.quicksum(T[0, k] * X_t[k, j] for k in range(K)) >= (1-z[i]) * xi[i][j])
    
    # 供給制約 
    # for k in range(K):
        # model.addConstr(gp.quicksum(X_t[k, j] for j in range(J)) <= M[k])
    
    #　時間上限
    model.Params.TimeLimit = 1800

    # 最適化の実行 
    model.optimize()
    
    # 結果の表示
    if model.status == GRB.OPTIMAL:
        optimal_z = np.array([z[i].x for i in range(n)])
        optimal_obj_value = model.objVal
        return optimal_z, optimal_obj_value
    else:
        print("最適解が見つかりませんでした。")
        return None, None
    
def dca0528(T, C, M, xi, epsilon, rho, max_iter=1000):

    '''
    K: 供給者数
    J: 需要者数
    n: シナリオ数
    '''
    np.random.seed(42)

    # 初期値の設定
    f_t_minus_1 = 0
    z_t_minus_1 = np.random.rand(len(xi))

    start = time.time()
    for iteration in range(max_iter):
        s_t_minus_1 = process_vector(z_t_minus_1, int(len(xi) * epsilon))

        X_t = solve_argmin_x(C, rho, s_t_minus_1, T, xi, z_t_minus_1, M)[0]
        z_t = solve_argmin_z(C, rho, s_t_minus_1, T, xi, X_t, M)[0]
        f_t = np.sum(C * X_t)

        if np.abs(f_t - f_t_minus_1) < 1e-4:
            print(f'{iteration+1}回目の更新で収束しました。')
            break

        f_t_minus_1 = f_t
        z_t_minus_1 = z_t
    end = time.time()

    print("最適解X:", X_t)
    print("最適解z：", z_t)
    # z_tの非ゼロ要素の個数を求める
    print("最適解zの非ゼロ要素数：", np.count_nonzero(z_t))
    print("最適解の目的関数値：", f_t)
    print("処理時間：", end-start, "秒")


# パラメータ
# シードの設定（再現性のため）
np.random.seed(42)

# 供給者数
K = 40
# 消費者数
J = 100
# シナリオ数
n = 1000

T = np.ones((1, K))
# print("T", T)
C = np.random.uniform(10, 50, size=(K, J))
# print("c", C)
M = np.array([100] * K)


# 消費者の需要を表すランダムベクトルの生成
xi = []
# making_xi_time_s = time.time()
for _ in range(n):
    mean = np.random.uniform(0, 10, J)  # 平均ベクトルをランダムに生成
    variance = np.random.uniform(1, 5, J)  # 分散ベクトルをランダムに生成
    consumer_demand = np.abs(np.random.normal(mean, np.sqrt(variance), J))  # 正規分布に従うランダムベクトルを生成
    xi.append(consumer_demand)
# making_xi_time_e = time.time()
# print("xi生成時間：", making_xi_time_e-making_xi_time_s, "秒")
xi = np.array(xi)
# print("xi", xi.shape)

dca0528(T, C, M, xi, epsilon=0.1, rho=10000, max_iter=1000)