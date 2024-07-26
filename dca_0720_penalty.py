import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
import json

def solve_opt(T:np.ndarray, C:np.ndarray, theta:np.array, r:np.array, rho:int, epsilon:float, s:np.array):
    '''
    凸最適化問題を解く \\
    minimize c^{\top}x \\
    subject to x \in X \\
    T_i x + M z_i 1 \geq r_i \\
    \sum _{i=1}^n z_i^2 \leq n * epsilon \\
    \sum_{i=1}^n z_i  - \sum_{i=1}^n z_i^2 \leq rho \\
    z \in [0, 1]

    return : X: np.array, z: np.array, obj: float
    '''
    solve_opt_start_time = time.time()
    # モデルの作成
    model = gp.Model("convex_optimization")
    # model.setParam('OutputFlag', 0)

    # 変数の設定
    X = model.addMVar((K, J), lb=0.0, vtype=GRB.CONTINUOUS, name="X")
    z = model.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="z")

    # 目的関数の設定
    obj = gp.quicksum(C[k, j] * X[k, j] for k in range(K) for j in range(J)) + rho * gp.quicksum(z[i] for i in range(n))- gp.quicksum(z[i] * rho * s[i] for i in range(n))
    model.setObjective(obj, GRB.MINIMIZE)

    # 制約条件の設定
    for i in range(n):
        model.addConstr(T @ X >= (1-z[i]) * r[i])
    
    # DC表現した制約
    # model.addConstr(gp.quicksum(z[i] for i in range(n)) - gp.quicksum(s[i] * z[i] for i in range(n)) <= rho)

    # 基数制約
    model.addConstr(gp.quicksum(z[i] for i in range(n)) <= int(n * epsilon))

    # 供給制約
    for k in range(K):
        model.addConstr(gp.quicksum(X[k, j] for j in range(J)) <= theta[k])

    # 時間上限
    model.Params.TimeLimit = 1200

    # 最適化
    model.optimize()

    # 解の取得
    optimal_X = np.array([[X[k, j].x for k in range(K)] for j in range(J)])
    optimal_z = np.array([z[i].x for i in range(n)])
    optimal_obj_value = model.objVal
    solve_opt_end_time = time.time()

    solve_opt_time = solve_opt_end_time - solve_opt_start_time
    print(solve_opt_time, "秒")

    return optimal_X, optimal_z, optimal_obj_value

def process_vector(z: np.array, K: int) -> np.array:
    '''
    ベクトルzを減少順に並び替え，z[K]まで1，それ以外を0にする
    return : np.array
    '''
    z = np.asarray(z, dtype=float)
    # wを減少順に並び替え
    sorted_indices = np.argsort(-np.abs(z))

    s = np.zeros_like(z)
    s[sorted_indices[:K]] = np.sign(z[sorted_indices[:K]])

    return s

def eval_X(T: np.ndarray, C: np.ndarray, theta: np.array, r: np.array, z_hat:np.array) -> float:
    '''
    与えられたz_hatに対するXの評価値を計算する
    minimize c^{\top}x \\
    subject to x \in X \\
    T_i x \geq r_i (1 - z_hat_i) \\
    return : np.array, float
    '''
    eval_start_time = time.time()
    # モデルの作成
    model = gp.Model("eval_X")

    # 変数の設定
    X = model.addMVar((K, J), lb=0.0, vtype=GRB.CONTINUOUS, name="X")

    # 目的関数の設定
    obj = gp.quicksum(C[k, j] * X[k, j] for k in range(K) for j in range(J))
    model.setObjective(obj, GRB.MINIMIZE)

    # 制約条件の設定
    for i in range(n):
        model.addConstr(T @ X >= (1-z_hat[i]) * r[i])
    
    # 供給制約
    for k in range(K):
        model.addConstr(gp.quicksum(X[k, j] for j in range(J)) <= theta[k])

    # 最適化
    model.optimize()

    # 解の取得
    optimal_X = np.array([[X[k, j].x for k in range(K)] for j in range(J)])
    optimal_obj_value = model.objVal

    eval_end_time = time.time()

    eval_time = eval_end_time - eval_start_time
    print(eval_time, "秒")

    return optimal_X, optimal_obj_value

def dca(T: np.ndarray, C: np.ndarray, theta: np.array, r: np.array, rho: int, epsilon: float, z_t_minus_1:np.array, max_iter: int) -> float:
    '''
    DCAを実行する
    1. 劣勾配を計算する
    2. 凸最適化問題を解く
    3. 評価値が悪化していたら終了，そうでなければ1に戻る
    '''
    
    start = time.time()
    iteration = 0
    K, J = C.shape
    n = len(r)
    f_t_minus_1 = np.inf
    f_t_hat_minus_1 = np.inf
    f_t_c_minus_1 = np.inf  
    k = int(n * epsilon)

    for iteration in range(max_iter):
        print("-----反復回数：", iteration+1, "回-----")

        # 劣勾配を計算する
        s_t_minus_1 = process_vector(z_t_minus_1, k)
        # print("s_t_minus_1:", s_t_minus_1)

        # 凸最適化問題を解く
        X_t, z_t, f_t = solve_opt(T, C, theta, r, rho, epsilon, s_t_minus_1)
        print("z_tの非ゼロ要素数：", np.count_nonzero(z_t))
        # f_t_c = np.sum(np.transpose(C) * X_t)  + np.sum(rho * z_t) - np.sum(rho * z_t ** 2)
        # print("最適値：",f_t_c)

        # zを0-1変数に戻す
        z_hat = process_vector(z_t, k)

        # Xを評価する
        X, f_t_hat = eval_X(T, C, theta, r, z_hat)

        z_t_diff = np.sum(z_t) - np.sum(z_t ** 2)
        z_t_minus_1_diff = np.sum(z_t_minus_1) - np.sum(np.array(z_t_minus_1) ** 2)

        if np.abs(z_t_diff - z_t_minus_1_diff) < 0.001:
            break

        # 評価値が悪化していたら終了
        # if f_t > f_t_minus_1:
        # if f_t_c_minus_1 < f_t_c:
        #     break
        # else:
        #     f_t_minus_1 = f_t
        #     f_t_hat_minus_1 = f_t_hat
        z_t_minus_1 = z_t

        # print("凸最適化のXを使う")
        # print("cx:", np.sum(np.transpose(C) * X_t))
        # print("cx+rho z:", np.sum(np.transpose(C) * X_t) + rho * np.sum(z_t))
        # print("cx+rho z-rho z^2:", np.sum(np.transpose(C) * X_t) + rho * np.sum(z_t) - rho * np.sum(z_t ** 2))
        # print("cx+rho z-2rho s z:", np.sum(np.transpose(C) * X_t) + rho * np.sum(z_t) - 2 * rho * np.sum(s_t_minus_1 * z_t))
        # print(" ")
        # print("評価値を使う")
        # print("cx:", np.sum(np.transpose(C) * X))
        # print("cx+rho z:", np.sum(np.transpose(C) * X) + rho * np.sum(z_t))
        # print("cx+rho z-rho z^2:", np.sum(np.transpose(C) * X) + rho * np.sum(z_t) - rho * np.sum(z_t ** 2))
        # print("cx+rho z-2rho s z:", np.sum(np.transpose(C) * X) + rho * np.sum(z_t) - 2 * rho * np.sum(s_t_minus_1 * z_t))

        print("z_1-z_k:", np.sum(z_t) - np.sum(s_t_minus_1 * z_t))

        print("f_t_hat:", f_t_hat)

    # print("最適解x:", X_t) 
    # print("最適解z:", z_t_minus_1)
    # print("最適解z_hat:", process_vector(z_t_minus_1, k))
    # print("最適値f：", f_t_minus_1)
    # print("最適値f：", f_t_hat_minus_1)
    print("反復回数：", iteration+1, "回")
    end = time.time()
    print("処理時間：", end-start, "秒")

# パラメータの設定
# シードの設定（再現性のため）
# np.random.seed(42)

# # 供給者数
# K = 40
# # 消費者数
# J = 100
# # シナリオ数
# n = 1000

# T = np.ones((1, K))
# # print("T", T)
# C = np.random.uniform(10, 50, size=(K, J))
# # print("c", C)
# theta = np.array([100] * K)

# # 消費者の需要を表すランダムベクトルの生成
# xi = []
# for _ in range(n):
#     mean = np.random.uniform(0, 10, J)  # 平均ベクトルをランダムに生成
#     variance = np.random.uniform(1, 5, J)  # 分散ベクトルをランダムに生成
#     consumer_demand = np.abs(np.random.normal(mean, np.sqrt(variance), J))  # 正規分布に従うランダムベクトルを生成
#     xi.append(consumer_demand)
# xi = np.array(xi)
with open("parameters/parameters40-100-1000a.json", "r") as f:
    data = json.load(f)
    
C = np.array(data["c"])
theta = np.array(data["theta"])
xi = np.array(data["xi"])

K = len(theta)
J = len(C[0])
n = len(xi)

T = np.ones((1, K))

z_t_minus_1 = [0.5] * n


# DCAの実行
dca(T, C, theta, xi, 1000, 0.1, z_t_minus_1, 10)