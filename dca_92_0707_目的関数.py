import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
import json

def subgradient(z_t_minus_1):
    # z_t_minus_1の各要素を2倍したnp.arrayを返す
    return 2 * np.array(z_t_minus_1)


# 凸最適化問題は最大Kノルムの設定に¥sum z_i <n*epsilonを追加したもの
def solv_main(T, C, M, xi, rho, epsilon, s_z_t_minus_1):
    '''
    K: 供給者数
    J: 需要者数
    n: シナリオ数
    '''
    K, J = C.shape
    n = len(xi)
    # モデルの作成
    model = gp.Model("DCA")
    
    # 変数の定義
    X = model.addMVar((K, J), lb=0.0, vtype=GRB.CONTINUOUS, name="X")
    z = model.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="z")
    
    # 目的関数の定義
    objective = gp.quicksum(C[k, j] * X[k, j] for k in range(K) for j in range(J)) + rho * gp.quicksum(z[i] for i in range(n))- gp.quicksum(z[i] * rho * s_z_t_minus_1[i] for i in range(n))
    model.setObjective(objective, sense=GRB.MINIMIZE)
    
    # 制約条件の定義
    for i in range(n):
        model.addConstr(T @ X >= (1-z[i]) * xi[i])   

    model.addConstr(gp.quicksum(z[i] for i in range(n)) <= int(n * epsilon))

    # 供給制約
    for k in range(K):
        model.addConstr(gp.quicksum(X[k, j] for j in range(J)) <= M[k])
        
    # 時間上限
    model.Params.TimeLimit = 3600
    
    # 最適化の実行
    model.optimize()
    
    # 結果の表示
    # if model.status == GRB.OPTIMAL:
    #     optimal_X = np.array([[X[k, j].x for j in range(J)] for k in range(K)]) 
    #     optimal_z = np.array([z[i].x for i in range(n)])
    #     optimal_obj_value = model.objVal
    #     return optimal_X, optimal_z, optimal_obj_value, seiyaku_time
    # else:
    #     print("最適解が見つかりませんでした。")
    #     return None, None, None, seiyaku_time

    optimal_X = np.array([[X[k, j].x for k in range(K)] for j in range(J)])
    optimal_z = np.array([z[i].x for i in range(n)])
    optimal_obj_value = model.objVal

    return optimal_X, optimal_z, optimal_obj_value

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

def solve_for_est(T, C, M, xi, epsilon, z_hat):
    '''
    K: 供給者数
    J: 需要者数
    n: シナリオ数
    '''
    K, J = C.shape
    n = len(xi)
    # モデルの作成
    model = gp.Model("DCA")
    
    # 変数の定義
    X = model.addMVar((K, J), lb=0.0, vtype=GRB.CONTINUOUS, name="X")
    
    # 目的関数の定義
    objective = gp.quicksum(C[k, j] * X[k, j] for k in range(K) for j in range(J)) 
    model.setObjective(objective, sense=GRB.MINIMIZE)
    
    # 制約条件の定義
    for i in range(n):
        model.addConstr(T @ X >= (1-z_hat[i]) * xi[i])   

    # model.addConstr(gp.quicksum(z_hat[i] for i in range(n)) <= int(n * epsilon))

    # 供給制約
    for k in range(K):
        model.addConstr(gp.quicksum(X[k, j] for j in range(J)) <= M[k])
        
    # 時間上限
    model.Params.TimeLimit = 3600
    
    # 最適化の実行
    model.optimize()
    
    # 結果の表示
    # if model.status == GRB.OPTIMAL:
    #     optimal_X = np.array([[X[k, j].x for j in range(J)] for k in range(K)]) 
    #     optimal_z = np.array([z[i].x for i in range(n)])
    #     optimal_obj_value = model.objVal
    #     return optimal_X, optimal_z, optimal_obj_value, seiyaku_time
    # else:
    #     print("最適解が見つかりませんでした。")
    #     return None, None, None, seiyaku_time

    optimal_X = np.array([[X[k, j].x for k in range(K)] for j in range(J)])
    optimal_obj_value = model.objVal

    return optimal_X, optimal_obj_value


def dca_92_h(T, C, M, xi, rho, epsilon, z_t_minus_1, varepsilon=0.001, max_iteration=10):
    # 初期値
    start = time.time()
    iteration = 0
    f_t_minus_1 = np.inf
    f_t_hat_minus_1 = np.inf
    f_t_c_minus_1 = np.inf
    for iteration in range(max_iteration):
    # while True:
        print("-----反復回数：", iteration+1, "回-----")

        # 劣勾配の計算
        s_t_minus_1 = subgradient(z_t_minus_1)
        # print("s_t_minus_1:", s_t_minus_1)
        
        # 最適化問題を解く
        X_t, z_t, f_t = solv_main(T, C, M, xi, rho, epsilon, s_t_minus_1)
        # print("z_t:", z_t)
        print(np.sum(z_t))
        # ゼロでないzの要素数
        print("z_tの非ゼロ要素数：", np.count_nonzero(z_t))
        # f_t_c = np.sum(np.transpose(C) * X_t)  + np.sum(rho * z_t) - np.sum(rho * z_t ** 2)
        # print("最適値：",f_t_c)

        # zをK番目まで1、それ以外を0にする
        z_hat = process_vector(z_t, int(len(xi) * epsilon))
        print("z_hatの非ゼロ要素数：", np.count_nonzero(z_hat))
        # print("z_hat:", z_hat)

        # Xを求める
        X, f_t_hat = solve_for_est(T, C, M, xi, epsilon, z_hat)
        print("f_t_hat:", f_t_hat)

        # 収束判定
        # rho = 500 で1386, 3回
        # if np.abs(f_t_minus_1 - f_t_hat) < varepsilon:

        # 100回行くけど1358に到達
        # if np.abs(f_t- f_t_hat) < varepsilon:
        # if f_t_minus_1 < f_t_hat:
        # if f_t_minus_1 < f_t:
        # if np.abs(f_t_hat_minus_1 - f_t_hat) < 1e-6:

        # if f_t_c_minus_1 < f_t_c:

        # 1416までしかいかない
        # if f_t_hat - f_t_minus_1 < 1e-10:

        # if f_t_hat_minus_1 < f_t_hat:

        # rho = 500で収束
        # if np.abs(f_t_minus_1 - f_t_c) < varepsilon:
            # print("解が収束しました。")
            # break
    
        # if iteration == max_iteration:
        #     print("最大反復回数に達しました。")
        #     break


        print("凸最適化のXを使う")
        print("cx:", np.sum(np.transpose(C) * X_t))
        print("cx+rho z:", np.sum(np.transpose(C) * X_t) + rho * np.sum(z_t))
        print("cx+rho z-rho z^2:", np.sum(np.transpose(C) * X_t) + rho * np.sum(z_t) - rho * np.sum(z_t ** 2))
        print("cx+rho z-2rho s z:", np.sum(np.transpose(C) * X_t) + rho * np.sum(z_t) - 2 * rho * np.sum(s_t_minus_1 * z_t))
        print(" ")
        print("評価値を使う")
        print("cx:", np.sum(np.transpose(C) * X))
        print("cx+rho z:", np.sum(np.transpose(C) * X) + rho * np.sum(z_t))
        print("cx+rho z-rho z^2:", np.sum(np.transpose(C) * X) + rho * np.sum(z_t) - rho * np.sum(z_t ** 2))
        print("cx+rho z-2rho s z:", np.sum(np.transpose(C) * X) + rho * np.sum(z_t) - 2 * rho * np.sum(s_t_minus_1 * z_t))
        print("z_t-z_t^2:", np.sum(z_t) - np.sum(z_t ** 2))
        print("z_t-2s_iz_i:", np.sum(z_t) - 2 * np.sum(s_t_minus_1 * z_t))
        print("f_t_hat:", f_t_hat)


        z_t_diff = np.sum(z_t) - np.sum(z_t ** 2)
        z_t_minus_1_diff = np.sum(z_t_minus_1) - np.sum(np.array(z_t_minus_1) ** 2)

        if np.abs(z_t_diff - z_t_minus_1_diff) < 0.001:
            break

        # 更新
        f_t_minus_1 = f_t
        f_t_hat_minus_1 = f_t_hat
        # f_t_c_minus_1 = f_t_c
        # f_t_minus_1 = f_t_c
        z_t_minus_1 = z_t
        # iteration += 1

    # print("最適解x:", X_t_hat) 
    # print("最適解z:", z_t_minus_1)
    # print("最適値z_01：", process_vector(z_t_minus_1, int(len(xi) * epsilon)))
    # print("最適値f：", f_t_minus_1)
    # print("最適値f_hat：", f_t_hat_minus_1)
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

# with open("parameters/parameters40-200-3000a.json", "r") as f:
#     data = json.load(f)
    
# C = np.array(data["c"])
# theta = np.array(data["theta"])
# xi = np.array(data["xi"])

# K = len(theta)
# J = len(C[0])
# n = len(xi)

# T = np.ones((1, K))

np.random.seed(42)

# # 供給者数
K = 40
# # 消費者数
J = 100
# # シナリオ数
n = 2000

T = np.ones((1, K))
# # print("T", T)
C = np.random.uniform(100, 500, size=(K, J))
# C = np.ones((K, J))
# # print("c", C)
theta = np.array([30000] * K)

# # 消費者の需要を表すランダムベクトルの生成
xi = []
for _ in range(n):
    mean = np.random.uniform(0, 10, 1)  # 平均ベクトルをランダムに生成
    variance = np.random.uniform(1, 5, 1)  # 分散ベクトルをランダムに生成
    consumer_demand = np.abs(np.random.normal(mean, np.sqrt(variance), 1)) * 1000 # 正規分布に従うランダムベクトルを生成
    xi.append(consumer_demand)
xi = np.array(xi)

z_t_minus_1 = [0.5] * n

dca_92_h(T, C, theta, xi, 100, 0.1, z_t_minus_1)