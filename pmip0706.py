import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
import json

def solv_pmip(T, C, M, epsilon, xi):
    '''
    K: 供給者数
    J: 需要者数
    n: シナリオ数
    '''
    
    K, J = C.shape
    n = len(xi)

    start = time.time()
    # モデルの定義
    model = gp.Model("PMIP")

    # 変数の定義
    X = model.addMVar((K, J), lb=0.0, vtype=GRB.CONTINUOUS, name="X")
    z = model.addVars(n, vtype=GRB.BINARY, name="z")

    # 目的関数の定義
    objective = gp.quicksum(C[k, j] * X[k, j] for k in range(K) for j in range(J))
    model.setObjective(objective, sense=GRB.MINIMIZE)

    # 制約条件の定義
    # rhs = []
    # for i in range(n):
    #     rhs.append()
    # print(len(rhs))
    start_seiyaku = time.time()
    for i in range(n):
        model.addConstr(T @ X >= (1-z[i]) * xi[i])    
    end_seiyaku = time.time()
    seiyaku_time = end_seiyaku-start_seiyaku
    print("制約条件追加の処理時間：", seiyaku_time)

    model.addConstr(gp.quicksum(z[i] for i in range(n)) <= int(n * epsilon))
    
    # 供給制約
    for k in range(K):
        model.addConstr(gp.quicksum(X[k, j] for j in range(J)) <= M[k])

    # 時間上限
    model.Params.TimeLimit = 3600
    # 最適化の実行
    model.optimize()

    # 結果の表示
    if model.status == GRB.OPTIMAL:
        optimal_X = np.array([[X[k, j].x for k in range(K)] for j in range(J)])
        optimal_z = np.array([z[i].x for i in range(n)])
        optimal_obj_value = model.objVal
        print("最適解X:", optimal_X)
        print("最適解z:", optimal_z)
        print(optimal_X.shape)
        print("最適解の目的関数値：", optimal_obj_value)
    
        end = time.time()
        print("処理時間：", end-start, "秒")
        return optimal_X, optimal_z, optimal_obj_value

        # optimal_zの値をファイルに書き込む
        # listとして保存
        # with open("optimal_z_0621.txt", mode="a") as f:
        #     f.write(str(K)+","+str(J)+"\n")
        #     f.write(str(optimal_z) + "\n")
    else:
        print("最適解は見つかりませんでした。")

    model.close()
    


# パラメータの設定
# シードの設定（再現性のため）
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
print(xi)

# with open("parameters/parameters40-100-1000a.json", "r") as f:
#     data = json.load(f)
    
# C = np.array(data["c"])
# theta = np.array(data["theta"])
# xi = np.array(data["xi"])

# K = len(theta)
# J = len(C[0])
# n = len(xi)

# T = np.ones((1, K))

solv_pmip(T, C, theta, 0.1, xi)