import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
from tqdm import tqdm

def solv_pmip(T, C, M, epsilon, xi):
    '''
    K: 供給者数
    J: 需要者数
    n: シナリオ数
    '''
    
    # h = xi[np.argsort(-np.asarray(norm_xi))]
    K, J = C.shape
    n = len(xi)
    p = int(n * epsilon)
    xi = np.array(xi)
    # norm_xi = []
    # for i in range(len(xi)):
    #     norm_xi.append(np.linalg.norm(xi[i]))
    # h = np.array(xi[np.argsort(-np.asarray(norm_xi))])
    h = np.sort(xi, axis=0)[::-1]

    start = time.time()
    # モデルの定義
    model = gp.Model("PMIPEX")

    # 変数の定義
    X = model.addMVar((K, J), lb=0.0, vtype=GRB.CONTINUOUS, name="X")
    z = model.addVars(n, vtype=GRB.BINARY, name="z")
    w = model.addVars(p, vtype=GRB.BINARY, name="w")

    # 目的関数の定義
    objective = gp.quicksum(C[k, j] * X[k, j] for k in range(K) for j in range(J))
    model.setObjective(objective, sense=GRB.MINIMIZE)

    # 制約条件の定義
    # rhs = []
    # for i in range(n):
    #     rhs.append()
    # print(len(rhs))
    start_seiyaku = time.time()
    # for i in range(n):
    model.addConstr(T @ X >= h[0] - gp.quicksum((h[i]-h[i+1])* w[i] for i in range(p)))  
    end_seiyaku = time.time()
    seiyaku_time = end_seiyaku-start_seiyaku
    print("制約条件追加の処理時間：", seiyaku_time)

    for i in range(p):
        model.addConstr(z[i] >= w[i])
        if i+1 < p:
            model.addConstr(w[i] - w[i+1] >= 0)
        else:
            model.addConstr(w[i] >= 0)
    
    model.addConstr(gp.quicksum(z[i] for i in range(n)) <= p)
    
    # 供給制約
    for k in range(K):
        model.addConstr(gp.quicksum(X[k, j] for j in range(J)) <= M[k])

    # 時間上限
    model.Params.TimeLimit = 300
    # 最適化の実行
    model.optimize()

    # 結果の表示
    if model.status == GRB.OPTIMAL:
        optimal_X = np.array([[X[k, j].x for k in range(K)] for j in range(J)])
        optimal_obj_value = model.objVal
        optimal_w = np.array([w[i].x for i in range(p)])
        optimal_z = np.array([z[i].x for i in range(n)])
        print("最適解X:", optimal_X)
        print("最適解wi:", optimal_w, optimal_w.shape)
        print("最適解zi:", optimal_z, optimal_z.shape)
        print("最適解の目的関数値：", optimal_obj_value)
        # C * X_tの値を計算
        print("C*X_t:", np.sum(C.T * optimal_X))
    else:
        print("最適解は見つかりませんでした。")

    model.close()
    end = time.time()
    print("処理時間：", end-start, "秒")
    return optimal_X, optimal_obj_value, optimal_z

# シードの設定（再現性のため）
np.random.seed(42)

# 供給者数
K = 40
# 消費者数
J = 100
# シナリオ数
n = 1000

T = np.ones((1, K))
C = np.random.uniform(10, 50, size=(K, J))
M = np.array([100] * K)


# 消費者の需要を表すランダムベクトルの生成
xi = []
making_xi_time_s = time.time()
for _ in range(n):
    mean = np.random.uniform(0, 10, J)  # 平均ベクトルをランダムに生成
    variance = np.random.uniform(1, 5, J)  # 分散ベクトルをランダムに生成
    consumer_demand = np.abs(np.random.normal(mean, np.sqrt(variance), J))  # 正規分布に従うランダムベクトルを生成
    xi.append(consumer_demand)
# making_xi_time_e = time.time()
# print("xi生成時間：", making_xi_time_e-making_xi_time_s, "秒")
print("xi:", xi)

# print("h:", h)

s = solv_pmip(T=T, C=C, M=M, epsilon=0.1, xi=xi)
# sに保存されている最適解を確認
# print("最適解X:", s[0])
print(s[0].shape)
# 制約条件を満たしているかを確認する関数
def check_constraints(X, z):
    # 制約条件のチェックを実装する
    for i in range(n):
        if T @ X >= (1-z[i]) * xi[i]:
            for k in range(K):
                if gp.quicksum(X[k, j] for j in range(J)) <= M[k]:
                    return True
                else:
                    return False
        else:
            return False
    # if solution > 0 and solution < 10:
    #     return True
    # else:
    #     return False

# 解の確認
# solution = 5  # 確認したい解の値を設定する
# if check_constraints(solution):
#     print("解は制約条件を満たしています")
# else:
#     print("解は制約条件を満たしていません")

check_constraints(s[0], s[2])