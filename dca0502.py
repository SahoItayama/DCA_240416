import gurobipy as gp
from gurobipy import GRB
import numpy as np  
import time
# import statistics
from tqdm import tqdm
from itertools import combinations

# 劣勾配を求める
# argmaxを使って求める
def calculate_argmax(w, K):
    n = len(w)
    max_value = float('-inf')
    max_combination = None

    for indices in combinations(range(n), K):
        s = np.zeros(n)
        s[list(indices)] = 1
        value = np.dot(w, s)
        # print("s:", s)
        # print("value:", value)
        
        if value > max_value:
            max_value = value
            max_combination = s
    
    return max_combination

# f^t = cx+ρ(|z|_1)-zs_z^{t-1}の計算
def solve_main(T, C, M, xi, rho, s_z_t_minus_1):
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
    objective = gp.quicksum(C[k, j] * X[k, j] for k in range(K) for j in range(J)) + rho * gp.quicksum(z[i] for i in range(n))- gp.quicksum(z[i] * s_z_t_minus_1[i] for i in range(n))
    model.setObjective(objective, sense=GRB.MINIMIZE)
    
    # 制約条件の定義
    # rhs = []
    # for i in range(n):
    #     rhs.append((1-z[i]) * xi[i])
    start_seiyaku = time.time()
    for i in range(n):
        model.addConstr(T @ X >= (1-z[i]) * xi[i])    
    end_seiyaku = time.time()
    seiyaku_time = end_seiyaku-start_seiyaku
    # print("制約条件追加の処理時間：", seiyaku_time, "秒")

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
        return optimal_X, optimal_z, optimal_obj_value, seiyaku_time
    else:
        print("最適解が見つかりませんでした。")
        return None, None, None, seiyaku_time


def dca(T, C, M, xi, rho, epsilon, z_t_minus_1, varepsilon=0.001, max_iteration=100):
    # 初期値
    # time_vec = []
    f_t_minus_1 = 0
    start = time.time()
    # z_t_minus_1を降順にソート
    z_t_minus_1 = np.sort(z_t_minus_1)[::-1]
    for iteration in range(max_iteration):
        print("反復回数：", iteration+1, "回")
        max_indices = np.argsort(-z_t_minus_1)[:int(len(xi)*epsilon)]
        argmax_s = np.zeros_like(z_t_minus_1)
        argmax_s[max_indices] = 1       
    
        # f^tを求める
        X_t, z_t, f_t, seiyaku_time = solve_main(T, C, M, xi, rho, argmax_s)
        #print(X_t, z_t, np.sum(z_t), f_t)
        print(z_t)
        print(seiyaku_time)
        # # 収束判定
        # if np.sum(z_t) == 0:
        #     print(f"{iteration+1}回で収束")
        #     break

        if np.abs(f_t-f_t_minus_1) < varepsilon:
            break
            
        # 次のイテレーションのために更新
        f_t_minus_1 = f_t
        z_t_minus_1 = z_t

    print("最適解x:", X_t) 
    print("最適解z:", z_t)
    print("最適値f：", f_t)
    # C * X_tの値を計算
    print("C*X_t:", np.sum(C.T * X_t))
    end = time.time()

    print("DCA処理時間：", end-start, "秒")

# シードの設定（再現性のため）
np.random.seed(42)

# 供給者数
K = 40
# 消費者数
J = 200
# シナリオ数
n = 3000

T = np.ones((1, K))
# print("T", T)
C = np.random.uniform(10, 50, size=(K, J))
# print("c", C)
M = np.array([100] * K)


# 消費者の需要を表すランダムベクトルの生成
xi = []
making_xi_time_s = time.time()
for _ in range(n):
    mean = np.random.uniform(0, 10, J)  # 平均ベクトルをランダムに生成
    variance = np.random.uniform(1, 5, J)  # 分散ベクトルをランダムに生成
    consumer_demand = np.abs(np.random.normal(mean, np.sqrt(variance), J))  # 正規分布に従うランダムベクトルを生成
    xi.append(consumer_demand)
making_xi_time_e = time.time()
# print("xi生成時間：", making_xi_time_e-making_xi_time_s, "秒")
xi = np.array(xi)
print("xi", xi.shape)
# norm_xi = []
# for i in range(len(xi)):
#     norm_xi.append(np.linalg.norm(xi[i]))
# print(np.asarray(norm_xi))
# print(norm_xi[312])
# print(np.argsort(-np.asarray(norm_xi)))
# h = xi[np.argsort(-np.asarray(norm_xi))]
# print(h)

for _ in tqdm(range(1)):
    dca(T=T, C=C, M=M, xi=xi, rho=10000, epsilon=0.1, z_t_minus_1=np.random.rand(n), varepsilon=1, max_iteration=100)
