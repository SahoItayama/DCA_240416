import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time

# ソフト閾値作用素の計算
def soft_thresholding(z, lambda_):
        if z <= -lambda_:
            return z + lambda_
        elif z >= lambda_:
            return z - lambda_
        else:
            return 0

# プロキシ関数の計算
# zの最大K要素のインデックス集合に含まれていればzを，含まれていなければソフト閾値作用素を適用したzを返す
def prox_operator(z, lambda_, K):
    sorted_indices = np.argsort(np.abs(z))[::-1]
    top_K_indices = sorted_indices[:K]
    z_prox = z.copy()
    for i in range(len(z)):
        if i not in top_K_indices:
            z_prox[i] = soft_thresholding(z[i], lambda_)
    return z_prox

# 最適化問題を解く
def solve_main(T, C, M, xi, epsilon, lambda_, z_0):
    '''
    K: 供給者数
    J: 需要者数
    n: シナリオ数
    '''
    K, J = C.shape
    n = len(xi)
    p = int(n * epsilon)

    # モデルの作成
    model = gp.Model('gist')

    # 変数の追加
    X = model.addMVar((K, J), lb=0.0, vtype=GRB.CONTINUOUS, name='X')
    # z = model.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='z')

    # 目的関数の追加
    # zを大きい順に並べ, pからnまでのzを足し合わせる
    z = np.array(z_0)
    z_sort = np.sort(z)
    T_K = gp.quicksum(z_sort[i] for i in range(p+1, n))
    objective  = gp.quicksum(C[k, j] * X[k, j] for k in range(K) for j in range(J)) + lambda_ * T_K
    model.setObjective(objective, GRB.MINIMIZE)

    # 制約条件の定義
    for i in range(n):
        model.addConstr(T @ X >= (1-z[i]) * xi[i])
    
    # for i in range(n-1):
    #     model.addConstr(z[i] >= z[i+1])

    # 供給制約
    for k in range(K):
        model.addConstr(gp.quicksum(X[k, j] for j in range(J)) <= M[k])

    # 時間上限
    model.Params.TimeLimit = 3600

    # 最適化の実行
    model.optimize()

    # 結果の表示
    if model.status == GRB.OPTIMAL:
        optimal_X = np.array([[X[k, j].x for j in range(J)] for k in range(K)]) 
        # optimal_z = np.array([z[i].x for i in range(n)])
        optimal_obj_value = model.objVal
        return optimal_X, optimal_obj_value
    else:
        print("最適解が見つかりませんでした。")
        return None, None
    
def gist(T, C, M, xi, epsilon, lambda_, max_iter, z_0):
    # 初期値
    f_t_minus_1 = 0
    z_t_minus_1 = z_0
    start = time.time()
    for iteration in range(max_iter):
        # zの近接写像を計算
        z_t = prox_operator(z_t_minus_1, lambda_, int(len(xi) * epsilon))

        # 最適化問題を解く
        X_t, obj_value = solve_main(T, C, M, xi, epsilon, lambda_, z_t)
        f_t = np.sum(C * X_t)

        if np.abs(f_t - f_t_minus_1) < 1e-6:
            print(f'{iteration+1}回目の更新で収束しました。')
            break

        f_t_minus_1 = f_t
        z_t_minus_1 = z_t
    end = time.time()

    print("最適解X:", X_t)
    print("X_tの形状：", X_t.shape)
    print("最適解z:", z_t)
    print("最適解zの非ゼロ要素数：", sum(i != 0 for i in z_t))
    print("最適解の目的関数値：", f_t)
    print("処理時間：", end-start, "秒")
    # 処理時間をファイルに書き込む
    # with open('gist_time.txt', 'a') as f:
        # f.write(str(J)+","+str(n)+"\n"+ str(end-start) + '\n')
    

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
making_xi_time_s = time.time()
for _ in range(n):
    mean = np.random.uniform(0, 10, J)  # 平均ベクトルをランダムに生成
    variance = np.random.uniform(1, 5, J)  # 分散ベクトルをランダムに生成
    consumer_demand = np.abs(np.random.normal(mean, np.sqrt(variance), J))  # 正規分布に従うランダムベクトルを生成
    xi.append(consumer_demand)
making_xi_time_e = time.time()
# print("xi生成時間：", making_xi_time_e-making_xi_time_s, "秒")
xi = np.array(xi)

z_0 = np.random.rand(n)

gist(T, C, M, xi, epsilon=0.1, lambda_=10, max_iter=1000, z_0=z_0)
