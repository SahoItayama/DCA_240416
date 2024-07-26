import gurobipy as gp
from gurobipy import GRB
import numpy as np  
import time
# import statistics
from tqdm import tqdm
import itertools

# f^t = cx+ρ(|z|_1)-zs_z^{t-1}の計算
def solve_main(T, C, M, xi, rho, s_z_t_minus_1):
    '''
    K: 供給者数
    J: 需要者数
    n: シナリオ数
    '''
    K, J = C.shape
    n = len(xi)
    # n = 3000
    # モデルの作成
    model = gp.Model("DCA")
    
    # 変数の定義
    X = model.addMVar((K, J), lb=0.0, vtype=GRB.CONTINUOUS, name="X")
    z = model.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="z")
    
    # 目的関数の定義
    objective = gp.quicksum(C[k, j] * X[k, j] for k in range(K) for j in range(J)) + rho * gp.quicksum(z[i] for i in range(n))- gp.quicksum(z[i] * rho * s_z_t_minus_1[i] for i in range(n))
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
    print("制約条件追加の処理時間：", seiyaku_time, "秒")

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

    optimal_X = np.array([[X[k, j].x for j in range(J)] for k in range(K)])
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

    return sorted_w, s

def solve_subproblem(wt, K):
    n = len(wt)

    # Gurobiモデルの作成
    model = gp.Model("subproblem")

    # 変数 s_i を追加
    s = model.addVars(n, lb=-1, ub=1, vtype=GRB.CONTINUOUS, name="s")

    # 目的関数を設定
    model.setObjective(gp.quicksum(wt[i] * s[i] for i in range(n)), GRB.MAXIMIZE)

    # 制約: sum |s_i| = K
    # model.addConstr(gp.quicksum(s[i] for i in range(n)) == K, "abs_sum")

    # 絶対値制約を設定
    abs_s = model.addVars(n, lb=0, ub=1, name="abs_s")
    for i in range(n):
        model.addConstr(abs_s[i] == gp.abs_(s[i]))

    # 制約: sum abs(s_i) = K
    model.addConstr(gp.quicksum(abs_s[i] for i in range(n)) == K, "abs_sum")

    # 最適化の実行
    model.optimize()

    # 最適解の取得
    if model.status == GRB.OPTIMAL:
        s_opt = [s[i].x for i in range(n)]
        return s_opt
    else:
        raise Exception("Optimal solution not found")
    
# def argmax_s(w, K):
#     max_sum = float('-inf')
#     max_s = None
    
#     for s in itertools.product([-1, 1], repeat=len(w)):
#         if sum(s) == K:
#             s_sum = sum(w[i] * s[i] for i in range(len(w)))
#             if s_sum > max_sum:
#                 max_sum = s_sum
#                 max_s = s
    
#     return max_s


def dca(T, C, M, xi, rho, epsilon, z_t_minus_1, varepsilon=0.001, max_iteration=100):
    # 初期値
    # time_vec = []
    f_t_minus_1 = 0
    s_z_t_minus_1 = np.zeros_like(z_t_minus_1)
    start = time.time()
    for iteration in range(max_iteration):
        print("反復回数：", iteration+1, "回")
        # _, s_z_t = process_vector(z_t_minus_1, int(len(xi)*epsilon))
        # print(s_z_t)
        # print(s_t_minus_1 - s_z_t)
        s_z_t = solve_subproblem(z_t_minus_1, int(len(xi)*epsilon))
        # s_z_t = argmax_s(z_t_minus_1, int(len(xi)*epsilon))
        # f^tを求める
        X_t, z_t, f_t = solve_main(T, C, M, xi, rho, s_z_t)
        # print(X_t, z_t, np.sum(z_t), f_t)
        print(z_t)
        # print(z_t==z_t_minus_1)
        # print(seiyaku_time)
        print("f_t:", f_t)
        # print("z_t-z_t_minus_1:", z_t - z_t_minus_1)
        # print("s_z_t_minus_1-s_Z_t:", s_z_t_minus_1 - s_z_t)

        if np.abs(np.sum(C * X_t) - f_t) < varepsilon:
            break
            
        # 次のイテレーションのために更新
        f_t_minus_1 = f_t
        z_t_minus_1 = z_t
        s_z_t_minus_1 = s_z_t

        # return X_t, z_t, f_t

    print("最適解x:", X_t) 
    print("最適解z:", z_t)
    print("最適値f：", f_t)
    end = time.time()
    print("DCA処理時間：", end-start, "秒")
    # 計算時間をファイルに書き込む
    with open("time_dca_0611.txt", mode="a") as f:
        f.write(str(K)+","+str(J)+"\n")
        f.write(str(end-start) + "\n") 
        f.write(str(f_t) + "\n")
    # np.savetxt("time_dca.csv", end-start, delimiter=",")
    print(np.sum(C * X_t))
    # print(np.sum(C @ X_t))
    print("Xの形", X_t.shape)
    print(np.sum(z_t))

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
# print("xi", xi)
# norm_xi = []
# for i in range(len(xi)):
#     norm_xi.append(np.linalg.norm(xi[i]))
# print(np.asarray(norm_xi))
# print(norm_xi[312])
# print(np.argsort(-np.asarray(norm_xi)))
# h = xi[np.argsort(-np.asarray(norm_xi))]
# print(h)

# for _ in tqdm(range(5)):
#     dca(T=T, C=C, M=M, xi=xi, rho=10000, epsilon=0.1, z_t_minus_1=np.random.rand(n), varepsilon=1, max_iteration=100)

# z_t_minus_1は0または1でランダムに設定，長さはn
# np.random.seed(42)
# z_t_minus_1 = np.random.randint(0, 2, n)

# dca(T=T, C=C, M=M, xi=xi, rho=10000, epsilon=0.1, z_t_minus_1=z_t_minus_1, varepsilon=1, max_iteration=100)

#optimal_XがT @ X>= (1-z[i]) * xi[i]を満たしているか確認
#for i in range(n):
#    print("T @ X - (1-z) * xi:", T @ X_t - (1-z_t[i]) * xi[i] >= 0)

# for i in range(100):
#     np.random.seed(i)
#     z_t_minus_1 = np.random.randint(0, 2, n)
#     dca(T=T, C=C, M=M, xi=xi, rho=10000, epsilon=0.1, z_t_minus_1=z_t_minus_1, varepsilon=1, max_iteration=100)

def create_random_array(n, epsilon):
    num_ones = int(n * epsilon)
    num_zeros = n - num_ones
    arr = np.array([1] * num_ones + [0] * num_zeros)
    np.random.shuffle(arr)
    return arr

# dca(T=T, C=C, M=M, xi=xi, rho=10000, epsilon=0.1, z_t_minus_1=z_t_minus_1, varepsilon=1, max_iteration=100)
# for i in range(1):
#     np.random.seed(i)
#     z_t_minus_1 = create_random_array(n, 0.1)
#     dca(T=T, C=C, M=M, xi=xi, rho=10000, epsilon=0.1, z_t_minus_1=z_t_minus_1, varepsilon=1, max_iteration=100)


# 初めのepsilon%の要素が1で残りが0の配列を作成
# epsilon = 0.1
# num_ones = int(n * epsilon)
# num_zeros = n - num_ones

# z_t_minus_1 = np.concatenate((np.ones(num_ones), np.zeros(num_zeros)))
# dca(T=T, C=C, M=M, xi=xi, rho=10000, epsilon=0.1, z_t_minus_1=z_t_minus_1, varepsilon=1, max_iteration=100)

# z_star
z_t_minus_1 = [-0.0, 1.0, 1.0, 1.0, 1.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 1.0, -0.0, -0.0, -0.0, 1.0, -0.0, 1.0, 0.0, -0.0, 1.0, -0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 1.0, -0.0, -0.0, 0.0, 0.0, 1.0, -0.0, -0.0, -0.0, -0.0, -0.0, 1.0, -0.0, 1.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 1.0, -0.0, -0.0, 1.0, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 1.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 1.0, 1.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0, -0.0, 1.0, -0.0, -0.0, -0.0, 0.0, -0.0, 1.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, 1.0, -0.0, 1.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, 1.0, -0.0, -0.0, -0.0, -0.0, 1.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 1.0, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 1.0, -0.0, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 1.0, -0.0, 1.0, 1.0, -0.0, 1.0, 1.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 1.0, -0.0, 0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 1.0, -0.0, 1.0, -0.0, -0.0, 1.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 1.0, 0.0, -0.0, -0.0, 1.0, -0.0, -0.0, 0.0, -0.0, -0.0, 1.0, 0.0, -0.0, 1.0, -0.0, -0.0, -0.0, 1.0, -0.0, -0.0, -0.0, -0.0, 1.0, -0.0, 0.0, -0.0, -0.0, 1.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, 1.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 1.0, -0.0, -0.0, -0.0, -0.0, -0.0, 1.0, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 1.0, 1.0, -0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 1.0, -0.0, -0.0, -0.0, 1.0, -0.0, -0.0, -0.0, -0.0, 1.0, -0.0, -0.0, -0.0, -0.0, 1.0, -0.0, 1.0, 0.0, 1.0, -0.0, -0.0, -0.0, 0.0, 1.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 1.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 1.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 1.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 1.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 1.0, -0.0, -0.0, -0.0, 1.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 1.0, -0.0, -0.0, 0.0, -0.0, -0.0, 1.0, -0.0, -0.0, -0.0, -0.0, -0.0, 1.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 1.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 1.0, -0.0, 0.0, 0.0, 0.0, -0.0, 0.0, -0.0, -0.0, 1.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 1.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, 1.0, 0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, -0.0, -0.0, 1.0, 0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 1.0, -0.0, -0.0, -0.0, 1.0, -0.0, -0.0, 1.0, -0.0, -0.0, 0.0, 0.0, -0.0, 1.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, 1.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 1.0, -0.0, -0.0, -0.0, 1.0, 0.0, -0.0, 1.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 1.0, 1.0, 0.0, -0.0, -0.0, -0.0, 1.0, -0.0, -0.0, 0.0, 1.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 1.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 1.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 1.0, -0.0, 0.0, 1.0, 0.0, 1.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 1.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 1.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 1.0, 0.0, 1.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 1.0, -0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 1.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 1.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 1.0, 1.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, 1.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, 1.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 1.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0]

# z_t_minus_1 = [0.5] * n
# print(z_t_minus_1)
dca(T=T, C=C, M=M, xi=xi, rho=10000, epsilon=0.1, z_t_minus_1=z_t_minus_1, varepsilon=1, max_iteration=100)

