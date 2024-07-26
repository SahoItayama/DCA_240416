import gurobipy as gp
from gurobipy import GRB
import numpy as np  
import time
# import statistics
from tqdm import tqdm

# f^t = cxの計算
def solve_main(T, C, M, xi, rho, s_z_t_minus_1, z_0):
    '''
    K: 供給者数
    J: 需要者数
    n: シナリオ数
    '''
    K, J = C.shape
    n = len(xi)
    z = np.array(z_0)
   
    # モデルの作成
    model = gp.Model("DCA")
    
    # 変数の定義
    X = model.addMVar((K, J), lb=0.0, vtype=GRB.CONTINUOUS, name="X")
    # z = model.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="z")
    
    # 目的関数の定義
    objective = gp.quicksum(C[k, j] * X[k, j] for k in range(K) for j in range(J))
    model.setObjective(objective, sense=GRB.MINIMIZE)
    
    # 制約条件の定義
    # rhs = []
    # for i in range(n):
    #     rhs.append((1-z[i]) * xi[i])
    # start_seiyaku = time.time()
    for i in range(n):
        model.addConstr(T @ X >= (1-z[i]) * xi[i])    
    # end_seiyaku = time.time()
    # seiyaku_time = end_seiyaku-start_seiyaku
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
        optimal_X = np.array([[X[k, j].x for j in range(J)] for k in range(K)]) 
        # optimal_z = np.array([z[i].x for i in range(n)])
        optimal_obj_value = model.objVal
        return optimal_X, optimal_obj_value
    else:
        print("最適解が見つかりませんでした。")
        return None, None

    
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

def dca(T, C, M, xi, rho, epsilon, z_t_minus_1, varepsilon=0.001, max_iteration=100, alpha=0.1):
    # 初期値
    # time_vec = []
    f_t_minus_1 = 0
    start = time.time()
    for iteration in range(max_iteration):
        print("反復回数：", iteration+1, "回")
        _, s_z_t_minus_1 = process_vector(z_t_minus_1, int(len(xi)*epsilon))
        print("s:", s_z_t_minus_1)
  
        # f^tを求める
        X_t, f_t = solve_main(T, C, M, xi, rho, s_z_t_minus_1, z_t_minus_1)
        #print(X_t, z_t, np.sum(z_t), f_t)
        print(f_t)
        # # 収束判定
        # if np.sum(z_t) == 0:
        #     print(f"{iteration+1}回で収束")
        #     break
        # v = T @ X_t
        # print(v)
        # print(len(v[0]))

        if np.abs(f_t-f_t_minus_1) < varepsilon:
            break
        
        # z_tを射影勾配法で求める
        y_t = np.zeros_like(z_t_minus_1)
        noise = np.random.normal(0, 0.1, n)  # Add Gaussian noise with mean 0 and standard deviation 0.1
        for i in range(n):
            if z_t_minus_1[i] >= alpha * (rho - s_z_t_minus_1[i]):
                # y_t[i] = z_t_minus_1[i] - alpha * (rho * (1-s_z_t_minus_1[i]) + noise[i])
                # print(z_t_minus_1[i])
                y_t[i] = z_t_minus_1[i] +  alpha * ((s_z_t_minus_1[i] - rho))
                # print(y_t[i])
            elif z_t_minus_1[i] < alpha * (-rho - s_z_t_minus_1[i]):
                # y_t[i] = z_t_minus_1[i] - alpha * (rho * (-1-s_z_t_minus_1[i]) + noise[i])
                y_t[i] = z_t_minus_1[i] +  alpha * ((s_z_t_minus_1[i] - rho))
                # print("b")
            else:
                # print("c")
                y_t[i] = 0
            # print(y_t[i])
        # for i in range(n):
        #     if z_t_minus_1[i] >= 0:
        #         y_t[i] = z_t_minus_1[i] - alpha * (rho * (1-s_z_t_minus_1[i]))
        #     else:
        #         y_t[i] = z_t_minus_1[i] - alpha * (rho * (-1-s_z_t_minus_1[i]))
        print(y_t)
        # print(len(y_t))

        # z_tの更新
        # z_t_minus_1を，T @ X >= (1-y_t[i]) * xi[i]を満たすならばy_t[i]の値に，満たさないならばT @ X >= (1-y_t[i]) * xi[i]を満たすような値に更新
        z_t = np.copy(z_t_minus_1)
        for i in range(n):
            if np.all(T @ X_t >= (1-y_t[i]) * xi[i]):
                z_t[i] = y_t[i]
            else:
                # print("False")
                if y_t[i] < 0:
                    z_t[i] = 0
                elif y_t[i] > 1:
                    z_t[i] = 1
                else:
                    z_t[i] = y_t[i]
        print("z_t:", z_t)
        # print(z_t==z_t_minus_1)
      

        # return X_t, z_t, f_t
        f_t_minus_1 = f_t
        z_t_minus_1 = z_t

    print("最適解x:", X_t) 
    print("最適解z:", z_t)
    print("最適値f：", f_t)
    end = time.time()
    print("DCA処理時間：", end-start, "秒")
    # 計算時間をファイルに書き込む
    with open("time_pdca_0612.txt", mode="a") as f:
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
for i in tqdm(range(1)):
    np.random.seed(i)
    z_t_minus_1 = create_random_array(n, 0.1)
    # print(z_t_minus_1)
    # z_t_minus_1 = np.random.randint(0, 2, n)
    dca(T=T, C=C, M=M, xi=xi, rho=10000, epsilon=0.1, z_t_minus_1=z_t_minus_1, varepsilon=1, max_iteration=100, alpha=1e-6)


# 初めのepsilon%の要素が1で残りが0の配列を作成
# epsilon = 0.1
# num_ones = int(n * epsilon)
# num_zeros = n - num_ones

# z_t_minus_1 = np.concatenate((np.ones(num_ones), np.zeros(num_zeros)))
# dca(T=T, C=C, M=M, xi=xi, rho=10000, epsilon=0.1, z_t_minus_1=z_t_minus_1, varepsilon=1, max_iteration=100)