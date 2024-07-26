import gurobipy as gp
from gurobipy import GRB
import numpy as np
from itertools import combinations

# 供給者数
K = 40
# 消費者数
J = 100
# シナリオ数
n = 1000

# データの定義
c = np.random.uniform(10, 50, size=(K, J)).tolist  # コスト係数のリスト
T = np.ones((J, K)).tolist  # 行列T
# v_bound = ...  # vの境界条件
xi = []  # xiのリスト
for _ in range(n):
    mean = np.random.uniform(0, 10, J)  # 平均ベクトルをランダムに生成
    variance = np.random.uniform(1, 5, J)  # 分散ベクトルをランダムに生成
    consumer_demand = np.abs(np.random.normal(mean, np.sqrt(variance), J))  # 正規分布に従うランダムベクトルを生成
    xi.append(consumer_demand)
epsilon = 0.05  # 制約条件の定数ε
pi = 1 / n  # piを1/nに設定

model = gp.Model("PMIP_with_Star_Inequalities")

# 変数の定義
x = model.addVars(len(c), vtype=GRB.CONTINUOUS, name="x")
v = model.addVar(vtype=GRB.CONTINUOUS, name="v")
z = model.addVars(n, vtype=GRB.BINARY, name="z")

# 目的関数の設定
model.setObjective(gp.quicksum(c[i] * x[i] for i in range(len(c))), GRB.MINIMIZE)

# 制約条件の設定
# (3) Tx - v = 0
for i in range(len(T)):
    model.addConstr(gp.quicksum(T[i][j] * x[j] for j in range(len(x))) - v == 0, name=f"Tx_v_{i}")

# (4) v + xi_i * z_i >= xi_i
for i in range(n):
    model.addConstr(v + xi[i] * z[i] >= xi[i], name=f"v_xi_z_{i}")

# (5) Σpi_i * z_i <= ε
model.addConstr(gp.quicksum(pi * z[i] for i in range(n)) <= epsilon, name="pi_z_epsilon")

# スター不等式の追加
# 全ての部分集合Tに対して不等式を追加
for l in range(1, n + 1):
    for subset in combinations(range(n), l):
        sorted_subset = sorted(subset)
        star_inequality = v
        for j in range(l):
            t_j = sorted_subset[j]
            t_j1 = sorted_subset[j + 1] if j + 1 < l else n
            star_inequality += (xi[t_j] - (xi[t_j1] if t_j1 < n else 0)) * z[t_j]
        model.addConstr(star_inequality >= xi[sorted_subset[0]], name=f"star_inequality_{subset}")

# モデルの最適化
model.optimize()

# 結果の表示
if model.status == GRB.OPTIMAL:
    print("Optimal solution found")
    for v in model.getVars():
        print(f"{v.varName}: {v.x}")
else:
    print("No optimal solution found")