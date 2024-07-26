import gurobipy as gp
from gurobipy import GRB

# モデルの作成
model = gp.Model("example_model")

# 変数の作成
x = model.addVar(name="x")
y = model.addVar(name="y")

# 目的関数の設定
model.setObjective(x + y, GRB.MAXIMIZE)

# 制約条件の追加
constraint1 = model.addConstr(x + y <= 10, name="constraint1")
constraint2 = model.addConstr(2*x - y >= 0, name="constraint2")
num_constraints = model.NumConstrs
print("Number of constraints:", num_constraints)

# モデルの更新
model.update()

# 制約式のprint
for constr in model.getConstrs():
    print(constr)

# 最適化の実行
model.optimize()

# 最適解の表示
if model.status == GRB.OPTIMAL:
    print("最適解が見つかりました:")
    print("x =", x.x)
    print("y =", y.x)
else:
    print("最適解が見つかりませんでした。")
