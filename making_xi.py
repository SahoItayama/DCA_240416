import numpy as np

np.random.seed(42)
xi = []
for _ in range(5):
    mean = np.random.uniform(0, 10, 5)  # 平均ベクトルをランダムに生成
    variance = np.random.uniform(1, 5, 5)  # 分散ベクトルをランダムに生成
    consumer_demand = np.abs(np.random.normal(mean, np.sqrt(variance), 5))  # 正規分布に従うランダムベクトルを生成
    xi.append(consumer_demand)

print(xi)

xi = np.array(xi)
norm_xi = []
for i in range(len(xi)):
    print(np.linalg.norm(xi[i]))
    norm_xi.append(np.linalg.norm(xi[i]))
# print(np.asarray(norm_xi))
# print(norm_xi[312])
# print(np.argsort(-np.asarray(norm_xi)))
h = xi[np.argsort(-np.asarray(norm_xi))]
print("h:", h)
print(h[0])


