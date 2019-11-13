import numpy as np
import matplotlib.pyplot as plt
import random
import math

SIZE = 50
OUT = 20
X = np.linspace(0, 10, 50)
Y = [3 * i + 10 for i in X[:-OUT]]
Y = Y + [random.randint(0, int(i)) for i in X[-OUT:]]

RANDOM_X = np.array(X)
RANDOM_Y = np.array(Y)

fig = plt.figure()
axe = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axe.scatter(RANDOM_X, RANDOM_Y)
plt.show()

# 使用RANSAC算法估算模型
# 迭代最大次数，每次得到更好的估计会优化iters的数值
iters = 1000
# 数据和模型之间可接受的差值
sigma = 0.25
# 最好模型的参数估计和内点数目
best_a = 0
best_b = 0
pretotal = 0
# 希望的得到正确模型的概率
P = 0.99

plt.ion()
fig, ax = plt.subplots()
for i in range(iters):
    # 随机在数据中红选出两个点去求解模型
    sample_index = random.sample(range(SIZE), 2)
    x_1 = RANDOM_X[sample_index[0]]
    x_2 = RANDOM_X[sample_index[1]]
    y_1 = RANDOM_Y[sample_index[0]]
    y_2 = RANDOM_Y[sample_index[1]]

    # y = ax + b 求解出a，b
    a = (y_2 - y_1) / (x_2 - x_1)
    b = y_1 - a * x_1

    # 算出内点数目
    total_inlier = 0
    for index in range(SIZE):
        y_estimate = a * RANDOM_X[index] + b
        if abs(y_estimate - RANDOM_Y[index]) < sigma:
            total_inlier = total_inlier + 1

    # 判断当前的模型是否比之前估算的模型好
    if total_inlier > pretotal:
        iters = math.log(1 - P) / math.log(1 - pow(total_inlier / (SIZE * 2), 2))
        pretotal = total_inlier
        best_a = a
        best_b = b

    # 判断是否当前模型已经符合超过一定规模的点
    if total_inlier > SIZE * 0.8:
        break
    x_line = np.linspace(np.min(RANDOM_X), np.max(RANDOM_X), 1000)
    y_line = best_a * x_line + best_b
    ax.plot(x_line, y_line, c='r')
    plt.title(str(i) + ' iterations', fontsize='xx-large')
    ax.scatter(RANDOM_X, RANDOM_Y)
    plt.pause(5)
    ax.cla()
