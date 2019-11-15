import numpy as np
import matplotlib.pyplot as plt

np.random.seed(125)
sample_num = 1000

x1 = np.random.multivariate_normal([0, 1], [[1, 0.5], [0.5, 1]], sample_num)
x2 = np.random.multivariate_normal([4, 7], [[1.7, 1.5], [1.5, 1.7]], sample_num)

real_xdata = np.vstack((x1, x2)).astype(np.float32)
real_label = np.hstack((np.zeros(sample_num), np.ones(sample_num)))

plt.figure(figsize=(10, 6))
plt.scatter(real_xdata[:, 0], real_xdata[:, 1],
            c=real_label, alpha=.4)
plt.show()


def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))


def logistic_regression(datax, label, num_steps, learning_rate):
    real_datax = np.mat(np.insert(datax, 0, 1, axis=1))
    real_label = np.mat(label).transpose()
    params = np.ones((np.shape(real_datax)[1], 1))
    plt.ion()
    fig, ax = plt.subplots()
    plt.rcParams['lines.markersize'] = 3
    for step in range(num_steps):
        scores = real_datax * params
        predictions = sigmoid(scores)
        params = params + learning_rate * real_datax.transpose() * (real_label - predictions)
        x1_min = np.min(datax[:, 0])
        x1_max = np.max(datax[:, 0])
        x2_min = np.min(datax[:, 1])
        x2_max = np.max(datax[:, 1])
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)

        x_line = np.linspace(x1_min, x1_max, 1000)
        y_line = (-params[0, 0] - params[1, 0] * x_line) / params[2, 0]
        plt.plot(x_line, y_line)
        plt.title(str(step) + ' iterations', fontsize='xx-large')
        plt.scatter(real_xdata[:, 0], real_xdata[:, 1],
                    c=label, alpha=.4)
        plt.pause(0.4)
        ax.cla()
    return params


if __name__ == "__main__":
    logistic_regression(real_xdata, real_label, 100, 0.01)
