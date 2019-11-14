import numpy as np
import sklearn.datasets
import sklearn.linear_model
import matplotlib.pyplot as plt
import time
np.random.seed(777)
X, y = sklearn.datasets.make_moons(500, noise=0.1)
trainset_length = len(X)
nn_input_dim = 2
nn_output_dim = 2
lr = 0.01
# reg_lambda = 0.01

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()


def build_model(nn_hidden_dim, num_passes=200, print_loss=False):
    W1 = np.random.randn(nn_input_dim, nn_hidden_dim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hidden_dim))
    W2 = np.random.randn(nn_hidden_dim, nn_output_dim) / np.sqrt(nn_hidden_dim)
    b2 = np.zeros((1, nn_output_dim))

    model = {}

    for i in range(num_passes):
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        delta3 = probs
        delta3[range(trainset_length), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        if print_loss and i % 10 == 0:
            print('Loss after iteration {}:{}'.format(i, calculate_loss(model)))
        time.sleep(0.2)
    return model


def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)

    z2 = a1.dot(W2) + b2

    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    log_probs = -np.log(probs[range(trainset_length), y])
    loss = np.sum(log_probs)

    return 1. / trainset_length * loss


if __name__ == '__main__':
    model = build_model(10,num_passes=1000,print_loss=True)
