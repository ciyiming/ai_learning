import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, structure, learning_rate, reg_lambda, iteration, threshold):
        self.structure = structure
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.iteration = iteration
        self.threshold = threshold
        self.layer = len(self.structure)
        model = {}
        np.random.seed(0)
        for i in range(self.layer - 1):
            model['W%s' % (i+1)] = np.random.randn(self.structure[i], self.structure[i+1]) / np.sqrt(self.structure[i])
            model['b%s' % (i+1)] = np.zeros((1, self.structure[i+1]))
        print('Model initialized as:\n', model)
        self.model = model

    def forward_propagation(self, dataset):
        fp_params = {'a1': dataset}
        for i in range(1, self.layer):
            fp_params['z%s' % (i+1)] = fp_params['a%s' % i].dot(self.model['W%s' % i]) + self.model['b%s' % i]
            fp_params['a%s' % (i+1)] = np.tanh(fp_params['z%s' % (i+1)])
        exp_scores = np.exp(fp_params['z%s' % self.layer])
        fp_params['a%s' % self.layer] = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return fp_params

    def backward_propagation(self, examples, labels):
        bp_params, derivates = {}, {}
        num_examples = len(examples)
        fp_params = self.forward_propagation(examples)
        last_delta = fp_params['a%s' % self.layer]
        last_delta[range(num_examples), labels] -= 1
        bp_params['delta%s' % self.layer] = last_delta
        for i in range(self.layer - 1, 0, -1):
            derivates['dW%s' % i] = fp_params['a%s' % i].T.dot(bp_params['delta%s' % (i+1)]) \
                                     + self.reg_lambda * self.model['W%s' % i]
            derivates['db%s' % i] = np.sum(bp_params['delta%s' % (i+1)], axis=0, keepdims=True)
            if i > 1:
                bp_params['delta%s' % i] = bp_params['delta%s' % (i+1)].dot(self.model['W%s' % i].T) \
                                                   * (1 - np.power(fp_params['a%s' % i], 2))
        return derivates

    def eval_loss(self, examples, labels):
        num_examples = len(examples)
        probs = self.forward_propagation(examples)['a%s' % self.layer]
        onehot_labels = np.zeros((probs.shape[0], probs.shape[1]))
        for i in range(len(labels)):
            onehot_labels[i][labels[i]] = 1
        corect_logprobs = -onehot_labels * np.log(probs)
        data_loss = np.sum(corect_logprobs)
        for p in self.model:
            if p[0] == 'W':
                data_loss += self.reg_lambda / 2 * np.sum(np.square(self.model[p]))
        return 1. / num_examples * data_loss

    def train_model(self, examples, labels, print_loss=True):
        print('Start training......')
        for i in range(self.iteration):
            derivates = self.backward_propagation(examples, labels)
            for j in range(1, self.layer):
                self.model['W%s' % j] -= self.learning_rate * derivates['dW%s' % j]
                self.model['b%s' % j] -= self.learning_rate * derivates['db%s' % j]
            if print_loss and i % 1000 == 0:
                loss = self.eval_loss(examples, labels)
                print("Loss after iteration %i: %f" % (i, loss))
                if loss < self.threshold:
                    break
        print('End training')
        return self.model

    def predict(self, data):
        probs = self.forward_propagation(data)['a%s' % self.layer]
        return np.argmax(probs, axis=1)


def feature_count(examples):
    return len(examples[0])


def class_count(labels):
    return len(np.unique(labels))


if __name__ == '__main__':
    X, y = sklearn.datasets.make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0,
                                                n_repeated=0, n_classes=3, n_clusters_per_class=1)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()
    nn_in_dim = feature_count(X)
    nn_out_dim = class_count(y)
    hidden_layers = input("Please input hidden layers' dim separate by whitespace: (eg. 3 3)")
    nn_structure = [nn_in_dim] + list(map(int, hidden_layers.split())) + [nn_out_dim]
    nn_lr = float(input("Please input learning rate: (eg. 0.01)"))
    nn_reg_lmbd = float(input("Please input regularization lambda: (eg. 0.1)"))
    nn_train_iter = int(input("Please input training iteration: (eg. 10000)"))
    nn_threshold = float(input("Please input threshold: (eg. 0.01)"))
    nn = NeuralNetwork(nn_structure, nn_lr, nn_reg_lmbd, nn_train_iter, nn_threshold)
    nn_model = nn.train_model(X, y)
    print('Find model:\n', nn_model)
