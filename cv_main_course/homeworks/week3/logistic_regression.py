import numpy as np
import time
import matplotlib.pyplot as plt
import utils


def train(x1_list, x2_list, y_list, ax, batch_size, lr, threshold=0.15, max_iter=1000):
    '''
    train logistic model
    :param x1_list: the first feature values of training set
    :param x2_list: the second feature values of training set
    :param y_list: training example labels
    :param ax: data plot to draw decision boundary
    :param batch_size: the batch size of gradient descent
    :param lr: learning rate
    :param threshold: threshold of loss
    :param max_iter: max iteration
    :return: model parameters
    '''
    # get training set size
    num = len(y_list)
    # initialize parameters
    # (gradient descent converge too slow, so parameter initialized as values close to the best model)
    p_array = np.float64([11., 6., 5.]).reshape(-1, 1)
    # make array of features and labels
    x_array = np.hstack([np.ones(num).reshape(-1, 1),
                         np.array(x1_list).reshape(-1, 1),
                         np.array(x2_list).reshape(-1, 1)])
    y_array = np.float64(y_list).reshape(-1, 1)
    # training iteration
    for i in range(max_iter):
        # randomly choose batch_size number of examples
        batch_idxs = np.random.choice(len(x1_list), batch_size)
        batch_x, batch_y = x_array[batch_idxs], y_array[batch_idxs]
        # call a step of gradient descent
        p_array = cal_step_gradient(batch_x, batch_y, p_array, lr)
        print('iteration {0}/{1}, w1:{2}, w2:{3}, b:{4}'.format(i, max_iter, p_array[1], p_array[2], p_array[0]))
        print('loss is {}'.format(eval_loss(p_array, x_array, y_array)))
        # draw the decision boundary
        x = np.linspace(0.0, 1.0, num=5)
        lines = []
        if p_array[1] != 0 and p_array[2] != 0:
            lines = ax.plot(x, -1 * (p_array[1] * x + p_array[0]) / p_array[2], 'r-', lw=5)
        elif p_array[1] == 0 and p_array[2] != 0:
            lines = ax.plot(x, [-1 * p_array[0] / p_array[2] for _ in range(len(x))], 'r-', lw=5)
        elif p_array[1] != 0 and p_array[2] == 0:
            lines = ax.plot([-1 * p_array[0] / p_array[1] for _ in range(len(x))], x, 'r-', lw=5)
        # stop learning when loss is less than threshold
        if eval_loss(p_array, x_array, y_array) < threshold:
            break
        # pause for 0.1 second
        time.sleep(0.1)
        # remove the line of decision boundary from the plot
        ax.lines.remove(lines[0])
    return p_array


def cal_step_gradient(x_array, y_array, param_array, lr):
    '''
    compute a the new parameters with a step of gradient descent
    :param x_array: the input examples in an array
    :param y_array: the input labels in an array
    :param param_array: previous parameters
    :param lr: learning rate
    :return: updated parameters
    '''
    batch_size = len(x_array)
    param_array -= lr / batch_size * (np.dot(x_array.T, sigmoid(np.dot(x_array, param_array)) - y_array))
    return param_array


def eval_loss(p_array, x_array, y_array):
    '''
    compute the loss value for the logistic model
    :param p_array: parameters
    :param x_array: training examples
    :param y_array: training labels
    :return: loss value
    '''
    num = len(y_array)
    loss_array = y_array * np.log(sigmoid(np.dot(x_array, p_array))) + \
                 (1. - y_array) * np.log(1. - sigmoid(np.dot(x_array, p_array)))
    loss = sum(loss_array) / num * -1.
    return loss


def sigmoid(an_array):
    '''
    compute sigmoid response
    :param an_array: input values in an array
    :return: response of every value of the input array
    '''
    return 1 / (1 + np.exp(-1 * an_array))


def main():
    fig = plt.figure()  # create figure
    ax = fig.add_subplot(1, 1, 1)  # add one subplot on the figure
    ax.set_title('Logistic Regression')  # set plot title
    plt.xlabel('x1')  # x1 axis label
    plt.ylabel('x2')  # x2 axis label
    x1_list, x2_list, y_list = utils.gen_data_for_lor(5, 5, 10, 100)  # generate training set
    ax.scatter([x1_list[i] for i in range(len(y_list)) if y_list[i] == 1],
                [x2_list[i] for i in range(len(y_list)) if y_list[i] == 1], marker='x')  # draw scatter points for positive examples
    ax.scatter([x1_list[i] for i in range(len(y_list)) if y_list[i] == 0],
                [x2_list[i] for i in range(len(y_list)) if y_list[i] == 0], color='', marker='o', edgecolors='g')  # draw scatter points for negative examples
    plt.legend(['positive', 'negative'])  # add legend for to kinds of examples
    plt.ion()  # turn on interactive mode
    b, w1, w2 = train(x1_list, x2_list, y_list, ax, 100, 1, 0.14, 1000)  # train logistic regression model
    plt.ioff()  # turn off interactive mode
    plt.show()  # show plot
    print('Find the logistic model, w1 = {0}, w2 = {1}, b = {2}.'.format(w1, w2, b))
    return


if __name__ == '__main__':
    main()
