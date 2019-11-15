import numpy as np
import time
import matplotlib.pyplot as plt
import utils


def train(x_list, y_list, ax, batch_size, lr, threshold=1000, max_iter=1000 ):
    '''
    :train model for given examples
    :param x_list: training examples
    :param y_list: training example labels
    :param ax: data plot to draw linear model
    :param batch_size: the batch size of gradient descent
    :param lr: learning rate
    :param threshold: threshold of loss
    :param max_iter: max iteration
    :return: model parameters
    '''
    w, b = 0, 0  # initialize parameters
    for i in range(max_iter):
        batch_idxs = np.random.choice(len(x_list), batch_size)  # randomly choose batch_size number of examples
        batch_x = [x_list[j] for j in batch_idxs]
        batch_y = [y_list[j] for j in batch_idxs]
        w, b = cal_step_gradient(batch_x, batch_y, w, b, lr)  # call a step of gradient descent
        print('iteration {0}/{1}, w:{2}, b:{3}'.format(i, max_iter, w, b))
        print('loss is {}'.format(eval_loss(w, b, x_list, y_list)))
        x = np.arange(1, 100)  # generate x axis values
        lines = ax.plot(x, w * x + b, 'r-', lw=5)  # draw the linear model computed by trained parameters
        if eval_loss(w, b, x_list, y_list) < threshold:  # stop learning when loss is less than threshold
            break
        time.sleep(0.1)  # pause for 0.1 second
        ax.lines.remove(lines[0])  # remove the line of linear model from the plot
    return w, b


def cal_step_gradient(x_list, y_list, w, b, lr):
    '''
    compute a the new parameters with a step of gradient descent
    :param x_list: the input examples
    :param y_list: the input labels
    :param w: previous parameter weight of the linear model
    :param b: previous parameter bias of the linear model
    :param lr: learning rate
    :return: parameter weight, parameter bias
    '''
    batch_size = len(x_list)
    b -= lr * sum(w * np.array(x_list) + b - np.array(y_list)) / batch_size
    w -= float(lr * np.dot(x_list, (w * np.array(x_list) + b - np.array(y_list)).reshape(-1, 1)) / batch_size)
    return w, b


def eval_loss(w, b, x_list, y_list):
    '''
    compute the loss value for the linear model
    :param w: parameter weight of the linear model
    :param b: parameter bias of the linear model
    :param x_list: the training examples
    :param y_list: labels of the training examples
    :return: loss value
    '''
    num = len(x_list)
    loss = sum((w * np.array(x_list) + b - np.array(y_list)) ** 2) / 2 / num
    return loss


def main():
    fig = plt.figure()  # create figure
    ax = fig.add_subplot(1, 1, 1)  # add one subplot on the figure
    ax.set_title('Linear Regression')  # set plot title
    plt.xlabel('x')  # x axis label
    plt.ylabel('y')  # y axis label
    x_list, y_list = utils.gen_data_for_lir(5, 35)  # generate training set
    ax.scatter(x_list, y_list)  # draw training set as scatter points
    plt.ion()  # turn on interactive mode
    w, b = train(x_list, y_list, ax, 100, 0.0010, 1000, 100000)  # train linear regression model
    plt.ioff()  # turn off interactive mode
    plt.show()  # show plot
    print('Find the linear model, w = {0}, b = {1}.'.format(w, b))
    return


if __name__ == '__main__':
    main()
