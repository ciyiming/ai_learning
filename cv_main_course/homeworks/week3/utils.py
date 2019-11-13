import random


def gen_data_for_lir(w, b, num=100):
    '''
    generate training data set for linear regression
    param w: a list of float of weight
    param b: a float of bias
    param num: number of the training examples
    return: a list of examples, a list of labels
    '''
    x_list = [random.randint(0, 100) * random.random() for _ in range(num)]
    y_list = [w * x + b + random.random() * random.randint(-1, 100) for x in x_list]
    return x_list, y_list


def gen_data_for_lor(w1, w2, b, num=100):
    '''
    generate training data set for logistic regression
    param w: a integer of weight
    param b: a integer of bias
    param num: number of the training examples
    return: a list of examples, a list of labels
    '''
    # generate positive examples
    positive_num = num // 2
    x1_list = [random.random() for _ in range(positive_num)]
    x2_list = [-1 * (b + w1 * x1) / w2 + random.random() for x1 in x1_list] if w2 > 0 \
              else [-1 * (b + w1 * x1) / w2 - random.random() for x1 in x1_list]
    y_list = [1 for _ in range(positive_num)]
    # generate negative examples
    negative_num = num - positive_num
    x1_list += [random.random() for _ in range(negative_num)]
    x2_list += [-1 * (b + w1 * x1) / w2 + random.random() for x1 in x1_list[positive_num:]] if w2 < 0 \
              else [-1 * (b + w1 * x1) / w2 - random.random() for x1 in x1_list[positive_num:]]
    y_list += [0 for _ in range(negative_num)]
    return x1_list, x2_list, y_list
