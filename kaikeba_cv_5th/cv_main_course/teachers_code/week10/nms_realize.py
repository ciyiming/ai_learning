import numpy as np

def NMS(lists, thre):
    # lists是多个bbox的列表，列表的元素是bbox的对角顶点坐标和score
    # thre是判断其他bbox和最佳bbox交叠区域大小的阈值，大于阈值会被抑制掉
    if len(lists) == 0:
        return np.array([], dtype=np.int32)
    lists = np.array(lists)
    res = []
    x1, y1, x2, y2, score = [lists[:, i] for i in range(5)]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)  # 计算bbox面积
    # get sorted index in ascending order
    idxs = np.argsort(score)  # y = argsort(x)是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]  # 选取score最高的索引
        res.append(i)  # 增加到res列表中

        # 非极大值抑制，找到相交区域
        xmin = np.maximum(x1[i], x1[idxs[:last]])  # np.maximum：(X, Y, out=None) X 与 Y 逐位比较取其大者；接受的两个参数，可以长度不一致
        ymin = np.maximum(y1[i], y1[idxs[:last]])  # 将sorce最高bbox的第一个顶点与其他bbox第一个顶点比较，选取其中的最大值
        xmax = np.minimum(x2[i], x2[idxs[:last]])  # 将sorce最高bbox的第二个顶点与其他bbox第一个顶点比较，选取其中的最小值
        ymax = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xmax - xmin + 1)  # 排除小于0的项
        h = np.maximum(0, ymax - ymin + 1)
        inner_area = w * h  # 计算相交区域面积
        iou = inner_area / (area[i] + area[idxs[:last]] - inner_area)  # 计算iou
        idxs = np.delete(idxs, np.concatenate(([last], np.where(iou > thre)[0])))  # 从索引列表中删掉iou过大的索引

                                         # here "where" will return us a tuple
                                         # [0] means to extract array from a tuple
# np.where(condition, x, y) 满足条件(condition)，输出x，不满足输出y。
# np.where(condition) 只有条件 (condition)，没有x和y，则输出满足条件 (即非0) 元素的坐标 (等价于numpy.nonzero)。这里的坐标以tuple的形式给出，通常原数组有多少维，输出的tuple中就包含几个数组，分别对应符合条件元素的各维坐标。
# [0]把tuple转换成列表，tuple中每一项只有一个元素，选取每个中第0个元素就成了一个列表
# np.concatenate数组拼接，另外需要指定拼接的方向，默认是 axis = 0，也就是说对0轴的数组对象进行纵向的拼接（纵向的拼接沿着axis= 1方向）；注：一般axis = 0，就是对该轴向的数组进行操作，操作方向是另外一个轴，即axis=1。
# numpy.delete(arr,obj,axis=None)
    #arr: 输入向量
    #obj: 表明哪一个子向量应该被移除。可以为整数或一个int型的向量
    #axis: 表明删除哪个轴的子向量，若默认，则返回一个被拉平的向量

    return np.array(res, dtype=np.int32)

print(NMS([[10,10,20,20,1], [9,9,21,21,0.9], [11,11,19,19,0.8], [1,1,9,9,0.7]], 0.5))
