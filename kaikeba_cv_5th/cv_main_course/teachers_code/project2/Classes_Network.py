import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3)
        #nn.Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True))
        # 参数：
        #   in_channel:输入数据的通道数，例RGB图片通道数为3；
        #   out_channel: 输出数据的通道数，这个根据模型调整；
        #   kennel_size: 卷积核大小，可以是int，或tuple；kennel_size=2,意味着卷积大小2， kennel_size=（2,3），意味着卷积在第一维度大小为2，在第二维度大小为3；
        #   stride：步长，默认为1，与kennel_size类似，stride=2,意味在所有维度步长为2， stride=（2,3），意味着在第一维度步长为2，意味着在第二维度步长为3；
        #   padding： 零填充
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)  #pooling 放在relu之前，和放在之后效果差距不大，但可以大大减小下一步的计算量
        self.relu1 = nn.ReLU(inplace=True)
        # inplace=True对从上层网络Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存，不用多存储其他变量
        # inplace=False不改变输入，生成新的变量存储输出
        self.conv2 = nn.Conv2d(3, 6, 3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(6 * 123 * 123, 150)
        # class torch.nn.Linear（in_features，out_features，bias = True ）
        # in_features - 每个输入样本的大小
        # out_features - 每个输出样本的大小
        # bias - 如果设置为False，则图层不会学习附加偏差。默认值：True
        self.relu3 = nn.ReLU(inplace=True)

        self.drop = nn.Dropout2d()  #问题：什么作用？

        self.fc2 = nn.Linear(150, 2)
        self.softmax1 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.relu2(x)

        # print(x.shape)
        x = x.view(-1, 6 * 123 * 123)  # view函数相当于numpy的reshape
        x = self.fc1(x)
        x = self.relu3(x)

        x = F.dropout(x, training=self.training)  # training=self.training将模型整体的training状态参数传入dropout函数

        x_classes = self.fc2(x)
        x_classes = self.softmax1(x_classes)

        return x_classes