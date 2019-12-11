import os
import matplotlib.pyplot as plt
from torch.utils.data import  DataLoader
import torch
from Classes_Network import *
from torchvision.transforms import transforms
from PIL import Image
import pandas as pd
import random
from torch import optim
from torch.optim import lr_scheduler
import copy

ROOT_DIR = '../Dataset/'
TRAIN_DIR = 'train/'
VAL_DIR = 'val/'
TRAIN_ANNO = 'Classes_train_annotation.csv'
VAL_ANNO = 'Classes_val_annotation.csv'
CLASSES = ['Mammals', 'Birds']

class MyDataset():

    def __init__(self, root_dir, annotations_file, transform=None):

        self.root_dir = root_dir
        self.annotations_file = annotations_file
        self.transform = transform

        if not os.path.isfile(self.annotations_file):
            print(self.annotations_file + 'does not exist!')
        self.file_info = pd.read_csv(annotations_file, index_col=0)
        self.size = len(self.file_info)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = self.file_info['path'][idx]
        if not os.path.isfile(image_path):
            print(image_path + '  does not exist!')
            return None

        image = Image.open(image_path).convert('RGB')
        label_class = int(self.file_info.iloc[idx]['classes']) #iloc通过行号来取行数据

        sample = {'image': image, 'classes': label_class}
        if self.transform:
            sample['image'] = self.transform(image)
        return sample

train_transforms = transforms.Compose([transforms.Resize((500, 500)), # Resize the input PIL Image to the given size.
                                       transforms.RandomHorizontalFlip(),# Horizontally flip the given PIL Image randomly with a given probability.
                                       # 问题：为什么要随机翻转，并没有增广？数据量虽然没有增多，但加入干扰使模型鲁棒性更强
                                       transforms.ToTensor(), # Convert a PIL Image or numpy.ndarray to tensor.
                                       ])
# torchvision.transforms.Compose(transforms) Composes several transforms together.
val_transforms = transforms.Compose([transforms.Resize((500, 500)),
                                     transforms.ToTensor()
                                     ])

train_dataset = MyDataset(root_dir= ROOT_DIR + TRAIN_DIR,
                          annotations_file= TRAIN_ANNO,
                          transform=train_transforms)

test_dataset = MyDataset(root_dir= ROOT_DIR + VAL_DIR,
                         annotations_file= VAL_ANNO,
                         transform=val_transforms)

train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset=test_dataset)
data_loaders = {'train': train_loader, 'val': test_loader}
# DataLoader将自定义的Dataset根据batch size大小、是否shuffle等封装成一个Batch Size大小的Tensor，用于后面的训练
# pytorch 的数据加载到模型的操作顺序是这样的：
# ① 创建一个 Dataset 对象
# ② 创建一个 DataLoader 对象
# ③ 循环这个 DataLoader 对象，将img, label加载到模型中进行训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def visualize_dataset():
    print(len(train_dataset))
    idx = random.randint(0, len(train_dataset))
    sample = train_loader.dataset[idx]
    print(idx, sample['image'].shape, CLASSES[sample['classes']])
    img = sample['image']
    plt.imshow(transforms.ToPILImage()(img))
    plt.show()
visualize_dataset()

def train_model(model, criterion, optimizer, scheduler, num_epochs=50):
    Loss_list = {'train': [], 'val': []}
    Accuracy_list_classes = {'train': [], 'val': []}

    best_model_wts = copy.deepcopy(model.state_dict()) # state_dict(destination=None, prefix='', keep_vars=False) Returns a dictionary containing a whole state of the module.
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-*' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 启用 BatchNormalization 和 Dropout
            else:
                model.eval() # 不启用 BatchNormalization 和 Dropout

            running_loss = 0.0
            corrects_classes = 0

            for idx,data in enumerate(data_loaders[phase]):
                #print(phase+' processing: {}th batch.'.format(idx))
                inputs = data['image'].to(device)
                labels_classes = data['classes'].to(device)
                optimizer.zero_grad()  # 直接把模型的参数梯度设成0，如果不清零，那么使用的这个grad就得同上一个mini-batch有关

                with torch.set_grad_enabled(phase == 'train'):  # 当数据是训练集
                    x_classes = model(inputs)  # 问题：为什么不是调用forward方法？
                    x_classes = x_classes.view(-1, 2)
                    _, preds_classes = torch.max(x_classes, 1)  # 返回每一行中最大值的那个元素，且返回其索引
                    loss = criterion(x_classes, labels_classes)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)  # loss.item()做什么的？loss中有数值和梯度，item()取出数值
                # 为什么要乘以inputs.size(0)？ 因为criterion求loss时自动除以了batch数，为了下一步计算整体的loss，先把batch数乘回来，再除以数据集总样本数，就得到整体的loss
                corrects_classes += torch.sum(preds_classes == labels_classes)

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            Loss_list[phase].append(epoch_loss)

            epoch_acc_classes = corrects_classes.double() / len(data_loaders[phase].dataset)
            epoch_acc = epoch_acc_classes

            Accuracy_list_classes[phase].append(100 * epoch_acc_classes)
            print('{} Loss: {:.4f}  Acc_classes: {:.2%}'.format(phase, epoch_loss,epoch_acc_classes))

            if phase == 'val' and epoch_acc > best_acc:

                best_acc = epoch_acc_classes
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Best val classes Acc: {:.2%}'.format(best_acc))

    model.load_state_dict(best_model_wts)  #load_state_dict(state_dict, strict=True)
    # Copies parameters and buffers from state_dict into this module and its descendants. If strict is True, then the keys of state_dict must exactly match the keys returned by this module’s state_dict() function.
    torch.save(model.state_dict(), 'best_model.pt')  #输出模型文件
    #torch.save(obj, f, pickle_module=<module 'pickle' from '/scratch/rzou/pt/v1.3.1-docs-env/lib/python3.7/pickle.py'>, pickle_protocol=2)
    print('Best val classes Acc: {:.2%}'.format(best_acc))
    return model, Loss_list,Accuracy_list_classes

network = Net().to(device) # Moves and/or casts the parameters and buffers. to(device=None, dtype=None, non_blocking=False)
optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1) # Decay LR by a factor of 0.1 every 1 epochs
model, Loss_list, Accuracy_list_classes = train_model(network, criterion, optimizer, exp_lr_scheduler, num_epochs=100)

x = range(0, 100)
y1 = Loss_list["val"]
y2 = Loss_list["train"]

plt.plot(x, y1, color="r", linestyle="-", marker="o", linewidth=1, label="val")
plt.plot(x, y2, color="b", linestyle="-", marker="o", linewidth=1, label="train")
plt.legend()
plt.title('train and val loss vs. epoches')
plt.ylabel('loss')
plt.savefig("train and val loss vs epoches.jpg")
plt.close('all') # 关闭图 0

y5 = Accuracy_list_classes["train"]
y6 = Accuracy_list_classes["val"]
plt.plot(x, y5, color="r", linestyle="-", marker=".", linewidth=1, label="train")
plt.plot(x, y6, color="b", linestyle="-", marker=".", linewidth=1, label="val")
plt.legend()
plt.title('train and val Classes_acc vs. epoches')
plt.ylabel('Classes_accuracy')
plt.savefig("train and val Classes_acc vs epoches.jpg")
plt.close('all')

############################################ Visualization ###############################################
def visualize_model(model):
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loaders['val']):
            inputs = data['image']
            labels_classes = data['classes'].to(device)

            x_classes = model(inputs.to(device))
            x_classes=x_classes.view( -1,2)
            _, preds_classes = torch.max(x_classes, 1)

            print(inputs.shape)
            plt.imshow(transforms.ToPILImage()(inputs.squeeze(0)))  #inputs.squeeze(0)作用：由于测试集是按batch为1加载的，所以data是一个1*3*500*500的4维数据，去掉第0维才能变成一张3通道的图片
            plt.title('predicted classes: {}\n ground-truth classes:{}'.format(CLASSES[preds_classes],CLASSES[labels_classes]))
            plt.show()

visualize_model(model)