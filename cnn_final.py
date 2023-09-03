import torchvision
import torchvision.transforms as transforms
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class Net(nn.Module):
    """define the CNN model"""
    conv_channel = 164  # 定义卷积层的输出通道数
    def __init__(self, classes=10):
        """定义初始化函数，classes为分类数，默认为10"""
        super(Net, self).__init__()  # 调用nn.Module的初始化函数
        self.classes = classes  # 将类别数保存在self.classes中
        # 定义第一个卷积层，输入通道数为1，输出通道数为16，卷积核大小为5，步长为1，填充为2
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        # 定义第二个卷积层，输入通道数为32，输出通道数为164，卷积核大小为5，步长为1，填充为2
        self.conv2 = nn.Conv2d(16, self.conv_channel, kernel_size=5, stride=1, padding=2)
        # 定义一个最大池化层，池化核大小为2，步长为2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # # 定义全连接层，输入节点数为164*7*7，输出节点数为分类数
        self.fc = nn.Linear(self.conv_channel*7*7, self.classes)
    
    def forward(self, x):
        """前向传播函数"""
        # 对输入x进行第一次卷积，然后使用ReLU激活函数，再进行最大池化
        x = self.pool(nn.functional.relu(self.conv1(x)))
        # 对x进行第二次卷积，然后使用ReLU激活函数，再进行最大池化
        x = self.pool(nn.functional.relu(self.conv2(x)))
        # 将x展开成一维向量
        x = x.view(-1, self.conv_channel*7*7)
        # 对x使用ReLU激活函数，然后进行全连接层的计算
        x = self.fc((x))
        return x  # 返回输出结果




def data_loader():
    """load MNIST dataset\n
    对数据进行预处理:归一化、数据增强、打乱数据"""
    print('\033[1;32m[NOTICE]\033[0m Start loading dataset')
    # 定义数据转换，将图像数据转换为tensor，并将像素值归一化
    # 利用transform模块进行数据增强，包括随机水平翻转、随机竖直翻转、随机旋转、随机裁剪等
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    # 加载训练数据集，并对数据进行预处理
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
    # 定义训练数据集的数据加载器，batch_size为每个batch的样本数，shuffle为是否打乱数据顺序
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

    # 加载测试数据集，并对数据进行预处理
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
    # 定义测试数据集的数据加载器，batch_size为每个batch的样本数，shuffle为是否打乱数据顺序
    test_loader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)
    print('\033[1;32m[NOTICE]\033[0m Finish loading dataset')
    return train_loader, test_loader



def train(train_loader, net, criterion, optimizer, device):
    """train the model, print loss and accuracy every 200 mini-batches"""
    net.to(device)
    print('\033[1;32m[NOTICE]\033[0m Start training')
    # loop over the dataset twice
    for epoch in range(2):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # backward
            loss = criterion(outputs, labels)
            loss.backward()
            # update the parameters
            optimizer.step()
            running_loss += loss.item()
            if i % 200 == 199:
                print('\t[%d][%5d] loss: %.3f accuracy: %.3f' % (epoch+1, i + 1, running_loss / 100, 100 * correct / total))
                running_loss = 0.0
                correct = 0
                total = 0
    
    print('\033[1;32m[NOTICE]\033[0m Finish train')


def test(test_loader, net, device):
    """test the model, print accuracy of each class and total accuracy"""
    correct_hole = 0    # 记录整体的正确率
    correct_list = [0] * 10 # 记录每个类别的正确率
    hole_list = [0] * 10    # 记录每个类别的样本数
    total = 0           # 记录整体的样本数
    net.to(device)
    print('\033[1;32m[NOTICE]\033[0m Start testing')
    # no_grad()表示在测试过程中不需要计算梯度，以节省内存
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # 计算整体的正确率
            correct_hole += (predicted == labels).sum().item()
            for i in range(len(labels)):
                hole_list[labels[i]] += 1
                # 计算每个类别的正确率
                if predicted[i] == labels[i]:
                    correct_list[labels[i]] += 1
    print('\tAccuracy of the network on the 10000 test images: %.3f %%' % (100 * correct_hole / total))
    for i in range(10):
        print('\tAccuracy of %d: %.3f %%' % (i, 100 * correct_list[i] / hole_list[i]))
    print('\033[1;32m[NOTICE]\033[0m Finish test')

if __name__ == '__main__':
    # load dataset
    train_loader, test_loader = data_loader()
    # create model instance
    net = Net(classes=10)
    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0008)
    # bind device, use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # train and test
    train(train_loader, net, criterion, optimizer, device)
    test(test_loader, net, device)
