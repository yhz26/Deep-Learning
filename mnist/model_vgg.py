import torch
import torch.nn as nn


class MyVGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(MyVGG16, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # 输入1通道，输出64通道
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)  # 池化层

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)

        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1024),  # 输入尺寸应为 256 * 3 * 3
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.maxpool1(x)

        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.maxpool2(x)

        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        x = self.relu7(self.conv7(x))
        x = self.maxpool3(x)

        # 展平卷积层输出 (batch_size, 256, 3, 3) -> (batch_size, 256*3*3)
        x = x.reshape(x.shape[0], -1)

        # 通过全连接层进行分类
        x = self.classifier(x)

        return x


