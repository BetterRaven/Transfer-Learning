from torch import nn
from function import ReverseLayerF


class Dense(nn.Module):
    # 全连接层
    def __init__(self, num_classes=1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1600, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        return self.fc(inputs)


class DaNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, 7),    # 输出为58*58
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     # 输出为29*29

            nn.Conv2d(16, 32, 5),   # 输出为25*25
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     # 输出为12*12

            nn.Conv2d(32, 64, 3),   # 输出为10*10
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     # 输出为5*5
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
        )

        # 领域鉴别器
        self.domain_discriminator = Dense(2)
        # 分类器
        self.classify = Dense(4)

    def forward(self, inputs, alpha=0):
        # 特征提取部分
        x = self.feature_extractor(inputs)
        feature = self.fc(x)

        # 计算反转梯度
        reversed_feature = ReverseLayerF.apply(feature, alpha)

        # 领域鉴别部分
        domain_output = self.domain_discriminator(reversed_feature)

        # 分类部分
        classify_output = self.classify(feature)

        return classify_output, domain_output
