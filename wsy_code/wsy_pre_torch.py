import torch
import torch.nn as nn
# import torch.functional as F
import torch.nn.functional as F # 这个才有log_softmax函数
class FCN(torch.nn.Module):
    # def __init__(self, data_dir, hidden_size, learning_rate):
    def __init__(self, hidden_size):
        super().__init__()
        # 将初始化参数设置为类属性
        # self.data_dir = data_dir
        self.hidden_size = hidden_size
        # self.learning_rate = learning_rate
        # 硬编码某些特定于数据集的属性
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        # 定义PyTorch模型
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes),
        )
    def forward(self, x):
        x = self.model(x) # 报错：Dimension out of range (expected to be in range of [-1, 0], but got 1)
        return F.log_softmax(x, dim=1)
