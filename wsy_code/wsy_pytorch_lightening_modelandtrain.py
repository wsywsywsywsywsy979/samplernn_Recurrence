# class FCN(pytorch_lightning.LightningModule):  
#     """
#     pytorch_lightning.LightningModule可以看作是对torch.nn.Module 的进一步封装
#     """
#     def __init__(self): # 实现模型
#         pass 
#     def training_step(self, batch, batch_idx): 
#         pass      
#     def configure_optimizers(self): 
#         pass

#     def forward(self): # 实现前向传播
#         pass

#------wsy add---------------------------------------------------------------------
import torch.functional as F
import pytorch_lightning
from wsy_pre_torch import FCN
import torch
PATH_DATASETS=""
# def Accuracy():
#     pass
from torchmetrics import Accuracy # pytorch有封装好的指标库
from wsy_pytorch_lightening_data import MNISTData
from trainer import * # 对于引入init中的类的方法
#-----------------------------------------------------------------------------------
class MNISTModel(pytorch_lightning.LightningModule):
    def __init__(self, data_dir=PATH_DATASETS, hidden_size=64, learning_rate=2e-4):
        super().__init__()
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.accuracy = Accuracy()
        # 定义PyTorch模型
        self.model = FCN(data_dir, hidden_size, learning_rate)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        重写training_step方法，定义训练的每一个step中进行的运算
        相比Pytorch，这里不需要设置model.train() optimizer.step() 等操作，Pytorch Lightning框架会自动调用
        最后将计算后的loss作为返回值
        """
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        如果需要在每个epoch后进行验证，可以重写validation_step方法
        """
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)
        return loss 

    def configure_optimizers(self): # 这个函数必须重写，因为优化器不能少
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer # 直接定义一个优化器返回

# 开始训练
# 目前有两个包装好的类，分别为模型和数据
    # 模型：包括了将Pytorch模型实例化以及具体的训练/验证/测试步骤
    # 数据：包括训练/验证/测试集划分及各自的DataLoader
def train():
    model = MNISTModel()
    data = MNISTData()
    trainer = Trainer()
    trainer.fit(model, data)
    # 设置最大迭代次数为一个比较大的100，使用early stopping来在收敛结束后自动停止训练
    # early_stopping = pytorch_lightning.callbacks.EarlyStopping('val_acc')
    # model = MNISTModel()
    # data = MNISTData()
    # trainer = Trainer(
    #     gpus=1,
    #     max_epochs=100,
    #     progress_bar_refresh_rate=10,
    #     callbacks=[early_stopping]
    # )
    # trainer.fit(model, data)

if __name__ == "__main__":
    train()
