#----wsy add------------------------
import pytorch_lightning 
PATH_DATASETS=""
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
BATCH_SIZE=16
#-----------------------------------
class MNISTData(pytorch_lightning.LightningDataModule):
    def __init__(self):
      self.data_dir = PATH_DATASETS
      self.transform = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),])

    def prepare_data(self):
        # 下载数据集
        torchvision.datasets.MNIST(self.data_dir, train=True, download=True)
        torchvision.datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = torchvision.datasets.MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = torchvision.datasets.MNIST(self.data_dir, train=False, transform=self.transform)
        
    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=BATCH_SIZE)
    
    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=BATCH_SIZE)

