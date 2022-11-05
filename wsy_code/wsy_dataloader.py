# 文件已通过基本运行测试！
# 在Pytorch中，dataloader要实现的分为两部分Dataset与DataLoader
## Dataset（继承torch.utils.data.DataSet类，读取训练数据）
#---------wsy add------------------
import torchvision
import torch
PATH_DATASETS="/home/wangshiyao/wangshiyao_space/exp4/samplernn-pytorch-master/wsy_code/datasets"
# transforms=torch.nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, 
#                                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, 
#                                 activation='relu', custom_encoder=None, custom_decoder=None)
from torchvision import transforms 
from torch.utils.data import DataLoader
BATCH_SIZE=256
import struct
import numpy as np
import array
def loadMNISTImages(file_name):
    """
    从提供的文件名加载图像
    """
    # 打开文件：
    image_file = open(file_name, 'rb')
    # 从文件中读取标头信息：
    head1 = image_file.read(4)
    head2 = image_file.read(4)
    head3 = image_file.read(4)
    head4 = image_file.read(4)
    # 格式化有用数据的标题信息:
    num_examples = struct.unpack('>I', head2)[0]
    num_rows     = struct.unpack('>I', head3)[0]
    num_cols     = struct.unpack('>I', head4)[0]
    # 将数据集初始化为零数组：
    dataset = np.zeros((num_rows*num_cols, num_examples))
    # 读取实际图像数据：
    images_raw  = array.array('B', image_file.read())
    image_file.close()
    # 按列排列数据:
    for i in range(num_examples):
        limit1 = num_rows * num_cols * i
        limit2 = num_rows * num_cols * (i + 1)
        dataset[:, i] = images_raw[limit1 : limit2]
    # 规范化并返回数据集:
    return dataset / 255
def loadMNISTLabels(file_name):
    """
    从提供的文件名加载图像标签:
    """
    # 打开文件：
    label_file = open(file_name, 'rb')
    # 从文件中读取标头信息：
    head1 = label_file.read(4)
    head2 = label_file.read(4)
    # 格式化有用数据的标题信息:
    num_examples = struct.unpack('>I', head2)[0]
    # 将数据集初始化为零数组：
    labels = np.zeros((num_examples, 1), dtype = np.int)
    # 读取标签数据：
    labels_raw = array.array('b', label_file.read())
    label_file.close()
    # 复制并返回标签数据：
    labels[:, 0] = labels_raw[:]
    return labels

#---------------------------------
### 常见数据集可以通过torchvision直接获取 # 已成功下载
# train_data = torchvision.datasets.MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
# test_data = torchvision.datasets.MNIST(PATH_DATASETS, train=False, download=True, transform=transforms.ToTensor())
### 自己的数据集
# class MyDataset(torch.utils.data.DataSet): 
class MyDataset(torch.utils.data.Dataset):  # wsy fix
    # def __init__(self, stage ...): 
    def __init__(self, stage,batch_size):  # TypeError: __init__() should return None, not 'tuple'
        #----------wsy add--------------------
        if stage=='train':
            self.data=loadMNISTImages("/home/wangshiyao/wangshiyao_space/exp4/samplernn-pytorch-master/wsy_code/datasets/MNIST/raw/train-images-idx3-ubyte")
            self.labels=loadMNISTLabels("/home/wangshiyao/wangshiyao_space/exp4/samplernn-pytorch-master/wsy_code/datasets/MNIST/raw/train-labels-idx1-ubyte")
            # return train_data,train_labels
            self.length=self.data.shape[1] # 设置样本数量为数据集的length
        elif stage=='test':
            self.data   = loadMNISTImages(r'/home/wangshiyao/wangshiyao_space/exp4/samplernn-pytorch-master/wsy_code/datasets/MNIST/raw/t10k-images-idx3-ubyte') 
            self.labels = loadMNISTLabels(r'/home/wangshiyao/wangshiyao_space/exp4/samplernn-pytorch-master/wsy_code/datasets/MNIST/raw/t10k-labels-idx1-ubyte')
            # return test_data,test_labels
            self.length=self.data.shape[1]
        # self.data=self.data.T # 感觉应该是每一列是一个样本来输入到网络中
        self.labels=self.labels.T
        self.batch_size=batch_size
        # self.idx=0 # 记录当前取到的样本索引
        #-------------------------------------
    def __len__(self):  # 因为此处没有定义
        return self.length # 60000

    def __getitem__(self,idx): # 此处需要自己定义如何获取数据，注意参数不可变
        # idx+=self.batch_size idx是以1位增量来迭代的 注意要返回for后面对应的变量数
        # i=idx*self.batch_size
        # x=self.data[i:i+self.batch_size,:] # 还是要每一步都看看
        # y=self.labels[i:i+self.batch_size,:]
        # x=self.data[:,i:i+self.batch_size]  # 每一列为一个样本
        # y=self.labels[:,i:i+self.batch_size]
        x=self.data[:,idx] # 改为每次只迭代一个样本
        y=self.labels[:,idx]
        return x,y
# train_data = MyDataset(stage = 'train', ...)
train_data_and_labels= MyDataset(stage = 'train',batch_size=64) # wsy fix
# val_data = MyDataset(satge = 'val', ...)
# val_data = MyDataset(stage = 'val')
#--------wsy add------------
test_data_and_labels=MyDataset(stage='test',batch_size=1)
#-----------------------------

# DataLoader & 实现LightningDataModule
## 使用Pytorch时，根据训练集、测试集等构建不同的DataLoader实例
train_loader = DataLoader(train_data_and_labels, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data_and_labels, batch_size=BATCH_SIZE)


