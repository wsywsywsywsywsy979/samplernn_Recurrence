#######################较为简略的训练code##############################################
import torch
from torch.utils.data import DataLoader
#------wsy add----------------
from torchvision import models
train_data=""
val_data=""
#------------------------------

# 数据集加载器
train_loader = DataLoader(dataset = train_data, batch_size = 128, shuffle = True) # dataset (Dataset): dataset from which to load the data.
val_loader = DataLoader(dataset = val_data, batch_size = 1, shuffle = False)
# 定义模型
# model = Net()
model=models.vgg16() # wst fix
# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
# 定义损失函数
criterion = torch.nn.CrossEntropyLoss()
# 定义训练epoch的次数
epochs = 500
# 判断设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 开始训练
for epoch in range(epochs):
    print('epoch {}'.format(epoch + 1))
    train_loss = 0
    train_acc = 0
    # 训练
    model.train()
    for i, (x, y) in enumerate(train_loader): # dataloader的使用！！！
        x = x.to(device)
        y = y.to(device)
        model = model.to(device)
        out = model(x)
        loss = criterion(out, y) # 预测和输出通过损失函数，所以应该是监督学习
        train_loss += loss.item()
        prediction = torch.max(out,1)[1] # 取概率最大的预测，所以最后一层应该是softmax层
        pred_correct = (prediction == y).sum()
        train_acc += pred_correct.item()
        optimizer.zero_grad() # 将梯度清零
        loss.backward()
        optimizer.step()
    print('train loss : {:.6f}, acc : {:.6f}'.format(train_loss / len(train_data), train_acc / len(train_data)))
    # 验证：防止过拟合
    model.eval()
    with torch.no_grad(): # 或者@torch.no_grad() 被他们包裹的代码块不需要计算梯度， 也不需要反向传播
        eval_loss = 0
        eval_acc = 0
        for i, (x, y) in enumerate(val_loader):
            out = model(x)
            loss = criterion(out, y)
            eval_loss += loss.item()
            prediction = torch.max(out, 1)[1]
            pred_correct = (prediction == y).sum()
            eval_acc += pred_correct.item()
        print('evaluation loss : {:.6f}, acc : {:.6f}'.format(eval_loss / len(val_data), eval_acc / len(val_data)))

