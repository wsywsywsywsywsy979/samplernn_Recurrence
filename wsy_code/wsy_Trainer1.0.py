import torch
#------wsy add-------------
train_data=""
val_data=""
#--------------------------
class Trainer(object):
    def __init__(self, model, criterion, optimizer, dataset, epochs, use_cuda = False):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataset = dataset
        self.use_cuda = use_cuda
        self.epochs = epochs
        self.iteration = 0
        
        if self.use_cuda:
            self.model = self.model.cuda()
    
    def run(self):
        for i in range(self.epochs):
            self.train()

    def train(self):
        self.model.train()
        for i, (x, y) in enumerate(self.dataset, self.iteration + 1):
            if self.use_cuda:
                x = x.cuda()
                y = y.cuda()
            # 每次前向传播过程就是一个函数闭包操作 !!!
            def closure():
                out = self.model(x)
                loss = self.criterion(out, y)
                loss.backward()
                return loss
            self.optimizer.zero_grad()
            self.optimizer.step(closure)
        self.iteration += i

    def evaluation(self):
        self.model.eval()
        with torch.no_grad(): # 或者@torch.no_grad() 被他们包裹的代码块不需要计算梯度， 也不需要反向传播
            eval_loss = 0
            eval_acc = 0
            for i, (x, y) in enumerate(self.dataset, self.iteration + 1):
                if self.use_cuda:
                    x = x.cuda()
                    y = y.cuda()
                out = self.model(x)
                loss = self.criterion(out, y)
                eval_loss += loss.item()
                prediction = torch.max(out, 1)[1]
                pred_correct = (prediction == y).sum()
                eval_acc += pred_correct.item()
            print('evaluation loss : {:.6f}, acc : {:.6f}'.format(eval_loss / len(val_data), eval_acc / len(val_data)))