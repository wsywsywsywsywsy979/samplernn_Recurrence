import torch
from torch.autograd import Variable
import heapq
# 基于torch.utils.trainer.Trainer代码 
# 允许对模型进行多个输入，不需要全部为张量。
class Trainer(object):
    def __init__(self, model, criterion, optimizer, dataset, cuda=False):
        self.model = model
        self.criterion = criterion # 损失函数
        self.optimizer = optimizer # 优化器
        self.dataset = dataset # 数据集
        self.cuda = cuda
        self.iterations = 0
        self.epochs = 0
        self.stats = {}
        self.plugin_queues = { # 感觉代码中实现的思想就是想用pytorch_lightening
            'iteration': [],
            'epoch': [],
            'batch': [],
            'update': [],
        }

    def register_plugin(self, plugin):
        plugin.register(self)

        intervals = plugin.trigger_interval # 这个成员变量应该是从原始的父类中继承的
        # intervals= 10 # wsy：根据对下方代码的理解，这个变量应该不仅仅是一个数值，所以不可如此简单赋值，所以下方全部去掉
        if not isinstance(intervals, list):
            intervals = [intervals]
        for (duration, unit) in intervals:
            queue = self.plugin_queues[unit]
            queue.append((duration, len(queue), plugin))

    def call_plugins(self, queue_name, time, *args):
        args = (time,) + args
        queue = self.plugin_queues[queue_name]
        if len(queue) == 0:
            return
        while queue[0][0] <= time:
            plugin = queue[0][2] # TrainingLossMonitor对象
            getattr(plugin, queue_name)(*args)
            for trigger in plugin.trigger_interval:
                if trigger[1] == queue_name:
                    interval = trigger[0]
            new_item = (time + interval, queue[0][1], plugin)
            heapq.heappushpop(queue, new_item) # queue是list类型，先进行heappush()（压入到list尾部），再进行heappop()（取list[0]）

    def run(self, epochs=1):
        for q in self.plugin_queues.values(): # 因为self.plugin_queues.values()是空的
            heapq.heapify(q) # 以线性时间讲一个列表转化为小根堆，因为定义的events优先级是数值越小优先级越高
        import os
        for self.epochs in range(self.epochs + 1, self.epochs + epochs + 1):
            print("epoch:",self.epochs) # wsy add
            self.train()
            self.call_plugins('epoch', self.epochs)
            #--------wsy add :后续调试发现源码中已经有随时观测和保存最小损失的模型了
            # if (self.epochs+1)%2==0: # 需修改数值
            #     path=os.path.join("/home/wangshiyao/wangshiyao_space/exp4/samplernn-pytorch-master/results",str(self.epochs+1))
            #     torch.save(self.model,path)

    def train(self):
        for (self.iterations, data) in \
                enumerate(self.dataset, self.iterations + 1):
            batch_inputs = data[: -1] # 查看后感觉应该是单样本进行训练
            batch_target = data[-1] # 这个应该类似y的作用
            # self.call_plugins( # 这个没有定义事件，所以注释掉
            #     'batch', self.iterations, batch_inputs, batch_target
            # )

            def wrap(input):
                if torch.is_tensor(input):
                    input = Variable(input)
                    if self.cuda:
                        input = input.cuda()
                return input
            batch_inputs = list(map(wrap, batch_inputs))

            batch_target = Variable(batch_target)
            if self.cuda:
                batch_target = batch_target.cuda()

            plugin_data = [None, None]

            def closure():
                batch_output = self.model(*batch_inputs)

                loss = self.criterion(batch_output, batch_target)
                loss.backward()

                if plugin_data[0] is None:
                    plugin_data[0] = batch_output.data
                    plugin_data[1] = loss.data

                return loss

            self.optimizer.zero_grad()
            self.optimizer.step(closure)
            self.call_plugins(
                'iteration', self.iterations, batch_inputs, batch_target,
                *plugin_data
            )
            self.call_plugins('update', self.iterations, self.model)
