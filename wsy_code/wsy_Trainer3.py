import heapq
import torch
#------wsy add-------------
queue=[]
import torch.nn.functional as F
#--------------------------
class Trainer(object):
    def __init__(self, model, criterion, optimizer, dataset, use_cuda=False):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataset = dataset 
        self.use_cuda = use_cuda
        #-----------wsy add--------
        if self.use_cuda:
            self.model.to('cuda')
            self.criterion.to('cuda') # Accracy也要放到cuda上
        #----------------------------------------
        self.iterations = 0 # 记录当前正在执行第几个个iteration
        self.epochs = 0
        # Trainer的状态，这里的状态包含了所有插件提供的状态
        self.stats = {}
        '''
        将插件的调用进行了分类:
            (1)iteration:一般是在完成一个batch 训练之后进行的事件调用序列（一般不改动网络或者优化器，如：计算准确率）调用序列；
            (2)batch 在进行batch 训练之前需要进行的事件调用序列
            (3)epoch 完成一个epoch 训练之后进行的事件调用序列
            (4)update 完成一个batch训练之后进行的事件(涉及到对网络或者优化器的改动,如:学习率的调整)
            注意，iteration 跟update 两种插件调用的时候传入的参数不一样,iteration 会传入batch output,loss 等训练过程中的数据,
            而update传入的的model ,方便对网络的修改 
        '''
        self.plugin_queues = { # 因为此处初始化之后就没有再增加修改过，所以下面注册时就都是空的
            'iteration': [],
            'epoch': [],
            'batch': [],
            'update': [],
        }
    
    # 注册插件：
    def register_plugin(self, plugin):
        plugin.register(self) # 这里将Trainer类的对象传递给plugin
        # 插件的触发间隔，一般形式为[(1, 'iteration'), (1, 'epoch')]
        intervals = plugin.trigger_interval
        if not isinstance(intervals, list): # 这里只是简单地封装到list中
            intervals = [intervals]
        for duration, unit in intervals:
            # unit是事件的触发类别
            event = self.plugin_queues[unit] # 当unit='iteration'和'epoch'时，event为空？
            # 添加事件
            '''
            duration：触发时间点，决定了之后比如在第几个iteration或epoch的时候触发事件. 这个值会在后面更新
            len(queue)可以理解为优先级（值越小，优先级越高）， 在相同duration的情况下，决定了调用的顺序
            '''
            event.append((duration, len(event), plugin))
    
    # 调用插件：
    def call_plugin(self, event_name, time, *args): # iteration：args[0]=x,args[1]=y, args[2]的shape和y一样不知道是啥，args[3]应该是loss
        # 这里的time指的是次数
        args = (time, ) + args
        event = self.plugin_queues[event_name]
        if len(event) == 0:
            return
        while event[0][0] <= time: # 事件队列的第一个事件的duration（即触发时间点）小于当前time的时候执行
            plugin = event[0][2] # 得到的是lossMonitor对象
            # 调用相关队列相应的方法，所以如果是继承Plugin类的插件，
            # 必须实现 iteration、batch、epoch和update中的至少一个且名字必须一致
            getattr(plugin, event_name)(*args)
            for trigger in plugin.trigger_interval:
                if trigger[1] == event_name:
                    interval = trigger[0]
            # 根据插件的事件触发间隔，来更新事件队列里的事件 duratio
            new_item = (time + interval, event[0][1], plugin)
            # 加入新的事件并弹出最小堆的堆头。最小堆重新排序
            heapq.heappushpop(queue, new_item)
            # event[0][0]=heapq.heappushpop(queue, new_item)[0] # wsy fix failed
            time-=1 # wsy add 如果不加，这里将一直死循环

    
    def run(self, epochs = 1):
        # 对四种事件的调用序列进行最小堆排序
        for q in self.plugin_queues.values():
            heapq.heapify(q)
        
        for i in range(1, epochs + 1): # epochs=100
            self.train()
            # 执行epoch的更新
            self.call_plugin('epoch', i)

    def train(self): # 调用enumerate 时用作迭代器的对象的__length__的方法需要可以使用
        for i, (x, y) in enumerate(self.dataset, self.iterations + 1): # 报错：TypeError: 'NoneType' object cannot be interpreted as an integer
            # 在每次获得batch samples后进行更新
            # self.call_plugin('batch', i, x, y) # i 居然等于1 因为i  # batch插件没有对应的event
            def wrap(input_data): # 把数据封装为tensor 并放置到cuda上
                if torch.is_tensor(input_data) and self.use_cuda:
                    input_data = input_data.cuda()
                return input_data
            #--------------wsy fix-------------
            # x = map(wrap, x)
            # y = map(wrap, y)
            x=wrap(x)
            x=x.to(torch.float32) # 注意是创建新的，不是在原始上改变
            y=torch.squeeze(y) 
            y=F.one_hot(torch.LongTensor(y),num_classes=10) # 将标签转为one-hot向量，因为网络的预测输出是one-hot
            y=wrap(y)  # y[0] ：(16,16,784)?  y[1]：（16,16,1）
            #-----------------------------
            # 给后续插件缓存部分数据， 这里是网络的输出和损失
            plugin_data = [None, None]
            def closure(): # 将训练过程封装为一个闭合函数
                out = self.model(x) # 报错：RuntimeError: expected scalar type Float but found Double
                loss = self.criterion(out, y)
                #------wsy add -----------
                loss.requires_grad_(True) 
                #-------------------------
                loss.backward()
                if plugin_data[0] is None:
                    plugin_data[0] = out.data
                    plugin_data[1] = loss.data
                return loss
            self.optimizer.zero_grad()
            self.optimizer.step(closure)
            
            self.call_plugin('iteration', self.iterations, x, y, *plugin_data)
            # self.call_plugin('update', i, self.model) # 这个没有定义event
            # wsy：fix 修改了缩进
            self.iterations += i 
            if self.iterations%100==0: 
                print("iterations",self.iterations)   