#--------wsy add------------
import torch
from torch.autograd import Variable
from wsy_dataloader import MyDataset
from torch.utils.data import DataLoader
from wsy_Trainer3 import Trainer
import torch.optim as opt
from torchmetrics import Accuracy
from wsy_pre_torch import FCN
BATCH_SIZE=16 # 
dataset=MyDataset(stage = 'train',batch_size=BATCH_SIZE)
model=FCN(hidden_size=256) # 注意输入的维度是28X28= 784
criterion=Accuracy() # 使用精度(就是计算正确预测的数量占总样本数的多少)作为评价指标
optimizer=opt.Adam(model.parameters(),lr=0.001) # 注意参数
train_loader=DataLoader(dataset, batch_size=BATCH_SIZE) # 这里的使用应该是没有问题的
#-------------------------
# 定义插件的基类：（其实都是代码都可以根据自己的理解进行改动的
class Plugin(object): # 
    def __init__(self, interval = None):
        if interval is None:
            interval = []
        self.trigger_interval = interval
    
    def register(self, trainer): # 感觉这里相当于把函数设置为了纯虚函数，不可以实例化。
        raise NotImplementedError

# 具体的插件类：实现每轮epoch ,iteration 应该做什么，可以通过 self._get_value()方法拿到具体实现的插件的值。
class Monitor(Plugin): 
    def __init__(self, running_average=True, epoch_average=True, smoothing=0.7,
                 precision=None, number_format=None, unit=''):
        '''
        para:
            running_average:
            epoch_average:
            smoothing:
            precision:数字输出精度
            number_format:  数字输出格式
            unit:
       '''
        if precision is None:
            precision = 4
        if number_format is None:
            number_format = '.{}f'.format(precision)
        # 输出格式：
        number_format = ':' + number_format
        # 给基类Plugin的interval复制：
        super(Monitor, self).__init__([(1, 'iteration'), (1, 'epoch')]) # 这里的定义应该是：每一个iteration和epoch都触发监控
        self.smoothing = smoothing # 0.7
        self.running_average = running_average # True
        self.epoch_average = epoch_average # True
        # 输出日志的格式  保存日志
        self.log_format = number_format
        self.log_unit = unit # ''
        self.log_epoch_fields = None
        self.log_iter_fields = ['{last' + number_format + '}' + unit] # [{last:.4f}]
        if self.running_average:
            self.log_iter_fields += ['({running_avg' + number_format + '}' + unit + ')'] # ['{last:.4f}', '({running_avg:.4f})']
        if self.epoch_average:
            self.log_epoch_fields = ['{running_avg' + number_format + '}' + unit] # ['{running_avg:.4f}']
        #--wsy add-------------------------
        # self.with_epoch_average=True # 因为默认的running_average和epoch_average都是true
        # self.with_running_average=True # 出现除0情况
        self.with_epoch_average=False
        self.with_running_average=False
        #-----------------------------------

    def register(self, trainer):
        self.trainer = trainer
        # 在这里给trainer的stats注册当前状态，比如log的格式等
        stats = self.trainer.stats.setdefault(self.stat_name, {})
        stats['log_format'] = self.log_format # ':.4f'
        stats['log_unit'] = self.log_unit # ''
        stats['log_iter_fields'] = self.log_iter_fields # ['{last:.4f}', '({running_avg:.4f})']
        if self.with_epoch_average:
            stats['log_epoch_fields'] = self.log_epoch_fields
        if self.with_epoch_average:
            stats['epoch_stats'] = (0, 0)
        #------------wsy add:发现stats设置之后后续并没有使用？-----------
        self.stats=stats
        #------------------------------------------------------------
    def iteration(self, *args):
        stats = self.trainer.stats.setdefault(self.stat_name, {})
        # 通过_get_value 方法拿到每个插件的值,放入到stats中
        #-------wsy fix----------------------------------
        # stats['last'] = self._get_value(*args) # 终于绕道到samplernn中遇到的问题了，只要这里解决了，就可以去看改samplernn了
        stats['last'] = self._get_value(args[4])
        #-----------------------------------------------
        if self.with_epoch_average:
            stats['epoch_stats'] = tuple(sum(t) for t in zip(stats['epoch_stats'], (stats['last'], 1)))

        if self.with_running_average:
            previous_avg = stats.get('running_avg', 0)
            stats['running_avg'] = previous_avg * self.smoothing + \
                                   stats['last'] * (1 - self.smoothing)
    def epoch(self, idx):
        # 每个epoch 进行的操作
        stats = self.trainer.stats.setdefault(self.stat_name, {})
        if self.with_epoch_average:
            # 如果需要计算每轮epoch 的精度等,需要 总数/轮数
            epoch_stats = stats['epoch_stats']
            stats['epoch_mean'] = epoch_stats[0] / epoch_stats[1]
        stats['epoch_stats'] = (0, 0)
    
class LossMonitor(Monitor): # 一个记录损失的日志类
    stat_name = 'loss'
    #该插件的作用为简单记录每次的loss
    def _get_value(self, loss):
        return loss.item()
    
from collections import defaultdict

class Logger(Plugin): # 日志类logger
    alignment = 4
    #不同字段之间的分隔符
    separator = '#' * 80
    def __init__(self, fields, interval=None):
        if interval is None:
            interval = [(1, 'iteration'), (1, 'epoch')]
        super(Logger, self).__init__(interval)
        # 需要打印的字段,如loss acc
        self.field_widths = defaultdict(lambda: defaultdict(int)) # wsy：应该是都设置为int长度
        self.fields = list(map(lambda f: f.split('.'), fields))
        # 遵循XPath路径的格式，以AccuracyMonitor为例子:
        #     如果想打印所有的状态:只需要令fields=[AccuracyMonitor.stat_name]，也就是，['accuracy']，
        #     如果想只打印AccuracyMonitor的子状态'last'，只需要设置为['accuracy.last'],而这里的split当然就是为了获得[['accuracy', 'last']]
        #         是为了之后的子状态解析（类似XPath路径解析）所使用的。
    def _join_results(self, results):
        # 这个函数主要是将获得的子状态的结果进行组装。
        joined_out = map(lambda i: (i[0], ' '.join(i[1])), results)
        joined_fields = map(lambda i: '{}: {}'.format(i[0], i[1]), joined_out)
        return '\t'.join(joined_fields)
    def log(self, msg):
        print(msg)
    def register(self, trainer):
        self.trainer = trainer
    def gather_stats(self):
        result = {}
        return result
    def _align_output(self, field_idx, output):
        #对其输出格式
        for output_idx, o in enumerate(output):
            if len(o) < self.field_widths[field_idx][output_idx]:
                num_spaces = self.field_widths[field_idx][output_idx] - len(o)
                output[output_idx] += ' ' * num_spaces
            else:
                self.field_widths[field_idx][output_idx] = len(o)
    def _gather_outputs(self, field, log_fields, stat_parent, stat, require_dict=False):
        # 这个函数是核心，负责将查找到的最底层的子模块的结果提取出来。
        output = []
        name = ''
        if isinstance(stat, dict):
            '''
            通过插件的子stat去拿到每一轮的信息,如LOSS等
            '''
            log_fields = stat.get(log_fields, [])
            name = stat.get('log_name', '.'.join(field))
            # 找到自定义的输出名称。y有时候我们并不像打印对应的Key出来，所以可以
            # 在写插件的时候增加多一个'log_name'的键值对，指定打印的名称。默认为
            # field的完整名字。传入的fileds为['accuracy.last']
            # 那么经过初始化之后，fileds=[['accuracy',
            # 'last']]。所以这里的'.'.join(fields)其实是'accuracy.last'。
            # 起到一个还原名称的作用。
            for f in log_fields:
                output.append(f.format(**stat))
        elif not require_dict:
            # 在这里的话，如果子模块stat不是字典且require_dict=False
            # 那么他就会以父模块的打印格式和打印单位作为输出结果的方式。
            name = '.'.join(field)
            number_format = stat_parent.get('log_format', '')
            unit = stat_parent.get('log_unit', '')
            fmt = '{' + number_format + '}' + unit
            output.append(fmt.format(stat))
        return name, output
    
    def _log_all(self, log_fields, prefix=None, suffix=None, require_dict=False):
        results = []
        for field_idx, field in enumerate(self.fields):
            parent, stat = None, self.trainer.stats
            for f in field:
                parent, stat = stat, stat[f]
            name, output = self._gather_outputs(field, log_fields,
                                                parent, stat, require_dict)
            if not output:
                continue
            self._align_output(field_idx, output)
            results.append((name, output))
        if not results:
            return
        output = self._join_results(results)
        loginfo = []

        if prefix is not None:
            loginfo.append(prefix)
            loginfo.append("\t")

        loginfo.append(output)
        if suffix is not None:
            loginfo.append("\t")
            loginfo.append(suffix)
        self.log("".join(loginfo))
    
    def iteration(self, *args):
        '''
        :param args:   ( i, batch_input, batch_target,*plugin_data) 的元祖
        :return:
        '''
        self._log_all('log_iter_fields',prefix="iteration:{}".format(args[0]))

    def epoch(self, epoch_idx):
        self._log_all('log_epoch_fields',
                      prefix=self.separator + '\nEpoch summary:',
                      suffix=self.separator,
                      require_dict=True)

class ValidationPlugin(Plugin): # evaluation 类
    
    def __init__(self, val_dataset, test_dataset):
        super().__init__([(1, 'epoch')])
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def register(self, trainer):
        self.trainer = trainer
        val_stats = self.trainer.stats.setdefault('validation_loss', {})
        val_stats['log_epoch_fields'] = ['{last:.4f}']
        test_stats = self.trainer.stats.setdefault('test_loss', {})
        test_stats['log_epoch_fields'] = ['{last:.4f}']

    def epoch(self, idx):
        self.trainer.model.eval()

        val_stats = self.trainer.stats.setdefault('validation_loss', {})
        val_stats['last'] = self._evaluate(self.val_dataset)
        test_stats = self.trainer.stats.setdefault('test_loss', {})
        test_stats['last'] = self._evaluate(self.test_dataset)

        self.trainer.model.train()
    
        def _evaluate(self, dataset):
            loss_sum = 0
        n_examples = 0
        for data in dataset:
            batch_inputs = data[: -1]
            batch_target = data[-1]
            batch_size = batch_target.size()[0]

            def wrap(input):
                if torch.is_tensor(input):
                    input = Variable(input, volatile=True)
                    if self.trainer.cuda:
                        input = input.cuda()
                return input
            batch_inputs = list(map(wrap, batch_inputs))

            batch_target = Variable(batch_target, volatile=True)
            if self.trainer.cuda:
                batch_target = batch_target.cuda()

            batch_output = self.trainer.model(*batch_inputs)
            loss_sum += self.trainer.criterion(batch_output, batch_target) \
                                    .data[0] * batch_size

            n_examples += batch_size

        return loss_sum / n_examples

trainer = Trainer(model,criterion,optimizer,train_loader,use_cuda=True) # wsy fix
trainer.register_plugin(LossMonitor()) 
trainer.register_plugin(Logger(['loss'])) # 增加了触发events
trainer.run(100)