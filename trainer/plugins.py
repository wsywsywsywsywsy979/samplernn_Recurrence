import matplotlib
matplotlib.use('Agg') # Agg 渲染器是非交互式的后端，没有GUI界面，所以不显示图片，它是用来生成图像文件。
from model import Generator
import torch
from torch.autograd import Variable
# from torch.utils.trainer.plugins.plugin import Plugin 逐步导入尝试，到报错：ModuleNotFoundError: No module named 'torch.utils.trainer'
# from torch.utils.trainer.plugins.monitor import Monitor
# from torch.utils.trainer.plugins import LossMonitor # 这个类目前只有mindspore中有了
# from librosa.output import write_wav # 报错：AttributeError: 'CacheManager' object has no attribute 'cachedir' 应该是版本问题，所以也需要修改，直接使用0.9.2版本
"""
librosa在0.8之后就把output下所有文件都删了，关于io都使用soundfile来实现
"""
import soundfile as sf
from matplotlib import pyplot
from glob import glob
import os
import pickle
import time
#-------wsy add----------------------------------------------------------------------------------------------------
class Plugin(object): # 插件的基类 果然查到的资料都还是要认真看一看，都是有用的
    def __init__(self, interval = None):
        if interval is None:
            interval = []
        self.trigger_interval = interval
    def register(self, trainer):
        raise NotImplementedError
class Monitor(Plugin): # 实现了每轮epoch ,iteration 应该做什么，可以通过 self._get_value()方法拿到具体实现的插件的值。
    def __init__(self, running_average=True, epoch_average=True, smoothing=0.7,
                 precision=None, number_format=None, unit=''):
        '''
        para：
            running_average:
            epoch_average:
            smoothing:
            precision:数字输出精度
            number_format:  数字输出格式
            unit:
       '''
        if precision is None:
            precision = 4
        if number_format is None: # .4f是从这来的
            number_format = '.{}f'.format(precision)
        # 输出格式
        number_format = ':' + number_format
        # 给基类Plugin的interval复制
        super(Monitor, self).__init__([(1, 'iteration'), (1, 'epoch')])
        self.smoothing = smoothing
        self.running_average = running_average
        self.epoch_average = epoch_average
        # 输出日志的格式  保存日志
        self.log_format = number_format
        self.log_unit = unit
        self.log_epoch_fields = None
        self.log_iter_fields = ['{last' + number_format + '}' + unit]
        if self.running_average: # 后续字典查找需要使用running_avg
            self.log_iter_fields += ['({running_avg' + number_format + '}' + unit + ')']
        if self.epoch_average:
            self.log_epoch_fields = ['{running_avg' + number_format + '}' + unit]
        #----wsy add-------------------------
        self.with_epoch_average=True
        # self.with_epoch_average=False
        self.with_running_average=True 
        # self.with_running_average=False
        #------------------------------------
    def register(self, trainer):
        self.trainer = trainer
        # 在这里给trainer的stats注册当前状态，比如log的格式等
        stats = self.trainer.stats.setdefault(self.stat_name, {})
        stats['log_format'] = self.log_format
        stats['log_unit'] = self.log_unit
        stats['log_iter_fields'] = self.log_iter_fields
        if self.with_epoch_average:
            stats['log_epoch_fields'] = self.log_epoch_fields
        if self.with_epoch_average:
            stats['epoch_stats'] = (0, 0)
    def iteration(self, *args):
        stats = self.trainer.stats.setdefault(self.stat_name, {})
        # 通过_get_value 方法拿到每个插件的值,放入到stats中
        #--------wsy fix-------------------------------
        # stats['last'] = self._get_value(*args) # 报错：TypeError: _get_value() takes 2 positional arguments but 6 were given
        stats['last'] = self._get_value(args[4])
        #----------------------------------------------
        if self.with_epoch_average:
            stats['epoch_stats'] = tuple(sum(t) for t in zip(stats['epoch_stats'], (stats['last'], 1)))

        if self.with_running_average:
            previous_avg = stats.get('running_avg', 0)
            stats['running_avg'] = previous_avg * self.smoothing + \
                                   stats['last'] * (1 - self.smoothing)
    def epoch(self, idx):
        # 每个epoch 进行的操作 每个epoch 共 1375个iteration
        stats = self.trainer.stats.setdefault(self.stat_name, {})
        if self.with_epoch_average:
            # 如果需要计算每轮epoch 的精度等,需要 总数/轮数
            epoch_stats = stats['epoch_stats']
            stats['epoch_mean'] = epoch_stats[0] / epoch_stats[1]
        stats['epoch_stats'] = (0, 0) # 清零操作？
class LossMonitor(Monitor): # 一个记录损失的日志类
    stat_name = 'loss'
    #该插件的作用为简单记录每次的loss
    def _get_value(self, loss):
        return loss.item()

from collections import defaultdict
class Logger(Plugin): # 日志类logger
    alignment = 4
    # 不同字段之间的分隔符
    separator = '#' * 80
    def __init__(self, fields, interval=None):
        if interval is None:
            interval = [(1, 'iteration'), (1, 'epoch')]
        super(Logger, self).__init__(interval)

        # 需要打印的字段,如loss acc
        self.field_widths = defaultdict(lambda: defaultdict(int))
        self.fields = list(map(lambda f: f.split('.'), fields))
        # 遵循XPath路径的格式，以AccuracyMonitor为例子：
        #   如果想打印所有的状态，只需要令fields=[AccuracyMonitor.stat_name]，也就是，['accuracy']，
        #   如果想只打印AccuracyMonitor的子状态'last'，只需要设置为['accuracy.last']
        #   这里的split当然就是为了获得[['accuracy', 'last']]，是为了之后的子状态解析（类似XPath路径解析）所使用的。
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
        # 对其输出格式
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
            for f in log_fields:
                output.append(f.format(**stat)) 
            # 报错：发生异常: KeyError(note: full exception trace is shown but execution is paused at: <module>)'running_avg' 这个报错只是单纯第因为字典中没有key
        elif not require_dict:
            # 如果子模块stat不是字典且require_dict=False，就会以父模块的打印格式和打印单位作为输出结果的方式。
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
                parent, stat = stat, stat[f] # 报错：KeyError: 'training_loss'
            name, output = self._gather_outputs(field, log_fields,
                                                parent, stat, require_dict)
            if not output:
                continue
            self._align_output(field_idx, output)
            results.append((name, output)) # training_loss :['8.8882', '(0.0889)'], time：[‘0s’] 终于开始成功有输出
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
        :param args:   ( i, batch_input, batch_target,*plugin_data) 的元组
        :return:
        '''
        self._log_all('log_iter_fields',prefix="iteration:{}".format(args[0]))

    def epoch(self, epoch_idx):
        self._log_all('log_epoch_fields',
                      prefix=self.separator + '\nEpoch summary:',
                      suffix=self.separator,
                      require_dict=True)
#-----------------------------------
class TrainingLossMonitor(LossMonitor):
    # class TrainingLossMonitor():
    stat_name = 'training_loss'
########################验证测试的过程（包含验证数据集，验证/测试损失，batch评估）##################################################
class ValidationPlugin(Plugin):
# class ValidationPlugin():
    def __init__(self, val_dataset, test_dataset):
        super().__init__([(1, 'epoch')]) # 报错：object.__init__() takes exactly one argument (the instance to initialize) 因为把继承的父类去掉了所以报错，应该是无法使用super了
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

    def _evaluate(self, dataset): # wsy：训练了一个epoch之后就是验证了
        loss_sum = 0
        n_examples = 0
        print("evaluate:") # wsy add
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
            #--------------wsy fix-------------------------------------------------
            # loss_sum += self.trainer.criterion(batch_output, batch_target) \
            #                         .data[0] * batch_size
            loss_sum += self.trainer.criterion(batch_output, batch_target) \
                                    .item() * batch_size
            # print("loss_sum：",loss_sum) # wsy add
            #-----------------------------------------------------------------
            n_examples += batch_size # 22625
        return loss_sum / n_examples
###############################监控时间的###########################################
class AbsoluteTimeMonitor(Monitor):
# class AbsoluteTimeMonitor():
    stat_name = 'time'
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('unit', 's')
        kwargs.setdefault('precision', 0)
        kwargs.setdefault('running_average', False)
        # kwargs.setdefault('epoch_average', False)
        kwargs.setdefault('epoch_average', True) # wsy fix
        super(AbsoluteTimeMonitor, self).__init__(*args, **kwargs) # 还是父类被我去掉了，所以不可再用super（×）
        self.start_time = None
    def _get_value(self, *args):
        if self.start_time is None:
            self.start_time = time.time()
        return time.time() - self.start_time
###############################监控何时进行保存的###########################################
class SaverPlugin(Plugin):
# class SaverPlugin():
    last_pattern = 'ep{}-it{}'
    best_pattern = 'best-ep{}-it{}'
    def __init__(self, checkpoints_path, keep_old_checkpoints):
        super().__init__([(1, 'epoch')]) # 去掉了父类（×）
        self.checkpoints_path = checkpoints_path
        self.keep_old_checkpoints = keep_old_checkpoints
        self._best_val_loss = float('+inf')
    def register(self, trainer):
        self.trainer = trainer
    def epoch(self, epoch_index):
        if not self.keep_old_checkpoints:
            self._clear(self.last_pattern.format('*', '*'))
        torch.save(
            self.trainer.model.state_dict(),
            os.path.join(
                self.checkpoints_path,
                self.last_pattern.format(epoch_index, self.trainer.iterations)
            )
        )
        cur_val_loss = self.trainer.stats['validation_loss']['last']
        if cur_val_loss < self._best_val_loss:
            self._clear(self.best_pattern.format('*', '*'))
            torch.save( # 这里已经在保存模型了
                self.trainer.model.state_dict(),
                os.path.join(
                    self.checkpoints_path,
                    self.best_pattern.format(
                        epoch_index, self.trainer.iterations
                    )
                )
            )
            self._best_val_loss = cur_val_loss
    def _clear(self, pattern):
        pattern = os.path.join(self.checkpoints_path, pattern)
        for file_name in glob(pattern):
            os.remove(file_name)
###############################监控合成音频的模型的###########################################
class GeneratorPlugin(Plugin):
# class GeneratorPlugin():
    pattern = 'ep{}-s{}.wav'
    def __init__(self, samples_path, n_samples, sample_length, sample_rate):
        super().__init__([(1, 'epoch')]) # 父类
        self.samples_path = samples_path
        self.n_samples = n_samples # 1
        self.sample_length = sample_length # 80000
        self.sample_rate = sample_rate # 16000
    def register(self, trainer):
        self.generate = Generator(trainer.model.model, trainer.cuda)
    def epoch(self, epoch_index):
        samples = self.generate(self.n_samples, self.sample_length) \
                      .cpu().float().numpy()
        for i in range(self.n_samples):
            # write_wav(
            #     os.path.join(
            #         self.samples_path, self.pattern.format(epoch_index, i + 1)
            #     ),
            #     samples[i, :], sr=self.sample_rate, norm=True
            # )
            sf.write(os.path.join(self.samples_path, self.pattern.format(epoch_index, i + 1)),samples[i, :], self.sample_rate)

###############################监控每一次迭代（iteration）的###########################################
class StatsPlugin(Plugin):
# class StatsPlugin():
    data_file_name = 'stats.pkl'
    plot_pattern = '{}.svg'
    def __init__(self, results_path, iteration_fields, epoch_fields, plots):
        super().__init__([(1, 'iteration'), (1, 'epoch')]) # 父类
        self.results_path = results_path
        self.iteration_fields = self._fields_to_pairs(iteration_fields)
        self.epoch_fields = self._fields_to_pairs(epoch_fields)
        self.plots = plots
        self.data = {
            'iterations': {
                field: []
                for field in self.iteration_fields + [('iteration', 'last')]
            },
            'epochs': {
                field: []
                for field in self.epoch_fields + [('iteration', 'last')]
            }
        }
    def register(self, trainer):
        self.trainer = trainer
    def iteration(self, *args):
        for (field, stat) in self.iteration_fields:
            self.data['iterations'][field, stat].append(
                self.trainer.stats[field][stat]
            )
        self.data['iterations']['iteration', 'last'].append(
            self.trainer.iterations
        )
    def epoch(self, epoch_index):
        for (field, stat) in self.epoch_fields:
            self.data['epochs'][field, stat].append(
                self.trainer.stats[field][stat]
            )
        self.data['epochs']['iteration', 'last'].append(
            self.trainer.iterations
        )
        data_file_path = os.path.join(self.results_path, self.data_file_name)
        with open(data_file_path, 'wb') as f:
            pickle.dump(self.data, f)
        for (name, info) in self.plots.items():
            x_field = self._field_to_pair(info['x'])
            try:
                y_fields = info['ys']
            except KeyError:
                y_fields = [info['y']]
            labels = list(map(
                lambda x: ' '.join(x) if type(x) is tuple else x,
                y_fields
            ))
            y_fields = self._fields_to_pairs(y_fields)
            try:
                formats = info['formats']
            except KeyError:
                formats = [''] * len(y_fields)
            pyplot.gcf().clear()
            for (y_field, format, label) in zip(y_fields, formats, labels):
                if y_field in self.iteration_fields:
                    part_name = 'iterations'
                else:
                    part_name = 'epochs'
                xs = self.data[part_name][x_field]
                ys = self.data[part_name][y_field]
                pyplot.plot(xs, ys, format, label=label) # wsy:此处因为一开始的设置，所以不会显示图片
            if 'log_y' in info and info['log_y']:
                pyplot.yscale('log')
            pyplot.legend()
            pyplot.savefig(
                os.path.join(self.results_path, self.plot_pattern.format(name))
            )
    @staticmethod
    def _field_to_pair(field):
        if type(field) is tuple:
            return field
        else:
            return (field, 'last')
    @classmethod
    def _fields_to_pairs(cls, fields):
        return list(map(cls._field_to_pair, fields))
###############################这个没看出来是干啥的###########################################
class CometPlugin(Plugin):
# class CometPlugin():
    def __init__(self, experiment, fields):
        super().__init__([(1, 'epoch')])
        self.experiment = experiment
        self.fields = [
            field if type(field) is tuple else (field, 'last')
            for field in fields
        ]
    def register(self, trainer):
        self.trainer = trainer
    def epoch(self, epoch_index):
        for (field, stat) in self.fields:
            self.experiment.log_metric(field, self.trainer.stats[field][stat])
        self.experiment.log_epoch_end(epoch_index)