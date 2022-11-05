# CometML needs to be imported first.
try:
    import comet_ml
except ImportError:
    pass
from model import SampleRNN, Predictor
from optim import gradient_clipping
from nn import sequence_nll_loss_bits
from trainer import Trainer
# from trainer.plugins import (
#     TrainingLossMonitor, ValidationPlugin, AbsoluteTimeMonitor, SaverPlugin,
#     GeneratorPlugin, StatsPlugin
# )
from trainer.plugins import(
    TrainingLossMonitor,ValidationPlugin,AbsoluteTimeMonitor,SaverPlugin,
    GeneratorPlugin,StatsPlugin,Logger) # wsy fix
from dataset import FolderDataset, DataLoader
import torch
# from torch.utils.trainer.plugins import Logger 这个应该是0.2.0版本的
from natsort import natsorted
# from functools import reduce
import os
import shutil
import sys
from glob import glob
import re
import argparse
default_params = {
    # model parameters
    'n_rnn': 1,
    'dim': 1024,
    'learn_h0': True,
    'q_levels': 256,
    'seq_len': 1024,
    'weight_norm': True,
    'batch_size': 128,
    'val_frac': 0.1,
    'test_frac': 0.1,
    # training parameters
    'keep_old_checkpoints': False,
    'datasets_path': 'datasets',
    # 'results_path': 'results',
    'results_path':'/home/wangshiyao/wangshiyao_space/exp4/samplernn-pytorch-master/results',
    # 'epoch_limit': 1000,
    "epoch_limit":5, # wsy：先改小，完整运行看看 10+30
    'resume': True,
    'sample_rate': 16000,
    'n_samples': 1,
    'sample_length': 80000,
    'loss_smoothing': 0.99,
    'cuda': True,
    'comet_key': None
}
tag_params = [
    'exp', 'frame_sizes', 'n_rnn', 'dim', 'learn_h0', 'q_levels', 'seq_len',
    'batch_size', 'dataset', 'val_frac', 'test_frac'
] # 这个应该是用于索引default_params词典的

def param_to_string(value): # 将参数转换为字符串
    if isinstance(value, bool):
        return 'T' if value else 'F'
    elif isinstance(value, list):
        return ','.join(map(param_to_string, value))
    else:
        return str(value)

def make_tag(params): # 制作本次实验存放生成结果的文件目录名称
    return '-'.join(
        key + ':' + param_to_string(params[key])
        for key in tag_params
        if key not in default_params or params[key] != default_params[key]
    )

def setup_results_dir(params): # 这个函数的功能就是创建特定的结果文件夹
    def ensure_dir_exists(path):
        if not os.path.exists(path):
            os.makedirs(path)
    tag = make_tag(params) # 此处的效果是把所有“输入”的参数都拼接成了一个字符串
    results_path = os.path.abspath(params['results_path'])
    ensure_dir_exists(results_path)
    results_path = os.path.join(results_path, tag)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    elif not params['resume']:
        shutil.rmtree(results_path)
        os.makedirs(results_path)
    for subdir in ['checkpoints', 'samples']:
        ensure_dir_exists(os.path.join(results_path, subdir))
    return results_path

def load_last_checkpoint(checkpoints_path): # 加最新保存的checkpoint
    checkpoints_pattern = os.path.join(
        checkpoints_path, SaverPlugin.last_pattern.format('*', '*')
    )
    checkpoint_paths = natsorted(glob(checkpoints_pattern))
    if len(checkpoint_paths) > 0:
        checkpoint_path = checkpoint_paths[-1]
        checkpoint_name = os.path.basename(checkpoint_path)
        match = re.match(
            SaverPlugin.last_pattern.format(r'(\d+)', r'(\d+)'),
            checkpoint_name
        )
        epoch = int(match.group(1))
        iteration = int(match.group(2))
        return (torch.load(checkpoint_path), epoch, iteration)
    else:
        return None

def tee_stdout(log_path): # 将输出重定向到log文件中（即命令行所示就是会写到log中的）
    log_file = open(log_path, 'a', 1)
    stdout = sys.stdout
    class Tee:
        def write(self, string):
            log_file.write(string)
            stdout.write(string)
        def flush(self):
            log_file.flush()
            stdout.flush()
    sys.stdout = Tee()

def make_data_loader(overlap_len, params):
    path = os.path.join(params['datasets_path'], params['dataset'])
    def data_loader(split_from, split_to, eval):
        dataset = FolderDataset(
            path, overlap_len, params['q_levels'], split_from, split_to
        )
        return DataLoader(
            dataset,
            batch_size=params['batch_size'], # 128
            seq_len=params['seq_len'], # 1024
            overlap_len=overlap_len, # 16
            shuffle=(not eval),
            drop_last=(not eval)
        )
    return data_loader

def init_comet(params, trainer):
    if params['comet_key'] is not None:
        from comet_ml import Experiment
        from trainer.plugins import CometPlugin
        experiment = Experiment(api_key=params['comet_key'], log_code=False)
        hyperparams = {
            name: param_to_string(params[name]) for name in tag_params
        }
        experiment.log_multiple_params(hyperparams)
        trainer.register_plugin(CometPlugin(
            experiment, [
                ('training_loss', 'epoch_mean'),
                'validation_loss',
                'test_loss'
            ]
        ))

def main(exp, frame_sizes, dataset, **params):
    params = dict(
        default_params, # 文件首部那些默认的参数
        exp=exp, frame_sizes=frame_sizes, dataset=dataset,
        **params
    )
    results_path = setup_results_dir(params)
    tee_stdout(os.path.join(results_path, 'log'))
    model = SampleRNN(
        frame_sizes=params['frame_sizes'], # 16
        n_rnn=params['n_rnn'], # 1
        dim=params['dim'], # 1024
        learn_h0=params['learn_h0'], # true
        q_levels=params['q_levels'], # 256
        weight_norm=params['weight_norm'] # true
    )
    predictor = Predictor(model)
    if params['cuda']: # 暂时不放在cuda上跑
        model = model.cuda() # 这里会卡住,好像是被挂起了 果然升级torch之后就不会卡了
        predictor = predictor.cuda()
    optimizer = gradient_clipping(torch.optim.Adam(predictor.parameters()))
    data_loader = make_data_loader(model.lookback, params)
    test_split = 1 - params['test_frac'] # 0.9
    val_split = test_split - params['val_frac'] # 0.8
    trainer = Trainer(
        predictor, sequence_nll_loss_bits, optimizer,
        data_loader(0, val_split, eval=False),
        cuda=params['cuda']
    )
    #-wsy add----
    # results_path="/home/wangshiyao/wangshiyao_space/exp4/samplernn-pytorch-master/results"
    #-----------
    checkpoints_path = os.path.join(results_path, 'checkpoints') # result_path有些奇怪
    checkpoint_data = load_last_checkpoint(checkpoints_path)
    if checkpoint_data is not None: # 如果之前有训练好的模型则加载
        (state_dict, epoch, iteration) = checkpoint_data
        trainer.epochs = epoch
        trainer.iterations = iteration
        predictor.load_state_dict(state_dict)
    trainer.register_plugin(TrainingLossMonitor( # 此处注释掉之后要看是哪里使用了smoothing
        smoothing=params['loss_smoothing'] # 0.99
    )) # 此处是报错 括号中要创建对象没有参数，并且该对象也不是很重要的样子，所以暂时先不注册。
    trainer.register_plugin(ValidationPlugin(
        data_loader(val_split, test_split, eval=True),
        data_loader(test_split, 1, eval=True)
    ))
    trainer.register_plugin(AbsoluteTimeMonitor())
    trainer.register_plugin(SaverPlugin(
        checkpoints_path, params['keep_old_checkpoints']
    ))
    trainer.register_plugin(GeneratorPlugin(
        os.path.join(results_path, 'samples'), params['n_samples'],
        params['sample_length'], params['sample_rate']
    ))
    trainer.register_plugin( # 此处已测试，6个参数报错不是这里的问题
        Logger([
            'training_loss',
            'validation_loss',
            'test_loss',
            'time'
        ])
    )
    # trainer.register_plugin( wsy fixed
    #     Logger(['loss'])
    # )
    trainer.register_plugin(StatsPlugin(
        results_path,
        iteration_fields=[
            'training_loss',
            ('training_loss', 'running_avg'),
            'time'
        ],
        epoch_fields=[
            'validation_loss',
            'test_loss',
            'time'
        ],
        plots={
            'loss': {
                'x': 'iteration',
                'ys': [
                    'training_loss',
                    ('training_loss', 'running_avg'),
                    'validation_loss',
                    'test_loss',
                ],
                'log_y': True
            }
        }
    ))
    init_comet(params, trainer)
    trainer.run(params['epoch_limit'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS
    )
    def parse_bool(arg):
        arg = arg.lower()
        if 'true'.startswith(arg):
            return True
        elif 'false'.startswith(arg):
            return False
        else:
            raise ValueError()
    # parser.add_argument('--exp', required=True, help='experiment name) 
    parser.add_argument('--exp',help='experiment name',default="TEST") # wsy fix
    # parser.add_argument(
    #     '--frame_sizes', nargs='+', type=int, required=True,
    #     help='frame sizes in terms of the number of lower tier frames, \
    #           starting from the lowest RNN tier'
    # )
    parser.add_argument(
        '--frame_sizes', nargs='+', type=int,
        help='frame sizes in terms of the number of lower tier frames, \
              starting from the lowest RNN tier',default=16
    )# wsy fix  frame_sizes:从最低RNN层开始，以较低层帧数表示的帧大小
    # parser.add_argument(
    #     '--dataset', required=True,
    #     help='dataset name - name of a directory in the datasets path \
    #           (settable by --datasets_path)'
    # ) 
    parser.add_argument(
        '--dataset', 
        help='dataset name - name of a directory in the datasets path \
              (settable by --datasets_path)',default="/home/wangshiyao/wangshiyao_space/exp4/samplernn-pytorch-master/datasets/piano"
    ) # wsy fix dataset:dataset name—数据集路径中目录的名称（可由--datasets_path设置）
    parser.add_argument(
        '--n_rnn', type=int, help='number of RNN layers in each tier',default=2
    ) # wsy fix n_rnn:每层中RNN层的数量
    parser.add_argument(
        '--dim', type=int, help='number of neurons in every RNN and MLP layer'
    ) # dim：每个RNN和MLP层中的神经元数量
    parser.add_argument(
        '--learn_h0', type=parse_bool,
        help='whether to learn the initial states of RNNs'
    ) # learn_h0：是否学习RNN的初始状态
    parser.add_argument(
        '--q_levels', type=int,
        help='number of bins in quantization of audio samples'
    ) # q_levels；音频样本量化中的bin的数量
    parser.add_argument(
        '--seq_len', type=int,
        help='how many samples to include in each truncated BPTT pass'
    ) # seq_len：每个截断的BPTT过程中要包含多少个样本 首先BPTT是lstm沿时间通道进行的反向传播，具体细节没有查到可以看懂的资料，后续借助weddl看懂
    parser.add_argument(
        '--weight_norm', type=parse_bool,
        help='whether to use weight normalization'
    ) # weight_norm：是否使用权重标准化
    parser.add_argument('--batch_size', type=int, help='batch size') 
    parser.add_argument(
        '--val_frac', type=float,
        help='fraction of data to go into the validation set'
    ) # val_frac：进入验证集的数据部分
    parser.add_argument(
        '--test_frac', type=float,
        help='fraction of data to go into the test set'
    ) # test_frac：进入测试集的数据部分
    parser.add_argument(
        '--keep_old_checkpoints', type=parse_bool,
        help='whether to keep checkpoints from past epochs'
    ) # keep_old_checkpoints：是否保留过去epoch的checkpoints
    parser.add_argument(
        '--datasets_path', help='path to the directory containing datasets'
    ) # datasets_path: 包含数据集的目录的路径
    parser.add_argument(
        '--results_path', help='path to the directory to save the results to'
    ) # results_path：将结果保存到的目录的路径
    parser.add_argument('--epoch_limit', help='how many epochs to run')
    # epoch_limit：要经历多少个epoch
    parser.add_argument(
        '--resume', type=parse_bool, default=True,
        help='whether to resume training from the last checkpoint'
    ) # resume：是否从最后一个checkpoint恢复训练
    parser.add_argument(
        '--sample_rate', type=int,
        help='sample rate of the training data and generated sound'
    ) # sample_rate：训练数据和生成声音的采样率
    parser.add_argument(
        '--n_samples', type=int,
        help='number of samples to generate in each epoch'
    ) # n_samples：每个epoch中要生成的样本数
    parser.add_argument(
        '--sample_length', type=int,
        help='length of each generated sample (in samples)'
    ) # sample_length：每个生成样本的长度（以样本为单位）
    parser.add_argument(
        '--loss_smoothing', type=float,
        help='smoothing parameter of the exponential moving average over \
              training loss, used in the log and in the loss plot'
    ) # loss_smoothing: 对数和损失图中使用的训练损失指数移动平均值的平滑参数
    parser.add_argument(
        '--cuda', type=parse_bool,
        help='whether to use CUDA'
    ) # cuda：是否使用CUDA
    parser.add_argument(
        '--comet_key', help='comet.ml API key'
    ) # comet_key：comet.ml API key
    parser.set_defaults(**default_params)
    # 前面有很多参数没有赋默认值，不是，是在文件首部有词典已经准备好了
    main(**vars(parser.parse_args()))
