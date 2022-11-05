import torch
from torch import nn
import math

class LearnedUpsampling1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        # self.conv_t = nn.ConvTranspose1d( # 使用了转置卷积来进行上采样
        #     in_channels=in_channels, # 1024
        #     out_channels=out_channels, # 1024
        #     kernel_size=kernel_size, # 16
        #     stride=kernel_size, # 16
        #     bias=False
        # )
        self.conv_t = nn.ConvTranspose1d( # wsy fix
            in_channels=int(in_channels), # 1024
            out_channels=int(out_channels), # 1024
            kernel_size=int(kernel_size), # 16
            stride=int(kernel_size), # 16
            bias=False
        )
        if bias:
            #self.bias = nn.Parameter(
            #    torch.FloatTensor(out_channels, kernel_size)
            #)
            self.bias = nn.Parameter( # wsy fix 
                torch.FloatTensor(int(out_channels), int(kernel_size))
            )
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        self.conv_t.reset_parameters()
        # nn.init.constant(self.bias, 0)
        nn.init.constant_(self.bias, 0)
    def forward(self, input):
        (batch_size, _, length) = input.size()
        (kernel_size,) = self.conv_t.kernel_size
        bias = self.bias.unsqueeze(0).unsqueeze(2).expand(
            batch_size, self.conv_t.out_channels,
            length, kernel_size
        ).contiguous().view(
            batch_size, self.conv_t.out_channels,
            length * kernel_size
        )
        return self.conv_t(input) + bias

def lecun_uniform(tensor):
    fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
    # nn.init.uniform(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
    nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in)) # 因为警告

def concat_init(tensor, inits):
    try:
        tensor = tensor.data # 获取tensor数据的方式！！
    except AttributeError:
        pass
    (length, fan_out) = tensor.size()
    fan_in = length // len(inits) # len(inits)=3
    chunk = tensor.new(fan_in, fan_out) # 1024 ，1024 所以这个model也是将音频进行了分chunk，不过处理数据集的时候就是安装chunk分的
    for (i, init) in enumerate(inits):
        init(chunk)
        tensor[i * fan_in : (i + 1) * fan_in, :] = chunk

def sequence_nll_loss_bits(input, target, *args, **kwargs):
    (_, _, n_classes) = input.size()
    return nn.functional.nll_loss(
        input.view(-1, n_classes), target.view(-1), *args, **kwargs
    ) * math.log(math.e, 2)