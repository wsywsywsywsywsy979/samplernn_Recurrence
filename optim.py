from torch.nn.functional import hardtanh
def gradient_clipping(optimizer, min=-1, max=1):
    class OptimizerWrapper(object):
        def step(self, closure):
            def closure_wrapper(): # 和封装的trainer一样将opt封装为一个闭合函数
                loss = closure()
                for group in optimizer.param_groups:
                    for p in group['params']:
                        hardtanh(p.grad, min, max, inplace=True)
                return loss
            return optimizer.step(closure_wrapper)
        def __getattr__(self, attr):
            return getattr(optimizer, attr)
    return OptimizerWrapper()
