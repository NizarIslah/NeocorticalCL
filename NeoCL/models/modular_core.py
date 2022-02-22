import torch
from torch import nn
from avalanche.models import IncrementalClassifier
from torch.nn import Linear, ReLU
from collections import OrderedDict, Union


class CustomModule(nn.Module):
    def __init__(self):
        super().__init__()
    
    def freeze_module(self):
        self.module_learned_buffer = torch.tensor(1.)  
        if self.args.keep_bn_in_eval_after_freeze:  
            bn_eval(self)
            # self.bn_eval_buffer=torch.tensor(1.)    
        for p in self.parameters():
            #setting requires_grad to False would prevent updating in the inner loop as well
            p.requires_grad = False
            #this would prevent updating in the outer loop:
            p.__class__ = FixedParameter      
        print(f'freezing module {self.name}')
        return True


class Conv_Module(CustomModule):
    def __init__(self,in_channels, out_channels, kernel_size, track_running_stats, pooling_kernel, pooling_stride=None, pooling_padding=0, affine_bn=True, momentum=1, decode=False, use_bn=True, **kwargs):
        super().__init__()
        self.module = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, **kwargs)),  
                    ('norm', nn.BatchNorm2d(out_channels, momentum=momentum, affine=affine_bn,
                        track_running_stats=track_running_stats)) if use_bn else ('norm', nn.Identity()),
                    ('relu', nn.ReLU()),
                    ('pool', nn.MaxPool2d(pooling_kernel,pooling_stride, pooling_padding))
                ]))

    def forward(self, x):
        return self.module(x)


class FC_Module(CustomModule):
    def __init__(self, in_size, out_size, momentum=1, use_bn=True, **kwargs):
        super().__init__()
        self.module = nn.Sequential(OrderedDict([
                ('linear', nn.Linear(in_size, out_size)),
                ('norm', nn.BatchNorm(out_size, momentum=momentum, track_running_stats=track_running_stats)) if use_bn else ('norm', nn.Identity()),
                ('relu', nn.ReLU())
            ]))

    def forward(self, x):
        return self.module(x)


def create_module():
    return


class BaseModularNN(nn.Module):
    def __init__(self, modules=Union[int, list]):
        super().__init__()
        if type(modules) == list:
            self.modules = nn.Sequential([create_module(module_name) for module_name in modules])  # module type specified by input
        else:
            self.modules = nn.Sequential([create_module() for _ in range(modules)])   # use default
        self.module_status = [0]*len(self.modules)  # 0 = active, 1 = learned/frozen

    def add_module(self):
        return

    def prune_module(self):
        return

    def functional_loss(self):
        return

    def structural_loss(self):
        return

    def loss(self):
        return

    def forward(self, x):
        for m, module in enumerate(self.modules):
            if self.module_status[m] > 0:
                x = self.modules[m](x)
        return x