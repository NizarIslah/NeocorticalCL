from avalanche.training.plugins import EWCPlugin
from collections import defaultdict
from typing import Dict, Tuple
import warnings
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from avalanche.models.utils import avalanche_forward
from avalanche.training.utils import copy_params_dict, zerolike_params_dict


class SparseEWCPlugin(EWCPlugin):
    def before_backward(self, strategy, **kwargs):
        """
        Compute EWC penalty and add it to the loss.
        """
        exp_counter = strategy.clock.train_exp_counter
        if exp_counter == 0:
            return

        penalty = torch.tensor(0).float().to(strategy.device)

        if self.mode == 'separate':
            for experience in range(exp_counter):
                for (_, cur_param), (_, saved_param), (_, imp) in zip(
                        strategy.model.named_parameters(),
                        self.saved_params[experience],
                        self.importances[experience]):
                    if cur_param.requires_grad:
                        if cur_param.size() != saved_param.size():
                            saved_size = saved_param.size(0)
                            temp = cur_param[:saved_size]
                            penalty += (imp * (temp - saved_param).pow(2)).sum()
                        else:
                            penalty += (imp * (cur_param - saved_param).pow(2)).sum()
        elif self.mode == 'online':
            prev_exp = exp_counter - 1
            for (_, cur_param), (_, saved_param), (_, imp) in zip(
                    strategy.model.named_parameters(),
                    self.saved_params[prev_exp],
                    self.importances[prev_exp]):
                if cur_param.requires_grad:
                    if cur_param.size() != saved_param.size():
                        saved_size = saved_param.size(0)
                        temp = cur_param[:saved_size]
                        penalty += (imp * (temp - saved_param).pow(2)).sum()
                    else:
                        penalty += (imp * (cur_param - saved_param).pow(2)).sum()
        else:
            raise ValueError('Wrong EWC mode.')

        strategy.loss += self.ewc_lambda * penalty

    def compute_importances(self, model, criterion, optimizer,
                            dataset, device, batch_size):
        """
        Compute EWC importance matrix for each parameter
        """
        model.eval()
        # Set RNN-like modules on GPU to training mode to avoid CUDA error
        if device == 'cuda':
            for module in model.modules():
                if isinstance(module, torch.nn.RNNBase):
                    warnings.warn(
                        'RNN-like modules do not support '
                        'backward calls while in `eval` mode on CUDA '
                        'devices. Setting all `RNNBase` modules to '
                        '`train` mode. May produce inconsistent '
                        'output if such modules have `dropout` > 0.'
                    )
                    module.train()

        # list of list
        importances = zerolike_params_dict(model)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        for i, batch in enumerate(dataloader):
            # get only input, target and task_id from the batch
            x, y, task_labels = batch[0], batch[1], batch[-1]
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = avalanche_forward(model, x, task_labels)
            loss = criterion(out, y)
            loss.backward()

            for (k1, p), (k2, imp) in zip(model.named_parameters(),
                                          importances):
                assert (k1 == k2)
                if p.grad is not None:
                    imp += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances:
            imp /= float(len(dataloader))

        return importances
