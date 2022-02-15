""" Gradient entropy
"""
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_gradient_entropy(model, batch, buffer, loss_fn, tasks, n_pairs=1, n_bins=10):
    grad_angle = torch.zeros(len(tasks)).to(device)
    for t, task in enumerate(tasks):
        x_t, y_t = buffer.sample(t, n_pairs*2)
        loss = loss_fn(model(x_t), y_t, reduction=None)
        grad = torch.autograd.grad(loss, inputs=torch.ones_like(loss))
        grad_angle[t] = torch.atan2(grad[1], grad[0])
    bins = torch.arange(0, 180, n_bins)  # restrict to prevent interference?
    discrete_probs = torch.hist(grad_angle, bins=bins, density=True)
    entropy = get_entropy(discrete_probs)
    return entropy


def optimizer_step(opt, params, grad_entropy, mask=False, threshold=0.2):
    if mask:
        for p in params:
            p.grad *= (p.grad <= threshold)
    else:
        for i, p in enumerate(params):
            p.grad *= (1-grad_entropy[i])
    opt.step()
    return opt


def get_entropy(probs):
    return -probs*probs.log().sum()