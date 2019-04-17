import torch
import torch.nn as nn


def is_numeric(s):
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit()


def get_mask(x):
    return (torch.ones_like(x).float() * (1 - torch.eq(x, 0.0)).float()).int()


def tensor_right_shift(tensor):
    temp = torch.zeros_like(tensor)
    temp[:, 1:, :] = tensor[:, :-1, :]
    return temp


def parse_activation(activation):
    if activation == 'relu':
        return nn.Relu()
    if activation == 'tanh':
        return nn.Tanh()
    if activation == 'sigmoid':
        return nn.Sigmoid()


def trace_back(var_grad_fn):
    print('Tracing back tensors:')
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                trace_back(n[0])
