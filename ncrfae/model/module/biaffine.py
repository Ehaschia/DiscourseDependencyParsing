# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class Biaffine(nn.Module):

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True, biaffine=True, identity=False):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.biaffine = biaffine
        self.weight = nn.Parameter(torch.Tensor(n_out, n_in + bias_x, n_in + bias_y))

        if identity:
            assert self.bias_x is False
            assert self.bias_y is False
            self.identity_reset()
        else:
            self.reset_parameters()

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def identity_reset(self):
        eye = torch.eye(self.n_in).to(self.weight.data.device)
        self.weight.data = eye.unsqueeze(0)
        self.weight.requires_grad = False
    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        if self.biaffine:
            # [batch_size, n_out, seq_len, seq_len]
            s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
            # remove dim 1 if n_out == 1
            s = s.squeeze(1)
        else:
            s = x + y.permute(0, 2, 1)
        return s
