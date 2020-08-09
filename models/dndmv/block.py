import torch
from torch import nn
from torch import Tensor
from typing import Sequence, Optional


class LSTMBlock(nn.Module):
    def __init__(self, in_dim, out_dim, bidirectional, num_layers, dropout):
        super().__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        if bidirectional:
            assert out_dim % 2 == 0
            out_dim //= 2
        self.nn = nn.LSTM(in_dim, out_dim, num_layers,
            dropout=dropout, bidirectional=bidirectional)

    def forward(self, h: Tensor, len_array: Tensor,
            batch_first: bool = True, enforce_sorted: bool = False):
        max_len, batch_size, *o = h.shape
        if batch_first:
            batch_size, max_len = max_len, batch_size
            h = h.transpose(0, 1).contiguous()

        h = nn.utils.rnn.pack_padded_sequence(h, len_array, False, enforce_sorted)
        h, (hx, cx) = self.nn(h)  # noqa
        h = nn.utils.rnn.pad_packed_sequence(h)[0]  # noqa

        diff = max_len - torch.max(len_array).item()
        if diff > 0:
            h = torch.cat([h, torch.zeros(diff, batch_size, *h.shape[2:], device=h.device)])

        if batch_first:
            h = h.transpose(0, 1).contiguous()
        cx = cx.transpose(0, 1).reshape(batch_size, self.num_layers, self.out_dim)

        return h, cx[:, -1]


def get_mlp_layer(dims: Sequence[int], dropout: float = 0., activate: str = 'relu', bias: bool = True,
        dropout_last: Optional[float] = None, activate_last: Optional[str] = None, bias_last: Optional[bool] = None):
    assert len(dims) >= 2, "at least two dims are required for one layer of linear as input dim and output dim."
    dropout_last = dropout_last if dropout_last is not None else dropout
    bias_last = bias_last if bias_last is not None else bias
    activate_last = activate_last if activate_last is not None else activate
    layers = nn.Sequential()

    activate_funcs = {'none': None, 'relu': nn.ReLU, 'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}
    activate = activate_funcs[activate]
    activate_last = activate_funcs[activate_last]

    in_dim = dims[0]
    for idx, out_dim in enumerate(dims[1:-1]):
        layers.add_module(f'FC:{idx}', nn.Linear(in_dim, out_dim))
        if activate != 'none':
            layers.add_module(f'A:{idx}', activate())
        if dropout > 0.:
            layers.add_module(f'D:{idx}', nn.Dropout(dropout))
        in_dim = out_dim

    layers.add_module(f'FC:{len(dims) - 1}', nn.Linear(dims[-2], dims[-1], bias_last))
    if activate_last != 'none':
        layers.add_module(f'A:{len(dims) - 1}', activate())
    if dropout_last > 0.:
        layers.add_module(f'D:{len(dims) - 1}', nn.Dropout(dropout_last))
    return layers
