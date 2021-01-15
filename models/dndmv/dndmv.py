import os
from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

# from utils.utils import make_mask
from typing import Dict, List, Union, Optional

from easydict import EasyDict
from models.dndmv.block import LSTMBlock
from utils import make_mask


@dataclass
class NeuralMOptions:
    num_lex: int = 0
    num_pos: int = 0
    num_tag: int = 0

    dim_pos_emb: int = 10
    # dim_word_emb: int = 200
    dim_valence_emb: int = 10

    lstm_dim_out: int = 10
    lstm_dropout: float = 0.
    lstm_layers: int = 1
    lstm_bidirectional: bool = True

    activation_func = 'relu'
    dim_hidden: int = 15
    dim_pre_out_decision: int = 5
    dim_pre_out_child: int = 12
    dim_pre_out_root: int = 12

    freeze_word_emb = False
    freeze_pos_emb = False
    freeze_out_pos_emb = False

    # if True, child_out_linear`s weight will be binded with POS emb and WORD emb(if available).
    #   described in NDMV, L-NDMV.
    # if False, child_out_linear`s weight will be randomness
    use_emb_as_w: bool = False


ACTIVATION_DICT = {'relu': F.relu, 'gelu': F.gelu, 'tanh': F.tanh, 'sigmoid': F.sigmoid}


class DiscriminativeNeuralDMV(nn.Module):
    """
    use POS,VALENCE. share LSTM in. not using LSTM out.
    """

    def __init__(self, cfg: Dict, emb: Dict[str, Tensor], mode: str = 'tdr'):
        super().__init__()
        self.cfg = EasyDict(cfg)
        self.mode = mode
        self.optimizer = None

        # emb
        if 'word' in emb:
            self.word_emb = nn.Embedding.from_pretrained(emb['word'], freeze=self.cfg.freeze_word_emb)
            assert self.word_emb.weight.shape[1] == self.cfg.dim_word_emb
        else:
            self.word_emb = nn.Embedding(self.cfg.num_lex + 2, self.cfg.dim_word_emb)
        self.emb_to_lstm = nn.Linear(self.cfg.dim_word_emb, self.cfg.lstm_dim_in)
        self.emb_to_cat = nn.Linear(self.cfg.dim_word_emb, self.cfg.dim_pos_emb)
        # chage dim same as word_emb due to the cluster
        if 'pos' in emb:
            if self.cfg.pca:
                u, s, v = torch.pca_lowrank(torch.tensor(emb['pos']), self.cfg.dim_pos_emb)
            else:
                u = emb['pos']
            self.pos_emb = nn.Embedding.from_pretrained(u, freeze=self.cfg.freeze_pos_emb)
            # assert self.pos_emb.weight.shape[1] == self.cfg.dim_pos_emb
        else:
            self.pos_emb = nn.Embedding(self.cfg.cluster, self.cfg.dim_pos_emb)
        self.emb_to_pos = nn.Linear(self.cfg.dim_word_emb, self.cfg.dim_pos_emb)

        if self.cfg.share_valence_emb and self.cfg.cv == 2:
            self.cv_emb = nn.Embedding(2, self.cfg.dim_valence_emb)
            self.dv_emb = self.cv_emb
        else:
            self.cv_emb = nn.Embedding(self.cfg.cv, self.cfg.dim_valence_emb)
            self.dv_emb = nn.Embedding(2, self.cfg.dim_valence_emb)

        self.deprel_emb = nn.Embedding(self.cfg.num_deprel, self.cfg.dim_deprel_emb)
        self.relation_emb = nn.Embedding(self.cfg.num_relation, self.cfg.dim_relation_emb)
        # self.emb_to_lstm = nn.Linear(self.cfg.dim_word_emb*3+self.cfg.dim_deprel_emb+self.cfg.dim_pos_emb*3,
        #                              self.cfg.lstm_dim_in)

        # shared layer
        self.lstm_s = LSTMBlock(self.cfg.lstm_dim_in, self.cfg.lstm_dim_out, self.cfg.lstm_bidirectional,
                                self.cfg.lstm_layers, self.cfg.lstm_dropout)

        # self.emb_dim = self.cfg.dim_word_emb + self.cfg.dim_pos_emb + self.cfg.dim_valence_emb
        self.emb_dim = self.cfg.dim_pos_emb*2 + self.cfg.dim_valence_emb + self.cfg.lstm_dim_out

        self.edus, self.edus_len = None, None
        self.sent_map = None

        self.left_right_linear = nn.Linear(self.emb_dim, 2 * self.cfg.dim_hidden)

        # separate layer
        if 'd' in self.mode:
            self.decision_linear = nn.Linear(self.cfg.dim_hidden, self.cfg.dim_pre_out_decision)
            self.decision_out_linear = nn.Linear(self.cfg.dim_pre_out_decision, 2)

        if 'r' in self.mode:
            self.rv_emb = nn.Embedding(1, self.cfg.dim_valence_emb)
            self.root_linear = nn.Linear(self.cfg.dim_hidden, self.cfg.dim_pre_out_root)
            self.root_out_linear = nn.Linear(self.cfg.dim_pre_out_root, 1)

        if 't' in self.mode:
            # if self.cfg.use_emb_as_w:
            #     w_dim = self.cfg.dim_word_emb + (self.cfg.dim_pos_emb if self.cfg.use_pair else 0)
            #     if self.cfg.dim_pre_out_child != w_dim:
            #         utils.ex.logger.warning(
            #             f"ignoring cfg.dim_pre_out_child because cfg.use_emb_as_w = True, using {w_dim}")
            #
            #     self.child_linear = nn.Linear(self.cfg.dim_hidden, w_dim)
            #     if 'out_pos' in emb:
            #         self.pos_emb_out = nn.Parameter(emb['out_pos'], requires_grad=not self.cfg.freeze_out_pos_emb)
            #     else:
            #         self.pos_emb_out = nn.Parameter(torch.empty(self.cfg.num_pos, w_dim), requires_grad=True)
            #         nn.init.normal_(self.pos_emb_out.data)
            # else:
            self.child_linear = nn.Linear(self.cfg.dim_hidden, self.cfg.dim_pre_out_child)
            self.child_out_linear = nn.Linear(self.cfg.dim_pre_out_child, self.cfg.cluster)
            # self.child_word_linear = nn.Linear(self.cfg.dim_word_emb, self.cfg.dim_pre_out_child)

        self.activate = ACTIVATION_DICT[self.cfg.activation_func]
        self.dropout = nn.Dropout(self.cfg.dropout)

        # self.word_idx, self.pos_idx = None, None

    def train_pipeline(self, arrays: Dict[str, Tensor], tag_array: Tensor, counts: List[Tensor], mode: str = 'tdr') \
            -> Tensor:
        len_array = arrays['len']
        max_len = arrays['edus_len'].size()[1]
        mask = make_mask(len_array, max_len)
        weight = arrays['weight']
        params = self(arrays, tag_array, mode)
        for i, m in enumerate(mode):
            if m == 't' and params[i] is not None:
                params[i] = self.transition_param_helper_2(params[i])
        for i, t in enumerate(params):
            weight_shape = tuple([-1] + [1] * (len(t.size())-1))
            params[i] = t * weight.reshape(weight_shape)
        loss = self.loss(params, counts, mask, mode)
        return loss

    def predict_pipeline(self, arrays: Dict[str, Tensor], tag_array: Tensor, mode: str = 'tdr') \
            -> Union[Tensor, List[Tensor]]:
        params = self(arrays, tag_array, mode)
        for i, m in enumerate(mode):
            if m == 't' and params[i] is not None:
                params[i] = self.transition_param_helper_2(params[i])
        return params if len(mode) != 1 else params[0]

    def optimize(self) -> None:
        self.optimizer.step()

    # a list of edu represent method
    def edu_feature_v1(self, arrays):

        first_word_emb = self.word_emb(arrays["first_word"])
        end_word_emb = self.word_emb(arrays["end_word"])
        head_word_emb = self.word_emb(arrays["head_word"])
        first_pos_emb = self.pos_emb(arrays["first_pos"])
        end_pos_emb = self.pos_emb(arrays["end_pos"])
        head_pos_emb = self.pos_emb(arrays["head_pos"])
        head_deprel_emb = self.deprel_emb(arrays["head_deprel"])

        # self.edus_len = arrays["edus_len"]
        # print(torch.sum(~torch.eq(arrays["edus"][:, :, torch.max(self.edus_len):], 7618)))
        # print(torch.sum(~torch.eq(arrays["edus"][:, :, torch.max(self.edus_len):], 7618)))
        # self.edus = arrays["edus"][:, :, :torch.max(self.edus_len)]
        # self.sent_map = arrays["sent_map"]

        edu_vectors = torch.cat([first_word_emb,
                                 end_word_emb,
                                 head_word_emb,
                                 first_pos_emb,
                                 end_pos_emb,
                                 head_pos_emb,
                                 head_deprel_emb], dim=2)
        return edu_vectors

    def bag_of_word(self):
        batch_size, max_len, word_len = self.edus.size()
        words = self.word_emb(self.edus.reshape(-1)).reshape(batch_size, max_len, word_len, self.cfg.dim_word_emb)
        edus_mask = make_mask(self.edus_len.view(-1)).view(batch_size, max_len, word_len, -1)
        bag_word = torch.sum(words * edus_mask, dim=2) / (self.edus_len + (self.edus_len == 0).long()).unsqueeze(-1)

        # edu_vectors = self.activate(self.dropout(self.emb_to_lstm(bag_word)))
        return bag_word

    def context_embedding(self, arrays):
        return arrays["context_embed"]

    def forward(self, arrays: Dict[str, Tensor], tag_array: Tensor, mode: str = 'tdr') -> List[Tensor]:
        len_array = arrays['len']
        max_len = arrays['edus_len'].size()[1]  # in sub_batch, max(len_array) != max_len
        batch_size = len(len_array)
        self.edus_len = arrays["edus_len"]
        self.edus = arrays["edus"][:, :, :torch.max(self.edus_len)]
        self.sent_map = arrays["sent_map"]

        edu_vectors = self.context_embedding(arrays)
        lstm_in = self.activate(self.dropout(self.emb_to_lstm(edu_vectors)))
        _, sent_emb = self.lstm_s(lstm_in, len_array)
        pos_embeds = self.pos_emb(tag_array)
        # if not self.cfg.pca:
        #     pos_embeds = self.emb_to_pos(pos_embeds)
        # here we try to concat the two embedding
        embs = [self.dropout(pos_embeds), self.dropout(self.emb_to_cat(edu_vectors)), sent_emb]

        params = {}
        if 'd' in mode and 'd' in self.mode:
            d_embs, d, v = self.prepare_decision(embs, None, batch_size, max_len, 'edv')
            params['d'] = self.real_forward(d_embs, d, v, 'decision').reshape(batch_size, max_len, 2, 2, 2)
            del d_embs, d, v
        if 't' in mode and 't' in self.mode:
            v_embs, d, v = self.prepare_transition(embs, None, batch_size, max_len, 'edv')
            h = self.real_forward(v_embs, d, v, 'transition')
            params['t'] = self.transition_param_helper(tag_array, h)
            # params['t'] = h # self.transition_param_helper(tag_array, h)
            del v_embs, d, v
        if 'r' in mode and 'r' in self.mode:
            r_embs, d, v = self.prepare_root(embs, None, batch_size, max_len, 'edv')
            params['r'] = self.real_forward(r_embs, d, v, 'root').view(batch_size, max_len)
            del r_embs, d, v
        return [params.get(m) for m in mode]

    def loss(self, predict_params: List[Tensor], counts: List[Tensor], mask, mode: str = 'tdr') -> Tensor:
        loss = torch.zeros(1, device=self.cfg.device)

        predict_params = {k: v for k, v in zip(mode, predict_params)}
        counts = {k: v for k, v in zip(mode, counts)}
        real_size = predict_params['d'].size()[1]
        if 'd' in mode and 'd' in self.mode:
            d_mask = self.prepare_decision(None, mask, None, None, mode='m')
            loss += self._loss(predict_params['d'], counts['d'][:, :real_size], d_mask)

        if 't' in mode and 't' in self.mode:
            t_mask = self.prepare_transition(None, mask, None, None, mode='m')
            loss += self._loss(predict_params['t'], counts['t'][:, :real_size, :real_size], t_mask)

        if 'r' in mode and 'r' in self.mode:
            r_mask = self.prepare_root(None, mask, None, None, mode='m')
            loss += self._loss(predict_params['r'], counts['r'][:, :real_size], r_mask)

        return loss / torch.sum(mask)

    def real_forward(self, emb_buffer: List[Tensor], direction: Tensor, valence: Tensor, mode: str):
        if mode == 'decision':
            v_emb = self.dv_emb
        elif mode == 'transition':
            v_emb = self.cv_emb
        elif mode == 'root':
            v_emb = self.rv_emb
        else:
            raise ValueError('bad mode')
        emb_buffer.append(v_emb(valence))
        h = torch.cat(emb_buffer, dim=-1)
        del emb_buffer

        h = self.dropout(h)
        left_right_h = self.activate(self.left_right_linear(h))
        left_h = left_right_h[:, :self.cfg.dim_hidden]
        right_h = left_right_h[:, self.cfg.dim_hidden:]

        left_h[direction == 1, :] = 0.
        right_h[direction == 0, :] = 0.
        h = left_h + right_h

        h = self.dropout(h)
        if mode == 'decision':
            h = self.decision_out_linear(self.activate(self.decision_linear(h)))
        elif mode == 'transition':
            # if self.cfg.use_emb_as_w:
            # all_pos = torch.arange(self.pos_emb.num_embeddings, device=self.cfg.device)
            # w = torch.cat([self.pos_emb_out, self.word_emb(self.word_idx)])
            # if self.cfg.use_pair:
            #    w_pos = torch.cat([self.pos_emb(all_pos), self.pos_emb(self.pos_idx)])
            #     w = torch.cat([w, w_pos], dim=1)
            # w = w.T
            # h = torch.mm(self.activate(self.child_linear(h)), w)
            # else:
            h = self.child_out_linear(self.activate(self.child_linear(h)))
            #

            # batch_size, max_len, _ = self.cs_embedding.size()
            #
            # # edu repesentation
            # edu_represtation = self.child_word_linear(self.cs_embedding).reshape(batch_size, max_len, 1, 1, -1).\
            #     expand(batch_size, max_len, 2, self.cfg.cv, -1)
            # h = torch.matmul(self.activate(self.child_linear(h)).reshape_as(edu_represtation).permute(0, 2, 3, 1, 4).
            #               reshape(-1, max_len, self.cfg.dim_pre_out_child),
            #               edu_represtation.permute(0, 2, 3, 4, 1).reshape(-1, self.cfg.dim_pre_out_child, max_len,)).\
            #     reshape(batch_size, 2, self.cfg.cv, max_len, max_len)
            # # h = torch.matmul(self.activate(self.child_linear(h)).reshape_as(edu_represtation).permute(0, 2, 1),
            # #                  edu_represtation)
            # length_mask = (self.edus_len == 0).long() * (torch.min(h)-1e20)
            # h = h + self.sent_map.view(batch_size, 1, 1, max_len, max_len)
            # # h = h + (2*torch.max(torch.abs(h))*self.sent_map).view(batch_size, 1, 1, max_len, max_len)
            # prob_h = torch.log_softmax((h + length_mask.view(batch_size, 1, 1, max_len, 1) +
            #                             length_mask.view(batch_size, 1, 1, 1, max_len)), dim=-1)
            # return prob_h.permute(0, 3, 4, 1, 2)

        elif mode == 'root':
            h = self.root_out_linear(self.activate(self.root_linear(h)))
        else:
            raise ValueError('bad mode')

        return torch.log_softmax(h, dim=-1)

    @staticmethod
    def _loss(forward_out: Tensor, target_count: Tensor, mask: Tensor) -> Tensor:
        # mask.to(torch.long) for compatibility (torch <= 1.2)
        forward_out = forward_out.flatten()
        target_count = target_count.flatten()
        mask = mask.flatten()
        batch_loss = -torch.sum(target_count * forward_out * mask.to(torch.float))
        return batch_loss

    def prepare_root(self, arrays_to_expand, mask, batch_size, max_len, mode: str = 'edvm') \
            -> Union[Tensor, List[Tensor]]:
        # arrays in arrays_to_expand should has shape (batch_size, hidden) or (batch_size, max_len, hidden)
        out = {}
        if 'e' in mode:
            expanded = []
            for array in arrays_to_expand:
                array = array.view(batch_size, -1, array.shape[-1])
                array = array.expand(-1, max_len, -1)
                array = array.reshape(batch_size * max_len, -1)
                expanded.append(array)
            out['e'] = expanded
        if 'd' in mode:
            out['d'] = torch.ones(batch_size * max_len, dtype=torch.long, device=self.cfg.device)
        if 'v' in mode:
            out['v'] = torch.zeros(batch_size * max_len, dtype=torch.long, device=self.cfg.device)
        if 'm' in mode:
            out['m'] = mask.view(-1)

        return [out.get(m) for m in mode] if len(mode) != 1 else out[mode]

    def prepare_decision(self, arrays_to_expand, mask, batch_size, max_len, mode: str = 'edvm') \
            -> Union[Tensor, List[Tensor]]:
        # arrays in arrays_to_expand should has shape (batch_size, hidden) or (batch_size, max_len, hidden)
        out = {}
        if 'e' in mode:
            expanded = []
            for array in arrays_to_expand:
                array = array.view(batch_size, -1, 1, array.shape[-1])
                array = array.expand(-1, max_len, 4, -1)
                array = array.reshape(batch_size * max_len * 4, -1)
                expanded.append(array)
            out['e'] = expanded
        if 'd' in mode:
            direction_array = torch.zeros(
                batch_size * max_len, 2, 2, dtype=torch.long, device=self.cfg.device)
            direction_array[:, 1, :] = 1
            direction_array = direction_array.view(-1)
            out['d'] = direction_array
        if 'v' in mode:
            valence_array = torch.zeros(
                batch_size * max_len * 2, 2, dtype=torch.long, device=self.cfg.device)
            valence_array[:, 1] = 1
            valence_array = valence_array.view(-1)
            out['v'] = valence_array
        if 'm' in mode:
            out['m'] = mask.unsqueeze(-1).expand(-1, -1, 8).reshape(-1)
        return [out.get(m) for m in mode] if len(mode) != 1 else out[mode]

    def prepare_transition(self, arrays_to_expand, mask, batch_size, max_len, mode: str = 'edvm') \
            -> Union[Tensor, List[Tensor]]:
        out = {}
        if 'e' in mode:
            expanded = []
            for array in arrays_to_expand:
                array = array.view(batch_size, -1, 1, array.shape[-1])
                array = array.expand(-1, max_len, 2 * self.cfg.cv, -1)
                array = array.reshape(batch_size * max_len * 2 * self.cfg.cv, -1)
                expanded.append(array)
            out['e'] = expanded
        if 'd' in mode:
            direction_array = torch.zeros(batch_size * max_len, 2, self.cfg.cv, dtype=torch.long,
                device=self.cfg.device)
            direction_array[:, 1, :] = 1
            direction_array = direction_array.view(-1)
            out['d'] = direction_array
        if 'v' in mode:
            if self.cfg.cv == 2:
                valence_array = torch.zeros(batch_size * max_len * 2, self.cfg.cv, dtype=torch.long,
                    device=self.cfg.device)
                valence_array[:, 1] = 1
                valence_array = valence_array.view(-1)
            else:
                valence_array = torch.zeros(batch_size * max_len * 2, dtype=torch.long, device=self.cfg.device)
            out['v'] = valence_array
        if 'm' in mode:
            out['m'] = (mask.unsqueeze(2) & mask.unsqueeze(1)).unsqueeze(3).expand(-1, -1, -1, self.cfg.cv).reshape(-1)
        return [out.get(m) for m in mode] if len(mode) != 1 else out[mode]

    def transition_param_helper(self, tag_array, forward_output):
        """convert (batch, seq_len, 2, self.cfg.cv, seq_len) to (batch, seq_len, seq_len, [direction,] self.cfg.cv)"""
        batch_size, max_len = tag_array.shape
        forward_output = forward_output.view(batch_size, max_len, 2, self.cfg.cv, self.cfg.cluster)
        index = tag_array.view(batch_size, 1, 1, 1, max_len).expand(-1, max_len, 2, self.cfg.cv, -1)
        param = torch.gather(forward_output, 4, index).permute(0, 1, 4, 2, 3).contiguous()
        return param

    def transition_param_helper_2(self, param: Tensor) -> Tensor:
        """only preserve valid direction"""
        batch_size, max_len, *_ = param.shape
        index = torch.ones(batch_size, max_len, max_len, 1, self.cfg.cv, dtype=torch.long,
                           device=self.cfg.device)
        for i in range(max_len):
            index[:, i, :i] = 0
        param = torch.gather(param, 3, index).squeeze(3)
        return param
