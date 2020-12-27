import os
from functools import lru_cache, partial
from itertools import zip_longest
from typing import Callable, List, Optional, Tuple, Dict

from easydict import EasyDict
import torch
from torch import Tensor, nn

import treesamplers
import utils
from utils.common import *
from utils.data import ConllDataset, prefetcher, ScidtbDataset
from utils.utils import push_front, scatter_add

try:
    import dmv_extension

    CUDA_BACKTRACKING_READY = True
except ImportError:
    dmv_extension = None
    CUDA_BACKTRACKING_READY = False


# SOFTMAX_EM_SIGMA_TYPE = Union[float, Tuple[float, float, int]]


# noinspection PyUnusedLocal
def dmv_config_template():
    num_tag: int = 0
    max_len: int = 10  # max len for all dataset
    cv: int = 1  # 1 or 2
    e_step_mode: str = 'em'  # em or viterbi
    count_smoothing: float = 1e-1  # smooth for init_param
    param_smoothing: float = 1e-1  # smooth for m_step

    # ===== extentions =====
    #
    # # see `Unambiguity Regularization for Unsupervised Learning of Probabilistic Grammars`
    # use_softmax_em: bool = False
    # # if Tuple[float, float, int], use annealing
    # # Tuple[float, float, int] means start_sigma, end_sigma, duration.
    # softmax_em_sigma: SOFTMAX_EM_SIGMA_TYPE = (1., 0., 100)
    # # if sigma bigger than this threshold, run viterbi to avoid overflow
    # softmax_em_sigma_threshold: float = 0.9
    # # if True, call softmax_em_step automatically when m_step is called.
    # softmax_em_auto_step: bool = True
    #
    # # backoff_rate*p(r|child_tag, dir, cv) + (1-backoff_rate)*p(r|parent_tag, child_tag, dir, cv)
    # # FIXME bad code
    # use_child_backoff: bool = False
    # child_backoff_rate: float = 0.33


class DMV(nn.Module):
    def __init__(self, cfg: Dict, mode: str = 'tdr'):

        super().__init__()
        self.cfg = EasyDict(cfg)
        self.mode = mode

        n, c = self.cfg.num_tag, self.cfg.cv

        self.root_param = nn.Parameter(torch.zeros(n), requires_grad=True) if 'r' in mode else None
        self.dec_param = nn.Parameter(torch.zeros(n, 2, 2, 2), requires_grad=True) if 'd' in mode else None
        self.trans_param = nn.Parameter(torch.zeros(n, n, 2, c), requires_grad=True) if 't' in mode else None

        self.function_mask_set = None  # for UD

    def forward(self, final_trans_scores: Tensor, final_dec_scores: Tensor, len_array: Tensor) \
            -> Tensor:
        """final_scores contain ROOT compared to scores
        :param final_trans_scores: Shape[batch, head, child, cv]
        :param final_dec_scores: Shape[batch, position, direction, dv, decision]
        :param len_array: Shape[batch]
        :return: likelihood for each instance
        """
        mode = 'sum' if self.cfg.e_step_mode == 'em' else 'max'
        # *_, prob = batch_inside(final_trans_scores, final_dec_scores, len_array, mode)
        *_, prob = batch_inside(final_trans_scores, final_dec_scores, len_array, mode)
        return prob

    def parse(self, final_trans_scores: Tensor, final_dec_scores: Tensor, len_array: Tensor,
              sent_map_array:Optional[Tensor]=None, paragraph_map_array:Optional[Tensor]=None) -> Tuple[List[List[int]], float]:
        """final_scores contain ROOT compared to scores"""

        ## BP to get counts
        # self.zero_grad()
        #
        # final_trans_scores = final_trans_scores.detach().requires_grad_()
        # final_dec_scores = final_dec_scores.detach().requires_grad_()
        #
        # *_, prob = batch_inside(final_trans_scores, final_dec_scores, len_array, 'max')
        # ll = torch.sum(prob)
        # ll.backward()
        #
        # result = final_trans_scores.grad.nonzero()
        # result = torch.split(result[:, 1:], torch.unique(result[:, 0], return_counts=True)[1].tolist())
        # result2 = []
        # for i, r in enumerate(result):
        #     r = r[torch.sort(r[:, 1])[1]][:, 0].tolist()
        #     assert len(r) == len_array[i]
        #     result2.append(r)
        # ll = ll.item()

        ## Directly calculate counts
        # heads, head_valences, valences = batch_parse(final_trans_scores, final_dec_scores, len_array)
        sent_map_np = sent_map_array.cpu().detach().numpy()
        paragraph_map_np = paragraph_map_array.cpu().detach().numpy()
        heads, head_valences, valences = batch_discourse_parse(final_trans_scores, final_dec_scores, len_array,
                                                               sent_map_np, paragraph_map_np)
        # heads = heads.detach().cpu().numpy()
        result2 = []
        len_array = len_array.cpu()
        for i, r in enumerate(heads):
            r = r[1:len_array[i] + 1]
            assert len(r) == len_array[i]
            result2.append(r)
        ll = 0  # TODO

        return result2, ll

    def build_scores(self, tag_array: Tensor, mode: str = 'tdr',
            given: Optional[List[Optional[Tensor]]] = None, using_fake: bool = True) \
            -> List[Optional[Tensor]]:
        """build scores (not final) using self.*_param"""
        # trans_scores:     batch, head, child, cv
        # dec_scores:       batch, head, direction, dv, decision
        # root_scores:      batch, child

        assert given is None or len(mode) == len(given), \
            "`mode` and `given` should have the same len if `given` is not None\n" \
            "None can be used as placeholder in `given`"

        data = {'t': self.trans_param, 'd': self.dec_param, 'r': self.root_param}
        if given:
            given = {m: (p if p is not None else data[m]) for m, p in zip(mode, given)}
        else:
            given = {m: data[m] for m in mode}
        out = {k: None for k in mode}

        n, c = self.cfg.num_tag, self.cfg.cv
        batch_size, length = tag_array.shape

        if given.get('t') is not None:
            t = given['t']
            t = t.unsqueeze(0).expand(batch_size, n, n, 2, c)
            head_pos_index = tag_array.view(*tag_array.shape, 1, 1, 1).expand(batch_size, length, n, 2, c)
            child_pos_index = tag_array.view(batch_size, 1, length, 1, 1).expand(batch_size, length, length, 2, c)
            trans_scores = torch.gather(torch.gather(t, 1, head_pos_index), 2, child_pos_index)
            index = torch.triu(torch.ones(length, length, dtype=torch.long, device=self.cfg.device)) \
                .view(1, length, length, 1, 1).expand(batch_size, length, length, 1, c)
            trans_scores = torch.gather(trans_scores, 3, index).squeeze(3)
            if using_fake:
                trans_scores[:, :, 0] = -1e20
                trans_scores[:, 0, :] = -1e20
            out['t'] = trans_scores

        if given.get('d') is not None:
            d = given['d']
            d = d.unsqueeze(0).expand(batch_size, n, 2, 2, 2)
            head_pos_index = tag_array.view(*tag_array.shape, 1, 1, 1).expand(batch_size, length, 2, 2, 2)
            dec_scores = torch.gather(d, 1, head_pos_index)
            if using_fake:
                dec_scores[:, 0] = 0
            out['d'] = dec_scores

        if given.get('r') is not None:
            r = given['r']
            r = r.unsqueeze(0).expand(batch_size, n)
            root_scores = torch.gather(r, 1, tag_array)
            out['r'] = root_scores

        return [out[m] for m in mode]

    @staticmethod
    def build_final_trans_scores(trans_scores: Tensor, root_scores: Tensor, using_fake: bool = True,
            require_grad: bool = False) -> Tensor:
        if not using_fake:
            trans_scores = push_front(trans_scores, -1e20, 1)
            trans_scores = push_front(trans_scores, -1e20, 2)
        else:
            root_scores = root_scores[:, 1:]

        trans_scores[:, 0, 1:, :] = root_scores.unsqueeze(-1)
        if require_grad:
            trans_scores.retain_grad()
        return trans_scores

    @staticmethod
    def build_final_dec_scores(dec_scores: Tensor, using_fake: bool = True,
            require_grad: bool = False) -> Tensor:
        if not using_fake:
            dec_scores = push_front(dec_scores, 0, 1)
        if require_grad:
            dec_scores.retain_grad()
        return dec_scores

    @staticmethod
    def merge_scores(group1, group2):
        out = []
        for g1, g2 in zip_longest(group1, group2):
            if g1 is not None:
                out.append(g1)
            elif g2 is not None:
                out.append(g2)
            else:
                raise ValueError("None in both group")
        return out

    def update_param_with_count(self, mode: str = 'tdr', smooth: Optional[float] = None,
            given: Optional[List[Optional[Tensor]]] = None, inplace: bool = True) \
            -> List[Optional[Tensor]]:
        # assume grad is calculated by `torch.sum(-ll).backward()`
        assert given is None or len(mode) == len(given), \
            "`mode` and `given` should have the same len if `given` is not None\n" \
            "None can be used as placeholder in `given`"

        data = {'t': self.trans_param, 'd': self.dec_param, 'r': self.root_param}
        data = [data[m] for m in mode]
        out = []
        smooth = smooth or self.cfg.param_smoothing

        for i, d in enumerate(data):
            count = given[i] if given else d.grad
            if count is not None:
                data = torch.log(count + smooth)
                out.append(data)
                if inplace:
                    d.data = data
            else:
                out.append(None)
        return out

    def normalize_param(self) -> None:
        # expect log(x) as input
        data = [(self.trans_param, 1), (self.dec_param, 3), (self.root_param, 0)]

        for d, axis in data:
            if d is not None:
                d.data = torch.log_softmax(d, dim=axis)

    # initializers

    def km_init(self, dataset: ConllDataset, getter: Optional[Callable] = None):
        if getter is None:
            def getter(a, b): return b

        harmonic_sum = [0., 1.]

        dec_param = self.dec_param.data.fill_(0.)
        root_param = self.root_param.data.fill_(0.)
        trans_param = self.trans_param.data.fill_(0.)

        def get_harmonic_sum(n):
            nonlocal harmonic_sum
            while n >= len(harmonic_sum):
                harmonic_sum.append(harmonic_sum[-1] + 1 / len(harmonic_sum))
            return harmonic_sum[n]

        def update_decision(_change, _norm_counter, _tag_array):
            for i in range(_tag_array.shape[1]):
                pos = _tag_array[:, i]
                for _direction in (0, 1):
                    if _change[i, _direction] > 0:
                        scatter_add(_norm_counter, (pos, _direction, NOCHILD, GO), 1)
                        scatter_add(_norm_counter, (pos, _direction, HASCHILD, GO), -1)
                        scatter_add(dec_param, (pos, _direction, HASCHILD, GO), _change[i, _direction])
                        scatter_add(_norm_counter, (pos, _direction, NOCHILD, STOP), -1)
                        scatter_add(_norm_counter, (pos, _direction, HASCHILD, STOP), 1)
                        scatter_add(dec_param, (pos, _direction, NOCHILD, STOP), 1)
                    else:
                        scatter_add(dec_param, (pos, _direction, NOCHILD, STOP), 1)

        def first_child_update(_norm_counter):
            all_param = dec_param.flatten()
            all_norm = _norm_counter.flatten()
            mask = (all_param <= 0) | (0 <= all_norm)
            ratio = - all_param / all_norm
            ratio[mask] = 1.
            return torch.min(ratio)

        norm_counter = torch.zeros_like(self.dec_param)
        change = torch.zeros((self.cfg.max_len, 2), device=self.dec_param.device)
        loader = dataset.get_dataloader(True, 10000, False, 0, self.cfg.num_worker, 1, self.cfg.max_len_train)
        loader = prefetcher(loader) if self.cfg.device != 'cpu' else loader
        for arrays in loader:
            tag_array = getter(arrays.word_array, arrays.pos_array)
            batch_size, word_num = tag_array.shape
            change.fill_(0.)
            scatter_add(root_param, [tag_array], 1. / word_num)
            if word_num > 1:
                for child_i in range(word_num):
                    child_sum = get_harmonic_sum(child_i - 0) + get_harmonic_sum(word_num - child_i - 1)
                    scale = (word_num - 1) / word_num / child_sum
                    for head_i in range(word_num):
                        if child_i == head_i:
                            continue
                        direction = 0 if head_i > child_i else 1
                        head_pos = tag_array[:, head_i]
                        child_pos = tag_array[:, child_i]
                        diff = scale / abs(head_i - child_i)
                        scatter_add(trans_param, (head_pos, child_pos, direction), diff)
                        change[head_i, direction] += diff
            update_decision(change, norm_counter, tag_array)

        trans_param += self.cfg.count_smoothing
        dec_param += self.cfg.count_smoothing
        root_param += self.cfg.count_smoothing

        es = first_child_update(norm_counter)
        norm_counter *= 0.9 * es
        dec_param += norm_counter

        root_param_sum = torch.sum(root_param)
        trans_param_sum = torch.sum(trans_param, dim=1, keepdim=True)
        decision_param_sum = torch.sum(dec_param, dim=3, keepdim=True)

        root_param /= root_param_sum
        trans_param /= trans_param_sum
        dec_param /= decision_param_sum

        torch.log(trans_param, out=trans_param)
        torch.log(root_param, out=root_param)
        torch.log(dec_param, out=dec_param)

    def good_init(self, dataset: ConllDataset, getter: Optional[Callable] = None):
        if getter is None:
            def getter(a, b): return b
        heads, valences, head_valences = self.batch_recovery(dataset)
        len_array = dataset.get_all_len().to(self.cfg.device)
        _loader = iter(dataset.get_dataloader(False, len(dataset), False, 0, 0))
        _loader = prefetcher(_loader) if self.cfg.device != 'cpu' else _loader
        _batch = next(_loader)
        tag_array = getter(_batch[2], _batch[1])
        r, t, d = self.calcu_viterbi_count(heads, head_valences, valences, len_array)

        r = self.get_tag_counter_root(r, tag_array, mode=1)
        t = self.get_tag_counter_trans(t, tag_array, mode=1)
        d = self.get_tag_counter_dec(d, tag_array, mode=1)
        torch.log(r + self.cfg.param_smoothing, out=self.root_param.data)
        torch.log(t + self.cfg.param_smoothing, out=self.trans_param.data)
        torch.log(d + self.cfg.param_smoothing, out=self.dec_param.data)
        del r, t, d
        self.normalize_param()

    def branching_init(self, dataset: ConllDataset, direction: str = 'right', getter: Optional[Callable] = None):
        assert direction in ('left', 'right')
        # direction: right-branching = left-head
        offset = 0 if direction == 'right' else 2
        for instance in dataset:
            parent_ids = list(range(offset, len(instance) + offset))
            for entry, parent_id in zip(instance, parent_ids):
                entry.misc = parent_id
            if direction == 'left':
                entry.misc = 0  # noqa
        self.good_init(dataset, getter)

    def bidirectional_branching_init(self, dataset: ConllDataset, getter: Optional[Callable] = None):
        _loader = dataset.get_dataloader(False, len(dataset), False, False, 0)
        _loader = prefetcher(_loader) if self.cfg.device != 'cpu' else _loader
        _batch = next(iter(_loader))
        tag_array = getter(_batch[2], _batch[1])
        len_array = dataset.get_all_len().to(self.cfg.device)
        # right-branching
        for instance in dataset:
            parent_ids = list(range(len(instance)))
            for entry, parent_id in zip(instance, parent_ids):
                entry.misc = parent_id
        heads, valences, head_valences = self.batch_recovery(dataset)
        r, t, d = self.calcu_viterbi_count(heads, head_valences, valences, len_array)

        # left-branching
        for instance in dataset:
            parent_ids = list(range(2, len(instance) + 2))
            for entry, parent_id in zip(instance, parent_ids):
                entry.misc = parent_id
            entry.misc = 0  # noqa
        heads, valences, head_valences = self.batch_recovery(dataset)
        r_l, t_l, d_l = self.calcu_viterbi_count(heads, head_valences, valences, len_array)
        r += r_l
        t += t_l
        d += d_l
        r = self.get_tag_counter_root(r, tag_array, mode=1)
        t = self.get_tag_counter_trans(t, tag_array, mode=1)
        d = self.get_tag_counter_dec(d, tag_array, mode=1)
        torch.log(r + self.cfg.param_smoothing, out=self.root_param.data)
        torch.log(t + self.cfg.param_smoothing, out=self.trans_param.data)
        torch.log(d + self.cfg.param_smoothing, out=self.dec_param.data)
        del r, t, d, r_l, d_l, t_l
        self.normalize_param()

    def random_init(self):
        if 'r' in self.mode:
            nn.init.normal_(self.root_param.data)
        if 't' in self.mode:
            nn.init.normal_(self.trans_param.data)
        if 'd' in self.mode:
            nn.init.normal_(self.dec_param.data)
        self.normalize_param()

    def uniform_init(self):
        if 'r' in self.mode:
            nn.init.constant_(self.root_param.data, 1.)
        if 't' in self.mode:
            nn.init.constant_(self.trans_param.data, 1.)
        if 'd' in self.mode:
            nn.init.constant_(self.dec_param.data, 1.)
        self.normalize_param()

    # tools
    def calcu_em_count_forward(self, final_trans_scores: Tensor, final_dec_scores: Tensor, len_array: Tensor,
            require_likelihood: bool = False):
        # TODO implement batch_outside
        raise NotImplementedError

    def calcu_viterbi_count_forward(self, final_trans_scores: Tensor, final_dec_scores: Tensor, len_array: Tensor,
            require_likelihood: bool = False):

        batch_size, fake_len, fake_len, cv = final_trans_scores.shape
        real_len = fake_len - 1
        device = final_trans_scores.device
        batch_likelihood = 0.

        batch_trans_trace = torch.zeros((batch_size, fake_len, fake_len, 2, cv), device=device)
        batch_dec_trace = torch.zeros((batch_size, fake_len, fake_len, 2, 2, 2), device=device)
        heads, head_valences, valences = batch_parse(final_trans_scores, final_dec_scores, len_array)

        ## cupy and for-loop version
        # batch_arange = cp.arange(batch_size)
        # for m in range(1, fake_len):
        #     h = heads[:, m]
        #     direction = (h <= m).astype(cpi)
        #     h_valence = head_valences[:, m]
        #     m_valence = valences[:, m]
        #     m_child_valence = h_valence if self.o.cv > 1 else cp.zeros_like(h_valence)
        #
        #     len_mask = ((h <= len_array) & (m <= len_array))
        #     if DEBUG and ((m <= len_array) & (h > len_array)).any():
        #         print('find bad arc')
        #
        #     batch_likelihood += cp.sum(dec_scores[batch_arange, m, 0, m_valence[:, 0], STOP][len_mask])
        #     batch_likelihood += cp.sum(dec_scores[batch_arange, m, 1, m_valence[:, 1], STOP][len_mask])
        #     self.batch_dec_trace[batch_arange, m - 1, m - 1, 0, m_valence[:, 0], STOP] = len_mask
        #     self.batch_dec_trace[batch_arange, m - 1, m - 1, 1, m_valence[:, 1], STOP] = len_mask
        #
        #     head_mask = h == 0
        #     mask = head_mask * len_mask
        #     if mask.any():
        #         # when use_torch_in_cupy_malloc(), mask.any()=False will raise a NullPointer error
        #         batch_likelihood += cp.sum(trans_scores[:, 0, m, 0][mask])
        #         cpx.scatter_add(self.root_counter, tag_array[:, m], mask)
        #
        #     head_mask = ~head_mask
        #     mask = head_mask * len_mask
        #     if mask.any():
        #         batch_likelihood += cp.sum(trans_scores[batch_arange, h, m, m_child_valence][mask])
        #         batch_likelihood += cp.sum(dec_scores[batch_arange, h, direction, h_valence, GO][mask])
        #         self.batch_trans_trace[batch_arange, h - 1, m - 1, direction, m_child_valence] = mask
        #         self.batch_dec_trace[batch_arange, h - 1, m - 1, direction, h_valence, GO] = mask
        #

        with torch.no_grad():
            _len_arange = torch.arange(fake_len, device=device).unsqueeze(0).expand(batch_size, -1)
            _batch_arange = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, fake_len)
            direction = (heads < _len_arange).to(torch.long).flatten()
            len_mask = ((heads <= len_array.unsqueeze(1)) & (_len_arange <= len_array.unsqueeze(1)))
            # len_mask2 = len_mask.clone()
            len_mask[:, 0] = False
            len_mask = len_mask.flatten()
            # len_mask = ((heads <= len_array.unsqueeze(1)) & (_len_arange <= len_array.unsqueeze(1))).flatten()
            heads = heads.flatten()
            m_child_valence = (head_valences if cv > 1 else torch.zeros_like(head_valences)).flatten()
            head_valences = head_valences.flatten()

            _len_arange = _len_arange.flatten()
            _batch_arange = _batch_arange.flatten()
            valences = valences.view(-1, 2)

            batch_dec_trace[_batch_arange, _len_arange, _len_arange, 0, valences[:, 0], STOP] = len_mask.to(torch.float)
            batch_dec_trace[_batch_arange, _len_arange, _len_arange, 1, valences[:, 1], STOP] = len_mask.to(torch.float)
            if require_likelihood:
                # dec_scores[:, m, 0, valences[:, m, 0], STOP]
                batch_likelihood += torch.sum(final_dec_scores[_batch_arange, _len_arange, 0, valences[:, 0], STOP][len_mask])
                batch_likelihood += torch.sum(final_dec_scores[_batch_arange, _len_arange, 1, valences[:, 1], STOP][len_mask])

            head_mask = heads != 0
            mask = head_mask & len_mask
            if require_likelihood:
                batch_likelihood += torch.sum(final_trans_scores[_batch_arange, heads, _len_arange, m_child_valence][mask])
                batch_likelihood += torch.sum(final_dec_scores[_batch_arange, heads, direction, head_valences.flatten(), GO][mask])
            batch_trans_trace[_batch_arange, heads, _len_arange, direction, m_child_valence] = mask.flatten().to(torch.float)
            batch_dec_trace[_batch_arange, heads, _len_arange, direction, head_valences, GO] = mask.flatten().to(torch.float)

            mask = (~head_mask) & len_mask
            if require_likelihood:
                batch_likelihood += torch.sum(final_trans_scores[_batch_arange, 0, _len_arange, NOCHILD][mask])
            batch_trans_trace[_batch_arange, 0, _len_arange, 1, NOCHILD] = mask.flatten().to(torch.float)

        transition_count = batch_trans_trace[:, 1:, 1:, :].clone()
        root_count = batch_trans_trace[:, 0, 1:, 1, NOCHILD].clone()
        dec_count = batch_dec_trace[:, 1:, 1:, :].clone()
        return transition_count, dec_count, root_count, batch_likelihood

    def calcu_viterbi_count(self, head: Tensor, head_valence: Tensor, valence: Tensor, len_array: Tensor) \
            -> Tuple[Tensor, Tensor, Tensor]:
        """use head, head_valence, valence to calculate viterbi traces"""
        batch_size, fake_len = head.shape
        real_len = fake_len - 1
        batch_arange = torch.arange(batch_size, device=head.device)

        root_trace = torch.zeros((batch_size, real_len), device=head.device)
        trans_trace = torch.zeros((batch_size, real_len, real_len, 2, self.cfg.cv), device=head.device)
        dec_trace = torch.zeros((batch_size, real_len, real_len, 2, 2, 2), device=head.device)

        with torch.no_grad():
            for m in range(1, fake_len):
                h = head[:, m]
                direction = (h <= m).to(torch.long)
                h_valence = head_valence[:, m]
                m_valence = valence[:, m]
                m_child_valence = h_valence if self.cfg.cv > 1 else torch.zeros_like(h_valence)
                len_mask = ((h <= len_array) & (m <= len_array)).to(dtype=torch.float)

                dec_trace[batch_arange, m - 1, m - 1, 0, m_valence[:, 0], STOP] = len_mask
                dec_trace[batch_arange, m - 1, m - 1, 1, m_valence[:, 1], STOP] = len_mask

                head_mask = h == 0
                mask = head_mask * len_mask
                root_trace[batch_arange, m - 1] = mask

                head_mask = ~head_mask
                mask = head_mask * len_mask
                trans_trace[batch_arange, h - 1, m - 1, direction, m_child_valence] = mask
                dec_trace[batch_arange, h - 1, m - 1, direction, h_valence, GO] = mask

        return root_trace, trans_trace, dec_trace

    @staticmethod
    def recovery_one(heads: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        left_most = np.arange(len(heads))
        right_most = np.arange(len(heads))
        for idx, each_head in enumerate(heads):
            if each_head == 0 or each_head == len(heads) + 1:  # skip head is ROOT
                continue
            each_head -= 1
            if idx < left_most[each_head]:
                left_most[each_head] = idx
            if idx > right_most[each_head]:
                right_most[each_head] = idx

        valences = npizeros((len(heads), 2))
        head_valences = npizeros(len(heads))

        for idx, each_head in enumerate(heads):
            each_head -= 1
            valences[idx, 0] = NOCHILD if left_most[idx] == idx else HASCHILD
            valences[idx, 1] = NOCHILD if right_most[idx] == idx else HASCHILD
            if each_head >= 0:
                if each_head > idx:
                    head_valences[idx] = NOCHILD if left_most[each_head] == idx else HASCHILD
                else:
                    head_valences[idx] = NOCHILD if right_most[each_head] == idx else HASCHILD
            else:
                head_valences[idx] = NOCHILD  # NO USAGE, just avoid out of bound
        return valences, head_valences

    def batch_recovery(self, dataset: ScidtbDataset, predicted='RB_RB_RB') -> Tuple[Tensor, Tensor, Tensor]:
        # utils.ex.logger.info(f"using {predicted} as predicted parse tree")
        print(f"using {predicted} as predicted parse tree")

        sampler = treesamplers.TreeSampler(predicted.split("_"))

        heads = npizeros((len(dataset), self.cfg.max_len + 1))
        valences = npizeros((len(dataset), self.cfg.max_len + 1, 2))
        head_valences = npizeros((len(dataset), self.cfg.max_len + 1))

        for idx, instance in enumerate(dataset):
            # one_heads = npasarray(list(map(int, getattr(instance, predicted))))
            one_heads = sampler.sample(inputs=instance.edu_ids, edus=instance.edus,
                                       edus_head=instance.edus_head, sbnds=instance.sbnds, pbnds=instance.pbnds,
                                       has_root=False)
            # one_heads = instance.arcs

            one_heads = np.asarray([rule[0] for rule in sorted(one_heads, key=lambda x: x[1])])
            one_valences, one_head_valences = self.recovery_one(one_heads)
            heads[idx, 1:len(instance) + 1] = one_heads
            valences[idx, 1:len(instance) + 1] = one_valences
            head_valences[idx, 1:len(instance) + 1] = one_head_valences

        heads = torch.tensor(heads, device=self.cfg.device)
        valences = torch.tensor(valences, device=self.cfg.device)
        head_valences = torch.tensor(head_valences, device=self.cfg.device)
        return heads, valences, head_valences

    def get_tag_counter_root(self, root_trace: Tensor, tag_array: Tensor,
            num_tag: Optional[int] = None, mode: int = 0) -> Tensor:
        """
        mode=0, sum in sentence,  trace[batch, real_len] to counter[batch, num_tag]
        mode=1, sum over batch, trace[batch, real_len] to counter[num_tag]
        """
        batch_size, max_len = tag_array.shape
        num_tag = num_tag or self.cfg.num_tag
        with torch.no_grad():
            if mode == 0:
                root_out = torch.zeros(batch_size, num_tag, device=root_trace.device)
                sentence_id = torch.arange(
                    batch_size, device=root_trace.device).unsqueeze(-1).expand(batch_size, max_len)
                index = sentence_id.flatten() * batch_size + tag_array.flatten()
            else:
                root_out = torch.zeros(num_tag, device=root_trace.device)
                index = tag_array.flatten()
            root_out.put_(index, root_trace, accumulate=True)
        return root_out

    def get_tag_counter_dec(self, dec_trace: Tensor, tag_array: Tensor,
            num_tag: Optional[int] = None, mode: int = 0) -> Tensor:
        """
        mode=0, sum in sentence,  trace[batch, sentence_len, ...] to counter[batch, num_tag, ...]
        mode=1, sum over batch, trace[batch, sentence_len, ...] to counter[num_tag, ...]
        """
        batch_size, max_len = tag_array.shape
        num_tag = num_tag or self.cfg.num_tag
        dec_post_dim = (2, 2, 2)
        with torch.no_grad():
            dec_trace = torch.sum(dec_trace, 2)
            if mode == 0:
                dec_out = torch.zeros(batch_size, num_tag, *dec_post_dim, device=dec_trace.device)
                sentence_id = torch.arange(
                    batch_size, device=dec_trace.device).unsqueeze(-1).expand(batch_size, max_len)
                index = sentence_id.flatten() * num_tag + tag_array.flatten()
            else:
                dec_out = torch.zeros(num_tag, *dec_post_dim, device=dec_trace.device)
                index = tag_array.flatten()
            index = (8 * index).unsqueeze(-1).repeat(1, 8)
            offset = torch.arange(8, device=dec_trace.device).unsqueeze(0)
            index += offset
            dec_out.put_(index, dec_trace, accumulate=True)
        return dec_out

    def get_tag_counter_trans(self, trans_trace: Tensor, tag_array: Tensor,
            num_tag: Optional[int] = None, mode: int = 0) -> Tensor:
        """
         mode=0, sum in sentence,  trace[batch, len, len, ...] to counter[batch, ntag, ntag, ...]
         mode=1, sum over batch, trace[batch, len, len, ...] to counter[ntag, ntag, ...]
         """
        batch_size, max_len = tag_array.shape
        num_tag = num_tag or self.cfg.num_tag
        trans_post_dim = (2, self.cfg.cv)
        with torch.no_grad():
            head_ids = tag_array.unsqueeze(2).expand(-1, -1, max_len).flatten()
            child_ids = tag_array.unsqueeze(1).expand(-1, max_len, -1).flatten()
            if mode == 0:
                trans_out = torch.zeros(batch_size, num_tag, num_tag, *trans_post_dim, device=trans_trace.device)
                sentence_id = torch.arange(
                    batch_size, device=trans_trace.device).unsqueeze(-1).unsqueeze(-1).expand(batch_size, max_len, max_len)
                index = sentence_id.flatten() * num_tag * num_tag + head_ids * num_tag + child_ids
            else:
                trans_out = torch.zeros(num_tag, num_tag, *trans_post_dim, device=trans_trace.device)
                index = head_ids * num_tag + child_ids
            index = (2 * self.cfg.cv * index).unsqueeze(-1).repeat(1, 2 * self.cfg.cv)
            offset = torch.arange(2 * self.cfg.cv, device=trans_trace.device).unsqueeze(0)
            index += offset
            trans_out.put_(index, trans_trace, accumulate=True)
        return trans_out

    def set_function_mask(self, functions_to_mask):
        self.function_mask_set = nn.Parameter(torch.tensor(functions_to_mask), requires_grad=False)
    #
    def function_mask(self, batch_scores, tag_array):
        tag_array = tag_array.unsqueeze(-1)
        mask_set = self.function_mask_set.view(1, 1, -1)
        in_mask = tag_array.eq(mask_set).any(dim=-1)
        batch_scores[in_mask] = -1e5

    def save(self, path):
        torch.save(self.state_dict(), os.path.join(path, 'dmv'))

    def load(self, path):
        self.load_state_dict(torch.load(os.path.join(path, 'dmv')))


class DMVProb(DMV):

    def forward(self, trans_scores, dec_scores, tag_prob, len_array):
        """
        :param trans_scores: Shape[batch, head, ntag, child, cv], child should be weighted avg of each tag
        :param dec_scores: Shape[batch, position, ntag, direction, dv, decision]
        :param tag_prob: Shape[batch, position]
        :param len_array: Shape[batch]
        :return: likelihood for each instance
        """
        mode = 'sum' if self.cfg.e_step_mode == 'em' else 'max'

        # tag_prob = torch.nn.functional.one_hot(tag_array, self.cfg.num_tag).to(torch.float)

        *_, prob = batch_inside_prob(trans_scores, dec_scores, tag_prob, len_array, mode, 'sum')
        return prob

    def parse(self, trans_scores, dec_scores, tag_prob, len_array):
        self.zero_grad()
        trans_scores.retain_grad()

        *_, prob = batch_inside_prob(trans_scores, dec_scores, tag_prob, len_array, 'max', 'max')
        ll = torch.sum(prob)
        ll.backward()

        result = trans_scores.grad.nonzero()
        result = torch.split(result[:, 1:], torch.unique(result[:, 0], return_counts=True)[1].tolist())
        result2 = []
        for i, r in enumerate(result):
            r = r[torch.sort(r[:, 1])[1]][:, 0].tolist()
            assert len(r) == len_array[i]
            result2.append(r)

        return result2, ll.item()

    def prepare_scores(self, tag_prob):
        """param[tag_num, ...] to param[batch, sentence_len, ...]"""
        # trans_scores:     batch, head, ntag, child, cv
        # dec_scores:       batch, head, ntag, direction, dv, decision
        n, c = self.cfg.num_tag, self.cfg.cv
        batch_size, fake_len, _ = tag_prob.shape

        trans_scores = self.trans_param.view(1, 1, n, 1, n, 2, c) \
            .expand(batch_size, fake_len, n, fake_len, n, 2, c)
        tag_prob = tag_prob.view(batch_size, 1, 1, fake_len, n, 1, 1)
        trans_scores = torch.logsumexp(trans_scores + torch.log(tag_prob), dim=4)

        d_indexer = (1 - np.tri(fake_len, k=-1, dtype=int)).reshape(1, fake_len, 1, fake_len, 1, 1)
        d_indexer = torch.tensor(d_indexer, device='cuda') \
            .expand(batch_size, fake_len, n, fake_len, 1, c)
        trans_scores = trans_scores.gather(4, d_indexer).squeeze(4)

        trans_scores[:, 0] = self.root_param.view(1, n, 1, 1)
        trans_scores[:, :, :, 0] = -1e30

        dec_scores = self.dec_param.view(1, 1, *self.dec_param.shape) \
            .expand(batch_size, fake_len, *self.dec_param.shape)

        return trans_scores, dec_scores

    def prepare_tag_array(self, tag_array, tag_prob):
        batch_size, _ = tag_array.shape
        tag_array = torch.cat([torch.zeros(batch_size, 1, dtype=torch.long, device='cuda'), tag_array], dim=1)
        tag_prob = torch.cat([torch.zeros(batch_size, 1, self.cfg.num_tag, device='cuda'), tag_prob], dim=1)
        tag_prob[:, 0, 0] = 1.
        return tag_array, tag_prob


@lru_cache()
def constituent_index(fake_len: int) -> Tuple[
    np.ndarray, np.ndarray, List[int], List[List[int]], List[List[int]],
    List[List[int]], List[List[int]], np.ndarray]:
    """generate span(left,right,direction) index"""

    id2span = []
    span2id = npiempty((fake_len, fake_len, 2))
    for left_idx in range(fake_len):
        for right_idx in range(left_idx, fake_len):
            for direction in range(2):
                span2id[left_idx, right_idx, direction] = len(id2span)
                id2span.append((left_idx, right_idx, direction))
    id2span = npasarray(id2span)

    basic_span = npiempty((2 * fake_len))
    for i in range(fake_len):
        basic_span[2 * i] = span2id[i, i, 0]
        basic_span[2 * i + 1] = span2id[i, i, 1]

    # the order of ijss is important
    ijss = []
    ikis = [[] for _ in range(len(id2span))]
    kjis = [[] for _ in range(len(id2span))]
    ikcs = [[] for _ in range(len(id2span))]
    kjcs = [[] for _ in range(len(id2span))]

    for length in range(1, fake_len):
        for i in range(fake_len - length):
            j = i + length
            id = span2id[i, j, 0]
            ijss.append(id)
            for k in range(i, j):
                # two complete spans to form an incomplete span
                ikis[id].append(span2id[i, k, 1])
                kjis[id].append(span2id[k + 1, j, 0])
                # one complete span, one incomplete span to form a complete span
                ikcs[id].append(span2id[i, k, 0])
                kjcs[id].append(span2id[k, j, 0])
            id = span2id[i, j, 1]
            ijss.append(id)
            for k in range(i, j + 1):
                # two complete spans to form an incomplete span
                if k < j and (i != 0 or k == 0):
                    ikis[id].append(span2id[i, k, 1])
                    kjis[id].append(span2id[k + 1, j, 0])
                # one incomplete span, one complete span to form a complete span
                if k > i:
                    ikcs[id].append(span2id[i, k, 1])
                    kjcs[id].append(span2id[k, j, 1])

    return span2id, id2span, ijss, ikcs, ikis, kjcs, kjis, basic_span

# check whether the span is valid
# the span should cross the bnds
def cross_check(bnds, left, right):
    inner = False
    for bnd_begin, bnd_end in bnds:
        if bnd_begin <= left <= right <= bnd_end:
            inner = True
            break
    return not inner

# consider sbnd boundary
def discourse_constituent_index(dlen: int, sbnds: List[Tuple[int, int]], pbnds: List[Tuple[int, int]]= None) -> Tuple[
    np.ndarray, np.ndarray, List[int], List[List[int]], List[List[int]],
    List[List[int]], List[List[int]], np.ndarray]:
    """generate span(left,right,direction) index"""

    id2span = []
    span2id = npiempty((dlen, dlen, 2))
    root_shifted_sbnds = [(begin+1, end+1) for begin, end in sbnds if begin >= 0]
    root_shifted_sbnds = [(0, 0)] + root_shifted_sbnds

    if pbnds is not None:
        root_shifted_pbnds = [(sbnds[p_begin][0]+1, sbnds[p_end][-1]+1) for p_begin, p_end in pbnds if p_begin >= 0]
        root_shifted_pbnds = [(0, 0)] + root_shifted_pbnds

    # this part consider inner sentence span
    # ignore -1
    for sbnd_begin, sbnd_end in root_shifted_sbnds:
        if sbnd_begin < 0:
            break
        for left_idx in range(sbnd_begin, sbnd_end+1):
            for right_idx in range(left_idx, sbnd_end+1):
                for direction in range(2): # 0 is left 1 is right
                    span2id[left_idx, right_idx, direction] = len(id2span)
                    id2span.append([left_idx, right_idx, direction])
    # this part consider inner paragraph, cross sentence span
    if pbnds is not None:
        for pbnd_begin, pbnd_end in root_shifted_pbnds:
            if pbnd_begin < -1:
                break
            for left_idx in range(pbnd_begin, pbnd_end+1):
                for right_idx in range(left_idx+1, pbnd_end+1):
                    # check whether valid
                    if not cross_check(root_shifted_sbnds, left_idx, right_idx):
                        continue

                    for direction in range(2):  # 0 is left 1 is right
                        span2id[left_idx, right_idx, direction] = len(id2span)
                        id2span.append([left_idx, right_idx, direction])

        # this part consider cross paragraph span
        for pbnd_begin, pbnd_end in root_shifted_pbnds:
            if pbnd_begin < -1:
                break
            for left_idx in range(pbnd_begin, pbnd_end+1):
                for right_idx in range(pbnd_end+1, dlen):
                    for direction in range(2):
                        span2id[left_idx, right_idx, direction] = len(id2span)
                        id2span.append([left_idx, right_idx, direction])
    else:
        # this part consider cross sentence span
        for sbnd_begin, sbnd_end in root_shifted_sbnds:
            if sbnd_begin < 0:
                break
            for left_idx in range(sbnd_begin, sbnd_end + 1):
                for right_idx in range(sbnd_end + 1, dlen):
                    for direction in range(2):
                        span2id[left_idx, right_idx, direction] = len(id2span)
                        id2span.append([left_idx, right_idx, direction])

    id2span = npasarray(id2span)

    basic_span = npiempty((2 * dlen))
    for i in range(dlen):
        basic_span[2 * i] = span2id[i, i, 0]
        basic_span[2 * i + 1] = span2id[i, i, 1]

    # the order of ijss is important
    ijss = []
    ikis = [[] for _ in range(len(id2span))]
    kjis = [[] for _ in range(len(id2span))]
    ikcs = [[] for _ in range(len(id2span))]
    kjcs = [[] for _ in range(len(id2span))]

    # we consider inner sentence then consider cross sentence
    # inner sentence same to origin
    for sbnd_begin, sbnd_end in root_shifted_sbnds:
        if sbnd_begin < -1:
            break
        span_len = sbnd_end - sbnd_begin + 1
        for length in range(1, span_len):
            for i in range(sbnd_begin, sbnd_begin+span_len - length):
                j = i + length
                id = span2id[i, j, 0]
                ijss.append(id)
                for k in range(i, j):
                    # two complete spans to form an incomplete span
                    ikis[id].append(span2id[i, k, 1])
                    kjis[id].append(span2id[k + 1, j, 0])
                    # one complete span, one incomplete span to form a complete span
                    ikcs[id].append(span2id[i, k, 0])
                    kjcs[id].append(span2id[k, j, 0])
                id = span2id[i, j, 1]
                ijss.append(id)
                for k in range(i, j + 1):
                    # two complete spans to form an incomplete span
                    if k < j and (i != 0 or k == 0):
                        ikis[id].append(span2id[i, k, 1])
                        kjis[id].append(span2id[k + 1, j, 0])
                    # one incomplete span, one complete span to form a complete span
                    if k > i:
                        ikcs[id].append(span2id[i, k, 1])
                        kjcs[id].append(span2id[k, j, 1])
    if pbnds is not None:
        # inner paragraph cross sentence part
        for pbnd_begin, pbnd_end in root_shifted_pbnds:
            if pbnd_begin < -1:
                break
            span_len = pbnd_end - pbnd_begin + 1
            for length in range(1, span_len):
                for i in range(pbnd_begin, pbnd_begin+span_len - length):
                    j = i + length
                    # in same span, continue
                    if not cross_check(root_shifted_sbnds, i, j):
                        continue
                    id = span2id[i, j, 0]
                    ijss.append(id)
                    for k in range(i, j):
                        # two complete spans to form an incomplete span
                        if cross_check(root_shifted_sbnds, k, k+1):
                            ikis[id].append(span2id[i, k, 1])
                            kjis[id].append(span2id[k + 1, j, 0])
                        # one complete span, one incomplete span to form a complete span
                        ikcs[id].append(span2id[i, k, 0])
                        kjcs[id].append(span2id[k, j, 0])
                    id = span2id[i, j, 1]
                    ijss.append(id)
                    for k in range(i, j + 1):
                        # two complete spans to form an incomplete span
                        if k < j and (i != 0 or k == 0) and cross_check(root_shifted_sbnds, k, k+1):
                            ikis[id].append(span2id[i, k, 1])
                            kjis[id].append(span2id[k + 1, j, 0])
                        # one incomplete span, one complete span to form a complete span
                        # check valid
                        if k > i:
                            ikcs[id].append(span2id[i, k, 1])
                            kjcs[id].append(span2id[k, j, 1])
    else:
        root_shifted_pbnds = root_shifted_sbnds

    # cross sentence part.
    # only consider complete inner sentence span to cross sentence span
    # ignore inner sentence span calculating
    for length in range(1, dlen):
        for i in range(0, dlen-length):
            j = i + length
            # in same span, continue
            if not cross_check(root_shifted_pbnds, i, j):
                continue
            id = span2id[i, j, 0]
            ijss.append(id)
            for k in range(i, j):
                # two complete spans to form an incomplete span
                if cross_check(root_shifted_pbnds, k, k+1):
                    ikis[id].append(span2id[i, k, 1])
                    kjis[id].append(span2id[k + 1, j, 0])
                # one complete span, one incomplete span to form a complete span
                ikcs[id].append(span2id[i, k, 0])
                kjcs[id].append(span2id[k, j, 0])
            id = span2id[i, j, 1]
            ijss.append(id)
            for k in range(i, j + 1):
                # two complete spans to form an incomplete span
                if k < j and (i != 0 or k == 0) and cross_check(root_shifted_pbnds, k, k+1):
                    ikis[id].append(span2id[i, k, 1])
                    kjis[id].append(span2id[k + 1, j, 0])
                # one incomplete span, one complete span to form a complete span
                if k > i:
                    ikcs[id].append(span2id[i, k, 1])
                    kjcs[id].append(span2id[k, j, 1])

    return span2id, id2span, ijss, ikcs, ikis, kjcs, kjis, basic_span


@lru_cache(maxsize=3)
def prepare_backtracking(batch_size, fake_len):
    """
    helper function to generate span idx.
    for backtracking
    """
    span2id, id2span, ijss, ikcs, ikis, kjcs, kjis, basic_span = constituent_index(fake_len)
    id2span = torch.tensor(id2span, device='cuda', dtype=torch.long)
    merged_is = []
    merged_is_index = []
    merged_cs = []
    merged_cs_index = []
    for i in range(len(id2span)):
        merged_is_index.append(len(merged_is))
        merged_is.extend(list(zip(ikis[i], kjis[i])))
        merged_cs_index.append(len(merged_cs))
        merged_cs.extend(list(zip(ikcs[i], kjcs[i])))
    merged_is = torch.tensor(merged_is, device='cuda', dtype=torch.long)
    merged_is_index = torch.tensor(merged_is_index, device='cuda', dtype=torch.long)
    merged_cs = torch.tensor(merged_cs, device='cuda', dtype=torch.long)
    merged_cs_index = torch.tensor(merged_cs_index, device='cuda', dtype=torch.long)
    shape = torch.tensor([batch_size, fake_len], device='cuda', dtype=torch.long)
    return merged_is, merged_is_index, merged_cs, merged_cs_index, id2span, shape


ALL = slice(None, None, None)


def get_many_span(table, spans, indexer=None):
    indexer = indexer or ALL
    to_stack = []
    for i in spans:
        assert table[i] is not None
        to_stack.append(table[i][indexer])
    return torch.stack(to_stack)


def batch_inside(trans_scores: Tensor, dec_scores: Tensor, len_array: Tensor, mode: str = 'sum') \
        -> Tuple[List[Optional[Tensor]], List[Optional[Tensor]], Tensor]:
    """
    decision valence means "whether at least one child is already generated" ? (from near to far)
    child valence means "whether more children need to be generated" ? (from far to near)
    """
    # trans_scores: batch, head, child, cv
    # dec_scores:   batch, head, direction, dv, decision
    op = partial(torch.logsumexp, dim=0) if mode == 'sum' else lambda x: torch.max(x, dim=0)[0]
    batch_size, fake_len, fake_len, cv = trans_scores.shape
    nspan = (fake_len + 1) * fake_len
    span2id, id2span, ijss, ikcs, ikis, kjcs, kjis, basic_span = constituent_index(fake_len)

    # complete/incomplete_table:   [nspan], batch, dv
    ictable: List[Optional[Tensor]] = [None for _ in range(nspan)]
    iitable: List[Optional[Tensor]] = [None for _ in range(nspan)]

    for bs in basic_span:
        ictable[bs] = dec_scores[:, id2span[bs, 0], id2span[bs, 2], :, STOP]

    for ij in ijss:
        l, r, direction = id2span[ij]

        # two complete span to form an incomplete span, and add a new arc.
        if direction == 0:
            h, m = r, l
            h_span_id, m_span_id = kjis[ij], ikis[ij]
        else:
            h, m = l, r
            h_span_id, m_span_id = ikis[ij], kjis[ij]

        """
        get head-side spans, test whether l == r to build decision scores.
        because head-side spans are complete span, if and only if l==r, head-side spans have no child generated.
        """
        h_valence = id2span[h_span_id]
        h_valence = torch.tensor(h_valence[:, 0] != h_valence[:, 1], dtype=torch.long)
        h_d_part = dec_scores[None, :, h, direction, h_valence, GO].transpose(0, 2)
        h_part = get_many_span(ictable, h_span_id, (ALL, HASCHILD, None)) + h_d_part
        m_part = get_many_span(ictable, m_span_id, (ALL, NOCHILD, None))

        t_part = trans_scores[None, :, h, m]
        iitable[ij] = op(h_part + m_part + t_part)

        # one complete span and one incomplete span to form an bigger complete span.
        if direction == 0:
            h_span_id, m_span_id = kjcs[ij], ikcs[ij]
        else:
            h_span_id, m_span_id = ikcs[ij], kjcs[ij]
        h_part = get_many_span(iitable, h_span_id)
        m_part = get_many_span(ictable, m_span_id, (ALL, NOCHILD, None))
        ictable[ij] = op(h_part + m_part)

    partition_score = [torch.sum(ictable[span2id[0, l, 1]][i, NOCHILD]) for i, l in enumerate(len_array)]
    partition_score = torch.stack(partition_score)
    return ictable, iitable, partition_score


def batch_discourse_inside(trans_scores: Tensor, dec_scores: Tensor, len_array: Tensor,
                           sbnds_list: List[List[Tuple[int, int]]], mode: str = 'sum'):
    # TODO here we not return the ic and iitable
    dlen_list = len_array.detach().cpu().numpy().tolist()
    partition_score_list = []
    for idx, dlen in enumerate(dlen_list):
        dlen += 1 # for root
        *_, partition_score = discourse_inside(trans_scores[idx, :dlen, :dlen, :],
                                               dec_scores[idx, :dlen, :dlen, :, :],
                                               sbnds_list[idx], mode=mode)
        partition_score_list.append(partition_score)
    partition_scores = torch.stack(partition_score_list)
    return None, None, partition_scores


def discourse_inside(trans_scores: Tensor, dec_scores: Tensor, sbnds: List[Tuple[int, int]], pbnds: List[Tuple[int, int]]=None, mode: str = 'sum')\
        -> Tuple[List[Optional[Tensor]], List[Optional[Tensor]], Tensor]:

    # trans_scores: head, child, cv
    # dec_scores:   head, direction, dv, decision
    op = partial(torch.logsumexp, dim=0) if mode == 'sum' else lambda x: torch.max(x, dim=0)[0]
    dlen, dlen, cv = trans_scores.shape
    nspan = (dlen + 1) * dlen
    span2id, id2span, ijss, ikcs, ikis, kjcs, kjis, basic_span = discourse_constituent_index(dlen, sbnds, pbnds)

    # complete/incomplete_table:   [nspan], batch, dv
    ictable: List[Optional[Tensor]] = [None for _ in range(nspan)]
    iitable: List[Optional[Tensor]] = [None for _ in range(nspan)]

    for bs in basic_span:
        ictable[bs] = dec_scores[id2span[bs, 0], id2span[bs, 2], :, STOP]

    for ij in ijss:
        l, r, direction = id2span[ij]

        # two complete span to form an incomplete span, and add a new arc.
        if direction == 0:
            h, m = r, l
            h_span_id, m_span_id = kjis[ij], ikis[ij]
        else:
            h, m = l, r
            h_span_id, m_span_id = ikis[ij], kjis[ij]

        """
        get head-side spans, test whether l == r to build decision scores.
        because head-side spans are complete span, if and only if l==r, head-side spans have no child generated.
        """
        h_valence = id2span[h_span_id]
        h_valence = torch.tensor(h_valence[:, 0] != h_valence[:, 1], dtype=torch.long)
        h_d_part = dec_scores[None, h, direction, h_valence, GO].transpose(0, 1)
        h_part = get_many_span(ictable, h_span_id, (HASCHILD, None)) + h_d_part
        m_part = get_many_span(ictable, m_span_id, (NOCHILD, None))

        t_part = trans_scores[None, h, m]
        iitable[ij] = op(h_part + m_part + t_part)

        # one complete span and one incomplete span to form an bigger complete span.
        if direction == 0:
            h_span_id, m_span_id = kjcs[ij], ikcs[ij]
        else:
            h_span_id, m_span_id = ikcs[ij], kjcs[ij]
        h_part = get_many_span(iitable, h_span_id)
        m_part = get_many_span(ictable, m_span_id, (NOCHILD, None))
        ictable[ij] = op(h_part + m_part)

    # partition_score = [torch.sum(ictable[span2id[0, l, 1]][i, NOCHILD]) for i, l in enumerate(len_array)]
    # partition_score = torch.stack(partition_score)
    partition_score = torch.sum(ictable[span2id[0, dlen-1, 1]][NOCHILD])
    return ictable, iitable, partition_score


def batch_parse(trans_scores, dec_scores, len_array):
    assert CUDA_BACKTRACKING_READY, "dmv_extension is required, see /extension"
    # trans_scores: batch, head, child, cv
    # dec_scores: batch, head, direction, dv, decision

    batch_size, fake_len, fake_len, cv = trans_scores.shape
    nspan = (fake_len + 1) * fake_len
    span2id, id2span, ijss, ikcs, ikis, kjcs, kjis, basic_span = constituent_index(fake_len)

    complete_table = torch.full((batch_size, nspan, 2), -np.inf, dtype=torch.float, device=trans_scores.device)
    incomplete_table = torch.full((batch_size, nspan, 2), -np.inf, dtype=torch.float, device=trans_scores.device)
    complete_backtrack = torch.full((batch_size, nspan, 2), -1, dtype=torch.long, device=trans_scores.device)
    incomplete_backtrack = torch.full((batch_size, nspan, 2), -1, dtype=torch.long, device=trans_scores.device)

    iids = id2span[basic_span]
    complete_table[:, basic_span, :] = dec_scores[:, iids[:, 0], iids[:, 2], :, STOP]

    for ij in ijss:
        l, r, direction = id2span[ij]

        # two complete span to form an incomplete span, and add a new arc.
        # two complete span to form an incomplete span, and add a new arc.
        if direction == 0:
            h, m = r, l
            h_span_id, m_span_id = kjis[ij], ikis[ij]
        else:
            h, m = l, r
            h_span_id, m_span_id = ikis[ij], kjis[ij]

        ## valence: far to near
        # span_i = complete_table[:, ikis[ij], NOCHILD, None] \
        #     + complete_table[:, kjis[ij], HASCHILD, None] \
        #     + trans_scores[:, h, m, None, :] \
        #     + dec_scores[:, r, direction, None, :, GO]

        ## valence: near to far
        h_valence = id2span[h_span_id]
        h_valence = torch.tensor(h_valence[:, 0] != h_valence[:, 1], dtype=torch.long)
        span_i = complete_table[:, h_span_id, HASCHILD, None] \
                 + complete_table[:, m_span_id, NOCHILD, None] \
                 + trans_scores[:, h, m, None, :] \
                 + dec_scores[:, h, direction, h_valence, None, GO]

        max_value, max_index = torch.max(span_i, dim=1)
        incomplete_backtrack[:, ij, :] = max_index
        incomplete_table[:, ij, :] = max_value

        # one complete span and one incomplete span to form bigger complete span
        if direction == 0:
            h_span_id, m_span_id = kjcs[ij], ikcs[ij]
        else:
            h_span_id, m_span_id = ikcs[ij], kjcs[ij]

        span_c = complete_table[:, m_span_id, NOCHILD, None] + incomplete_table[:, h_span_id, :]
        max_value, max_index = torch.max(span_c, dim=1)
        complete_backtrack[:, ij, :] = max_index
        complete_table[:, ij, :] = max_value

    merged_is, merged_is_index, merged_cs, merged_cs_index, id2span, shape = prepare_backtracking(batch_size, fake_len)
    root_id = torch.tensor([span2id[0, l, 1] for l in len_array], device='cuda', dtype=torch.long)

    heads, head_valences, valences = dmv_extension.backtracking(incomplete_backtrack, complete_backtrack,
        merged_is, merged_is_index, merged_cs, merged_cs_index, root_id, id2span, shape, False)
    return heads, head_valences, valences


def batch_discourse_parse(trans_scores, dec_scores, len_array, sbnds_list, pbnds_list):
    len_list = len_array.detach().cpu().numpy().tolist()
    batch_trans_scores_np = trans_scores.detach().cpu().numpy()
    batch_dec_scores_np = dec_scores.detach().cpu().numpy()
    batch_heads, batch_head_valences, batch_valences = [], [], []
    for idx, dlen in enumerate(len_list):
        # here consider the root
        dlen += 1
        heads, head_valences, valences = discourse_parse(batch_trans_scores_np[idx][:dlen, :dlen, :],
                                                         batch_dec_scores_np[idx][:dlen], sbnds_list[idx], pbnds_list[idx])
        batch_heads.append(heads)
        batch_head_valences.append(valences)
        batch_valences.append(valences)
    return batch_heads, batch_head_valences, batch_valences


# not batch discourse parse on cpu
def discourse_parse(trans_scores, dec_scores, sbnds, pbnds=None):
    # trans_scores: head, child, cv
    # dec_scores: head, direction, dv, decision

    dlen, dlen, cv = trans_scores.shape
    nspan = (dlen + 1) * dlen
    span2id, id2span, ijss, ikcs, ikis, kjcs, kjis, basic_span = discourse_constituent_index(dlen, sbnds, pbnds)

    complete_table = np.full((nspan, 2), -np.inf)
    incomplete_table = np.full((nspan, 2), -np.inf)
    complete_backtrack = np.full((nspan, 2), -1)
    incomplete_backtrack = np.full((nspan, 2), -1)

    iids = id2span[basic_span]
    complete_table[basic_span, :] = dec_scores[iids[:, 0], iids[:, 2], :, STOP]

    for ij in ijss:
        l, r, direction = id2span[ij]

        # two complete span to form an incomplete span, and add a new arc.
        if direction == 0:
            h, m = r, l
            h_span_id, m_span_id = kjis[ij], ikis[ij]
        else:
            h, m = l, r
            h_span_id, m_span_id = ikis[ij], kjis[ij]

        ## valence: far to near
        # span_i = complete_table[:, ikis[ij], NOCHILD, None] \
        #     + complete_table[:, kjis[ij], HASCHILD, None] \
        #     + trans_scores[:, h, m, None, :] \
        #     + dec_scores[:, r, direction, None, :, GO]

        ## valence: near to far
        h_valence = id2span[h_span_id]
        h_valence = (h_valence[:, 0] != h_valence[:, 1]) * 1
        span_i = complete_table[h_span_id, HASCHILD, None] \
                 + complete_table[m_span_id, NOCHILD, None] \
                 + trans_scores[h, m, None, :] \
                 + dec_scores[h, direction, h_valence, None, GO]

        max_value = np.max(span_i, axis=0)
        max_index = np.argmax(span_i, axis=0)
        incomplete_backtrack[ij, :] = max_index
        incomplete_table[ij, :] = max_value

        # one complete span and one incomplete span to form bigger complete span
        if direction == 0:
            h_span_id, m_span_id = kjcs[ij], ikcs[ij]
        else:
            h_span_id, m_span_id = ikcs[ij], kjcs[ij]

        span_c = complete_table[m_span_id, NOCHILD, None] + incomplete_table[h_span_id, :]
        max_value = np.max(span_c, axis=0)
        max_index = np.argmax(span_c, axis=0)
        complete_backtrack[ij, :] = max_index
        complete_table[ij, :] = max_value
    heads = -np.ones((dlen))
    head_valences = np.zeros((dlen))
    valences = np.zeros((dlen, 2))
    root_id = span2id[(0, dlen - 1, 1)]
    discourse_backtracking(incomplete_backtrack, complete_backtrack, root_id, 0, 1, heads, head_valences,
                           valences, ikcs, ikis, kjcs, kjis, id2span, span2id)
    return heads, head_valences, valences


def discourse_backtracking(incomplete_backtrack, complete_backtrack, span_id, decision_valence, complete,
                 heads, head_valences, valences, ikcs, ikis, kjcs, kjis, id_2_span, span_2_id):
    (l, r, dir) = id_2_span[span_id]
    if l == r:
        valences[l, dir] = decision_valence
        return
    if complete:
        if dir == 0:
            k = complete_backtrack[span_id, decision_valence]
            # print 'k is ', k, ' complete left'
            k_span = k
            left_span_id = ikcs[span_id][k_span]
            right_span_id = kjcs[span_id][k_span]
            discourse_backtracking(incomplete_backtrack, complete_backtrack, left_span_id, 0, 1, heads,
                                   head_valences, valences, ikcs, ikis, kjcs, kjis, id_2_span, span_2_id)
            discourse_backtracking(incomplete_backtrack, complete_backtrack, right_span_id, decision_valence, 0, heads,
                                   head_valences, valences, ikcs, ikis, kjcs, kjis, id_2_span, span_2_id)
            return
        else:
            num_k = len(ikcs[span_id])
            k = complete_backtrack[span_id, decision_valence]
            # print 'k is ', k, ' complete right'
            k_span = k
            left_span_id = ikcs[span_id][k_span]
            right_span_id = kjcs[span_id][k_span]
            discourse_backtracking(incomplete_backtrack, complete_backtrack, left_span_id, decision_valence,
                                   0, heads, head_valences, valences, ikcs, ikis, kjcs, kjis, id_2_span, span_2_id)
            discourse_backtracking(incomplete_backtrack, complete_backtrack, right_span_id, 0, 1,
                                   heads, head_valences, valences, ikcs, ikis, kjcs, kjis, id_2_span, span_2_id)
            return
    else:
        if dir == 0:

            k = incomplete_backtrack[span_id, decision_valence]
            # print 'k is ', k, ' incomplete left'
            heads[l] = r
            head_valences[l] = decision_valence
            left_span_id = ikis[span_id][k]
            right_span_id = kjis[span_id][k]
            discourse_backtracking(incomplete_backtrack, complete_backtrack, left_span_id, 0, 1, heads,
                                   head_valences, valences, ikcs, ikis, kjcs, kjis, id_2_span, span_2_id)
            discourse_backtracking(incomplete_backtrack, complete_backtrack, right_span_id, 1, 1, heads,
                                   head_valences, valences, ikcs, ikis, kjcs, kjis, id_2_span, span_2_id)
            return
        else:
            k = incomplete_backtrack[span_id, decision_valence]
            # print 'k is', k, ' incomplete right'
            heads[r] = l
            head_valences[r] = decision_valence
            left_span_id = ikis[span_id][k]
            right_span_id = kjis[span_id][k]
            discourse_backtracking(incomplete_backtrack, complete_backtrack, left_span_id, 1, 1, heads,
                                   head_valences, valences, ikcs, ikis, kjcs, kjis, id_2_span, span_2_id)
            discourse_backtracking(incomplete_backtrack, complete_backtrack, right_span_id, 0, 1, heads,
                                   head_valences, valences, ikcs, ikis, kjcs, kjis, id_2_span, span_2_id)
            return


def batch_inside_prob(trans_scores, dec_scores, tag_prop, len_array, mode='sum', mode2='sum'):
    """

    a,b,c in graph mean `idx`, while in formula mean `tag at idx`
    A,B,C mean `tag's probability at idx`
    ii mean `inside incomplete score`, ic mean `inside complete score`
    head tag(c in following graph) prob is not contained in ii, ic

    * SITUATION1: TWO COMPLETE SPAN -> ONE INCOMPLETE SPAN

        a     b      c
        |-----+------|    ic[c] + ic[a] + d[c] + t[c,a] = ii[c,a]
        |--^      ^--|
        ^------------|

    * SITUATION2: ONE COMPLETE SPAN + ONE INCOMPLETE SPAN -> ONE COMPLETE SPAN

        a     b      c
        |-----+------|    SUM_b { B * { ii[c,b] + ic[b] } } = ic[c]
           ^--^------|

    Finally we need ic[*ROOT*], we can sum out `a` in SITUATION 1 (`b` in S2):
    to save memory:
        SUM_a { A * (ic[a] + t[c,a]) } + ic[c] + d[c] = sumed_ii[c]
        sumed_ii[c] + SUM_b { B * ic[b] } = ic[c]

    :param trans_scores: Shape[batch, head, ntag, child, cv], head/child mean idx in tag seq
    :param dec_scores: Shape[batch, head, ntag, direction, dv, decision]
    :param tag_prop: Shape[batch, head, ntag]
    :param len_array: Shape[batch]
    :param mode: `sum` or `max` for spans
    :param mode2: `sum` or `max` for tag prob
    :returns: tuple(ictable, iitable, likelihood_for_each_sentence)
    """

    op1 = partial(torch.logsumexp, dim=0) if mode == 'sum' else lambda x: torch.max(x, dim=0)[0]

    batch_size, fake_len, ntag, *_, cv = trans_scores.shape
    nspan = (fake_len + 1) * fake_len
    span2id, id2span, ijss, ikcs, ikis, kjcs, kjis, basic_span = constituent_index(fake_len)

    # complete_table:   [nspan], batch, ntag, valence
    # incomplete_table: [nspan], batch, ntag, valence
    ictable: List[Optional[Tensor]] = [None for _ in range(nspan)]
    iitable: List[Optional[Tensor]] = [None for _ in range(nspan)]

    for bs in basic_span:
        ictable[bs] = dec_scores[:, id2span[bs, 0], :, id2span[bs, 2], :, STOP]

    for ij in ijss:
        l, r, direction = id2span[ij]

        ## two complete span to form an incomplete span, and add a new arc.
        if direction == 0:
            h, m = r, l
            h_span_id, m_span_id = kjis[ij], ikis[ij]
        else:
            h, m = l, r
            h_span_id, m_span_id = ikis[ij], kjis[ij]

        # h_part: nspan, batch, ntag
        h_part = get_many_span(ictable, h_span_id, (ALL, ALL, HASCHILD))
        # m_prob: batch, ntag
        m_prob = tag_prop[:, m, :]
        # m_part: nspan, batch, ntag
        m_part = get_many_span(ictable, m_span_id, (ALL, ALL, NOCHILD))
        # d_part: batch, ntag, valence
        d_part = dec_scores[:, h, :, direction, :, GO]
        # t_part: batch, ntag  (already sum out m`s ntag)
        t_part = trans_scores[:, h, :, m]

        if mode2 == 'sum':
            # nspan, batch
            sumed_ii = torch.matmul(m_part.unsqueeze(2), m_prob.unsqueeze(2)).squeeze()
        elif mode2 == 'max':
            # nspan, batch, 1
            m_index = torch.argmax(m_prob, dim=1).view(1, -1, 1).expand(m_part.shape[0], -1, 1)
            # nspan, batch
            sumed_ii = torch.gather(m_part, 2, m_index).squeeze()
        else:
            raise ValueError("bad mode2")
        # nspan, batch, ntag
        sumed_ii = sumed_ii.unsqueeze(2) + t_part.unsqueeze(0)
        # nspan, batch, ntag, valence
        sumed_ii = sumed_ii.unsqueeze(3) + h_part.unsqueeze(3) + d_part.unsqueeze(0)
        # batch, ntag, valence
        iitable[ij] = op1(sumed_ii)

        ## one complete span and one incomplete span to form an bigger complete span.
        if direction == 0:
            h_span_id, m_span_id = kjcs[ij], ikcs[ij]
        else:
            h_span_id, m_span_id = ikcs[ij], kjcs[ij]

        # h_part: nspan, batch, ntag, valence
        h_part = get_many_span(iitable, h_span_id)
        # m_prob_indexer: nspan
        m_prob_index = torch.tensor([id2span[i, direction] for i in m_span_id], device=h_part.device, dtype=torch.long)
        # m_prob: nspan, batch, ntag
        m_prob = tag_prop[:, m_prob_index].transpose(0, 1)
        # m_part: nspan, batch, ntag
        m_part = get_many_span(ictable, m_span_id, (ALL, ALL, NOCHILD))

        if mode2 == 'sum':
            # nspan, batch
            ic = torch.sum(m_part * m_prob, dim=2)
        elif mode2 == 'max':
            # nspan, batch, 1
            m_index = torch.argmax(m_prob, dim=2, keepdim=True)
            # nspan, batch
            ic = torch.gather(m_part, 2, m_index).squeeze()
        else:
            raise ValueError("bad mode2")

        # nspan, batch, ntag, valence
        ic = ic.view(*ic.shape, 1, 1) + h_part
        # batch, ntag, valence
        ictable[ij] = op1(ic)

    partition_score = [op1(ictable[span2id[0, l, 1]][i, :, NOCHILD]) for i, l in enumerate(len_array)]
    partition_score = torch.stack(partition_score)
    return ictable, iitable, partition_score


# # debug batch inside
# if __name__ == '__main__':
#     torch.random.manual_seed(46)
#
#     dlen = 10
#     cv = 2
#     dv = 2
#     sbnds = [(0, 1), (2, 4), (5, 8)]
#     pbnds = [(0, 0), (1, 2)]
#
#     tscore = -1.0 * torch.rand(dlen, dlen, cv)
#     dscore = -1.0 * torch.zeros(dlen, 2, dv, 2)
#
#     # gold = batch_inside(tscore.unsqueeze(0),
#     #                     dscore.unsqueeze(0),
#     #                     torch.tensor([dlen]).long())
#     tscore_np = tscore.numpy()
#     dscore_np = dscore.numpy()
#     res = discourse_inside(tscore, dscore, sbnds)
#     idx = discourse_parse(tscore_np, dscore_np, sbnds)
#     print(res)
#     print(idx[0])