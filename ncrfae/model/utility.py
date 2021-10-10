import itertools
from itertools import groupby

import numpy as np
import torch
from torch import autograd, nn as nn
from torch.autograd import Variable

from model.definition import LOGZERO
import treetk

EMPTY = -1


def memoize(func):
    mem = {}

    def helper(*args, **kwargs):
        key = (args, frozenset(kwargs.items()))
        if key not in mem:
            mem[key] = func(*args, **kwargs)
        return mem[key]

    return helper


def get_state_code(i, j, k):
    return i * 4 + j * 2 + k


def _is_valid_signature(sig, max_dep_length, length_constraint_on_root):
    left_index, right_index, bL, bR, is_simple = sig

    if bL != bR and is_simple:
        len = right_index - left_index
        if len > max_dep_length:
            if left_index == 0:
                if length_constraint_on_root:
                    return False
            else:
                return False

    if left_index == 0 and bL:
        return False
    if right_index - left_index == 1:
        if is_simple:
            if bL and bR:
                return False
            else:
                return True
        else:
            return False

    if bL and bR and is_simple:
        return False

    return (bL == is_simple) or (bR == is_simple)


@memoize
def constituent_indexes(sent_len, is_multi_root=False, max_dependency_len=None, length_constraint_on_root=False):
    if max_dependency_len is None or max_dependency_len < 0 or max_dependency_len > sent_len:
        max_dependency_len = sent_len

    seed_spans = []
    base_left_spans = []
    crt_id = 0
    id_span_map = {}

    for left_index in range(sent_len):
        for right_index in range(left_index + 1, sent_len):
            for c in range(8):
                id_span_map[crt_id] = (left_index, right_index, c)
                crt_id += 1

    span_id_map = {v: k for k, v in id_span_map.items()}

    for i in range(1, sent_len):
        ids = span_id_map[(i - 1, i, get_state_code(0, 0, 1))]
        seed_spans.append(ids)

        ids = span_id_map[(i - 1, i, get_state_code(0, 1, 1))]
        base_left_spans.append(ids)

    base_right_spans = []
    for i in range(2, sent_len):
        ids = span_id_map[(i - 1, i, get_state_code(1, 0, 1))]
        base_right_spans.append(ids)

    ijss = []
    ikss = [[] for _ in range(crt_id)]
    kjss = [[] for _ in range(crt_id)]

    left_spans = set()
    right_spans = set()
    for length in range(2, sent_len):
        for i in range(1, sent_len - length + 1):
            j = i + length
            for (bl, br) in list(itertools.product([0, 1], repeat=2)):
                ids = span_id_map[(i - 1, j - 1, get_state_code(bl, br, 0))]

                for (b, s) in list(itertools.product([0, 1], repeat=2)):
                    for k in range(i + 1, j):
                        sig_left = (i - 1, k - 1, bl, b, 1)
                        sig_right = (k - 1, j - 1, 1 - b, br, s)

                        if (_is_valid_signature(sig_left, max_dependency_len, length_constraint_on_root) and
                                _is_valid_signature(sig_right, max_dependency_len, length_constraint_on_root)):
                            if is_multi_root or ((i > 1) or (
                                                    (i == 1) and (j == sent_len) and (bl == 0) and (b == 1) and (
                                                br == 1)) or ((i == 1) and (k == 2) and (bl == 0) and (b == 0) and (
                                        br == 0))):
                                ids1 = span_id_map[(i - 1, k - 1, get_state_code(bl, b, 1))]
                                ikss[ids].append(ids1)
                                ids2 = span_id_map[(k - 1, j - 1, get_state_code(1 - b, br, s))]
                                kjss[ids].append(ids2)
                if len(ikss[ids]) >= 1:
                    ijss.append(ids)

            if length <= max_dependency_len or (i == 1 and not length_constraint_on_root):
                ids = span_id_map[(i - 1, j - 1, get_state_code(0, 1, 1))]
                ijss.append(ids)
                left_spans.add(ids)
                if i != 1:
                    ids = span_id_map[(i - 1, j - 1, get_state_code(1, 0, 1))]
                    ijss.append(ids)
                    right_spans.add(ids)

    return seed_spans, base_left_spans, base_right_spans, left_spans, right_spans, ijss, ikss, kjss, id_span_map, span_id_map


def logsumexp(a, axis=None):
    a_max = amax(a, axis=axis, keepdim=True)
    a_max[~isfinite(a_max)] = 0
    res = torch.log(asum(torch.exp(a - a_max), axis=axis, keepdim=True)) + a_max
    if isinstance(axis, tuple):
        for x in reversed(axis):
            res.squeeze_(x)
    else:
        res.squeeze_(axis)
    return res


def logaddexp(a, b):
    max_ab = torch.max(a, b)
    max_ab[~isfinite(max_ab)] = 0
    return torch.log(torch.add(torch.exp(a - max_ab), torch.exp(b - max_ab))) + max_ab


def asum(a, axis=None, keepdim=False):
    if isinstance(axis, tuple):
        for x in reversed(axis):
            a = a.sum(x, keepdim=keepdim)
    else:
        a = a.sum(axis, keepdim=keepdim)
    return a


def amax(a, axis=None, keepdim=False):
    if isinstance(axis, tuple):
        for x in reversed(axis):
            a, index = a.max(x, keepdim=keepdim)
    else:
        a, index = a.max(axis, keepdim=True)
    return a


def isfinite(a):
    return (a != np.inf) & (a != -np.inf) & (a != np.nan) & (a != -np.nan)


def decoding_batch(weights, is_multi_root, max_dependency_len, length_constraint_on_root):
    batch_size, sentence_len, _ = weights.shape

    inside_table = np.empty((batch_size, sentence_len * sentence_len * 8), dtype=np.float64)
    inside_table.fill(LOGZERO)

    (seed_spans, base_left_spans, base_right_spans, left_spans, right_spans,
     ijss, ikss, kjss, id_span_map, span_id_map) = constituent_indexes(sentence_len, is_multi_root, max_dependency_len,
                                                                       length_constraint_on_root)
    kbc = np.zeros_like(inside_table, dtype=int)

    for ii in seed_spans:
        inside_table[:, ii] = 0.0
        kbc[:, ii] = EMPTY

    for ii in base_right_spans:
        (l, r, c) = id_span_map[ii]
        inside_table[:, ii] = weights[:, r, l]
        kbc[:, ii] = EMPTY

    for ii in base_left_spans:
        (l, r, c) = id_span_map[ii]
        inside_table[:, ii] = weights[:, l, r]
        kbc[:, ii] = EMPTY

    for ij in ijss:
        (l, r, c) = id_span_map[ij]
        if ij in left_spans:
            ids = span_id_map.get((l, r, get_state_code(0, 0, 0)), -1)
            prob = inside_table[:, ids] + weights[:, l, r]
            inside_table[:, ij] = np.maximum(inside_table[:, ij], prob)
        elif ij in right_spans:
            ids = span_id_map.get((l, r, get_state_code(0, 0, 0)), -1)
            prob = inside_table[:, ids] + weights[:, r, l]
            inside_table[:, ij] = np.maximum(inside_table[:, ij], prob)
        else:
            beta_ik, beta_kj = inside_table[:, ikss[ij]], inside_table[:, kjss[ij]]
            probs = beta_ik + beta_kj
            inside_table[:, ij] = np.max(probs, axis=1)
            kbc[:, ij] = np.argmax(probs, axis=1)

    id1 = span_id_map.get((0, sentence_len - 1, get_state_code(0, 1, 0)), -1)
    id2 = span_id_map.get((0, sentence_len - 1, get_state_code(0, 1, 1)), -1)

    score1 = inside_table[:, id1]
    score2 = inside_table[:, id2]

    root_prob1 = score1
    root_prob2 = score2

    best_score = np.maximum(root_prob1, root_prob2)
    mask = np.equal(best_score, root_prob1)

    root_id = np.empty((batch_size, 1), dtype=int)
    root_id.fill(id2)
    root_id[mask] = id1

    best_tree = back_trace_batch(kbc, root_id, sentence_len, is_multi_root, max_dependency_len,
                                 length_constraint_on_root)

    return best_score, best_tree


def back_trace_batch(kbc, root_id, sentence_len, is_multi_root, max_dependency_len, length_constraint_on_root):
    (seed_spans, base_left_spans, base_right_spans, left_spans, right_spans,
     ijss, ikss, kjss, id_span_map, span_id_map) = constituent_indexes(sentence_len, is_multi_root, max_dependency_len,
                                                                       length_constraint_on_root)

    batch_size, num_ctt = kbc.shape
    in_tree = np.full((batch_size, num_ctt), EMPTY)
    trees = np.full((batch_size, sentence_len), EMPTY)

    for sent_id in range(batch_size):
        current_sent_root_id = root_id[sent_id][0]
        in_tree[sent_id, current_sent_root_id] = 1

        (l, r, c) = id_span_map[current_sent_root_id]
        if current_sent_root_id in base_left_spans or current_sent_root_id in left_spans:
            trees[sent_id, r] = l

        if current_sent_root_id in base_right_spans or current_sent_root_id in right_spans:
            trees[sent_id, l] = r

        for ij in reversed(ijss):
            non = in_tree[sent_id, ij]
            (l, r, c) = id_span_map[ij]
            if non != EMPTY:
                if ij in left_spans:
                    ids = span_id_map.get((l, r, get_state_code(0, 0, 0)), -1)
                    in_tree[sent_id, ids] = 1
                    trees[sent_id, r] = l
                elif ij in right_spans:
                    ids = span_id_map.get((l, r, get_state_code(0, 0, 0)), -1)
                    in_tree[sent_id, ids] = 1
                    trees[sent_id, l] = r
                else:
                    iks, kjs = ikss[ij], kjss[ij]
                    k = kbc[sent_id, ij]
                    in_tree[sent_id, iks[k]] = 1
                    in_tree[sent_id, kjs[k]] = 1

        for ij in base_left_spans:
            non = in_tree[sent_id, ij]
            (l, r, c) = id_span_map[ij]
            if non != EMPTY:
                trees[sent_id, r] = l

        for ij in base_right_spans:
            non = in_tree[sent_id, ij]
            (l, r, c) = id_span_map[ij]
            if non != EMPTY:
                trees[sent_id, l] = r

    return trees


def to_scalar(var):
    """change the first element of a tensor to scalar
    """
    if isinstance(var, autograd.Variable):
        return var.view(-1).data.tolist()[0]
    else:
        return var.view(-1).tolist()[0]


def init_linear(input_linear):
    """
    Initialize linear transformation
    """
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform_(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()


def init_lstm(input_lstm):
    """
    Initialize lstm
    """
    for ind in range(0, input_lstm.num_layers):
        weight = eval('input_lstm.weight_ih_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform_(weight, -bias, bias)
        weight = eval('input_lstm.weight_hh_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform_(weight, -bias, bias)

    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.bias_ih_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            weight = eval('input_lstm.bias_hh_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1


def check_nan_exists(x):
    x = x.data if isinstance(x, Variable) else x
    y = (x != x)
    return y.any()


def construct_batches_by_length(data_set, batch_size):
    data_set = sorted(data_set, key=lambda s: len(s))
    grouped = [list(g) for k, g in groupby(data_set, lambda s: len(s))]
    batch_data = []
    for group in grouped:
        sub_batch_data = get_batch_data(group, batch_size)
        batch_data.extend(sub_batch_data)
    return batch_data


def construct_batches(data_set, batch_size):
    batch_data = []
    sub_batch_data = get_batch_data(data_set, batch_size)
    batch_data.extend(sub_batch_data)
    return batch_data


def get_batch_data(data_set, batch_size):
    batch_data = []
    len_data = len(data_set)
    num_batch = len_data // batch_size
    if not len_data % batch_size == 0:
        num_batch += 1

    for i in range(num_batch):
        start_idx = i * batch_size
        end_idx = min(len_data, (i + 1) * batch_size)
        batch_data.append(data_set[start_idx:end_idx])
    return batch_data


def compute_tree_distance(arcs1, arcs2, coef):
    """
    :type arcs1: list of (int, int)
    :type arcs2: list of (int, int)
    :type coef: float
    :rtype: float
    """
    assert len(arcs1) == len(arcs2)

    dtree1 = treetk.arcs2dtree(arcs1)
    dtree2 = treetk.arcs2dtree(arcs2)

    dist = 0.0
    for d in range(len(dtree1.tokens)):
        if d == 0:
            continue
        h1, _ = dtree1.get_head(d)
        h2, _ = dtree2.get_head(d)
        if h1 != h2:
            dist += 1.0
    dist = coef * dist
    return dist

def flatten_lists(list_of_lists):
    """
    :type list_of_lists: list of list of T
    :rtype: list of T
    """
    return [elem for lst in list_of_lists for elem in lst]