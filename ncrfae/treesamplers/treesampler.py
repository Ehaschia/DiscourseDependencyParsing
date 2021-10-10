import copy

import numpy as np
from model.eisner import EisnerDecoder
import random

class TreeSampler(object):

    def __init__(self, sampler_names):
        """
        :type sampler_names: list of str
        """
        assert len(sampler_names) == 3
        assert sampler_names[1] != "RB2"
        assert sampler_names[2] != "RB2"

        self.sampler_names = sampler_names

        self.sampler_s = self.get_sampler(sampler_names[0])
        self.sampler_p = self.get_sampler(sampler_names[1])
        self.sampler_d = self.get_sampler(sampler_names[2])

        if self.sampler_s is not None:
            self.use_sbnds = True
        else:
            self.use_sbnds = False

        if self.sampler_p is not None:
            self.use_pbnds = True
        else:
            self.use_pbnds = False

        self.arc_score = None

    def get_sampler(self, sampler_name):
        """
        :type sampler_name: str
        :rtype: BaseTreeSampler
        """
        if sampler_name == "X":
            return None
        elif sampler_name == "BU":
            return BU()
        elif sampler_name == "RB":
            return RB()
        elif sampler_name == "LB":
            return LB()
        else:
            raise ValueError("Invalid sampler_name=%s" % sampler_name)

    def sample(self, inputs, edus, edus_head, sbnds, pbnds, has_root=False):
        """
        :type inputs: list of int
        :type edus: list of list of str
        :type edus_head: list of (str, str, str)
        :type sbnds: list of (int, int), or None
        :type pbnds: list of (int, int), or None
        :rtype: list of str
        """

        rules = []
        head = inputs[0] if has_root else 0
        assert len(inputs) > 1
        self.arc_score = np.random.rand(len(inputs), len(inputs))
        noroot_inputs = copy.deepcopy(inputs)[1:] if has_root else copy.deepcopy(inputs)
        noroot_edus = None
        if edus is not None:
            noroot_edus = copy.deepcopy(edus)[1:] if has_root else copy.deepcopy(edus)
        noroot_edus_head = None
        if edus_head is not None:
            noroot_edus_head = copy.deepcopy(edus_head)[1:] if has_root else copy.deepcopy(edus_head)

        # Sentence-level sampling
        if self.use_sbnds:
            assert sbnds is not None
            target_bnds = sbnds
            s_rules, noroot_inputs = self.apply_sampler(
                                    sampler=self.sampler_s,
                                    inputs=noroot_inputs,
                                    edus=noroot_edus,
                                    edus_head=noroot_edus_head,
                                    target_bnds=target_bnds)
            rules += s_rules

        # Paragraph-level sampling
        if self.use_pbnds:
            assert pbnds is not None
            if self.use_sbnds:
                target_bnds = pbnds
            else:
                target_bnds = [(sbnds[b][0],sbnds[e][1]) for b,e in pbnds]
            p_rules, noroot_inputs = self.apply_sampler(
                                    sampler=self.sampler_p,
                                    inputs=noroot_inputs,
                                    edus=None,
                                    edus_head=None,
                                    target_bnds=target_bnds)
            rules += p_rules

        # Document-level sampling
        d_rules, noroot_inputs = self.sampler_d.sample(inputs=noroot_inputs,
                                                       edus=None,
                                                       edus_head=None,
                                                       arc_score=self.arc_score) # list of str
        rules += d_rules
        rules.append((head, noroot_inputs, "<root>"))
        return rules


    def apply_sampler(self, sampler, inputs, edus, edus_head, target_bnds):
        """
        :type sampler: BaseTreeSampler
        :type inputs: list of int/str
        :type edus: list of list of str
        :type edus_head: list of (str, str, str)
        :type target_bnds: list of (int, int)
        :rtype: list of str
        """
        outputs = [] # list of str
        heads = []
        for begin_i, end_i in target_bnds:
            if begin_i == end_i:
                heads.append(inputs[begin_i])
                rules = []
            else:
                rules, head = sampler.sample(
                                inputs=inputs[begin_i:end_i+1],
                                edus=edus[begin_i:end_i+1] if edus is not None else None,
                                edus_head=edus_head[begin_i:end_i+1] if edus_head is not None else None,
                                arc_score=self.arc_score) # list of str
                heads.append(head)
            outputs += rules
        return outputs, heads

class BU(object):
    def __int__(self):
        pass

    def sample(self, inputs, edus, edus_head, arc_score):
        arcs_holder = []
        head = self.rec_sample(inputs, arcs_holder)
        return arcs_holder, head

    def rec_sample(self, inputs, arcs):
        n = len(inputs)

        if n == 1:
            return inputs[0]

        root_idx = random.choice([i for i in range(n)])
        if root_idx == 0:
            child_idx = 1
        elif root_idx == n - 1:
            child_idx = root_idx - 1
        else:
            child_idx = random.choice([-1, 1])
            child_idx = root_idx + child_idx

        arcs.append((inputs[root_idx], inputs[child_idx], '*'))
        head = self.rec_sample(inputs[0:child_idx] + inputs[child_idx+1:], arcs)

        return head

class RB(object):
    """
    Right Branching
    """

    def __init__(self):
        pass

    def sample(self, inputs, edus, edus_head, arc_score):
        """
        :type inputs: list of int/str
        :type edus: list of list of str
        :type edus_head: list of (str, str, str)
        :rtype: list of str
        """

        if len(inputs) == 1:
            return [], inputs[0]
        head = inputs[0]
        rules = []
        for i in range(1, len(inputs)):
            rules.append((inputs[i-1], inputs[i], "*"))
        return rules, head

class LB(object):
    """
    Left Branching
    """

    def __init__(self):
        pass

    def sample(self, inputs, edus, edus_head, arc_score):
        """
        :type inputs: list of int/str
        :type edus: list of list of str
        :type edus_head: list of (str, str, str)
        :rtype: list of str
        """

        if len(inputs) == 1:
            return [], inputs[0]
        head = inputs[-1]
        rules = []
        for i in reversed(range(0, len(inputs)-1)):
            rules.append((inputs[i+1], inputs[i], "*"))
        return rules, head
