import copy

import numpy as np
from decoders.eisner import EisnerDecoder

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
        elif sampler_name == "TD":
            return TD()
        elif sampler_name == "RB":
            return RB()
        elif sampler_name == "LB":
            return LB()
        elif sampler_name == "RB2":
            return RB2()
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
        if edus_head is None:
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
                # sexp = inputs[begin_i] # int/str
                # sexp = str(sexp)
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
    """
    Random Bottom-Up
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
        x = copy.deepcopy(inputs)
        while len(x) > 1:
            i = np.random.randint(0, len(x)-1)
            lhs = "%s" % x[i]
            rhs = "%s" % x[i+1]
            merged = "( %s %s )" % (lhs, rhs)
            x[i] = merged
            x.pop(i+1)
        sexp = x[0]
        sexp = sexp.split()
        return sexp


class TD(object):
    def __int__(self):
        pass

    def sample(self, inputs, edus, edus_head, arc_score):
        dep, head = self.rec_sample(inputs, arc_score)
        return dep, head

    # TODO implement
    def rec_sample(self, inputs, arc_score):

        n = len(inputs)

        if len(inputs) == 1:
            return [], inputs[0]

        # head_idx = np.random.randint(0, n)
        # head = inputs[head_idx]
        # rules = []
        # # left children and right children
        # if head_idx != 0:
        #     lc_num = np.random.randint(1, len(inputs[:head_idx]))
        #     lcs_idx = np.random.permutation(head_idx)[lc_num]
        #     for lc_idx in lcs_idx:
        #         rules.append((head, inputs[lc_idx], "*"))
        #     for i in range()
        #
        # if head_idx+1 != n:
        #     rc_num = np.random.randint(1, len(inputs[head_idx+1:]))
        # random init a arc score matrix
        eisner = EisnerDecoder()
        arcs, head = eisner.decode_without_root(arc_score, inputs, gold_heads=None)
        arcs = [(l, r, "*") for l, r in arcs]
        return arcs, head

class RB(object):
    """
    Right Branching
    """

    def __init__(self):
        pass

    # def sample(self, inputs, edus, edus_head):
    #     """
    #     :type inputs: list of int/str
    #     :type edus: list of list of str
    #     :type edus_head: list of (str, str, str)
    #     :rtype: list of str
    #     """
    #     x = copy.deepcopy(inputs)
    #     while len(x) > 1:
    #         lhs = x[-2]
    #         rhs = x[-1]
    #         merged = "( %s %s )" % (lhs, rhs)
    #         x[-2] = merged
    #         x.pop(-1)
    #     sexp = x[0]
    #     sexp = sexp.split()
    #     return sexp

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


    # def sample(self, inputs, edus, edus_head):
    #     """
    #     :type inputs: list of int/str
    #     :type edus: list of list of str
    #     :type edus_head: list of (str, str, str)
    #     :rtype: list of str
    #     """
    #     x = copy.deepcopy(inputs)
    #     while len(x) > 1:
    #         lhs = x[0]
    #         rhs = x[1]
    #         merged = "( %s %s )" % (lhs, rhs)
    #         x[0] = merged
    #         x.pop(1)
    #     sexp = x[0]
    #     sexp = sexp.split()
    #     return sexp

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

class RB2(object):
    """
    Syntax-Aware Right Branching
    """

    def __init__(self):
        self.sampler = RB()

    def sample(self, inputs, edus, edus_head, arc_score):
        """
        :type inputs: list of int
        :type edus: list of list of str
        :type edus_head: list of (str, str, str)
        :rtype: list of str
        """
        # Find the position of the EDU with the ROOT head
        root_position = None
        for edu_i in range(len(inputs)):
            if edus_head[edu_i][2] == "ROOT":
                root_position = edu_i
                break

        if (root_position is not None): #and (len(inputs) == 3 or len(inputs) == 4):
            if root_position == 0:
                sexp_lhs = []
            elif root_position == 1:
                sexp_lhs = [str(inputs[0])]
            else:
                sexp_lhs = self.sampler.sample(inputs[:root_position], edus=None, edus_head=None)

            if root_position == len(inputs)-1:
                sexp_rhs = [str(inputs[-1])]
            else:
                sexp_rhs = self.sampler.sample(inputs[root_position:], edus=None, edus_head=None)

            sexp = sexp_lhs + sexp_rhs
            if len(sexp_lhs) != 0 and len(sexp_rhs) != 0:
                sexp = ["("] + sexp + [")"]
            return sexp
        # if len(inputs) == 3:
        #     # if edus_head[0][2] == "ROOT":
        #     # if (edus[1][0] in ["--", "-lrb-"]) and (edus[1][-1] in ["--", "-rrb-"]):
        #     #     # B が記号で囲まれる
        #     #     sexp = "( ( %d %d ) %d )" % (inputs[0], inputs[1], inputs[2])
        #     #     sexp = sexp.split()
        #     #     return sexp
        #     if edus_head[2][2] == "ROOT":
        #         # C が ROOT
        #         sexp = "( ( %d %d ) %d )" % (inputs[0], inputs[1], inputs[2])
        #         sexp = sexp.split()
        #         return sexp
        #     # elif (edus[0][0] in ["--", "-lrb-"]) and (edus[1][-1] in ["--", "-rrb-"]):
        #     #     # ( A B ) が記号で囲まれる
        #     #     sexp = "( ( %d %d ) %d )" % (inputs[0], inputs[1], inputs[2])
        #     #     sexp = sexp.split()
        #     #     return sexp
        #     # elif edus_head[1][2] == "ROOT":
        #     #     if edus[0][-1] != ",":
        #     #         # print("B が ROOT かつ、A が ',' で終わらない")
        #     #         sexp = "( ( %d %d ) %d )" % (inputs[0], inputs[1], inputs[2])
        #     #         sexp = sexp.split()
        #     #         return sexp
        #     # その他
        #     sexp = "( %d ( %d %d ) )" % (inputs[0], inputs[1], inputs[2])
        #     sexp = sexp.split()
        #     return sexp
        # elif len(inputs) == 4:
        #     # if (edus[1][0] in ["--", "-lrb-"]) and (edus[1][-1] in ["--", "-rrb-"]):
        #     #     sexp = "( ( %d %d ) ( %d %d ) )" % (inputs[0], inputs[1], inputs[2], inputs[3])
        #     #     sexp = sexp.split()
        #     #     return sexp
        #     # elif (edus[2][0] in ["--", "-lrb-"]) and (edus[2][-1] in ["--", "-rrb-"]):
        #     #     sexp = "( %d ( ( %d %d ) %d ) )" % (inputs[0], inputs[1], inputs[2], inputs[3])
        #     #     sexp = sexp.split()
        #     #     return sexp
        #     if edus_head[2][2] == "ROOT":
        #         sexp = "( ( %d %d ) ( %d %d ) )" % (inputs[0], inputs[1], inputs[2], inputs[3])
        #         sexp = sexp.split()
        #         return sexp
        #     elif edus_head[3][2] == "ROOT":
        #         sexp = "( ( %d ( %d %d ) ) %d )" % (inputs[0], inputs[1], inputs[2], inputs[3])
        #         sexp = sexp.split()
        #         return sexp
        #     # その他
        #     sexp = "( %d ( %d ( %d %d ) ) )" % (inputs[0], inputs[1], inputs[2], inputs[3])
        #     sexp = sexp.split()
        #     return sexp
        else:
            sexp = self.sampler.sample(inputs, edus, edus_head)
            # x = copy.deepcopy(inputs)
            # while len(x) > 1:
            #     lhs = x[-2]
            #     rhs = x[-1]
            #     merged = "( %s %s )" % (lhs, rhs)
            #     x[-2] = merged
            #     x.pop(-1)
            # sexp = x[0]
            # sexp = sexp.split()
            return sexp


