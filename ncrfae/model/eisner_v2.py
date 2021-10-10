import numpy as np
import torch
import torch.nn as nn
# from itertools import chain

RIGHT = 0
LEFT = 1
COMPLETE = 0
INCOMPLETE = 1


class IncrementalEisnerDecoder(object):

    def __init__(self):
        self.decoder = EisnerDecoder()
        self.summer = EisnerSummer()
    #
    # def decode(self,
    #         arc_scores,
    #         edu_ids,
    #         sbnds,
    #         pbnds,
    #         use_sbnds,
    #         use_pbnds,
    #         gold_heads=None):
    #     """
    #     :type arc_scores: numpy.ndarray(shape=(n_edus, n_edus), dtype="float")
    #     :type edu_ids: list of int
    #     :type sbnds: list of (int, int)
    #     :type pbnds: list of (int, int)
    #     :type use_sbnds: bool
    #     :type use_pbnds: bool
    #     :type gold_heads: numpy.ndarray(shape=(n_edus, n_edus), dtype=np.int32)
    #     :rtype: list of (int, int)
    #     """
    #     assert edu_ids[0] == 0 # NOTE
    #
    #     arcs = []
    #
    #     # Exclude ROOT
    #     new_edu_ids = edu_ids[1:]
    #
    #     # Sentence-level parsing
    #     if use_sbnds:
    #         target_bnds = sbnds
    #         sub_arcs, new_edu_ids, _, _ = self.apply_decoder(
    #                                             arc_scores=arc_scores,
    #                                             edu_ids=new_edu_ids,
    #                                             target_bnds=target_bnds,
    #                                             gold_heads=gold_heads)
    #         arcs.extend(sub_arcs)
    #
    #     # Paragraph-level parsing
    #     if use_pbnds:
    #         if use_sbnds:
    #             target_bnds = pbnds
    #         else:
    #             target_bnds = [(sbnds[b][0],sbnds[e][1]) for b,e in pbnds]
    #         sub_arcs, new_edu_ids, _, _ = self.apply_decoder(
    #                                             arc_scores=arc_scores,
    #                                             edu_ids=new_edu_ids,
    #                                             target_bnds=target_bnds,
    #                                             gold_heads=gold_heads)
    #         arcs.extend(sub_arcs)
    #
    #     # Document-level parsing
    #     sub_arcs, head, _, _ = self.decoder.decode_without_root(
    #                                             arc_scores=arc_scores,
    #                                             edu_ids=new_edu_ids,
    #                                             gold_heads=gold_heads)
    #     arcs.extend(sub_arcs)
    #
    #     # Root attachment
    #     arcs.append((0, head))
    #
    #     return arcs

    def global_decode(self, arc_scores, edu_ids, sbnds, pbnds, use_sbnds, use_pbnds, gold_heads=None):
        assert edu_ids[0] == 0

        chart = {}
        back_ptr = {}
        # Exclude ROOT
        new_edu_ids = edu_ids[1:]

        # Sentence-level parsing
        target_bnds = None
        if use_sbnds:
            target_bnds = sbnds
            sub_arcs, tmp, sub_chart, sub_back_ptr = self.apply_sentence_level_decoder(
                arc_scores=arc_scores, edu_ids=new_edu_ids, target_bnds=target_bnds, gold_heads=gold_heads)
            self.merge_chart(chart, sub_chart, target_bnds)
            back_ptr = sub_back_ptr

        # Paragraph-level parsing
        if use_pbnds:
            paragraph_chart = {}
            sub_arcs, tmp, sub_chart, sub_back_ptr = self.apply_paragraph_level_decoder(
                arc_scores=arc_scores, edu_ids=new_edu_ids, pbnds=pbnds, sbnds=sbnds, gold_heads=gold_heads,
                chart=chart, back_ptr=back_ptr)
            new_pbnds = []
            for begin_i, end_i in pbnds:
                new_pbnds.append([sbnds[begin_i][0], sbnds[end_i][1]])

            self.merge_chart(paragraph_chart, sub_chart, new_pbnds)
            chart = paragraph_chart
            back_ptr = sub_back_ptr

        assert target_bnds is not None
        # Document-level parsing
        arcs, score = self.decoder.discourse_level_decode(arc_scores=arc_scores, edu_ids=edu_ids,
                                                          chart=chart, back_ptr=back_ptr, target_bnds=target_bnds,
                                                          gold_heads=gold_heads)
        return arcs

    # Alert here we edit chart inner class
    def merge_chart(self, chart, sub_chart, target_bnds):
        for begin_i, end_i in target_bnds:
            i1 = begin_i+1
            i3 = end_i+1
            for i2 in range(i1, i3+1):
                chart[i1, i2, LEFT, COMPLETE] = sub_chart[i1, i2, LEFT, COMPLETE]
                # chart[i1, i2, LEFT, INCOMPLETE] = sub_chart[i1, i2, LEFT, INCOMPLETE]
                chart[i2, i3, RIGHT, COMPLETE] = sub_chart[i2, i3, RIGHT, COMPLETE]
                # chart[i2, i3, RIGHT, INCOMPLETE] = sub_chart[i2, i3, RIGHT, INCOMPLETE]
                # chart[i2, i2, LEFT, COMPLETE] = sub_chart[i2, i2, LEFT, COMPLETE]
                # chart[i2, i2, RIGHT, COMPLETE] = sub_chart[i2, i2, RIGHT, COMPLETE]

    def partition(self,
                  arc_scores,
                  edu_ids,
                  sbnds,
                  pbnds,
                  use_sbnds,
                  use_pbnds):
        assert edu_ids[0] == 0

        chart = {}
        # Exclude ROOT
        new_edu_ids = edu_ids[1:]

        # Sentence-level parsing
        if use_sbnds:
            target_bnds = sbnds
            sub_chart = self.apply_sentence_level_summer(arc_scores=arc_scores, edu_ids=new_edu_ids, target_bnds=target_bnds)
            self.merge_chart(chart, sub_chart, target_bnds)

        # Paragraph-level parsing
        if use_pbnds:
            chart = self.apply_paragraph_level_summer(arc_scores=arc_scores,
                                                      edu_ids=new_edu_ids,
                                                      pbnds=pbnds, sbnds=sbnds,
                                                      chart=chart)

        # Document-level parsing
        score = self.summer.discourse_level_summ(arc_scores=arc_scores,
                                                 edu_ids=edu_ids,
                                                 chart=chart,
                                                 target_bnds=sbnds)
        return score

    # this method
    def partition_all(self, arc_scores, edu_ids, sbnds, pbnds,
                      use_sbnds, use_pbnds):
        return self.summer.summ(arc_scores, edu_ids)

    def apply_sentence_level_summer(self, arc_scores, edu_ids, target_bnds):
        """
        :type arc_scores: numpy.ndarray(shape=(n_edus, n_edus), dtype="float")
        :type edu_ids: list of int
        :type target_bnds: list of (int, int)
        :rtype: list of (int, int), list of int
        """
        chart = {}
        for begin_i, end_i in target_bnds:
            if begin_i == end_i:
                chart[begin_i+1, begin_i+1, LEFT, COMPLETE] = 0
                chart[begin_i+1, begin_i+1, RIGHT, COMPLETE] = 0
            else:
                sub_chart = self.summer.sentence_level_summ(arc_scores=arc_scores, edu_ids=edu_ids[begin_i:end_i+1])

                for key, score in sub_chart.items():
                    chart[key] = score
        return chart

    def apply_paragraph_level_summer(self, arc_scores, edu_ids, sbnds, pbnds, chart):
        """
        :type arc_scores: numpy.ndarray(shape=(n_edus, n_edus), dtype="float")
        :type edu_ids: list of int
        :type sbnds: list of (int, int)
        :type pbnds: list of (int, int)
        :chart: dict of [(int, int, int, int): float]
        :rtype: dict of [(int, int, int, int): float]
        """
        for begin_i, end_i in pbnds:

            # get target_bnds
            target_bnds = sbnds[begin_i:end_i+1]

            if begin_i == end_i:
                # chart[begin_i+1, begin_i+1, LEFT, COMPLETE] = 0
                # chart[begin_i+1, begin_i+1, RIGHT, COMPLETE] = 0
                pass
            else:
                chart = self.summer.paragraph_level_summ(arc_scores=arc_scores, target_bnds=target_bnds, chart=chart,
                                                         edu_ids=edu_ids[sbnds[begin_i][0]:sbnds[end_i][1]+1])
        return chart

    def batch_summer(self, arc_scores, edu_ids):
        return self.summer.batch_summ(arc_scores, edu_ids)

    def apply_sentence_level_decoder(self, arc_scores, edu_ids, target_bnds, gold_heads):
        """
        :type arc_scores: numpy.ndarray(shape=(n_edus, n_edus), dtype="float")
        :type edu_ids: list of int
        :type target_bnds: list of (int, int)
        :type gold_heads: numpy.ndarray(shape=(n_edus, n_edus), dtype=np.int32)
        :rtype: list of (int, int), list of int
        """
        arcs = [] # list of (int, int)
        new_edu_ids = [] # list of int
        chart = {}
        back_ptr = {}
        for begin_i, end_i in target_bnds:
            if begin_i == end_i:
                sub_arcs = []
                head = edu_ids[begin_i]
                chart[begin_i+1, begin_i+1, LEFT, COMPLETE] = 0
                chart[begin_i+1, begin_i+1, RIGHT, COMPLETE] = 0
            else:
                sub_arcs, head, sub_chart, sub_back_ptr = self.decoder.sentence_level_decode(
                    arc_scores=arc_scores, edu_ids=edu_ids[begin_i:end_i+1], gold_heads=gold_heads)
                # transfer sub_chart to global chart
                # sub_edu_ids = edu_ids[begin_i: end_i+1]
                for key, score in sub_chart.items():
                    chart[key] = score

                for key, id in sub_back_ptr.items():
                    back_ptr[key] = id

            arcs.extend(sub_arcs)
            new_edu_ids.append(head)

        return arcs, new_edu_ids, chart, back_ptr

    def apply_paragraph_level_decoder(self, arc_scores, edu_ids, sbnds, pbnds, gold_heads, chart, back_ptr):
        """
        :type arc_scores: numpy.ndarray(shape=(n_edus, n_edus), dtype="float")
        :type edu_ids: list of int
        :type sbnds: list of (int, int)
        :type pbnds: list of (int, int)
        :type gold_heads: numpy.ndarray(shape=(n_edus, n_edus), dtype=np.int32)
        :chart: dict of [(int, int, int, int): int]
        :back_ptr: dict of [(int, int, int, int): int]
        :rtype: list of (int, int), list of int
        """
        arcs = [] # list of (int, int)
        new_edu_ids = [] # list of int
        for begin_i, end_i in pbnds:

            # get target_bnds
            target_bnds = sbnds[begin_i:end_i+1]

            if begin_i == end_i:
                sub_arcs = []
                head = edu_ids[begin_i]
                # chart[begin_i+1, begin_i+1, LEFT, COMPLETE] = 0
                # chart[begin_i+1, begin_i+1, RIGHT, COMPLETE] = 0
            else:
                sub_arcs, head, chart, back_ptr = self.decoder.paragraph_level_decode(
                    arc_scores=arc_scores, edu_ids=edu_ids[sbnds[begin_i][0]:sbnds[end_i][1]+1], gold_heads=gold_heads,
                    chart=chart, back_ptr=back_ptr, target_bnds=target_bnds)

            arcs.extend(sub_arcs)
            new_edu_ids.append(head)

        return arcs, new_edu_ids, chart, back_ptr


class EisnerDecoder(object):

    def __init__(self):
        pass

    def decode(self,
               arc_scores,
               edu_ids,
               gold_heads=None):
        """
        :type arc_scores: numpy.ndarray(shape=(n_edus, n_edus), dtype="float")
        :type edu_ids: list of int
        :type gold_heads: numpy.ndarray(shape=(n_edus, n_edus), dtype=np.int32)
        :rtype: list of (int, int)
        """
        assert edu_ids[0] == 0 # NOTE: Including the Root

        # Initialize charts
        chart = {} # {(int, int, int, int): float}
        back_ptr = {} # {(int, int, int, int): float}

        length = len(edu_ids)

        # Base case
        for i in range(length):
            chart[i, i, LEFT, COMPLETE] = 0.0
            chart[i, i, RIGHT, COMPLETE] = 0.0
            chart[i, i, LEFT, INCOMPLETE] = 0.0
            chart[i, i, RIGHT, INCOMPLETE] = 0.0
        for i in range(length):
            chart[0, i, LEFT, INCOMPLETE] = -np.inf

        # General case (without ROOT)
        for d in range(1, length):
            for i1 in range(1, length - d): # NOTE
                i3 = i1 + d

                # Incomplete span
                # Left tree
                max_score = -np.inf
                memo = None
                arc_score = arc_scores[edu_ids[i3], edu_ids[i1]]
                if gold_heads is not None:
                    if gold_heads[edu_ids[i1]] != edu_ids[i3]:
                        arc_score += 1.0
                for i2 in range(i1, i3):
                    score = arc_score \
                            + chart[i1, i2, RIGHT, COMPLETE] \
                            + chart[i2+1, i3, LEFT, COMPLETE]
                    if max_score < score:
                        max_score = score
                        memo = i2
                chart[i1, i3, LEFT, INCOMPLETE] = max_score
                back_ptr[i1, i3, LEFT, INCOMPLETE] = memo
                # Right tree
                max_score = -np.inf
                memo = None
                arc_score = arc_scores[edu_ids[i1], edu_ids[i3]]
                if gold_heads is not None:
                    if gold_heads[edu_ids[i3]] != edu_ids[i1]:
                        arc_score += 1.0
                for i2 in range(i1, i3):
                    score = arc_score \
                            + chart[i1, i2, RIGHT, COMPLETE] \
                            + chart[i2+1, i3, LEFT, COMPLETE]
                    if max_score < score:
                        max_score = score
                        memo = i2
                chart[i1, i3, RIGHT, INCOMPLETE] = max_score
                back_ptr[i1, i3, RIGHT, INCOMPLETE] = memo

                # Complete span
                # Left tree
                max_score = -np.inf
                memo = None
                for i2 in range(i1, i3):
                    score = chart[i1, i2, LEFT, COMPLETE] \
                            + chart[i2, i3, LEFT, INCOMPLETE]
                    if max_score < score:
                        max_score = score
                        memo = i2
                chart[i1, i3, LEFT, COMPLETE] = max_score
                back_ptr[i1, i3, LEFT, COMPLETE] = memo
                # Right tree
                max_score = -np.inf
                memo = None
                for i2 in range(i1, i3):
                    score = chart[i1, i2+1, RIGHT, INCOMPLETE] + chart[i2+1, i3, RIGHT, COMPLETE]
                    if max_score < score:
                        max_score = score
                        memo = i2 + 1
                chart[i1, i3, RIGHT, COMPLETE] = max_score
                back_ptr[i1, i3, RIGHT, COMPLETE] = memo

        # ROOT attachment
        # arcs = self.recover_tree(back_ptr, 0, length-1, RIGHT, COMPLETE, arcs=None) # NOTE
        max_score = -np.inf
        memo = None
        for i2 in range(1, length):
            arc_score = arc_scores[edu_ids[0], edu_ids[i2]]
            score = arc_score \
                    + chart[0, 0, RIGHT, COMPLETE] \
                    + chart[1, i2, LEFT, COMPLETE] \
                    + chart[i2, length - 1, RIGHT, COMPLETE]
            if max_score < score:
                max_score = score
                memo = i2
        chart[0, length-1, RIGHT, COMPLETE] = max_score
        back_ptr[0, length-1, RIGHT, COMPLETE] = memo
        head = memo

        # Recovering dependency arcs
        arcs = [(0, head)]
        arcs = self.recover_tree(back_ptr, 1, head, LEFT, COMPLETE, arcs=arcs)
        arcs = self.recover_tree(back_ptr, head, length-1, RIGHT, COMPLETE, arcs=arcs)

        # Shifting: local position -> global position
        arcs = [(edu_ids[h], edu_ids[d]) for h,d in arcs]
        return arcs, max_score

    def sentence_level_decode(self, arc_scores, edu_ids, gold_heads=None):
        """
        :type arc_scores: numpy.ndarray(shape=(n_edus, n_edus), dtype="float")
        :type edu_ids: list of int
        :type gold_heads: numpy.ndarray(shape=(n_edus, n_edus), dtype=np.int32)
        :rtype: list of (int, int)
        """
        assert edu_ids[0] != 0  # NOTE: Without the Root


        # Initialize charts
        chart = {}  # {(int, int, int, int): float}
        back_ptr = {}  # {(int, int, int, int): float}

        length = len(edu_ids)

        # Base case
        for i in edu_ids:
            chart[i, i, LEFT, COMPLETE] = 0.0
            chart[i, i, RIGHT, COMPLETE] = 0.0
            chart[i, i, LEFT, INCOMPLETE] = 0.0
            chart[i, i, RIGHT, INCOMPLETE] = 0.0

        # General case
        for d in range(1, length):
            # for i1 in range(0, length - d):  # NOTE: index "0" does NOT represent ROOT
            for i1 in range(edu_ids[0], edu_ids[-1]-d+1):
                i3 = i1 + d

                # Incomplete span
                # Left tree
                max_score = -np.inf
                memo = None
                # arc_score = arc_scores[edu_ids[i3], edu_ids[i1]]
                arc_score = arc_scores[i3, i1]
                # if gold_heads is not None:
                #     if gold_heads[edu_ids[i1]] != edu_ids[i3]:
                #         arc_score += 1.0
                for i2 in range(i1, i3):
                    score = arc_score + chart[i1, i2, RIGHT, COMPLETE] + chart[i2 + 1, i3, LEFT, COMPLETE]
                    if max_score < score:
                        max_score = score
                        memo = i2
                chart[i1, i3, LEFT, INCOMPLETE] = max_score

                back_ptr[i1, i3, LEFT, INCOMPLETE] = memo
                # Right tree
                max_score = -np.inf
                memo = None
                # arc_score = arc_scores[edu_ids[i1], edu_ids[i3]]
                arc_score = arc_scores[i1, i3]
                # if gold_heads is not None:
                #     if gold_heads[edu_ids[i3]] != edu_ids[i1]:
                #         arc_score += 1.0
                for i2 in range(i1, i3):
                    score = arc_score \
                            + chart[i1, i2, RIGHT, COMPLETE] \
                            + chart[i2 + 1, i3, LEFT, COMPLETE]
                    if max_score < score:
                        max_score = score
                        memo = i2
                chart[i1, i3, RIGHT, INCOMPLETE] = max_score

                back_ptr[i1, i3, RIGHT, INCOMPLETE] = memo

                # Complete span
                # Left tree
                max_score = -np.inf

                memo = None
                for i2 in range(i1, i3):
                    score = chart[i1, i2, LEFT, COMPLETE] \
                            + chart[i2, i3, LEFT, INCOMPLETE]
                    if max_score < score:
                        max_score = score
                        memo = i2
                chart[i1, i3, LEFT, COMPLETE] = max_score

                back_ptr[i1, i3, LEFT, COMPLETE] = memo
                # Right tree
                max_score = -np.inf
                memo = None
                for i2 in range(i1, i3):
                    score = chart[i1, i2 + 1, RIGHT, INCOMPLETE] \
                            + chart[i2 + 1, i3, RIGHT, COMPLETE]
                    if max_score < score:
                        max_score = score
                        memo = i2 + 1
                chart[i1, i3, RIGHT, COMPLETE] = max_score
                back_ptr[i1, i3, RIGHT, COMPLETE] = memo

        # ROOT identification
        max_score = -np.inf
        memo = None
        for i2 in range(edu_ids[0], edu_ids[-1]+1):
        # for i2 in range(0, length):
            # score = chart[0, i2, LEFT, COMPLETE] + chart[i2, length - 1, RIGHT, COMPLETE]
            score = chart[edu_ids[0], i2, LEFT, COMPLETE] + chart[i2, edu_ids[-1], RIGHT, COMPLETE]
            if max_score < score:
                max_score = score
                memo = i2
        head = memo

        # Recovering dependency arcs
        # arcs = self.recover_tree(back_ptr, 0, head, LEFT, COMPLETE, arcs=None)
        arcs = self.recover_tree(back_ptr, edu_ids[0], head, LEFT, COMPLETE, arcs=None)
        # arcs = self.recover_tree(back_ptr, head, length - 1, RIGHT, COMPLETE, arcs=arcs)
        arcs = self.recover_tree(back_ptr, head, edu_ids[-1], RIGHT, COMPLETE, arcs=arcs)

        # Shifting: local position -> global position
        # arcs = [(edu_ids[h], edu_ids[d]) for h, d in arcs]
        # head = edu_ids[head]
        return arcs, head, chart, back_ptr


    def paragraph_level_decode(self, arc_scores, edu_ids, chart=None, back_ptr=None, target_bnds=None, gold_heads=None):
        """
        do paragraph level decode based on sentence level result
        :type arc_scores: numpy.ndarray(shape=(n_edus, n_edus), dtype="float")
        :type edu_ids: list of int
        :type chart: dict of [(int, int, int, int): int]
        :type back_ptr: dict of [(int, int, int, int): int]
        :type target_bnds: list of (int, int)
        :type gold_heads: numpy.ndarray(shape=(n_edus, n_edus), dtype=np.int32)
        :rtype: list of (int, int)
        """
        assert edu_ids[0] != 0  # NOTE: Without the Root
        assert chart is not None
        assert back_ptr is not None
        assert target_bnds is not None

        length = len(edu_ids)

        # Base case
        # for i in edu_ids:
        #     chart[i, i, LEFT, COMPLETE] = 0.0
        #     chart[i, i, RIGHT, COMPLETE] = 0.0
        #     chart[i, i, LEFT, INCOMPLETE] = 0.0
        #     chart[i, i, RIGHT, INCOMPLETE] = 0.0

        # General case
        for d in range(1, length):
            # for i1 in range(0, length - d):  # NOTE: index "0" does NOT represent ROOT
            for i1 in range(edu_ids[0], edu_ids[-1]-d+1):
                i3 = i1 + d

                decoded = False
                for begin_i, end_i in target_bnds:
                    if begin_i+1 <=i1 <= i3 <= end_i+1:
                        decoded = True
                        break
                if decoded:
                    continue

                # Incomplete span
                # Left tree
                max_score = -np.inf
                score = -np.inf
                memo = None
                # arc_score = arc_scores[edu_ids[i3], edu_ids[i1]]
                arc_score = arc_scores[i3, i1]
                # if gold_heads is not None:
                #     if gold_heads[edu_ids[i1]] != edu_ids[i3]:
                #         arc_score += 1.0
                for i2 in range(i1, i3):
                    if (i1, i2, RIGHT, COMPLETE) in chart and \
                            (i2 + 1, i3, LEFT, COMPLETE) in chart:
                        score = arc_score \
                                + chart[i1, i2, RIGHT, COMPLETE] \
                                + chart[i2 + 1, i3, LEFT, COMPLETE]

                    if max_score < score:
                        max_score = score
                        memo = i2
                if max_score != -np.inf:
                    chart[i1, i3, LEFT, INCOMPLETE] = max_score
                    back_ptr[i1, i3, LEFT, INCOMPLETE] = memo
                # Right tree
                max_score = -np.inf
                score = -np.inf
                memo = None
                # arc_score = arc_scores[edu_ids[i1], edu_ids[i3]]
                arc_score = arc_scores[i1, i3]
                # if gold_heads is not None:
                #     if gold_heads[edu_ids[i3]] != edu_ids[i1]:
                #         arc_score += 1.0
                for i2 in range(i1, i3):
                    if (i1, i2, RIGHT, COMPLETE) in chart and \
                        (i2 + 1, i3, LEFT, COMPLETE) in chart:
                        score = arc_score \
                                + chart[i1, i2, RIGHT, COMPLETE] \
                                + chart[i2 + 1, i3, LEFT, COMPLETE]
                    if max_score < score:
                        max_score = score
                        memo = i2
                if max_score != -np.inf:
                    chart[i1, i3, RIGHT, INCOMPLETE] = max_score
                    back_ptr[i1, i3, RIGHT, INCOMPLETE] = memo

                # Complete span
                # Left tree
                max_score = -np.inf
                score = -np.inf
                memo = None
                for i2 in range(i1, i3):
                    if (i1, i2, LEFT, COMPLETE) in chart and \
                        (i2, i3, LEFT, INCOMPLETE) in chart:
                        score = chart[i1, i2, LEFT, COMPLETE] \
                                + chart[i2, i3, LEFT, INCOMPLETE]
                    if max_score < score:
                        max_score = score
                        memo = i2
                if max_score != -np.inf:
                    chart[i1, i3, LEFT, COMPLETE] = max_score
                    back_ptr[i1, i3, LEFT, COMPLETE] = memo
                # Right tree
                max_score = -np.inf
                score = -np.inf
                memo = None
                for i2 in range(i1, i3):
                    if (i1, i2 + 1, RIGHT, INCOMPLETE) in chart \
                            and (i2 + 1, i3, RIGHT, COMPLETE) in chart:
                        score = chart[i1, i2 + 1, RIGHT, INCOMPLETE] + chart[i2 + 1, i3, RIGHT, COMPLETE]
                    if max_score < score:
                        max_score = score
                        memo = i2 + 1
                if max_score != -np.inf:
                    chart[i1, i3, RIGHT, COMPLETE] = max_score
                    back_ptr[i1, i3, RIGHT, COMPLETE] = memo

        # ROOT identification
        max_score = -np.inf
        memo = None
        for i2 in range(edu_ids[0], edu_ids[-1]+1):
            score = chart[edu_ids[0], i2, LEFT, COMPLETE] + chart[i2, edu_ids[-1], RIGHT, COMPLETE]
            if max_score < score:
                max_score = score
                memo = i2
        head = memo

        # Recovering dependency arcs
        arcs = self.recover_tree(back_ptr, edu_ids[0], head, LEFT, COMPLETE, arcs=None)
        arcs = self.recover_tree(back_ptr, head, edu_ids[-1], RIGHT, COMPLETE, arcs=arcs)

        # Shifting: local position -> global position
        # arcs = [(edu_ids[h], edu_ids[d]) for h, d in arcs]
        # head = edu_ids[head]
        return arcs, head, chart, back_ptr

    def discourse_level_decode(self, arc_scores, edu_ids, target_bnds, gold_heads=None, chart=None, back_ptr=None):
        # assert edu_ids[0] == 0  # NOTE: Including the Root
        # Initialize charts
        chart = {} if chart is None else chart # {(int, int, int, int): float}
        back_ptr = {} if back_ptr is None else back_ptr  # {(int, int, int, int): float}
        edu_ids = edu_ids[1:]
        length = len(edu_ids)

        # Base case
        chart[0, 0, LEFT, COMPLETE] = 0.0
        chart[0, 0, RIGHT, COMPLETE] = 0.0
        chart[0, 0, LEFT, INCOMPLETE] = 0.0
        chart[0, 0, RIGHT, INCOMPLETE] = 0.0

        for i in range(length):
            chart[0, i, LEFT, INCOMPLETE] = -np.inf

        # General case (without ROOT)
        for d in range(1, length):
            # for i1 in range(1, length - d):  # NOTE
            for i1 in range(edu_ids[0], edu_ids[-1]-d+1):  # NOTE
                i3 = i1 + d

                decoded = False
                for begin_i, end_i in target_bnds:
                    if begin_i+1 <=i1 <= i3 <= end_i+1:
                        decoded = True
                        break
                if decoded:
                    continue

                # Incomplete span
                # Left tree
                max_score = -np.inf
                score = -np.inf
                memo = None
                # arc_score = arc_scores[edu_ids[i3], edu_ids[i1]]
                arc_score = arc_scores[i3, i1]
                # if gold_heads is not None:
                #     if gold_heads[edu_ids[i1]] != edu_ids[i3]:
                #         arc_score += 1.0
                for i2 in range(i1, i3):
                    # if i2 not in candidate_i2:
                    #     continue
                    if (i1, i2, RIGHT, COMPLETE) in chart and \
                            (i2 + 1, i3, LEFT, COMPLETE) in chart:
                        score = arc_score \
                                + chart[i1, i2, RIGHT, COMPLETE] \
                                + chart[i2 + 1, i3, LEFT, COMPLETE]

                    if max_score < score:
                        max_score = score
                        memo = i2
                if max_score != -np.inf:
                    chart[i1, i3, LEFT, INCOMPLETE] = max_score
                    back_ptr[i1, i3, LEFT, INCOMPLETE] = memo
                # Right tree
                max_score = -np.inf
                score = -np.inf
                memo = None
                arc_score = arc_scores[i1, i3]
                # arc_score = arc_scores[edu_ids[i1], edu_ids[i3]]
                if gold_heads is not None:
                    if gold_heads[edu_ids[i3]] != edu_ids[i1]:
                        arc_score += 1.0

                for i2 in range(i1, i3):
                    # if i2 not in candidate_i2:
                    #     continue
                    if (i1, i2, RIGHT, COMPLETE) in chart and \
                        (i2 + 1, i3, LEFT, COMPLETE) in chart:
                        score = arc_score \
                                + chart[i1, i2, RIGHT, COMPLETE] \
                                + chart[i2 + 1, i3, LEFT, COMPLETE]
                    if max_score < score:
                        max_score = score
                        memo = i2
                if max_score != -np.inf:
                    chart[i1, i3, RIGHT, INCOMPLETE] = max_score
                    back_ptr[i1, i3, RIGHT, INCOMPLETE] = memo

                # Complete span
                # Left tree
                max_score = -np.inf
                score = -np.inf
                memo = None
                for i2 in range(i1, i3):
                    if (i1, i2, LEFT, COMPLETE) in chart and \
                        (i2, i3, LEFT, INCOMPLETE) in chart:
                        score = chart[i1, i2, LEFT, COMPLETE] \
                                + chart[i2, i3, LEFT, INCOMPLETE]
                    if max_score < score:
                        max_score = score
                        memo = i2
                if max_score != -np.inf:
                    chart[i1, i3, LEFT, COMPLETE] = max_score
                    back_ptr[i1, i3, LEFT, COMPLETE] = memo
                # Right tree
                max_score = -np.inf
                score = -np.inf
                memo = None
                for i2 in range(i1, i3):
                    if (i1, i2 + 1, RIGHT, INCOMPLETE) in chart \
                            and (i2 + 1, i3, RIGHT, COMPLETE) in chart:
                        score = chart[i1, i2 + 1, RIGHT, INCOMPLETE] + chart[i2 + 1, i3, RIGHT, COMPLETE]
                    if max_score < score:
                        max_score = score
                        memo = i2 + 1
                if max_score != -np.inf:
                    chart[i1, i3, RIGHT, COMPLETE] = max_score
                    back_ptr[i1, i3, RIGHT, COMPLETE] = memo

        # ROOT attachment
        max_score = -np.inf
        memo = None
        for i2 in range(edu_ids[0], edu_ids[-1]+1):
            # arc_score = arc_scores[edu_ids[0], edu_ids[i2]]
            arc_score = arc_scores[0, i2]
            score = arc_score \
                    + chart[0, 0, RIGHT, COMPLETE] \
                    + chart[edu_ids[0], i2, LEFT, COMPLETE] \
                    + chart[i2, edu_ids[-1], RIGHT, COMPLETE]
            if max_score < score:
                max_score = score
                memo = i2
        # chart[0, length - 1, RIGHT, COMPLETE] = max_score
        chart[0, edu_ids[-1], RIGHT, COMPLETE] = max_score
        # back_ptr[0, length - 1, RIGHT, COMPLETE] = memo
        back_ptr[0, edu_ids[-1], RIGHT, COMPLETE] = memo
        head = memo

        # Recovering dependency arcs
        arcs = [(0, head)]
        # arcs = self.recover_tree(back_ptr, 1, head, LEFT, COMPLETE, arcs=arcs)
        arcs = self.recover_tree(back_ptr, 1, head, LEFT, COMPLETE, arcs=arcs)
        # arcs = self.recover_tree(back_ptr, head, length - 1, RIGHT, COMPLETE, arcs=arcs)
        arcs = self.recover_tree(back_ptr, head, edu_ids[-1], RIGHT, COMPLETE, arcs=arcs)

        # Shifting: local position -> global position
        # arcs = [(edu_ids[h], edu_ids[d]) for h, d in arcs]
        return arcs, max_score


    def recover_tree(self, back_ptr, i1, i3, direction, complete, arcs=None):
        """
        :type back_ptr: {(int, int, int, int): int}
        :type i1: int
        :type i3: int
        :type direction: int
        :type complete: int
        :type arcs: list of (int, int)
        :rtype: list of (int, int)
        """
        if arcs is None:
            arcs = []

        if i1 == i3:
            return arcs

        i2 = back_ptr[i1, i3, direction, complete]
        if complete == COMPLETE:
            if direction == LEFT:
                arcs = self.recover_tree(back_ptr, i1, i2, LEFT, COMPLETE, arcs=arcs)
                arcs = self.recover_tree(back_ptr, i2, i3, LEFT, INCOMPLETE, arcs=arcs)
            else:
                arcs = self.recover_tree(back_ptr, i1, i2, RIGHT, INCOMPLETE, arcs=arcs)
                arcs = self.recover_tree(back_ptr, i2, i3, RIGHT, COMPLETE, arcs=arcs)
        else:
            if direction == LEFT:
                arcs.append((i3, i1))
                arcs = self.recover_tree(back_ptr, i1, i2, RIGHT, COMPLETE, arcs=arcs)
                arcs = self.recover_tree(back_ptr, i2+1, i3, LEFT, COMPLETE, arcs=arcs)
            else:
                arcs.append((i1, i3))
                arcs = self.recover_tree(back_ptr, i1, i2, RIGHT, COMPLETE, arcs=arcs)
                arcs = self.recover_tree(back_ptr, i2+1, i3, LEFT, COMPLETE, arcs=arcs)
        return arcs

class EisnerSummer(object):

    def discourse_level_summ(self, arc_scores, edu_ids, target_bnds, chart=None):
        assert edu_ids[0] == 0  # NOTE: Including the Root
        # Initialize charts
        chart = {} if chart is None else chart # {(int, int, int, int): float}
        length = len(edu_ids)

        # Base case
        chart[0, 0, LEFT, COMPLETE] = 0.0
        chart[0, 0, RIGHT, COMPLETE] = 0.0
        chart[0, 0, LEFT, INCOMPLETE] = 0.0
        chart[0, 0, RIGHT, INCOMPLETE] = 0.0

        for i in range(length):
            chart[0, i, LEFT, INCOMPLETE] = -np.inf

        # candidate_i2 = set(i+1 for i in chain.from_iterable(target_bnds))
        # General case (without ROOT)
        for d in range(1, length):
            # for i1 in range(1, length - d):  # NOTE
            for i1 in range(edu_ids[0], edu_ids[-1]-d+1):  # NOTE
                i3 = i1 + d

                same_span = False
                for begin_i, end_i in target_bnds:
                    if begin_i+1 <=i1 <= i3 <= end_i+1:
                        same_span = True
                        break
                if same_span:
                    continue

                # Incomplete span
                # Left tree
                scores = []
                arc_score = arc_scores[i3, i1]
                # arc_score = arc_scores[edu_ids[i3], edu_ids[i1]]
                for i2 in range(i1, i3):
                    if (i1, i2, RIGHT, COMPLETE) in chart and \
                            (i2 + 1, i3, LEFT, COMPLETE) in chart:
                        score = arc_score \
                                + chart[i1, i2, RIGHT, COMPLETE] \
                                + chart[i2 + 1, i3, LEFT, COMPLETE]
                        scores.append(score)
                if len(scores) != 0:
                    chart[i1, i3, LEFT, INCOMPLETE] = self.logsumexp_scala(*scores)
                # Right tree
                scores = []
                arc_score = arc_scores[i1, i3]
                # arc_score = arc_scores[edu_ids[i1], edu_ids[i3]]
                for i2 in range(i1, i3):
                    if (i1, i2, RIGHT, COMPLETE) in chart and \
                        (i2 + 1, i3, LEFT, COMPLETE) in chart:
                        score = arc_score \
                                + chart[i1, i2, RIGHT, COMPLETE] \
                                + chart[i2 + 1, i3, LEFT, COMPLETE]
                        scores.append(score)
                if len(scores) != 0:
                    chart[i1, i3, RIGHT, INCOMPLETE] = self.logsumexp_scala(*scores)

                # Complete span
                # Left tree
                scores = []
                for i2 in range(i1, i3):
                    if (i1, i2, LEFT, COMPLETE) in chart and \
                        (i2, i3, LEFT, INCOMPLETE) in chart:
                        score = chart[i1, i2, LEFT, COMPLETE] \
                                + chart[i2, i3, LEFT, INCOMPLETE]
                        scores.append(score)
                if len(scores) != 0:
                    chart[i1, i3, LEFT, COMPLETE] = self.logsumexp_scala(*scores)
                # Right tree
                scores = []
                for i2 in range(i1, i3):
                    if (i1, i2 + 1, RIGHT, INCOMPLETE) in chart \
                            and (i2 + 1, i3, RIGHT, COMPLETE) in chart:
                        score = chart[i1, i2 + 1, RIGHT, INCOMPLETE] + chart[i2 + 1, i3, RIGHT, COMPLETE]
                        scores.append(score)
                if len(scores) != 0:
                    chart[i1, i3, RIGHT, COMPLETE] = self.logsumexp_scala(*scores)

        # ROOT attachment
        scores = []
        # alert here length -1 == edu_ids[-1]
        assert length - 1 == edu_ids[-1]
        for i2 in range(1, length):
            arc_score = arc_scores[0, i2]
            # arc_score = arc_scores[edu_ids[0], edu_ids[i2]]
            score = arc_score \
                    + chart[0, 0, RIGHT, COMPLETE] \
                    + chart[1, i2, LEFT, COMPLETE] \
                    + chart[i2, edu_ids[-1], RIGHT, COMPLETE]
            scores.append(score)
        chart[0, length - 1, RIGHT, COMPLETE] = self.logsumexp_scala(*scores)

        return chart[0, length - 1, RIGHT, COMPLETE]

    # calculate the score of all candidate tree
    def summ(self,
            arc_scores,
            edu_ids):
        """
        :type arc_scores: torch.tensor(shape=(n_edus, n_edus), dtype="float")
        :type edu_ids: list of int
        :rtype: torch.tensor
        """
        assert edu_ids[0] == 0 # NOTE: Including the Root

        # Initialize charts
        chart = {} # {(int, int, int, int): float}

        length = len(edu_ids)

        # Base case
        for i in range(length):
            chart[i, i, LEFT, COMPLETE] = 0.0
            chart[i, i, RIGHT, COMPLETE] = 0.0
            chart[i, i, LEFT, INCOMPLETE] = 0.0
            chart[i, i, RIGHT, INCOMPLETE] = 0.0
        for i in range(length):
            chart[0, i, LEFT, INCOMPLETE] = -np.inf

        # General case (without ROOT)
        for d in range(1, length):
            for i1 in range(1, length - d): # NOTE
                i3 = i1 + d

                # Incomplete span
                # Left tree
                scores = []
                arc_score = arc_scores[edu_ids[i3], edu_ids[i1]]
                for i2 in range(i1, i3):
                    score = arc_score \
                            + chart[i1, i2, RIGHT, COMPLETE] \
                            + chart[i2+1, i3, LEFT, COMPLETE]
                    scores.append(score)
                chart[i1, i3, LEFT, INCOMPLETE] = self.logsumexp_scala(*scores)
                # Right tree
                scores = []
                arc_score = arc_scores[edu_ids[i1], edu_ids[i3]]
                for i2 in range(i1, i3):
                    score = arc_score \
                            + chart[i1, i2, RIGHT, COMPLETE] \
                            + chart[i2+1, i3, LEFT, COMPLETE]
                    scores.append(score)
                chart[i1, i3, RIGHT, INCOMPLETE] = self.logsumexp_scala(*scores)

                # Complete span
                # Left tree
                scores = []
                for i2 in range(i1, i3):
                    score = chart[i1, i2, LEFT, COMPLETE] \
                            + chart[i2, i3, LEFT, INCOMPLETE]
                    scores.append(score)
                chart[i1, i3, LEFT, COMPLETE] = self.logsumexp_scala(*scores)
                # Right tree
                scores = []
                for i2 in range(i1, i3):
                    score = chart[i1, i2+1, RIGHT, INCOMPLETE] \
                            + chart[i2+1, i3, RIGHT, COMPLETE]
                    scores.append(score)
                chart[i1, i3, RIGHT, COMPLETE] = self.logsumexp_scala(*scores)

        # ROOT attachment
        scores = []
        for i2 in range(1, length):
            arc_score = arc_scores[edu_ids[0], edu_ids[i2]]
            score = arc_score \
                    + chart[0, 0, RIGHT, COMPLETE] \
                    + chart[1, i2, LEFT, COMPLETE] \
                    + chart[i2, length-1, RIGHT, COMPLETE]
            scores.append(score)
        chart[0, length-1, RIGHT, COMPLETE] = self.logsumexp_scala(*scores)

        return chart[0, length-1, RIGHT, COMPLETE]

    # calculate the score of all candidate tree
    def batch_summ(self, arc_scores, edu_ids):
        assert edu_ids[0] == 0 # NOTE: Including the Root

        # Initialize charts
        chart = {} # {(int, int, int, int): float}

        length = len(edu_ids)

        # Base case
        for i in range(length):
            chart[i, i, LEFT, COMPLETE] = 0.0
            chart[i, i, RIGHT, COMPLETE] = 0.0
            chart[i, i, LEFT, INCOMPLETE] = 0.0
            chart[i, i, RIGHT, INCOMPLETE] = 0.0
        for i in range(length):
            chart[0, i, LEFT, INCOMPLETE] = -np.inf

        # General case (without ROOT)
        for d in range(1, length):
            for i1 in range(1, length - d): # NOTE
                i3 = i1 + d

                # Incomplete span
                # Left tree
                scores = []
                arc_score = arc_scores[:, edu_ids[i3], edu_ids[i1]]
                for i2 in range(i1, i3):
                    score = arc_score \
                            + chart[i1, i2, RIGHT, COMPLETE] \
                            + chart[i2+1, i3, LEFT, COMPLETE]
                    scores.append(score)
                chart[i1, i3, LEFT, INCOMPLETE] = self.logsumexp_scala(*scores)
                # Right tree
                scores = []
                arc_score = arc_scores[:, edu_ids[i1], edu_ids[i3]]
                for i2 in range(i1, i3):
                    score = arc_score \
                            + chart[i1, i2, RIGHT, COMPLETE] \
                            + chart[i2+1, i3, LEFT, COMPLETE]
                    scores.append(score)
                chart[i1, i3, RIGHT, INCOMPLETE] = self.logsumexp_scala(*scores)

                # Complete span
                # Left tree
                scores = []
                for i2 in range(i1, i3):
                    score = chart[i1, i2, LEFT, COMPLETE] \
                            + chart[i2, i3, LEFT, INCOMPLETE]
                    scores.append(score)
                chart[i1, i3, LEFT, COMPLETE] = self.logsumexp_scala(*scores)
                # Right tree
                scores = []
                for i2 in range(i1, i3):
                    score = chart[i1, i2+1, RIGHT, INCOMPLETE] \
                            + chart[i2+1, i3, RIGHT, COMPLETE]
                    scores.append(score)
                chart[i1, i3, RIGHT, COMPLETE] = self.logsumexp_scala(*scores)

        # ROOT attachment
        # arcs = self.recover_tree(back_ptr, 0, length-1, RIGHT, COMPLETE, arcs=None) # NOTE
        # sum_score = 0.0
        scores = []
        for i2 in range(1, length):
            arc_score = arc_scores[:, edu_ids[0], edu_ids[i2]]
            score = arc_score \
                    + chart[0, 0, RIGHT, COMPLETE] \
                    + chart[1, i2, LEFT, COMPLETE] \
                    + chart[i2, length-1, RIGHT, COMPLETE]
            scores.append(score)
        chart[0, length-1, RIGHT, COMPLETE] = self.logsumexp_scala(*scores)

        # ROOT identification

        return chart[0, length-1, RIGHT, COMPLETE]
    #
    # def sum_without_root(self, arc_scores, edu_ids):
    #     assert edu_ids[0] != 0  # NOTE: Without the Root
    #
    #     if len(edu_ids) == 1:
    #         return [], edu_ids[0]
    #
    #     # Initialize charts
    #     chart = {}  # {(int, int, int, int): float}
    #
    #     length = len(edu_ids)
    #
    #     # Base case
    #     for i in range(length):
    #         chart[i, i, LEFT, COMPLETE] = 0.0
    #         chart[i, i, RIGHT, COMPLETE] = 0.0
    #         chart[i, i, LEFT, INCOMPLETE] = 0.0
    #         chart[i, i, RIGHT, INCOMPLETE] = 0.0
    #
    #     # General case
    #     for d in range(1, length):
    #         for i1 in range(0, length - d):  # NOTE: index "0" does NOT represent ROOT
    #             i3 = i1 + d
    #
    #             # Incomplete span
    #             # Left tree
    #             scores = []
    #             arc_score = arc_scores[edu_ids[i3], edu_ids[i1]]
    #             for i2 in range(i1, i3):
    #                 score = arc_score + chart[i1, i2, RIGHT, COMPLETE] + chart[i2 + 1, i3, LEFT, COMPLETE]
    #                 scores.append(score)
    #             chart[i1, i3, LEFT, INCOMPLETE] = self.logsumexp_scala(*scores)
    #             # Right tree
    #             scores = []
    #             arc_score = arc_scores[edu_ids[i1], edu_ids[i3]]
    #
    #             for i2 in range(i1, i3):
    #                 score = arc_score \
    #                         + chart[i1, i2, RIGHT, COMPLETE] \
    #                         + chart[i2 + 1, i3, LEFT, COMPLETE]
    #                 scores.append(score)
    #             chart[i1, i3, RIGHT, INCOMPLETE] = self.logsumexp_scala(*scores)
    #
    #             # Complete span
    #             # Left tree
    #             scores = []
    #             for i2 in range(i1, i3):
    #                 score = chart[i1, i2, LEFT, COMPLETE] \
    #                         + chart[i2, i3, LEFT, INCOMPLETE]
    #                 scores.append(score)
    #             chart[i1, i3, LEFT, COMPLETE] = self.logsumexp_scala(*scores)
    #             # Right tree
    #             scores = []
    #             for i2 in range(i1, i3):
    #                 score = chart[i1, i2 + 1, RIGHT, INCOMPLETE] \
    #                         + chart[i2 + 1, i3, RIGHT, COMPLETE]
    #                 scores.append(score)
    #             chart[i1, i3, RIGHT, COMPLETE] = self.logsumexp_scala(*scores)
    #
    #     return chart


    def sentence_level_summ(self, arc_scores, edu_ids):
        assert edu_ids[0] != 0  # NOTE: Without the Root

        if len(edu_ids) == 1:
            return [], edu_ids[0]

        # Initialize charts
        chart = {}  # {(int, int, int, int): float}

        length = len(edu_ids)

        # Base case
        for i in edu_ids:
            chart[i, i, LEFT, COMPLETE] = 0.0
            chart[i, i, RIGHT, COMPLETE] = 0.0
            chart[i, i, LEFT, INCOMPLETE] = 0.0
            chart[i, i, RIGHT, INCOMPLETE] = 0.0

        # General case
        for d in range(1, length):
            # for i1 in range(0, length - d):  # NOTE: index "0" does NOT represent ROOT
            for i1 in range(edu_ids[0], edu_ids[-1]-d+1):
                i3 = i1 + d

                # Incomplete span
                # Left tree
                scores = []
                # arc_score = arc_scores[edu_ids[i3], edu_ids[i1]]
                arc_score = arc_scores[i3, i1]
                for i2 in range(i1, i3):
                    score = arc_score + chart[i1, i2, RIGHT, COMPLETE] + chart[i2 + 1, i3, LEFT, COMPLETE]
                    scores.append(score)
                chart[i1, i3, LEFT, INCOMPLETE] = self.logsumexp_scala(*scores)
                # Right tree
                scores = []
                arc_score = arc_scores[i1, i3]
                # arc_score = arc_scores[edu_ids[i1], edu_ids[i3]]

                for i2 in range(i1, i3):
                    score = arc_score \
                            + chart[i1, i2, RIGHT, COMPLETE] \
                            + chart[i2 + 1, i3, LEFT, COMPLETE]
                    scores.append(score)
                chart[i1, i3, RIGHT, INCOMPLETE] = self.logsumexp_scala(*scores)

                # Complete span
                # Left tree
                scores = []
                for i2 in range(i1, i3):
                    score = chart[i1, i2, LEFT, COMPLETE] \
                            + chart[i2, i3, LEFT, INCOMPLETE]
                    scores.append(score)
                chart[i1, i3, LEFT, COMPLETE] = self.logsumexp_scala(*scores)
                # Right tree
                scores = []
                for i2 in range(i1, i3):
                    score = chart[i1, i2 + 1, RIGHT, INCOMPLETE] \
                            + chart[i2 + 1, i3, RIGHT, COMPLETE]
                    scores.append(score)
                chart[i1, i3, RIGHT, COMPLETE] = self.logsumexp_scala(*scores)

        return chart

    def paragraph_level_summ(self, arc_scores, edu_ids, chart, target_bnds):
        """
        do paragraph level decode based on sentence level result
        :type arc_scores: numpy.ndarray(shape=(n_edus, n_edus), dtype="float")
        :type edu_ids: list of int
        :type chart: dict of [(int, int, int, int): float]
        :type target_bnds: list of (int, int)
        :rtype: dict of [(int, int, int, int): float]
        """
        assert edu_ids[0] != 0  # NOTE: Without the Root
        assert chart is not None

        length = len(edu_ids)

        # General case
        for d in range(1, length):
            # for i1 in range(0, length - d):  # NOTE: index "0" does NOT represent ROOT
            for i1 in range(edu_ids[0], edu_ids[-1]-d+1):
                i3 = i1 + d

                same_span = False
                for begin_i, end_i in target_bnds:
                    if begin_i+1 <=i1 <= i3 <= end_i+1:
                        same_span = True
                        break
                if same_span:
                    continue

                # Incomplete span
                # Left tree
                scores = []
                arc_score = arc_scores[i3, i1]
                for i2 in range(i1, i3):
                    if (i1, i2, RIGHT, COMPLETE) in chart and \
                            (i2 + 1, i3, LEFT, COMPLETE) in chart:
                        score = arc_score \
                                + chart[i1, i2, RIGHT, COMPLETE] \
                                + chart[i2 + 1, i3, LEFT, COMPLETE]
                        scores.append(score)
                if len(scores) != 0:
                    chart[i1, i3, LEFT, INCOMPLETE] = self.logsumexp_scala(*scores)
                # Right tree
                scores = []
                # arc_score = arc_scores[edu_ids[i1], edu_ids[i3]]
                arc_score = arc_scores[i1, i3]
                for i2 in range(i1, i3):
                    if (i1, i2, RIGHT, COMPLETE) in chart and \
                        (i2 + 1, i3, LEFT, COMPLETE) in chart:
                        score = arc_score \
                                + chart[i1, i2, RIGHT, COMPLETE] \
                                + chart[i2 + 1, i3, LEFT, COMPLETE]
                        scores.append(score)
                if len(scores) != 0:
                    chart[i1, i3, RIGHT, INCOMPLETE] = self.logsumexp_scala(*scores)

                # Complete span
                # Left tree
                scores = []
                for i2 in range(i1, i3):
                    if (i1, i2, LEFT, COMPLETE) in chart and \
                        (i2, i3, LEFT, INCOMPLETE) in chart:
                        score = chart[i1, i2, LEFT, COMPLETE] \
                                + chart[i2, i3, LEFT, INCOMPLETE]
                        scores.append(score)
                if len(scores) != 0:
                    chart[i1, i3, LEFT, COMPLETE] = self.logsumexp_scala(*scores)
                # Right tree
                scores = []
                for i2 in range(i1, i3):
                    if (i1, i2 + 1, RIGHT, INCOMPLETE) in chart \
                            and (i2 + 1, i3, RIGHT, COMPLETE) in chart:
                        score = chart[i1, i2 + 1, RIGHT, INCOMPLETE] + chart[i2 + 1, i3, RIGHT, COMPLETE]
                        scores.append(score)
                if len(scores) != 0:
                    chart[i1, i3, RIGHT, COMPLETE] = self.logsumexp_scala(*scores)

        return chart

    @staticmethod
    def logsumexp_scala(*inputs) -> torch.Tensor:
        if len(inputs) == 1:
            return inputs[0]
        return torch.logsumexp(torch.stack(inputs), dim=0)

    # @staticmethod
    # def logsumexp_scala(*inputs):
    #     from scipy.special import logsumexp
    #     if len(inputs) == 1:
    #         return inputs[0]
    #     return logsumexp(inputs, axis=0)


if __name__ == '__main__':

    eisner = IncrementalEisnerDecoder()
    # for k in range(50):
    #     np.random.seed(k)
    #     input_mat_score = np.random.random([10, 10])
    #     pbnds = [(0, 1), (2, 8)]
    #     sbnds = [(0, 1), (2, 4), (5, 8)]
    #     edu_ids = list(range(10))
    #     t = eisner.global_decode(input_mat_score, edu_ids, sbnds, pbnds, True, True, None)
    #     t1 = eisner.decode(input_mat_score, edu_ids, sbnds, pbnds, True, True, None)
    #     score = 0.0
    #     for id1, id2 in t:
    #         score += input_mat_score[id1, id2]
    #     score1 = 0.0
    #     for id1, id2 in t1:
    #         score1 += input_mat_score[id1, id2]
    #     if (score - score1) < -1e-5:
    #         print("Bug here")
    #         print(t)
    #         print(score)
    #         print(t1)
    #         print(score1)
    #         print(k)

    np.random.seed(11)
    input_mat_score = np.random.random([10, 10])
    sbnds = [(0, 1), (2, 4), (5, 8)]
    pbnds = [(0, 0), (1, 2)]
    edu_ids = list(range(10))
    t = eisner.global_decode(input_mat_score, edu_ids, sbnds, pbnds, True, True, None)
    # s = eisner.partition(input_mat_score, edu_ids, sbnds, pbnds, True, True)
    print(t)
    # print(s)
