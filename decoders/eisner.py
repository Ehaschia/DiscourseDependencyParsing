import numpy as np

RIGHT = 0
LEFT = 1
COMPLETE = 0
INCOMPLETE = 1

class IncrementalEisnerDecoder(object):

    def __init__(self):
        self.decoder = EisnerDecoder()

    def decode(self,
            arc_scores,
            edu_ids,
            sbnds,
            pbnds,
            use_sbnds,
            use_pbnds,
            gold_heads=None):
        """
        :type arc_scores: numpy.ndarray(shape=(n_edus, n_edus), dtype="float")
        :type edu_ids: list of int
        :type sbnds: list of (int, int)
        :type pbnds: list of (int, int)
        :type use_sbnds: bool
        :type use_pbnds: bool
        :type gold_heads: numpy.ndarray(shape=(n_edus, n_edus), dtype=np.int32)
        :rtype: list of (int, int)
        """
        assert edu_ids[0] == 0 # NOTE

        arcs = []

        # Exclude ROOT
        new_edu_ids = edu_ids[1:]

        # Sentence-level parsing
        if use_sbnds:
            target_bnds = sbnds
            sub_arcs, new_edu_ids = self.apply_decoder(
                                                arc_scores=arc_scores,
                                                edu_ids=new_edu_ids,
                                                target_bnds=target_bnds,
                                                gold_heads=gold_heads)
            arcs.extend(sub_arcs)

        # Paragraph-level parsing
        if use_pbnds:
            if use_sbnds:
                target_bnds = pbnds
            else:
                target_bnds = [(sbnds[b][0],sbnds[e][1]) for b,e in pbnds]
            sub_arcs, new_edu_ids = self.apply_decoder(
                                                arc_scores=arc_scores,
                                                edu_ids=new_edu_ids,
                                                target_bnds=target_bnds,
                                                gold_heads=gold_heads)
            arcs.extend(sub_arcs)

        # Document-level parsing
        sub_arcs, head = self.decoder.decode_without_root(
                                                arc_scores=arc_scores,
                                                edu_ids=new_edu_ids,
                                                gold_heads=gold_heads)
        arcs.extend(sub_arcs)

        # Root attachment
        arcs.append((0, head))

        return arcs

    # tmp
    def partition(self,
                  arc_scores,
                  edu_ids,
                  sbnds):
        return self.decoder.sum_without_root(arc_scores=arc_scores, edu_ids=edu_ids[1:], target_bnds=sbnds)

    def apply_decoder(self,
                    arc_scores,
                    edu_ids,
                    target_bnds,
                    gold_heads):
        """
        :type arc_scores: numpy.ndarray(shape=(n_edus, n_edus), dtype="float")
        :type edu_ids: list of int
        :type target_bnds: list of (int, int)
        :type gold_heads: numpy.ndarray(shape=(n_edus, n_edus), dtype=np.int32)
        :rtype: list of (int, int), list of int
        """
        arcs = [] # list of (int, int)
        new_edu_ids = [] # list of int

        for begin_i, end_i in target_bnds:
            if begin_i == end_i:
                sub_arcs = []
                head = edu_ids[begin_i]
            else:
                sub_arcs, head = self.decoder.decode_without_root(
                                                    arc_scores=arc_scores,
                                                    edu_ids=edu_ids[begin_i:end_i+1],
                                                    gold_heads=gold_heads)
            arcs.extend(sub_arcs)
            new_edu_ids.append(head)
        return arcs, new_edu_ids

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
                    score = chart[i1, i2+1, RIGHT, INCOMPLETE] \
                            + chart[i2+1, i3, RIGHT, COMPLETE]
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
                    + chart[i2, length-1, RIGHT, COMPLETE]
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
        return arcs

    # calculate the score of all candidate tree
    def summ(self,
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
                sum_score = 0.0
                arc_score = arc_scores[edu_ids[i3], edu_ids[i1]]
                for i2 in range(i1, i3):
                    score = arc_score \
                            + chart[i1, i2, RIGHT, COMPLETE] \
                            + chart[i2+1, i3, LEFT, COMPLETE]
                    sum_score += score
                chart[i1, i3, LEFT, INCOMPLETE] = sum_score
                # Right tree
                sum_score = 0.0
                arc_score = arc_scores[edu_ids[i1], edu_ids[i3]]
                for i2 in range(i1, i3):
                    score = arc_score \
                            + chart[i1, i2, RIGHT, COMPLETE] \
                            + chart[i2+1, i3, LEFT, COMPLETE]
                    sum_score += score
                chart[i1, i3, RIGHT, INCOMPLETE] = sum_score

                # Complete span
                # Left tree
                sum_score = 0.0
                for i2 in range(i1, i3):
                    score = chart[i1, i2, LEFT, COMPLETE] \
                            + chart[i2, i3, LEFT, INCOMPLETE]
                    sum_score += score
                chart[i1, i3, LEFT, COMPLETE] = sum_score
                # Right tree
                sum_score = 0.0
                for i2 in range(i1, i3):
                    score = chart[i1, i2+1, RIGHT, INCOMPLETE] \
                            + chart[i2+1, i3, RIGHT, COMPLETE]
                    sum_score += score
                chart[i1, i3, RIGHT, COMPLETE] = sum_score

        # ROOT attachment
        # arcs = self.recover_tree(back_ptr, 0, length-1, RIGHT, COMPLETE, arcs=None) # NOTE
        sum_score = 0.0
        for i2 in range(1, length):
            arc_score = arc_scores[edu_ids[0], edu_ids[i2]]
            score = arc_score \
                    + chart[0, 0, RIGHT, COMPLETE] \
                    + chart[1, i2, LEFT, COMPLETE] \
                    + chart[i2, length-1, RIGHT, COMPLETE]
            sum_score += score
        chart[0, length-1, RIGHT, COMPLETE] = sum_score
        return sum_score

    def decode_without_root(self,
                            arc_scores,
                            edu_ids,
                            gold_heads=None):
        """
        :type arc_scores: numpy.ndarray(shape=(n_edus, n_edus), dtype="float")
        :type edu_ids: list of int
        :type gold_heads: numpy.ndarray(shape=(n_edus, n_edus), dtype=np.int32)
        :rtype: list of (int, int)
        """
        assert edu_ids[0] != 0 # NOTE: Without the Root

        if len(edu_ids) == 1:
            return [], edu_ids[0]

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

        # General case
        for d in range(1, length):
            for i1 in range(0, length - d): # NOTE: index "0" does NOT represent ROOT
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
                    score = chart[i1, i2+1, RIGHT, INCOMPLETE] \
                            + chart[i2+1, i3, RIGHT, COMPLETE]
                    if max_score < score:
                        max_score = score
                        memo = i2 + 1
                chart[i1, i3, RIGHT, COMPLETE] = max_score
                back_ptr[i1, i3, RIGHT, COMPLETE] = memo

        # ROOT identification
        max_score = -np.inf
        memo = None
        for i2 in range(0, length):
            score = chart[0, i2, LEFT, COMPLETE] \
                    + chart[i2, length-1, RIGHT, COMPLETE]
            if max_score < score:
                max_score = score
                memo = i2
        head = memo

        # Recovering dependency arcs
        arcs = self.recover_tree(back_ptr, 0, head, LEFT, COMPLETE, arcs=None)
        arcs = self.recover_tree(back_ptr, head, length-1, RIGHT, COMPLETE, arcs=arcs)

        # Shifting: local position -> global position
        arcs = [(edu_ids[h], edu_ids[d]) for h,d in arcs]
        head = edu_ids[head]
        return arcs, head

    def sum_without_root(self,
                         arc_scores,
                         edu_ids,
                         target_bnds):
        """
        :type arc_scores: numpy.ndarray(shape=(n_edus, n_edus), dtype="float")
        :type edu_ids: list of int
        :type target_bnds: list of (int, int)
        :rtype: tuple (List[float], List[head])
        """
        # Initialize charts
        chart = {}  # {(int, int, int, int): float}
        back_ptr = {} # {(int, int, int, int): float}

        length = len(edu_ids)
        target_bnds = [(edu_ids[left], edu_ids[right]) for left, right in target_bnds]
        # Base case
        for i in range(1, length+1):
            chart[i, i, LEFT, COMPLETE] = 0.0
            chart[i, i, RIGHT, COMPLETE] = 0.0
            chart[i, i, LEFT, INCOMPLETE] = 0.0
            chart[i, i, RIGHT, INCOMPLETE] = 0.0
        # same block
        for left, right in target_bnds:
            right += 1
            span = right - left
            for d in range(1, span):
                for i1 in range(left, right - d):  # NOTE: index "0" does NOT represent ROOT
                    i3 = i1 + d

                    # Incomplete span
                    # Left tree
                    sum_score = 0.0
                    memo = None
                    arc_score = arc_scores[i3, i1]

                    for i2 in range(i1, i3):
                        score = arc_score \
                                + chart[i1, i2, RIGHT, COMPLETE] \
                                + chart[i2 + 1, i3, LEFT, COMPLETE]
                        sum_score += score
                    chart[i1, i3, LEFT, INCOMPLETE] = sum_score
                    back_ptr[i1, i3, LEFT, INCOMPLETE] = memo
                    # Right tree
                    sum_score = 0.0
                    memo = None
                    arc_score = arc_scores[i1, i3]

                    for i2 in range(i1, i3):
                        score = arc_score \
                                + chart[i1, i2, RIGHT, COMPLETE] \
                                + chart[i2 + 1, i3, LEFT, COMPLETE]
                        sum_score += score
                    chart[i1, i3, RIGHT, INCOMPLETE] = sum_score
                    back_ptr[i1, i3, RIGHT, INCOMPLETE] = memo

                    # Complete span
                    # Left tree
                    sum_score = 0.0
                    memo = None
                    for i2 in range(i1, i3):
                        score = chart[i1, i2, LEFT, COMPLETE] \
                                + chart[i2, i3, LEFT, INCOMPLETE]
                        sum_score += score
                    chart[i1, i3, LEFT, COMPLETE] = sum_score
                    back_ptr[i1, i3, LEFT, COMPLETE] = memo
                    # Right tree
                    sum_score = 0.0
                    memo = None
                    for i2 in range(i1, i3):
                        score = chart[i1, i2 + 1, RIGHT, INCOMPLETE] \
                                + chart[i2 + 1, i3, RIGHT, COMPLETE]
                        sum_score += score
                    chart[i1, i3, RIGHT, COMPLETE] = sum_score
                    back_ptr[i1, i3, RIGHT, COMPLETE] = memo

        high_chart = {} # {(int, int, int), float}
        for i in range(length):
            high_chart[i, i, i] = 0.0
        for left, right in target_bnds:
            for k in range(left, right+1):
                high_chart[left, k, right] = chart[left, k, LEFT, COMPLETE] \
                                             + chart[k, right, RIGHT, COMPLETE]

        left_idxs = [left for left, right in target_bnds]
        right_idxs = [right for left, right in target_bnds]
        for span_length in range(1, len(target_bnds)):
            for left_idx in range(0, len(target_bnds)-span_length):
                for left_span in range(0, span_length):
                    left_left = left_idxs[left_idx]
                    left_right = right_idxs[left_idx+left_span]
                    right_right = right_idxs[left_idx+span_length]
                    for k_left in range(left_left, left_right+1):
                        for k_right in range(left_right+1, right_right+1):
                            if (left_left, k_left, right_right) not in high_chart:
                                high_chart[left_left, k_left, right_right] = 0.0
                            if (left_left, k_right, right_right) not in high_chart:
                                high_chart[left_left, k_right, right_right] = 0.0
                            high_chart[left_left, k_left, right_right] += (high_chart[left_left, k_left, left_right]
                                                                           + high_chart[left_right+1, k_right, right_right]
                                                                           + arc_scores[k_right, k_left])
                            high_chart[left_left, k_right, right_right] += (high_chart[left_left, k_left, left_right]
                                                                           + high_chart[left_right+1, k_right, right_right]
                                                                           + arc_scores[k_left, k_right])

        sum_score = 0.0
        for k in range(edu_ids[0], edu_ids[-1]+1):
            sum_score += high_chart[edu_ids[0], k, edu_ids[-1]]
        return sum_score

    def sum_temp(self, arc_scores, edu_ids):
        """
        :type arc_scores: numpy.ndarray(shape=(n_edus, n_edus), dtype="float")
        :type edu_ids: list of int
        :rtype: float
        """

        # Initialize charts
        chart = {}  # {(int, int, int, int): float}
        back_ptr = {}  # {(int, int, int, int): float}

        length = len(edu_ids)

        # Base case
        for i in range(length):
            chart[i, i, LEFT, COMPLETE] = 0.0
            chart[i, i, RIGHT, COMPLETE] = 0.0
            chart[i, i, LEFT, INCOMPLETE] = 0.0
            chart[i, i, RIGHT, INCOMPLETE] = 0.0

        # General case
        for d in range(1, length):
            for i1 in range(0, length - d):  # NOTE: index "0" does NOT represent ROOT
                i3 = i1 + d

                # Incomplete span
                # Left tree
                sum_score = 0.0
                memo = None
                arc_score = arc_scores[edu_ids[i3], edu_ids[i1]]

                for i2 in range(i1, i3):
                    score = arc_score \
                            + chart[i1, i2, RIGHT, COMPLETE] \
                            + chart[i2 + 1, i3, LEFT, COMPLETE]
                    sum_score += score
                chart[i1, i3, LEFT, INCOMPLETE] = sum_score
                back_ptr[i1, i3, LEFT, INCOMPLETE] = memo
                # Right tree
                sum_score = 0.0
                arc_score = arc_scores[edu_ids[i1], edu_ids[i3]]
                for i2 in range(i1, i3):
                    score = arc_score \
                            + chart[i1, i2, RIGHT, COMPLETE] \
                            + chart[i2 + 1, i3, LEFT, COMPLETE]
                    sum_score+= score
                chart[i1, i3, RIGHT, INCOMPLETE] = sum_score

                # Complete span
                # Left tree
                sum_score = 0.0
                for i2 in range(i1, i3):
                    score = chart[i1, i2, LEFT, COMPLETE] \
                            + chart[i2, i3, LEFT, INCOMPLETE]
                    sum_score += score
                chart[i1, i3, LEFT, COMPLETE] = sum_score
                # Right tree
                sum_score = 0.0
                for i2 in range(i1, i3):
                    score = chart[i1, i2 + 1, RIGHT, INCOMPLETE] \
                            + chart[i2 + 1, i3, RIGHT, COMPLETE]
                    sum_score += score
                chart[i1, i3, RIGHT, COMPLETE] = sum_score

        # ROOT identification
        sum_score = 0.0

        for i2 in range(0, length):
            score = chart[0, i2, LEFT, COMPLETE] \
                    + chart[i2, length - 1, RIGHT, COMPLETE]
            sum_score += score

        return sum_score

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

