import numpy as np
from chainer import cuda

RIGHT = 0
LEFT = 1
COMPLETE = 0
INCOMPLETE = 1

class IncrementalEisnerDecoder(object):

    def __init__(self):
        self.decoder = EisnerDecoder()

    def decode(self,
            model,
            edu_ids,
            edu_vectors,
            same_sent_map,
            sbnds,
            pbnds,
            use_sbnds,
            use_pbnds,
            gold_heads=None):
        """
        :type model: Model
        :type edu_ids: list of int
        :type edu_vectors: Variable(shape=(n_edus, bilstm_dim + tempfeat1_dim), dtype=np.float32)
        :type same_sent_map: numpy.ndarray(shape=(n_edus, n_edus), dtype=np.int32)
        :type sbnds: list of (int, int)
        :type pbnds: list of (int, int)
        :type use_sbnds: bool
        :type use_pbnds: bool
        :type gold_heads: numpy.ndarray(shape=(n_edus,n_edus), dtype=np.int32)
        :rtype: list of (int, int)
        """
        assert edu_ids[0] == 0 # NOTE
        assert edu_vectors.shape[0] == len(edu_ids) # NOTE

        arcs = []

        # Exclude ROOT
        new_edu_ids = edu_ids[1:]

        # Sentence-level parsing
        if use_sbnds:
            target_bnds = sbnds
            sub_arcs, new_edu_ids = self.apply_decoder(
                                                target_bnds=target_bnds,
                                                model=model,
                                                edu_ids=new_edu_ids,
                                                edu_vectors=edu_vectors,
                                                same_sent_map=same_sent_map,
                                                gold_heads=gold_heads)
            arcs.extend(sub_arcs)

        # Paragraph-level parsing
        if use_pbnds:
            if use_sbnds:
                target_bnds = pbnds
            else:
                target_bnds = [(sbnds[b][0],sbnds[e][1]) for b,e in pbnds]
            sub_arcs, new_edu_ids = self.apply_decoder(
                                                target_bnds=target_bnds,
                                                model=model,
                                                edu_ids=new_edu_ids,
                                                edu_vectors=edu_vectors,
                                                same_sent_map=same_sent_map,
                                                gold_heads=gold_heads)
            arcs.extend(sub_arcs)

        # Document-level parsing
        sub_arcs, head = self.decoder.decode_without_root(
                                                model=model,
                                                edu_ids=new_edu_ids,
                                                edu_vectors=edu_vectors,
                                                same_sent_map=same_sent_map,
                                                gold_heads=gold_heads)
        arcs.extend(sub_arcs)

        # Root attachment
        arcs.append((0, head))

        return arcs

    def apply_decoder(self,
                    target_bnds,
                    model,
                    edu_ids,
                    edu_vectors,
                    same_sent_map,
                    gold_heads):
        """
        :type target_bnds: list of (int, int)
        :type model: Model
        :type edu_ids: list of int (length=n_edus)
        :type edu_vectors: Variable(shape=(n_edus*, bilstm_dim + tempfeat1_dim), dtype=np.float32)
        :type same_sent_map: numpy.ndarray(shape=(n_edus*, n_edus*), dtype=np.int32)
        :type gold_heads: numpy.ndarray(shape=(n_edus*, n_edus*), dtype=np.int32)
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
                                                    model=model,
                                                    edu_ids=edu_ids[begin_i:end_i+1],
                                                    edu_vectors=edu_vectors,
                                                    same_sent_map=same_sent_map,
                                                    gold_heads=gold_heads)
            arcs.extend(sub_arcs)
            new_edu_ids.append(head)
        return arcs, new_edu_ids

class EisnerDecoder(object):

    def __init__(self):
        pass

    def decode(self,
            model,
            edu_ids,
            edu_vectors,
            same_sent_map,
            gold_heads=None):
        """
        :type model: Model
        :type edu_ids: list of int (length=n_edus)
        :type edu_vectors: Variable(shape=(n_edus, bilstm_dim + tempfeat1_dim), dtype=np.float32)
        :type same_sent_map: numpy.ndarray(shape=(n_edus, n_edus), dtype=np.int32)
        :type gold_heads: numpy.ndarray(shape=(n_edus, n_edus), dtype=np.int32)
        :rtype: list of (int, int)
        """
        assert edu_ids[0] == 0 # NOTE
        assert edu_vectors.shape[0] == len(edu_ids) # NOTE

        # Precompute arc scores to avoid redundant calculation
        arc_scores = self.precompute_arc_scores(model=model,
                                                edu_ids=edu_ids,
                                                edu_vectors=edu_vectors,
                                                same_sent_map=same_sent_map)

        # Initialize charts
        chart = {} # {(int, int, int, int): float}
        back_ptr = {} # {(int, int, int, int): float}

        n_edus = len(edu_ids)

        # Base case
        for i in range(n_edus):
            chart[i, i, LEFT, COMPLETE] = 0.0
            chart[i, i, RIGHT, COMPLETE] = 0.0
            chart[i, i, LEFT, INCOMPLETE] = 0.0
            chart[i, i, RIGHT, INCOMPLETE] = 0.0
        for i in range(n_edus):
            chart[0, i, LEFT, INCOMPLETE] = -np.inf

        # General case (without ROOT)
        for d in range(1, n_edus):
            for i1 in range(1, n_edus - d): # NOTE
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
        # arcs = self.recover_tree(back_ptr, 0, n_edus-1, RIGHT, COMPLETE, arcs=None) # NOTE
        max_score = -np.inf
        memo = None
        for i2 in range(1, n_edus):
            arc_score = arc_scores[edu_ids[0], edu_ids[i2]]
            score = arc_score \
                    + chart[0, 0, RIGHT, COMPLETE] \
                    + chart[1, i2, LEFT, COMPLETE] \
                    + chart[i2, n_edus-1, RIGHT, COMPLETE]
            if max_score < score:
                max_score = score
                memo = i2
        chart[0, n_edus-1, RIGHT, COMPLETE] = max_score
        back_ptr[0, n_edus-1, RIGHT, COMPLETE] = memo
        head = memo

        # Recovering dependency arcs
        arcs = [(0, head)]
        arcs = self.recover_tree(back_ptr, 1, head, LEFT, COMPLETE, arcs=arcs)
        arcs = self.recover_tree(back_ptr, head, n_edus-1, RIGHT, COMPLETE, arcs=arcs)

        # Shifting: local position -> global position
        arcs = [(edu_ids[h], edu_ids[d]) for h,d in arcs]
        return arcs

    def decode_without_root(self,
                            model,
                            edu_ids,
                            edu_vectors,
                            same_sent_map,
                            gold_heads=None):
        """
        :type model: Model
        :type edu_ids: list of int (length=n_edus)
        :type edu_vectors: Variable(shape=(n_edus*, bilstm_dim + tempfeat1_dim), dtype=np.float32)
        :type same_sent_map: numpy.ndarray(shape=(n_edus*, n_edus*), dtype=np.int32)
        :type gold_heads: numpy.ndarray(shape=(n_edus*, n_edus*), dtype=np.int32)
        :rtype: list of (int, int)
        """
        assert edu_ids[0] != 0 # NOTE

        if len(edu_ids) == 1:
            return [], edu_ids[0]

        # Precompute arc scores to avoid redundant calculation
        arc_scores = self.precompute_arc_scores(model=model,
                                                edu_ids=edu_ids,
                                                edu_vectors=edu_vectors,
                                                same_sent_map=same_sent_map)

        # Initialize charts
        chart = {} # {(int, int, int, int): float}
        back_ptr = {} # {(int, int, int, int): float}

        n_edus = len(edu_ids)

        # Base case
        for i in range(n_edus):
            chart[i, i, LEFT, COMPLETE] = 0.0
            chart[i, i, RIGHT, COMPLETE] = 0.0
            chart[i, i, LEFT, INCOMPLETE] = 0.0
            chart[i, i, RIGHT, INCOMPLETE] = 0.0

        # General case
        for d in range(1, n_edus):
            for i1 in range(0, n_edus - d): # NOTE: index "0" does NOT represent ROOT
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
        for i2 in range(0, n_edus):
            score = chart[0, i2, LEFT, COMPLETE] \
                    + chart[i2, n_edus-1, RIGHT, COMPLETE]
            if max_score < score:
                max_score = score
                memo = i2
        head = memo

        # Recovering dependency arcs
        arcs = self.recover_tree(back_ptr, 0, head, LEFT, COMPLETE, arcs=None)
        arcs = self.recover_tree(back_ptr, head, n_edus-1, RIGHT, COMPLETE, arcs=arcs)

        # Shifting: local position -> global position
        arcs = [(edu_ids[h], edu_ids[d]) for h,d in arcs]
        head = edu_ids[head]
        return arcs, head

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

    def precompute_arc_scores(self, model, edu_ids, edu_vectors, same_sent_map):
        """
        :type model: Model
        :type edu_ids: list of int (length=n_edus)
        :type edu_vectors: Variable(shape=(n_edus*, bilstm_dim + tempfeat1_dim), dtype=np.float32)
        :type same_sent_map: numpy.ndarray(shape=(n_edus*, n_edus*), dtype=np.int32)
        :rtype: {(int, int): float}
        """
        result = {} # {(int, int): float}

        n_edus = len(edu_ids)

        # Aggregating patterns
        arcs = []
        for h in range(0, n_edus):
            for d in range(0, n_edus):
                if h == d:
                    continue
                arc = (h, d)
                if arc in arcs:
                    continue
                arcs.append(arc)

        # Shifting: local position -> global position
        arcs = [(edu_ids[h], edu_ids[d]) for h,d in arcs]

        # Scoring
        arc_scores = model.forward_arcs_for_attachment(
                                edu_vectors=edu_vectors,
                                same_sent_map=same_sent_map,
                                batch_arcs=[arcs],
                                aggregate=False) # (1, n_arcs, 1)
        arc_scores = cuda.to_cpu(arc_scores.data)[0] # (n_arcs, 1)
        for arc_i, arc in enumerate(arcs):
            result[arc] = float(arc_scores[arc_i])

        return result

