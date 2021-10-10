import torch

from model import utility
from model.definition import LOGZERO
import numpy as np
EPS = 1e-12


class CRFAE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, crf_weights, joint_prior_weights, sentence_len, batch_size,
                is_multi_root, max_dependency_len, use_gpu, length_constraint_on_root):
        # sentence_len = kwargs['sentence_len']
        # batch_size = kwargs['batch_size']
        # is_multi_root = kwargs['is_multi_root']
        # max_dependency_len = kwargs['max_dependency_len']
        # use_gpu = kwargs['use_gpu']
        # length_constraint_on_root = kwargs['length_constraint_on_root']

        def dp_inside_batch(weights):
            """

            :param weights:  batch_size * seq_length * seq_length
            :return:
            """
            inside_table = torch.DoubleTensor(batch_size, sentence_len * sentence_len * 8)
            inside_table.fill_(LOGZERO)

            if use_gpu and torch.cuda.is_available():
                inside_table = inside_table.cuda()

            m = sentence_len
            (seed_spans, base_left_spans, base_right_spans, left_spans, right_spans,
             ijss, ikss, kjss, id_span_map, span_id_map) = utility.constituent_indexes(
                m, is_multi_root, max_dependency_len, length_constraint_on_root
            )

            for ii in seed_spans:
                inside_table[:, ii] = 0.0

            for ii in base_left_spans:
                (l, r, c) = id_span_map[ii]
                inside_table[:, ii] = weights[:, l, r]

            for ii in base_right_spans:
                (l, r, c) = id_span_map[ii]
                inside_table[:, ii] = weights[:, r, l]

            for ij in ijss:
                (l, r, c) = id_span_map[ij]
                if ij in left_spans:
                    ids = span_id_map.get((l, r, utility.get_state_code(0, 0, 0)), -1)
                    # print(inside_table[:, ids])
                    # print(weights[:, l, r])
                    prob = inside_table[:, ids] + weights[:, l, r]
                    inside_table[:, ij] = utility.logaddexp(inside_table[:, ij], prob)
                elif ij in right_spans:
                    ids = span_id_map.get((l, r, utility.get_state_code(0, 0, 0)), -1)
                    prob = inside_table[:, ids] + weights[:, r, l]
                    inside_table[:, ij] = utility.logaddexp(inside_table[:, ij], prob)
                else:
                    beta_ik, beta_kj = inside_table[:, ikss[ij]], inside_table[:, kjss[ij]]
                    probs = (beta_ik + beta_kj)  # B * k
                    probs = utility.logsumexp(probs, axis=1)
                    inside_table[:, ij] = utility.logaddexp(inside_table[:, ij], probs)

            id1 = span_id_map.get((0, m - 1, utility.get_state_code(0, 1, 0)), -1)
            id2 = span_id_map.get((0, m - 1, utility.get_state_code(0, 1, 1)), -1)

            score1 = inside_table[:, id1]  # B * 1
            score2 = inside_table[:, id2]  # B * 1
            ll = utility.logaddexp(score1, score2)
            return inside_table, ll.contiguous().view(batch_size, 1, 1)

        def decoding(weights):
            best_score, tree = utility.decoding_batch(weights.cpu().detach().numpy(), is_multi_root,
                                                      max_dependency_len, length_constraint_on_root)

            best_score = torch.DoubleTensor(best_score)
            tree = torch.LongTensor(tree)

            if use_gpu:
                best_score = best_score.cuda()
                tree = tree.cuda()

            return best_score, tree

        inside_table, log_partition = dp_inside_batch(crf_weights)
        best_score, best_tree = decoding(joint_prior_weights)
        configs = torch.tensor([sentence_len, batch_size, int(is_multi_root), max_dependency_len, int(use_gpu),
                      length_constraint_on_root])
        ctx.save_for_backward(configs, inside_table, crf_weights, log_partition, best_tree)
        return log_partition, best_score, best_tree

    # def diff(self):
    #     outside_table = self.dp_outside_batch(self.inside_table, self.crf_weights)
    #
    #     (seed_spans, base_left_spans, base_right_spans, left_spans, right_spans,
    #      ijss, ikss, kjss, id_span_map, span_id_map) = utility.constituent_indexes(self.sentence_len,
    #                                                                                self.is_multi_root,
    #                                                                                self.max_dependency_len,
    #                                                                                self.length_constraint_on_root)
    #
    #     counts = self.inside_table + outside_table  # shape is batch_size * num_span
    #     part_count = torch.DoubleTensor(self.batch_size, self.sentence_len, self.sentence_len)
    #     part_count.fill_(LOGZERO)
    #
    #     for left_index in range(self.sentence_len):
    #         for right_index in range(left_index + 1, self.sentence_len):
    #             span_id = span_id_map.get((left_index, right_index, utility.get_state_code(0, 1, 1)))
    #             if span_id is not None:
    #                 part_count[:, left_index, right_index] = counts[:, span_id]
    #
    #             span_id = span_id_map.get((left_index, right_index, utility.get_state_code(1, 0, 1)))
    #             if span_id is not None:
    #                 part_count[:, right_index, left_index] = counts[:, span_id]
    #
    #     if self.use_gpu and torch.cuda.is_available():
    #         part_count = part_count.cuda()
    #
    #     alpha = part_count - self.log_partition
    #     diff = torch.exp(alpha)
    #
    #     return diff
    @staticmethod
    def backward(ctx, *grad_output):
        configs, inside_table, crf_weights, log_partition, best_tree = ctx.saved_tensors

        sentence_len, batch_size, is_multi_root, max_dependency_len, use_gpu, \
        length_constraint_on_root = list(configs.cpu().detach().numpy().astype(np.int32))

        def dp_outside_batch(inside_table, weights):
            outside_table = torch.DoubleTensor(batch_size, sentence_len * sentence_len * 8)
            outside_table.fill_(LOGZERO)

            if use_gpu and torch.cuda.is_available():
                outside_table = outside_table.cuda()

            m = sentence_len

            (seed_spans, base_left_spans, base_right_spans, left_spans, right_spans,
             ijss, ikss, kjss, id_span_map, span_id_map) = utility.constituent_indexes(m, is_multi_root,
                                                                                       max_dependency_len,
                                                                                       length_constraint_on_root)

            id1 = span_id_map.get((0, m - 1, utility.get_state_code(0, 1, 0)), -1)
            id2 = span_id_map.get((0, m - 1, utility.get_state_code(0, 1, 1)), -1)
            outside_table[:, id1] = 0.0
            outside_table[:, id2] = 0.0

            for ij in reversed(ijss):
                (l, r, c) = id_span_map[ij]
                if ij in left_spans:
                    prob = outside_table[:, ij] + weights[:, l, r]
                    ids = span_id_map.get((l, r, utility.get_state_code(0, 0, 0)), -1)
                    outside_table[:, ids] = utility.logaddexp(outside_table[:, ids], prob)
                elif ij in right_spans:
                    prob = outside_table[:, ij] + weights[:, r, l]
                    ids = span_id_map.get((l, r, utility.get_state_code(0, 0, 0)), -1)
                    outside_table[:, ids] = utility.logaddexp(outside_table[:, ids], prob)
                else:
                    K = len(ikss[ij])
                    alpha_ij = outside_table[:, ij].contiguous().view(batch_size, 1)  # N * 1
                    beta_left = inside_table[:, ikss[ij]]  # N * K

                    new_right = alpha_ij + beta_left  # N * K
                    outside_table[:, kjss[ij]] = utility.logaddexp(outside_table[:, kjss[ij]], new_right)

                    # Problem: The span_id in ikss[ij] may be duplicated.
                    for i in range(K):
                        ik = ikss[ij][i]
                        kj = kjss[ij][i]
                        alpha_ij = outside_table[:, ij]
                        beta_right = inside_table[:, kj]
                        new_left = alpha_ij + beta_right
                        outside_table[:, ik] = utility.logaddexp((outside_table[:, ik]), new_left)

            for ij in base_left_spans:
                (l, r, c) = id_span_map[ij]
                prob = outside_table[:, ij] + weights[:, l, r]
                ids = span_id_map.get((l, r, utility.get_state_code(0, 0, 1)), -1)
                outside_table[:, ids] = utility.logaddexp(outside_table[:, ids], prob)

            for ij in base_right_spans:
                (l, r, c) = id_span_map[ij]
                prob = outside_table[:, ij] + weights[:, r, l]
                ids = span_id_map.get((l, r, utility.get_state_code(0, 0, 1)), -1)
                outside_table[:, ids] = utility.logaddexp(outside_table[:, ids], prob)

            return outside_table

        def diff():
            outside_table = dp_outside_batch(inside_table, crf_weights)

            (seed_spans, base_left_spans, base_right_spans, left_spans, right_spans,
             ijss, ikss, kjss, id_span_map, span_id_map) = utility.constituent_indexes(sentence_len,
                                                                                       is_multi_root,
                                                                                       max_dependency_len,
                                                                                       length_constraint_on_root)

            counts = inside_table + outside_table  # shape is batch_size * num_span
            part_count = torch.DoubleTensor(batch_size, sentence_len, sentence_len)
            part_count.fill_(LOGZERO)

            for left_index in range(sentence_len):
                for right_index in range(left_index + 1, sentence_len):
                    span_id = span_id_map.get((left_index, right_index, utility.get_state_code(0, 1, 1)))
                    if span_id is not None:
                        part_count[:, left_index, right_index] = counts[:, span_id]

                    span_id = span_id_map.get((left_index, right_index, utility.get_state_code(1, 0, 1)))
                    if span_id is not None:
                        part_count[:, right_index, left_index] = counts[:, span_id]

            if use_gpu and torch.cuda.is_available():
                part_count = part_count.cuda()

            alpha = part_count - log_partition
            diff = torch.exp(alpha)

            return diff

        expected_count = diff()
        grad_partition, grad_score, _ = grad_output

        grad_partition = grad_partition.contiguous().view(batch_size, 1, 1)

        gd_partition_over_w = expected_count * grad_partition
        gd_partition_over_w[:, :, 0].fill_(0.0)
        for i in range(sentence_len):
            gd_partition_over_w[:, i, i].fill_(0.0)

        grad_score = grad_score.contiguous().view(batch_size, 1, 1)
        gd_score_over_joint_w = torch.zeros(batch_size, sentence_len, sentence_len)
        batch_index = torch.arange(batch_size).contiguous().view(-1, 1).long()  # B * 1
        head_index = best_tree[:, 1:]  # B * (N - 1)
        child_index = torch.arange(1, sentence_len).contiguous().view(1, -1).long()  # 1 * (N - 1)

        if use_gpu:
            gd_score_over_joint_w = gd_score_over_joint_w.cuda()
            batch_index = batch_index.cuda()
            head_index = head_index.cuda()
            child_index = child_index.cuda()

        gd_score_over_joint_w[batch_index.detach(), head_index.detach(), child_index.detach()] = 1.

        gd_score_over_joint_w = gd_score_over_joint_w.double() * grad_score

        return gd_partition_over_w, gd_score_over_joint_w, None, None, None, None, None, None

    # def dp_inside_batch(self, weights):
    #     """
    #
    #     :param weights:  batch_size * seq_length * seq_length
    #     :return:
    #     """
    #     inside_table = torch.DoubleTensor(self.batch_size, self.sentence_len * self.sentence_len * 8)
    #     inside_table.fill_(LOGZERO)
    #
    #     if self.use_gpu and torch.cuda.is_available():
    #         inside_table = inside_table.cuda()
    #
    #     m = self.sentence_len
    #     (seed_spans, base_left_spans, base_right_spans, left_spans, right_spans,
    #      ijss, ikss, kjss, id_span_map, span_id_map) = utility.constituent_indexes(
    #         m, self.is_multi_root, self.max_dependency_len, self.length_constraint_on_root
    #     )
    #
    #     for ii in seed_spans:
    #         inside_table[:, ii] = 0.0
    #
    #     for ii in base_left_spans:
    #         (l, r, c) = id_span_map[ii]
    #         inside_table[:, ii] = weights[:, l, r]
    #
    #     for ii in base_right_spans:
    #         (l, r, c) = id_span_map[ii]
    #         inside_table[:, ii] = weights[:, r, l]
    #
    #     for ij in ijss:
    #         (l, r, c) = id_span_map[ij]
    #         if ij in left_spans:
    #             ids = span_id_map.get((l, r, utility.get_state_code(0, 0, 0)), -1)
    #             # print(inside_table[:, ids])
    #             # print(weights[:, l, r])
    #             prob = inside_table[:, ids] + weights[:, l, r]
    #             inside_table[:, ij] = utility.logaddexp(inside_table[:, ij], prob)
    #         elif ij in right_spans:
    #             ids = span_id_map.get((l, r, utility.get_state_code(0, 0, 0)), -1)
    #             prob = inside_table[:, ids] + weights[:, r, l]
    #             inside_table[:, ij] = utility.logaddexp(inside_table[:, ij], prob)
    #         else:
    #             beta_ik, beta_kj = inside_table[:, ikss[ij]], inside_table[:, kjss[ij]]
    #             probs = (beta_ik + beta_kj)  # B * k
    #             probs = utility.logsumexp(probs, axis=1)
    #             inside_table[:, ij] = utility.logaddexp(inside_table[:, ij], probs)
    #
    #     id1 = span_id_map.get((0, m - 1, utility.get_state_code(0, 1, 0)), -1)
    #     id2 = span_id_map.get((0, m - 1, utility.get_state_code(0, 1, 1)), -1)
    #
    #     score1 = inside_table[:, id1]  # B * 1
    #     score2 = inside_table[:, id2]  # B * 1
    #     ll = utility.logaddexp(score1, score2)
    #     return inside_table, ll.contiguous().view(self.batch_size, 1, 1)
    #
    # def dp_outside_batch(self, inside_table, weights):
    #     outside_table = torch.DoubleTensor(self.batch_size, self.sentence_len * self.sentence_len * 8)
    #     outside_table.fill_(LOGZERO)
    #
    #     if self.use_gpu and torch.cuda.is_available():
    #         outside_table = outside_table.cuda()
    #
    #     m = self.sentence_len
    #
    #     (seed_spans, base_left_spans, base_right_spans, left_spans, right_spans,
    #      ijss, ikss, kjss, id_span_map, span_id_map) = utility.constituent_indexes(m, self.is_multi_root,
    #                                                                                self.max_dependency_len,
    #                                                                                self.length_constraint_on_root)
    #
    #     id1 = span_id_map.get((0, m - 1, utility.get_state_code(0, 1, 0)), -1)
    #     id2 = span_id_map.get((0, m - 1, utility.get_state_code(0, 1, 1)), -1)
    #     outside_table[:, id1] = 0.0
    #     outside_table[:, id2] = 0.0
    #
    #     for ij in reversed(ijss):
    #         (l, r, c) = id_span_map[ij]
    #         if ij in left_spans:
    #             prob = outside_table[:, ij] + weights[:, l, r]
    #             ids = span_id_map.get((l, r, utility.get_state_code(0, 0, 0)), -1)
    #             outside_table[:, ids] = utility.logaddexp(outside_table[:, ids], prob)
    #         elif ij in right_spans:
    #             prob = outside_table[:, ij] + weights[:, r, l]
    #             ids = span_id_map.get((l, r, utility.get_state_code(0, 0, 0)), -1)
    #             outside_table[:, ids] = utility.logaddexp(outside_table[:, ids], prob)
    #         else:
    #             K = len(ikss[ij])
    #             alpha_ij = outside_table[:, ij].contiguous().view(self.batch_size, 1)  # N * 1
    #             beta_left = inside_table[:, ikss[ij]]  # N * K
    #
    #             new_right = alpha_ij + beta_left  # N * K
    #             outside_table[:, kjss[ij]] = utility.logaddexp(outside_table[:, kjss[ij]], new_right)
    #
    #             # Problem: The span_id in ikss[ij] may be duplicated.
    #             for i in range(K):
    #                 ik = ikss[ij][i]
    #                 kj = kjss[ij][i]
    #                 alpha_ij = outside_table[:, ij]
    #                 beta_right = inside_table[:, kj]
    #                 new_left = alpha_ij + beta_right
    #                 outside_table[:, ik] = utility.logaddexp((outside_table[:, ik]), new_left)
    #
    #     for ij in base_left_spans:
    #         (l, r, c) = id_span_map[ij]
    #         prob = outside_table[:, ij] + weights[:, l, r]
    #         ids = span_id_map.get((l, r, utility.get_state_code(0, 0, 1)), -1)
    #         outside_table[:, ids] = utility.logaddexp(outside_table[:, ids], prob)
    #
    #     for ij in base_right_spans:
    #         (l, r, c) = id_span_map[ij]
    #         prob = outside_table[:, ij] + weights[:, r, l]
    #         ids = span_id_map.get((l, r, utility.get_state_code(0, 0, 1)), -1)
    #         outside_table[:, ids] = utility.logaddexp(outside_table[:, ids], prob)
    #
    #     return outside_table
    #
    # def decoding(self, weights):
    #     best_score, tree = utility.decoding_batch(weights.cpu().numpy(), self.is_multi_root,
    #                                               self.max_dependency_len, self.length_constraint_on_root)
    #
    #     best_score = torch.DoubleTensor(best_score)
    #     tree = torch.LongTensor(tree)
    #
    #     if self.use_gpu:
    #         best_score = best_score.cuda()
    #         tree = tree.cuda()
    #
    #     return best_score, tree
