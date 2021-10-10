from typing import Tuple, List

import torch
from torch import nn
import torch.distributions as D
import torch.nn.functional as F
from torch.nn import Parameter

from model import utility
from model.context_sensitive_encoder import CSEncoder
from model.dep_dataset import SentencePreprocessor, DiscoursePreprocessor
from model.eisner import IncrementalEisnerDecoder
import numpy as np

ACTIVATION_DICT = {'relu': F.relu, 'gelu': F.gelu, 'tanh': F.tanh, 'sigmoid': F.sigmoid}

class CSNCRFAE(nn.Module):
    def __init__(self, preprocessor: DiscoursePreprocessor, tagset_size, embedding_dim, hidden_dim,
                 dropout_ratio, prior_weight, recons_dim, act_func='tanh',
                 use_gpu=False, is_multi_root=False, max_dependency_len=10, length_constraint_on_root=False):
        super(CSNCRFAE, self).__init__()

        self.preprocessor = preprocessor

        self.use_gpu = use_gpu
        self.is_multi_root = is_multi_root
        self.max_dependency_len = max_dependency_len
        self.length_constraint_on_root = length_constraint_on_root

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.em_proj = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.activation = ACTIVATION_DICT[act_func]

        self.dropout = nn.Dropout(p=dropout_ratio)

        self.prior_factor = prior_weight

        self.tagset_size = tagset_size

        self.batch_size = 1  # temp
        self.seq_length = 1  # temp

        # self.multinomial = nn.Parameter(torch.ones(tagset_size, tagset_size) / tagset_size)
        # self.multinomial.requires_grad = False
        self.f2prob1 = nn.Linear(hidden_dim, recons_dim)
        self.f2prob2 = nn.Linear(hidden_dim, recons_dim)

        self.score_layer = nn.Linear(6 * hidden_dim, 1)
        self.esiner = IncrementalEisnerDecoder()
        self.best_tree = None

    def get_batch_seq_size(self, sentence):
        """
        set batch size and sequence length
        """
        batch_size, seq_length, _ = sentence.size()
        self.seq_length = seq_length
        self.batch_size = batch_size
        batch_seq_size = {'sentence_len': self.seq_length,
                          'batch_size': self.batch_size,
                          'is_multi_root': self.is_multi_root,
                          'max_dependency_len': self.max_dependency_len,
                          'use_gpu': self.use_gpu,
                          'length_constraint_on_root': self.length_constraint_on_root}

        return batch_seq_size

    def init(self):
        utility.init_linear(self.score_layer)
        utility.init_linear(self.em_proj)
        utility.init_linear(self.f2prob1)
        utility.init_linear(self.f2prob2)

    def forward(self, sentences, enable_prior=True):
        embeds = self.preprocessor.process_cs_batch(sentences)
        self.get_batch_seq_size(embeds)

        d_embeds = self.dropout(embeds)
        hidden_out = self.activation(self.em_proj(d_embeds))

        x_next = torch.zeros_like(hidden_out)
        x_next[:, 0:self.seq_length - 1, :] = hidden_out[:, 1:self.seq_length, :]
        x_next = x_next[:, :, None, :].repeat(1, 1, self.seq_length, 1)

        x_before = torch.zeros_like(hidden_out)
        x_before[:, 1:self.seq_length, :] = hidden_out[:, 0:self.seq_length - 1, :]
        x_before = x_before[:, :, None, :].repeat(1, 1, self.seq_length, 1)

        x = hidden_out[:, :, None, :].repeat(1, 1, self.seq_length, 1)  # B * N * [N] * H
        f = torch.cat([
            x,
            x_before,
            x_next,
            torch.transpose(x, 1, 2),
            torch.transpose(x_before, 1, 2),
            torch.transpose(x_next, 1, 2)
        ], dim=3)  # B * N * N * 6H

        crf_weights = self.score_layer(f).view(self.batch_size, self.seq_length, self.seq_length)

        # ALERT method 1 use prior_weight not hard prior
        prior_weight = torch.cat([s.prior_weight()[None, :, :] for s in sentences])
        prior_weight = (self.prior_factor if enable_prior else 0.) * prior_weight / self.seq_length

        prior_weight = prior_weight.cuda() if self.use_gpu else prior_weight
        prior_weight.requires_grad = True

        recons_weights = self.get_recons_weight_v2(hidden_out)

        joint_weights = crf_weights + recons_weights
        joint_prior_weights = joint_weights + prior_weight

        # log_partition, best_score, best_tree = self.crfae(crf_weights.double(), joint_prior_weights.double(),
        #                                                   self.seq_length, self.batch_size, self.is_multi_root,
        #                                                   self.max_dependency_len, self.use_gpu,
        #                                                   self.length_constraint_on_root)
        crf_weight_list = torch.split(crf_weights, 1)
        joint_prior_weight_list = torch.split(joint_prior_weights, 1)
        self.best_tree = []
        log_partitions = []
        for crf_weight, joint_prior_weight in zip(crf_weight_list, joint_prior_weight_list):
            edu_ids = [i for i in range(crf_weight.size()[-1])]
            arcs, \
            max_score, \
            log_partition = self.esiner.decoder.decode_and_sum(crf_weight.squeeze(0),
                                                               joint_prior_weight.squeeze(0),
                                                               edu_ids)
            log_partitions.append(log_partition)
            pred_tree = [node[0] for node in sorted(arcs, key=lambda x: x[1])]
            self.best_tree.append([-1] + pred_tree)
        log_partitions = torch.stack(log_partitions, dim=0)

        # TODO: current implementation is supervised training
        # Supervised HACK
        joint_prior_weights = joint_prior_weights.double()

        heads = torch.LongTensor([s.heads for s in sentences])
        heads_token_id = heads[:, 1:]
        sent_idx = torch.arange(self.batch_size).contiguous().view(-1, 1).long()
        children_token_id = torch.arange(self.seq_length).contiguous().view(1, -1).long()

        if self.use_gpu:
            sent_idx = sent_idx.cuda()
            heads_token_id = heads_token_id.cuda()
            children_token_id = children_token_id.cuda()

        best_scores = torch.sum(joint_prior_weights[sent_idx, heads_token_id, children_token_id[:, 1:]], dim=1)
        # HACK END
        # return None
        return -(best_scores - log_partitions).mean()

    def decoding(self, sentence, enable_prior=True):
        self.forward(sentence, enable_prior=enable_prior)
        return self.best_tree

    # gaussian method
    def get_recons_weight(self, input: torch.Tensor) -> torch.Tensor:

        mus = self.f2prob1(input)
        # todo not debug here
        logvars = torch.diag_embed(torch.zeros_like(mus) + 1.0)
        point = self.f2prob1(input)
        batch_list = torch.split(point, 1)
        batch_mu_list = torch.split(mus, 1)
        batch_logvar_list = torch.split(logvars, 1)
        batch_holder = []
        for idx, sent in enumerate(batch_list):
            sent_holder = []
            sent_mu = torch.split(batch_mu_list[idx].squeeze(0), 1)
            sent_logvar = torch.split(batch_logvar_list[idx].squeeze(0), 1)
            for mu, logvar in zip(sent_mu, sent_logvar):
                gaussian = D.MultivariateNormal(mu.squeeze(0), logvar.squeeze(0))
                sent_log_prob = gaussian.log_prob(sent.squeeze(0))
                sent_holder.append(sent_log_prob)
            batch_holder.append(torch.stack(sent_holder, dim=0))
        return torch.stack(batch_holder, dim=0)

    # porb method
    def get_recons_weight_v2(self, input: torch.Tensor) -> torch.Tensor:

        # similar to self-attention?
        # todo try copy self-attention like method
        p1 = self.f2prob1(input)
        p2 = self.f2prob2(input)

        score = torch.matmul(p1, p2.permute(0, 2, 1))
        prob = torch.log_softmax(score, dim=-1)
        return prob




class CSNCRFAE_v2(nn.Module):
    def __init__(self, preprocessor: DiscoursePreprocessor, tagset_size, embedding_dim, hidden_dim,
                 dropout_ratio, prior_weight, recons_dim, act_func='tanh',
                 use_gpu=False, is_multi_root=False, max_dependency_len=10, length_constraint_on_root=False):
        super(CSNCRFAE_v2, self).__init__()

        self.preprocessor = preprocessor

        self.use_gpu = use_gpu
        self.is_multi_root = is_multi_root
        self.max_dependency_len = max_dependency_len
        self.length_constraint_on_root = length_constraint_on_root

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.em2crf1 = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.em2crf2 = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.activation = ACTIVATION_DICT[act_func]

        self.dropout = nn.Dropout(p=dropout_ratio)

        self.prior_factor = prior_weight

        self.tagset_size = tagset_size

        self.batch_size = 1  # temp
        self.seq_length = 1  # temp
        self.crie = torch.nn.CrossEntropyLoss(reduction='none')

    def get_batch_seq_size(self, sentence):
        """
        set batch size and sequence length
        """
        batch_size, seq_length, _ = sentence.size()
        self.seq_length = seq_length
        self.batch_size = batch_size
        batch_seq_size = {'sentence_len': self.seq_length,
                          'batch_size': self.batch_size,
                          'is_multi_root': self.is_multi_root,
                          'max_dependency_len': self.max_dependency_len,
                          'use_gpu': self.use_gpu,
                          'length_constraint_on_root': self.length_constraint_on_root}

        return batch_seq_size

    def init(self):
        utility.init_linear(self.score_layer)
        utility.init_linear(self.em2crf1)
        utility.init_linear(self.em2crf2)
        utility.init_linear(self.f2prob1)
        utility.init_linear(self.f2prob2)

    def real_forward(self, sentences, enable_prior=True):
        # r_embed, p_embed = self.preprocessor.process_cs_batch(sentences)
        r_embed, p_embed = self.preprocessor.process_ss_batch(sentences)

        self.get_batch_seq_size(r_embed)

        d_r_embeds = self.dropout(r_embed)
        d_hidden_out = self.activation(self.em2crf1(d_r_embeds))

        d_p_embeds = self.dropout(p_embed)
        p_hidden_out = self.activation(self.em2crf2(d_p_embeds))

        score_mat = torch.matmul(d_hidden_out, p_hidden_out.permute(0, 2, 1))
        prior_weight = torch.cat([s.prior_weight()[None, :, :] for s in sentences])
        prior_weight = (self.prior_factor if enable_prior else 0.) * prior_weight

        prior_weight = prior_weight.cuda() if self.use_gpu else prior_weight

        joint_score_mat = prior_weight + score_mat

        return joint_score_mat

    def forward(self, sentences, enable_prior=True):
        score_mat = self.real_forward(sentences, enable_prior)
        batch, length = score_mat.size()[:2]
        gold = torch.tensor([sent.heads for sent in sentences]).long().to(score_mat.device)
        ll = self.crie(score_mat.view(-1, length), gold.view(-1)).reshape(batch, length)
        return ll[:, 1:].mean()

    def decoding(self, sentence, enable_prior=True):
        score_mat = self.real_forward(sentence, enable_prior=enable_prior)
        best_tree = score_mat.argmax(dim=-1)
        return best_tree

    # gaussian method
    def get_recons_weight(self, input: torch.Tensor) -> torch.Tensor:

        mus = self.f2prob1(input)
        # todo not debug here
        logvars = torch.diag_embed(torch.zeros_like(mus) + 1.0)
        point = self.f2prob1(input)
        batch_list = torch.split(point, 1)
        batch_mu_list = torch.split(mus, 1)
        batch_logvar_list = torch.split(logvars, 1)
        batch_holder = []
        for idx, sent in enumerate(batch_list):
            sent_holder = []
            sent_mu = torch.split(batch_mu_list[idx].squeeze(0), 1)
            sent_logvar = torch.split(batch_logvar_list[idx].squeeze(0), 1)
            for mu, logvar in zip(sent_mu, sent_logvar):
                gaussian = D.MultivariateNormal(mu.squeeze(0), logvar.squeeze(0))
                sent_log_prob = gaussian.log_prob(sent.squeeze(0))
                sent_holder.append(sent_log_prob)
            batch_holder.append(torch.stack(sent_holder, dim=0))
        return torch.stack(batch_holder, dim=0)

    # porb method
    def get_recons_weight_v2(self, input: torch.Tensor) -> torch.Tensor:

        # similar to self-attention?
        # todo try copy self-attention like method
        p1 = self.f2prob1(input)
        p2 = self.f2prob2(input)

        score = torch.matmul(p1, p2.permute(0, 2, 1))
        prob = torch.log_softmax(score, dim=-1)
        return prob

class CSNCRFAE_lstm(nn.Module):
    def __init__(self, preprocessor: DiscoursePreprocessor, tagset_size, embedding_dim, hidden_dim, rnn_layers,
                 dropout_ratio, prior_weight, recons_dim, act_func='tanh',
                 use_gpu=False, is_multi_root=False, max_dependency_len=10, length_constraint_on_root=False):
        super(CSNCRFAE_lstm, self).__init__()

        self.preprocessor = preprocessor

        self.use_gpu = use_gpu
        self.is_multi_root = is_multi_root
        self.max_dependency_len = max_dependency_len
        self.length_constraint_on_root = length_constraint_on_root

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.em2h1 = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.em2h2 = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.activation = ACTIVATION_DICT[act_func]
        self.dropout = nn.Dropout(p=dropout_ratio)

        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2,
                            num_layers=rnn_layers, bidirectional=True)

        self.prior_factor = prior_weight

        self.tagset_size = tagset_size

        self.batch_size = 1  # temp
        self.seq_length = 1  # temp
        self.crie = torch.nn.CrossEntropyLoss(reduction='none')
        self.esiner = IncrementalEisnerDecoder()


    def get_batch_seq_size(self, sentence):
        """
        set batch size and sequence length
        """
        batch_size, seq_length, _ = sentence.size()
        self.seq_length = seq_length
        self.batch_size = batch_size
        batch_seq_size = {'sentence_len': self.seq_length,
                          'batch_size': self.batch_size,
                          'is_multi_root': self.is_multi_root,
                          'max_dependency_len': self.max_dependency_len,
                          'use_gpu': self.use_gpu,
                          'length_constraint_on_root': self.length_constraint_on_root}

        return batch_seq_size

    def init(self):
        pass

    def real_forward(self, sentences, enable_prior=False):
        r_embed, p_embed = self.preprocessor.process_cs_batch(sentences)
        # r_embed, p_embed = self.preprocessor.process_ss_batch(sentences)
        batch_size, length, _ = r_embed.size()
        self.get_batch_seq_size(r_embed)

        d_r_embeds = self.dropout(r_embed)
        r_lstm_in = self.activation(self.em2h1(d_r_embeds))

        d_p_embeds = self.dropout(p_embed)
        p_lstm_in = self.activation(self.em2h2(d_p_embeds))


        # BiLSTM
        p_packed_edu = nn.utils.rnn.pack_padded_sequence(p_lstm_in,
                                                       torch.tensor([length]*batch_size),
                                                       batch_first=True,
                                                       enforce_sorted=False)
        p_packed_output, hidden = self.lstm(p_packed_edu)
        p_lstm_out, _ = nn.utils.rnn.pad_packed_sequence(p_packed_output, batch_first=True)

        r_packed_edu = nn.utils.rnn.pack_padded_sequence(r_lstm_in,
                                                       torch.tensor([length]*batch_size),
                                                       batch_first=True,
                                                       enforce_sorted=False)
        r_packed_output, hidden = self.lstm(r_packed_edu)
        r_lstm_out, _ = nn.utils.rnn.pad_packed_sequence(r_packed_output, batch_first=True)

        score_mat = torch.matmul(self.dropout(r_lstm_out),
                                 self.dropout(p_lstm_out).permute(0, 2, 1))
        prior_weight = torch.cat([s.prior_weight()[None, :, :] for s in sentences])
        prior_weight = (self.prior_factor if enable_prior else 0.) * prior_weight

        prior_weight = prior_weight.cuda() if self.use_gpu else prior_weight

        joint_score_mat = prior_weight + score_mat

        return joint_score_mat

    def forward(self, sentences, enable_prior=False):
        score_mat = self.real_forward(sentences, enable_prior)
        batch, length = score_mat.size()[:2]
        gold = torch.tensor([sent.heads for sent in sentences]).long().to(score_mat.device)
        ll = self.crie(score_mat.view(-1, length), gold.view(-1)).reshape(batch, length)
        return ll[:, 1:].mean()

    # def decoding(self, sentence, enable_prior=True):
    #     score_mat = self.real_forward(sentence, enable_prior=enable_prior)
    #     best_tree = score_mat.argmax(dim=-1)
    #     return best_tree.detach().cpu().numpy()

    # def decoding(self, sentence, enable_prior=True):
    #     batch_score_mat = torch.log_softmax(self.real_forward(sentence, enable_prior=enable_prior), dim=-1)
    #     score_mat_np = batch_score_mat.permute(0, 2, 1).detach().cpu().numpy()
    #     best_tree = []
    #     for score_matrix in score_mat_np:
    #         edu_ids = [i for i in range(score_matrix.shape[-1])]
    #         arcs, max_score = self.esiner.decoder.decode(score_matrix,edu_ids)
    #         pred_tree = [node[0] for node in sorted(arcs, key=lambda x: x[1])]
    #         best_tree.append([-1] + pred_tree)
    #     return best_tree

    def decoding(self, sentences):
        batch_score_mat = self.real_forward(sentences, enable_prior=False)
        best_tree = []
        batch_score_mat_np = batch_score_mat.permute(0, 2, 1).detach().cpu().numpy()
        for idx, (sentence, score_mat_np) in enumerate(zip(sentences, batch_score_mat_np)):
            unlabeled_arcs = self.esiner.decode(arc_scores=score_mat_np, edu_ids=sentence.raw.edu_ids,
                                                sbnds=sentence.raw.sbnds, pbnds=sentence.raw.pbnds,
                                                use_sbnds=True, use_pbnds=False)
            pred_arcs = [arc[0] for arc in sorted(unlabeled_arcs, key=lambda x: x[1])]
            best_tree.append([0] + pred_arcs)
        return best_tree


class CSNCRFAE_d(nn.Module):
    def __init__(self, preprocessor: DiscoursePreprocessor, tagset_size, embedding_dim, hidden_dim,
                 dropout_ratio, prior_weight, recons_dim, act_func='tanh',
                 use_gpu=False, is_multi_root=False, max_dependency_len=10, length_constraint_on_root=False):
        super(CSNCRFAE_d, self).__init__()

        self.preprocessor = preprocessor

        self.use_gpu = use_gpu
        self.is_multi_root = is_multi_root
        self.max_dependency_len = max_dependency_len
        self.length_constraint_on_root = length_constraint_on_root

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.em2crf1 = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.em2crf2 = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.activation = ACTIVATION_DICT[act_func]

        self.dropout = nn.Dropout(p=dropout_ratio)

        self.prior_factor = prior_weight

        self.tagset_size = tagset_size

        self.batch_size = 1  # temp
        self.seq_length = 1  # temp
        self.crie = torch.nn.CrossEntropyLoss(reduction='none')
        self.esiner = IncrementalEisnerDecoder()

    def get_batch_seq_size(self, sentence):
        """
        set batch size and sequence length
        """
        batch_size, seq_length, _ = sentence.size()
        self.seq_length = seq_length
        self.batch_size = batch_size
        batch_seq_size = {'sentence_len': self.seq_length,
                          'batch_size': self.batch_size,
                          'is_multi_root': self.is_multi_root,
                          'max_dependency_len': self.max_dependency_len,
                          'use_gpu': self.use_gpu,
                          'length_constraint_on_root': self.length_constraint_on_root}

        return batch_seq_size

    def init(self):
        utility.init_linear(self.score_layer)
        utility.init_linear(self.em2crf1)
        utility.init_linear(self.em2crf2)
        utility.init_linear(self.f2prob1)
        utility.init_linear(self.f2prob2)

    def real_forward(self, sentences, enable_prior=True):
        # r_embed, p_embed = self.preprocessor.process_cs_batch(sentences)
        r_embed, p_embed = self.preprocessor.process_ss_batch(sentences)

        self.get_batch_seq_size(r_embed)

        d_r_embeds = self.dropout(r_embed)
        d_hidden_out = self.activation(self.em2crf1(d_r_embeds))

        d_p_embeds = self.dropout(p_embed)
        p_hidden_out = self.activation(self.em2crf2(d_p_embeds))

        score_mat = torch.matmul(d_hidden_out, p_hidden_out.permute(0, 2, 1))
        prior_weight = torch.cat([s.prior_weight()[None, :, :] for s in sentences])
        prior_weight = (self.prior_factor if enable_prior else 0.) * prior_weight

        prior_weight = prior_weight.cuda() if self.use_gpu else prior_weight

        joint_score_mat = prior_weight + score_mat

        return joint_score_mat

    def forward(self, sentences, enable_prior=True):
        score_mat = self.real_forward(sentences, enable_prior)
        batch, length = score_mat.size()[:2]
        gold = torch.tensor([sent.heads for sent in sentences]).long().to(score_mat.device)
        ll = self.crie(score_mat.view(-1, length), gold.view(-1)).reshape(batch, length)
        return ll[:, 1:].mean()

    def decoding(self, sentence, enable_prior=True):
        batch_score_mat = torch.log_softmax(self.real_forward(sentence, enable_prior=enable_prior), dim=-1)
        score_mat_np = batch_score_mat.permute(0, 2, 1).detach().cpu().numpy()
        best_tree = []
        for score_matrix in score_mat_np:
            edu_ids = [i for i in range(score_matrix.shape[-1])]
            arcs, max_score = self.esiner.decoder.decode(score_matrix,edu_ids)
            pred_tree = [node[0] for node in sorted(arcs, key=lambda x: x[1])]
            best_tree.append([-1] + pred_tree)
        return best_tree


class CSNCRFAE_biaffine(nn.Module):
    def __init__(self, preprocessor: DiscoursePreprocessor, tagset_size, embedding_dim, hidden_dim,
                 dropout_ratio, prior_weight, recons_dim, act_func='tanh',
                 use_gpu=False, is_multi_root=False, max_dependency_len=10, length_constraint_on_root=False):
        super(CSNCRFAE_biaffine, self).__init__()

        self.preprocessor = preprocessor

        self.use_gpu = use_gpu
        self.is_multi_root = is_multi_root
        self.max_dependency_len = max_dependency_len
        self.length_constraint_on_root = length_constraint_on_root

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.em2crf1 = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.em2crf2 = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.activation = ACTIVATION_DICT[act_func]

        self.biaffine = Parameter(torch.zeros(self.hidden_dim, self.hidden_dim))
        self.dropout = nn.Dropout(p=dropout_ratio)

        self.prior_factor = prior_weight

        self.tagset_size = tagset_size

        self.batch_size = 1  # temp
        self.seq_length = 1  # temp
        self.crie = torch.nn.CrossEntropyLoss(reduction='none')
        self.esiner = IncrementalEisnerDecoder()

    def get_batch_seq_size(self, sentence):
        """
        set batch size and sequence length
        """
        batch_size, seq_length, _ = sentence.size()
        self.seq_length = seq_length
        self.batch_size = batch_size
        batch_seq_size = {'sentence_len': self.seq_length,
                          'batch_size': self.batch_size,
                          'is_multi_root': self.is_multi_root,
                          'max_dependency_len': self.max_dependency_len,
                          'use_gpu': self.use_gpu,
                          'length_constraint_on_root': self.length_constraint_on_root}

        return batch_seq_size

    def init(self):
        nn.init.xavier_normal_(self.biaffine)
        # nn.init.xavier_normal_(self.em2crf1)
        # nn.init.xavier_normal_(self.em2crf2)
        # utility.init_linear(self.score_layer)
        # utility.init_linear(self.em2crf1)
        # utility.init_linear(self.em2crf2)
        # utility.init_linear(self.f2prob1)
        # utility.init_linear(self.f2prob2)

    def real_forward(self, sentences, enable_prior=True):
        # r_embed, p_embed = self.preprocessor.process_cs_batch(sentences)
        r_embed, p_embed = self.preprocessor.process_ss_batch(sentences)

        self.get_batch_seq_size(r_embed)

        d_r_embeds = self.dropout(r_embed)
        d_hidden_out = self.activation(self.em2crf1(d_r_embeds))

        d_p_embeds = self.dropout(p_embed)
        p_hidden_out = self.activation(self.em2crf2(d_p_embeds))

        # score_mat = torch.matmul(d_hidden_out, p_hidden_out.permute(0, 2, 1))
        score_mat = torch.matmul(d_hidden_out,
                                 torch.matmul(self.biaffine.view(1, 1, self.hidden_dim, self.hidden_dim),
                                              p_hidden_out.unsqueeze(-1)).squeeze(-1).permute(0, 2, 1))
        prior_weight = torch.cat([s.prior_weight()[None, :, :] for s in sentences])
        prior_weight = (self.prior_factor if enable_prior else 0.) * prior_weight

        prior_weight = prior_weight.cuda() if self.use_gpu else prior_weight

        joint_score_mat = prior_weight + score_mat

        return joint_score_mat

    def forward(self, sentences, enable_prior=True):
        score_mat = self.real_forward(sentences, enable_prior)
        batch, length = score_mat.size()[:2]
        gold = torch.tensor([sent.heads for sent in sentences]).long().to(score_mat.device)
        ll = self.crie(score_mat.view(-1, length), gold.view(-1)).reshape(batch, length)
        return ll[:, 1:].mean()

    def decoding(self, sentence, enable_prior=True):
        batch_score_mat = torch.log_softmax(self.real_forward(sentence, enable_prior=enable_prior), dim=-1)
        score_mat_np = batch_score_mat.permute(0, 2, 1).detach().cpu().numpy()
        best_tree = []
        for score_matrix in score_mat_np:
            edu_ids = [i for i in range(score_matrix.shape[-1])]
            arcs, max_score = self.esiner.decoder.decode(score_matrix,edu_ids)
            pred_tree = [node[0] for node in sorted(arcs, key=lambda x: x[1])]
            best_tree.append([-1] + pred_tree)
        best_tree_2 = batch_score_mat.argmax(dim=-1).detach().cpu().numpy()
        return best_tree, best_tree_2


class CSNCRFAE_margin(nn.Module):
    def __init__(self, preprocessor: DiscoursePreprocessor, tagset_size, embedding_dim, hidden_dim,
                 dropout_ratio, prior_weight, recons_dim, act_func='tanh',
                 use_gpu=False, is_multi_root=False, max_dependency_len=10, length_constraint_on_root=False):
        super(CSNCRFAE_margin, self).__init__()

        self.preprocessor = preprocessor

        self.use_gpu = use_gpu
        self.is_multi_root = is_multi_root
        self.max_dependency_len = max_dependency_len
        self.length_constraint_on_root = length_constraint_on_root

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.em2crf1 = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.em2crf2 = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.activation = ACTIVATION_DICT[act_func]

        self.dropout = nn.Dropout(p=dropout_ratio)

        self.prior_factor = prior_weight

        self.tagset_size = tagset_size

        self.batch_size = 1  # temp
        self.seq_length = 1  # temp
        self.crie = torch.nn.CrossEntropyLoss(reduction='none')
        self.esiner = IncrementalEisnerDecoder()

    def get_batch_seq_size(self, sentence):
        """
        set batch size and sequence length
        """
        batch_size, seq_length, _ = sentence.size()
        self.seq_length = seq_length
        self.batch_size = batch_size
        batch_seq_size = {'sentence_len': self.seq_length,
                          'batch_size': self.batch_size,
                          'is_multi_root': self.is_multi_root,
                          'max_dependency_len': self.max_dependency_len,
                          'use_gpu': self.use_gpu,
                          'length_constraint_on_root': self.length_constraint_on_root}

        return batch_seq_size

    def init(self):
        pass

        # nn.init.xavier_normal_(self.em2crf1)
        # nn.init.xavier_normal_(self.em2crf2)
        # utility.init_linear(self.score_layer)
        # utility.init_linear(self.em2crf1)
        # utility.init_linear(self.em2crf2)
        # utility.init_linear(self.f2prob1)
        # utility.init_linear(self.f2prob2)

    def real_forward(self, sentences, enable_prior=True):
        # r_embed, p_embed = self.preprocessor.process_cs_batch(sentences)
        r_embed, p_embed = self.preprocessor.process_ss_batch(sentences)

        self.get_batch_seq_size(r_embed)

        d_r_embeds = self.dropout(r_embed)
        d_hidden_out = self.activation(self.em2crf1(d_r_embeds))

        d_p_embeds = self.dropout(p_embed)
        p_hidden_out = self.activation(self.em2crf2(d_p_embeds))

        score_mat = torch.matmul(d_hidden_out, p_hidden_out.permute(0, 2, 1))
        prior_weight = torch.cat([s.prior_weight()[None, :, :] for s in sentences])
        prior_weight = (self.prior_factor if enable_prior else 0.) * prior_weight

        prior_weight = prior_weight.cuda() if self.use_gpu else prior_weight

        joint_score_mat = prior_weight + score_mat

        return joint_score_mat

    def forward(self, sentences, enable_prior=True):
        batch_score_mat = self.real_forward(sentences, enable_prior).permute(0, 2, 1)
        # gold = torch.tensor([sent.heads for sent in sentences]).long().to(score_mat.device)
        # ll = self.crie(score_mat.view(-1, length), gold.view(-1)).reshape(batch, length)
        batch_score_mat_np = batch_score_mat.detach().cpu().numpy()
        loss = 0.0
        for idx, (score_mat_np, score_mat) in enumerate(zip(batch_score_mat_np, batch_score_mat)):
            edu_ids = [i for i in range(score_mat_np.shape[-1])]
            arcs, max_score = self.esiner.decoder.decode(score_mat_np, edu_ids)

            pred_score= self.calculate_score(score_mat, arcs)
            golden_score = self.calculate_score(score_mat, sentences[idx].raw.arcs)
            margin = utility.compute_tree_distance(arcs, sentences[idx].raw.arcs, 1.0)
            # calculate score
            loss += torch.clamp(pred_score + margin - golden_score, 0.0, 10000000.0)
        return loss.mean()

    @staticmethod
    def calculate_score(matrix: torch.Tensor, tree: List[Tuple[int, int]])-> torch.Tensor:
        score = 0.0
        for arc in tree:
            score += matrix[arc[0]][arc[1]]
        return score

    def decoding(self, sentence, enable_prior=True):
        batch_score_mat = self.real_forward(sentence, enable_prior=enable_prior).permute(0, 2, 1)
        score_mat_np = batch_score_mat.detach().cpu().numpy()
        best_tree = []
        for score_matrix in score_mat_np:
            edu_ids = [i for i in range(score_matrix.shape[-1])]
            arcs, max_score = self.esiner.decoder.decode(score_matrix, edu_ids)
            pred_tree = [node[0] for node in sorted(arcs, key=lambda x: x[1])]
            best_tree.append([-1] + pred_tree)
        best_tree_2 = batch_score_mat.argmax(dim=-1).detach().cpu().numpy()
        return best_tree, best_tree_2


class JAP(nn.Module):
    def __init__(self, preprocessor, word_dim, postag_dim, deprel_dim, lstm_dim, mlp_dim, device):
        super(JAP, self).__init__()
        self.preprocessor = preprocessor
        self.device = device

        # Word embedding
        self.word_dim = word_dim
        self.postag_dim = postag_dim
        self.deprel_dim = deprel_dim

        # BiLSTM over EDUs
        self.lstm_dim = lstm_dim
        self.bilstm_dim = lstm_dim + lstm_dim

        # MLP
        self.mlp_dim = mlp_dim
        self.n_relations = 17 # len(self.vocab_relation)

        self.encoder = CSEncoder('bert', self.device)

        self.W_edu = nn.Linear(self.word_dim, self.lstm_dim)

        self.bilstm = nn.LSTM(num_layers=1,
                              input_size=self.lstm_dim,
                              hidden_size=self.lstm_dim,
                              bidirectional=True,
                              dropout=0.0)
        # MLPs
        self.W1_a = nn.Linear(self.bilstm_dim +
                              self.bilstm_dim,
                              self.mlp_dim)

        self.W2_a = nn.Linear(self.mlp_dim, 1)

        self.W1_r = nn.Linear(self.bilstm_dim +
                              self.bilstm_dim,
                              self.mlp_dim)

        self.W2_r = nn.Linear(self.mlp_dim, self.n_relations)

        self.esiner = IncrementalEisnerDecoder()

    def forward_edus(self, sentences):
        # edu_vectors, _ = self.preprocessor.process_cs_batch(sentences)
        # edu_vectors, _ = self.preprocessor.process_ss_batch(sentences)
        edu_vectors = self.preprocessor.process_sm_batch(sentences)
        # edu_vectors = self.preprocessor.process_map_batch(sentences)
        edu_vectors = F.relu(self.W_edu(edu_vectors)) # (n_edus, word_dim)

        batch_size, length, _ = edu_vectors.size()
        # BiLSTM
        packed_edu = nn.utils.rnn.pack_padded_sequence(edu_vectors,
                                                       torch.tensor([length]*batch_size),
                                                       batch_first=True,
                                                       enforce_sorted=False)
        packed_output, hidden = self.bilstm(packed_edu)
        states, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        return states

    def forward_arcs_for_attachment(
                    self,
                    edu_vectors,
                    batch_arcs,
                    aggregate=True):

        batch_size = len(batch_arcs)
        n_arcs = len(batch_arcs[0])
        # total_arcs = batch_size * n_arcs
        # for arcs in batch_arcs:
        #     assert len(arcs) == n_arcs

        # Reshape
        flatten_batch_arcs = utility.flatten_lists(batch_arcs) # total_arcs * (int, int)
        batch_head, batch_dep = zip(*flatten_batch_arcs)
        batch_head = list(batch_head) # total_arcs * int
        batch_dep = list(batch_dep) # total_arcs * int

        # Feature extraction
        batch_head_vectors = torch.embedding(edu_vectors, torch.tensor(batch_head).to(self.device)) # (total_arcs, bilstm_dim + tempfeat1_dim)
        batch_dep_vectors = torch.embedding(edu_vectors, torch.tensor(batch_dep).to(self.device)) # (total_arcs, bilstm_dim + tempfeat1_dim)

        batch_arc_vectors = torch.cat([batch_head_vectors,
                                      batch_dep_vectors],
                                      dim=1)

        # MLP (Attachment Scoring)
        arc_scores = self.W2_a(F.dropout(F.relu(self.W1_a(batch_arc_vectors)), p=0.2)) # (total_arcs, 1)
        arc_scores = torch.reshape(arc_scores, (batch_size, n_arcs, 1)) # (batch_size, n_arcs, 1)

        # Aggregate
        if aggregate:
            tree_scores = torch.sum(arc_scores, dim=1) # (batch_size, 1)
            return tree_scores
        else:
            return arc_scores # (batch_size, n_arcs, 1)

    def forward_arcs_for_labeling(self, edu_vectors, batch_arcs):

        batch_size = len(batch_arcs)
        n_arcs = len(batch_arcs[0])
        # total_arcs = batch_size * n_arcs
        # for arcs in batch_arcs:
        #     assert len(arcs) == n_arcs

        # Reshape
        flatten_batch_arcs = utility.flatten_lists(batch_arcs)  # total_arcs * (int, int)
        batch_head, batch_dep = zip(*flatten_batch_arcs)
        batch_head = list(batch_head)  # total_arcs * int
        batch_dep = list(batch_dep)  # total_arcs * int

        # Feature extraction
        batch_head_vectors = torch.embedding(edu_vectors, torch.tensor(batch_head).to(self.device))
        batch_dep_vectors = torch.embedding(edu_vectors, torch.tensor(batch_dep).to(self.device))

        batch_arc_vectors = torch.cat([batch_head_vectors,
                                      batch_dep_vectors],
                                      dim=1)

        # MLP (labeling)
        logits = self.W2_r(F.dropout(F.relu(self.W1_r(batch_arc_vectors)), p=0.2))
        logits = logits.reshape(batch_size, n_arcs, self.n_relations)

        return logits

    def forward(self, sentences):
        batch_edu_vectors = self.forward_edus(sentences)

        loss, loss_attachment = 0.0, 0.0

        for idx, (sentence, edu_vectors) in enumerate(zip(sentences, batch_edu_vectors)):
            with torch.no_grad():
                pos_arcs = [(h,d) for h,d,l in sentence.raw.arcs]
                arcs_scores = self.precompute_all_arc_scores(edu_ids=sentence.raw.edu_ids, edu_vectors=edu_vectors)
                gold_heads = -np.ones((len(sentence.raw.edus),), dtype=np.int32)
                for h,d,l in sentence.raw.arcs:
                    gold_heads[d] = h
                neg_arcs = self.esiner.global_decode(arc_scores=arcs_scores, edu_ids=sentence.raw.edu_ids,
                                              sbnds=sentence.raw.sbnds, pbnds=sentence.raw.pbnds,
                                              use_sbnds=True, use_pbnds=False, gold_heads=gold_heads)
                margin = utility.compute_tree_distance(pos_arcs, neg_arcs, coef=1.0)
            pred_scores = self.forward_arcs_for_attachment(edu_vectors=edu_vectors, batch_arcs=[pos_arcs, neg_arcs],
                                                           aggregate=True)
            # # Labeling
            # pred_relations = self.forward_arcs_for_labeling(edu_vectors=edu_vectors, batch_arcs=[pos_arcs])
            # pred_relations = pred_relations[0]

            # Attachment Loss
            loss_attachment += torch.clamp(pred_scores[1] + margin - pred_scores[0], 0.0, 10000000.0)

            loss += loss_attachment
        return loss

    def decoding(self, sentences):
        batch_edu_vectors = self.forward_edus(sentences)
        best_tree = []
        for idx, (sentence, edu_vectors) in enumerate(zip(sentences, batch_edu_vectors)):
            arc_scores = self.precompute_all_arc_scores(edu_ids=sentence.raw.edu_ids, edu_vectors=edu_vectors)
            unlabeled_arcs = self.esiner.global_decode(arc_scores=arc_scores, edu_ids=sentence.raw.edu_ids,
                                                sbnds=sentence.raw.sbnds, pbnds=sentence.raw.pbnds,
                                                use_sbnds=True, use_pbnds=False)
            pred_arcs = [arc[0] for arc in sorted(unlabeled_arcs, key=lambda x: x[1])]
            best_tree.append([0] + pred_arcs)
        return best_tree

    def precompute_all_arc_scores(self, edu_ids, edu_vectors):
        n_edus = len(edu_ids)

        result = np.zeros((n_edus, n_edus), dtype="float")

        # Aggregating patterns
        arcs = []
        for h in range(0, n_edus):
            for d in range(0, n_edus):
                if h == d:
                    continue
                arc = (h, d)
                arcs.append(arc)

        # Shifting: local position -> global position
        arcs = [(edu_ids[h], edu_ids[d]) for h, d in arcs]

        # Scoring
        arc_scores = self.forward_arcs_for_attachment(
            edu_vectors=edu_vectors,
            batch_arcs=[arcs],
            aggregate=False)
        arc_scores = arc_scores.cpu().numpy()[0]
        for arc_i, (h, d) in enumerate(arcs):
            result[h, d] = float(arc_scores[arc_i])
        return result

# fintuen bert
class CSNCRFAE_lstm_f(nn.Module):
    def __init__(self, preprocessor: DiscoursePreprocessor, tagset_size, embedding_dim, hidden_dim, rnn_layers,
                 dropout_ratio, prior_weight, recons_dim, act_func='tanh',
                 use_gpu=False, is_multi_root=False, max_dependency_len=10, length_constraint_on_root=False):
        super(CSNCRFAE_lstm_f, self).__init__()

        self.preprocessor = preprocessor

        self.use_gpu = use_gpu
        self.is_multi_root = is_multi_root
        self.max_dependency_len = max_dependency_len
        self.length_constraint_on_root = length_constraint_on_root

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.em2h1 = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.em2h2 = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.activation = ACTIVATION_DICT[act_func]
        self.dropout = nn.Dropout(p=dropout_ratio)

        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2,
                            num_layers=rnn_layers, bidirectional=True)

        self.prior_factor = prior_weight

        self.tagset_size = tagset_size

        self.batch_size = 1  # temp
        self.seq_length = 1  # temp
        self.crie = torch.nn.CrossEntropyLoss(reduction='none')
        self.esiner = IncrementalEisnerDecoder()
        self.encoder = CSEncoder('bert')


    def get_batch_seq_size(self, sentence):
        """
        set batch size and sequence length
        """
        batch_size, seq_length, _ = sentence.size()
        self.seq_length = seq_length
        self.batch_size = batch_size
        batch_seq_size = {'sentence_len': self.seq_length,
                          'batch_size': self.batch_size,
                          'is_multi_root': self.is_multi_root,
                          'max_dependency_len': self.max_dependency_len,
                          'use_gpu': self.use_gpu,
                          'length_constraint_on_root': self.length_constraint_on_root}

        return batch_seq_size

    def init(self):
        pass

    def real_forward(self, sentences, enable_prior=False):
        # r_embed, p_embed = self.preprocessor.process_cs_batch(sentences)
        # r_embed, p_embed = self.preprocessor.process_ss_batch(sentences)
        r_embed, p_embed = [], []
        for sent in sentences:
            embed1, embed2 = self.encoder.grad_encode(sent.raw)
            r_embed.append(embed1)
            p_embed.append(embed2)
        r_embed = torch.stack(r_embed, dim=0)
        p_embed = torch.stack(p_embed, dim=0)
        batch_size, length, _ = r_embed.size()
        self.get_batch_seq_size(r_embed)

        d_r_embeds = self.dropout(p_embed)
        r_lstm_in = self.activation(self.em2h1(d_r_embeds))

        d_p_embeds = self.dropout(p_embed)
        p_lstm_in = self.activation(self.em2h2(d_p_embeds))


        # BiLSTM
        p_packed_edu = nn.utils.rnn.pack_padded_sequence(p_lstm_in,
                                                       torch.tensor([length]*batch_size),
                                                       batch_first=True,
                                                       enforce_sorted=False)
        p_packed_output, hidden = self.lstm(p_packed_edu)
        p_lstm_out, _ = nn.utils.rnn.pad_packed_sequence(p_packed_output, batch_first=True)

        r_packed_edu = nn.utils.rnn.pack_padded_sequence(r_lstm_in,
                                                       torch.tensor([length]*batch_size),
                                                       batch_first=True,
                                                       enforce_sorted=False)
        r_packed_output, hidden = self.lstm(r_packed_edu)
        r_lstm_out, _ = nn.utils.rnn.pad_packed_sequence(r_packed_output, batch_first=True)

        score_mat = torch.matmul(self.dropout(r_lstm_out),
                                 self.dropout(p_lstm_out).permute(0, 2, 1))
        prior_weight = torch.cat([s.prior_weight()[None, :, :] for s in sentences])
        prior_weight = (self.prior_factor if enable_prior else 0.) * prior_weight

        prior_weight = prior_weight.cuda() if self.use_gpu else prior_weight

        joint_score_mat = prior_weight + score_mat

        return joint_score_mat

    def forward(self, sentences, enable_prior=False):
        score_mat = self.real_forward(sentences, enable_prior)
        batch, length = score_mat.size()[:2]
        gold = torch.tensor([sent.heads for sent in sentences]).long().to(score_mat.device)
        ll = self.crie(score_mat.view(-1, length), gold.view(-1)).reshape(batch, length)
        return ll[:, 1:].mean()

    def decoding(self, sentences):
        batch_score_mat = self.real_forward(sentences, enable_prior=False)
        best_tree = []
        batch_score_mat_np = batch_score_mat.permute(0, 2, 1).detach().cpu().numpy()
        for idx, (sentence, score_mat_np) in enumerate(zip(sentences, batch_score_mat_np)):
            unlabeled_arcs = self.esiner.decode(arc_scores=score_mat_np, edu_ids=sentence.raw.edu_ids,
                                                sbnds=sentence.raw.sbnds, pbnds=sentence.raw.pbnds,
                                                use_sbnds=True, use_pbnds=False)
            pred_arcs = [arc[0] for arc in sorted(unlabeled_arcs, key=lambda x: x[1])]
            best_tree.append([0] + pred_arcs)
        return best_tree

class CSNCRFAE_lstm_m(nn.Module):
    def __init__(self, preprocessor: DiscoursePreprocessor, tagset_size, embedding_dim, hidden_dim, rnn_layers,
                 dropout_ratio, prior_weight, recons_dim, act_func='tanh',
                 use_gpu=False, is_multi_root=False, max_dependency_len=10, length_constraint_on_root=False):
        super(CSNCRFAE_lstm_m, self).__init__()

        self.preprocessor = preprocessor

        self.use_gpu = use_gpu
        self.is_multi_root = is_multi_root
        self.max_dependency_len = max_dependency_len
        self.length_constraint_on_root = length_constraint_on_root

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.em2h1 = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.em2h2 = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.activation = ACTIVATION_DICT[act_func]
        self.dropout = nn.Dropout(p=dropout_ratio)

        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2,
                            num_layers=rnn_layers, bidirectional=True)

        self.prior_factor = prior_weight

        self.tagset_size = tagset_size

        self.batch_size = 1  # temp
        self.seq_length = 1  # temp
        self.crie = torch.nn.CrossEntropyLoss(reduction='none')
        self.esiner = IncrementalEisnerDecoder()

    def get_batch_seq_size(self, sentence):
        """
        set batch size and sequence length
        """
        batch_size, seq_length, _ = sentence.size()
        self.seq_length = seq_length
        self.batch_size = batch_size
        batch_seq_size = {'sentence_len': self.seq_length,
                          'batch_size': self.batch_size,
                          'is_multi_root': self.is_multi_root,
                          'max_dependency_len': self.max_dependency_len,
                          'use_gpu': self.use_gpu,
                          'length_constraint_on_root': self.length_constraint_on_root}

        return batch_seq_size

    def init(self):
        pass

    def real_forward(self, sentences, enable_prior=False):
        r_embed = self.preprocessor.process_mean_batch(sentences)
        # r_embed = self.preprocessor.process_sap_batch(sentences)
        # r_embed = self.preprocessor.process_map_batch(sentences)
        # r_embed, p_embed = self.preprocessor.process_ss_batch(sentences)
        batch_size, length, _ = r_embed.size()
        self.get_batch_seq_size(r_embed)

        d_r_embeds = self.dropout(r_embed)
        r_lstm_in = self.activation(self.em2h1(d_r_embeds))

        d_p_embeds = self.dropout(r_embed)
        p_lstm_in = self.activation(self.em2h2(d_p_embeds))

        # BiLSTM
        p_packed_edu = nn.utils.rnn.pack_padded_sequence(p_lstm_in,
                                                         torch.tensor([length] * batch_size),
                                                         batch_first=True,
                                                         enforce_sorted=False)
        p_packed_output, hidden = self.lstm(p_packed_edu)
        p_lstm_out, _ = nn.utils.rnn.pad_packed_sequence(p_packed_output, batch_first=True)

        r_packed_edu = nn.utils.rnn.pack_padded_sequence(r_lstm_in,
                                                         torch.tensor([length] * batch_size),
                                                         batch_first=True,
                                                         enforce_sorted=False)
        r_packed_output, hidden = self.lstm(r_packed_edu)
        r_lstm_out, _ = nn.utils.rnn.pad_packed_sequence(r_packed_output, batch_first=True)

        score_mat = torch.matmul(self.dropout(r_lstm_out),
                                 self.dropout(p_lstm_out).permute(0, 2, 1))
        prior_weight = torch.cat([s.prior_weight()[None, :, :] for s in sentences])
        prior_weight = (self.prior_factor if enable_prior else 0.) * prior_weight

        prior_weight = prior_weight.cuda() if self.use_gpu else prior_weight

        joint_score_mat = prior_weight + score_mat

        return joint_score_mat

    def forward(self, sentences, enable_prior=False):
        score_mat = self.real_forward(sentences, enable_prior)
        batch, length = score_mat.size()[:2]
        gold = torch.tensor([sent.heads for sent in sentences]).long().to(score_mat.device)
        ll = self.crie(score_mat.view(-1, length), gold.view(-1)).reshape(batch, length)
        return ll[:, 1:].mean()

    def decoding(self, sentences):
        batch_score_mat = self.real_forward(sentences, enable_prior=False)
        best_tree = []
        batch_score_mat_np = batch_score_mat.permute(0, 2, 1).detach().cpu().numpy()
        for idx, (sentence, score_mat_np) in enumerate(zip(sentences, batch_score_mat_np)):
            unlabeled_arcs = self.esiner.global_decode(arc_scores=score_mat_np, edu_ids=sentence.raw.edu_ids,
                                                sbnds=sentence.raw.sbnds, pbnds=sentence.raw.pbnds,
                                                use_sbnds=True, use_pbnds=False)
            pred_arcs = [arc[0] for arc in sorted(unlabeled_arcs, key=lambda x: x[1])]
            best_tree.append([0] + pred_arcs)
        return best_tree
