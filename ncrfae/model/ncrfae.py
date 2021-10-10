import torch
from torch import nn
import torch.distributions as D

from model import crfae
from model import utility
from model.dep_dataset import SentencePreprocessor, DiscoursePreprocessor
from model.eisner import IncrementalEisnerDecoder

class NCRFAE(nn.Module):
    def __init__(self, preprocessor: DiscoursePreprocessor, tagset_size, embedding_dim, hidden_dim, rnn_layers,
                 dropout_ratio, prior_weight, recons_dim,
                 use_gpu=False, is_multi_root=False, max_dependency_len=10, length_constraint_on_root=False):
        super(NCRFAE, self).__init__()

        self.preprocessor = preprocessor

        self.use_gpu = use_gpu
        self.is_multi_root = is_multi_root
        self.max_dependency_len = max_dependency_len
        self.length_constraint_on_root = length_constraint_on_root

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.word_embeds = nn.Embedding(tagset_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=rnn_layers, bidirectional=True)

        self.dropout1 = nn.Dropout(p=dropout_ratio)
        self.dropout2 = nn.Dropout(p=dropout_ratio)

        self.prior_factor = prior_weight

        self.tagset_size = tagset_size

        self.batch_size = 1  # temp
        self.seq_length = 1  # temp

        # self.multinomial = nn.Parameter(torch.ones(tagset_size, tagset_size) / tagset_size)
        # self.multinomial.requires_grad = False
        self.f2prob1 = nn.Linear(hidden_dim, recons_dim)
        self.f2prob2 = nn.Linear(hidden_dim, recons_dim)

        self.score_layer = nn.Linear(6 * hidden_dim, 1)
        # self.crfae = crfae.CRFAE.apply
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
        utility.init_lstm(self.lstm)
        utility.init_linear(self.score_layer)

    def forward(self, sentences, hidden=None, enable_prior=True):
        sents, mask = self.preprocessor.process_discourse_batch(sentences)
        self.get_batch_seq_size(sents)

        batch_size, max_len, word_len = sents.size()
        edu_length = torch.sum(mask, dim=-1)
        words = self.word_embeds(sents.reshape(-1)).reshape(batch_size, max_len, word_len, -1)
        embeds = torch.sum(words * mask.unsqueeze(-1), dim=2) / (edu_length + (edu_length == 0).long()).unsqueeze(-1)
        # embeds = self.word_embeds(sents)
        d_embeds = self.dropout1(embeds)
        lstm_out, hidden = self.lstm(d_embeds, hidden)  # B * N * H
        d_lstm_out = self.dropout2(lstm_out)  # B * N * H`

        x_next = torch.zeros_like(d_lstm_out)
        x_next[:, 0:self.seq_length - 1, :] = d_lstm_out[:, 1:self.seq_length, :]
        x_next = x_next[:, :, None, :].repeat(1, 1, self.seq_length, 1)

        x_before = torch.zeros_like(d_lstm_out)
        x_before[:, 1:self.seq_length, :] = d_lstm_out[:, 0:self.seq_length - 1, :]
        x_before = x_before[:, :, None, :].repeat(1, 1, self.seq_length, 1)

        x = d_lstm_out[:, :, None, :].repeat(1, 1, self.seq_length, 1)  # B * N * [N] * H
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

        recons_weights = self.get_recons_weight_v2(d_lstm_out)

        joint_weights = crf_weights + recons_weights
        joint_prior_weights = joint_weights + prior_weight

        # log_partition, best_score, best_tree = self.crfae(crf_weights.double(), joint_prior_weights.double(),
        #                                                   self.seq_length, self.batch_size, self.is_multi_root,
        #                                                   self.max_dependency_len, self.use_gpu,
        #                                                   self.length_constraint_on_root)
        # crf_weight_list = torch.split(crf_weights, 1)
        # joint_prior_weight_list = torch.split(joint_prior_weights, 1)
        # self.best_tree = []
        # log_partitions = []
        # for crf_weight, joint_prior_weight in zip(crf_weight_list, joint_prior_weight_list):
        #     edu_ids = [i for i in range(crf_weight.size()[-1])]
        #     # show in the method, here we IGNORE the weight from root to its child
        #     arcs, best_score = self.esiner.decoder.decode(joint_prior_weight.squeeze(0), edu_ids)
        #     # Include the log_partition
        #     log_partition = self.esiner.decoder.summ(crf_weight.squeeze(0), edu_ids)
        #     log_partitions.append(log_partition)
        #     pred_tree = [node[0] for node in sorted(arcs, key=lambda x: x[1])]
        #     self.best_tree.append([-1] + pred_tree)
        # log_partitions = torch.stack(log_partitions, dim=0)
        edu_ids = [i for i in range(crf_weights.size()[-1])]
        log_partitions = self.esiner.decoder.batch_summ(crf_weights, edu_ids)

        # TODO: current implementation is supervised training
        # Supervised HACK
        # joint_prior_weights = joint_prior_weights.double()

        heads = torch.LongTensor([s.heads for s in sentences])
        heads_token_id = heads[:, 1:]
        sent_idx = torch.arange(self.batch_size).contiguous().view(-1, 1).long()
        children_token_id = torch.arange(self.seq_length).contiguous().view(1, -1).long()

        if self.use_gpu:
            sent_idx = sent_idx.cuda()
            heads_token_id = heads_token_id.cuda()
            children_token_id = children_token_id.cuda()

        best_scores = torch.logsumexp(joint_prior_weights[sent_idx, heads_token_id, children_token_id[:, 1:]], dim=1)
        # HACK END
        # return None
        return -(best_scores - log_partitions).mean()

    def decode(self, sentences, hidden=None, enable_prior=True):
        sents, mask = self.preprocessor.process_discourse_batch(sentences)
        self.get_batch_seq_size(sents)

        batch_size, max_len, word_len = sents.size()
        edu_length = torch.sum(mask, dim=-1)
        words = self.word_embeds(sents.reshape(-1)).reshape(batch_size, max_len, word_len, -1)
        embeds = torch.sum(words * mask.unsqueeze(-1), dim=2) / (edu_length + (edu_length == 0).long()).unsqueeze(
            -1)
        # embeds = self.word_embeds(sents)
        d_embeds = self.dropout1(embeds)
        lstm_out, hidden = self.lstm(d_embeds, hidden)  # B * N * H
        d_lstm_out = self.dropout2(lstm_out)  # B * N * H

        x_next = torch.zeros_like(d_lstm_out)
        x_next[:, 0:self.seq_length - 1, :] = d_lstm_out[:, 1:self.seq_length, :]
        x_next = x_next[:, :, None, :].repeat(1, 1, self.seq_length, 1)

        x_before = torch.zeros_like(d_lstm_out)
        x_before[:, 1:self.seq_length, :] = d_lstm_out[:, 0:self.seq_length - 1, :]
        x_before = x_before[:, :, None, :].repeat(1, 1, self.seq_length, 1)

        x = d_lstm_out[:, :, None, :].repeat(1, 1, self.seq_length, 1)  # B * N * [N] * H
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

        recons_weights = self.get_recons_weight_v2(d_lstm_out)

        joint_weights = crf_weights + recons_weights
        joint_prior_weights = joint_weights + prior_weight

        # log_partition, best_score, best_tree = self.crfae(crf_weights.double(), joint_prior_weights.double(),
        #                                                   self.seq_length, self.batch_size, self.is_multi_root,
        #                                                   self.max_dependency_len, self.use_gpu,
        #                                                   self.length_constraint_on_root)
        crf_weight_list = torch.split(crf_weights, 1)
        joint_prior_weight_list = torch.split(joint_prior_weights, 1)
        best_tree = []

        edu_ids = [i for i in range(crf_weights.size()[-1])]
        for crf_weight, joint_prior_weight in zip(crf_weight_list, joint_prior_weight_list):
            arcs, best_score = self.esiner.decoder.decode(joint_prior_weight.squeeze(0), edu_ids)
            pred_tree = [node[0] for node in sorted(arcs, key=lambda x: x[1])]
            best_tree.append([-1] + pred_tree)
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






