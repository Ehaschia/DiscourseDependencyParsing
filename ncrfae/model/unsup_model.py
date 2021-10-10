import os
import pickle

from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Embedding

from model.eisner_v2 import IncrementalEisnerDecoder
from model.module import CHAR_LSTM, MLP, Biaffine, BiLSTM
from model.module.dropout import IndependentDropout, SharedDropout
from treesamplers import TreeSampler, NegativeTreeSampler
import math
from sklearn.cluster import KMeans

from utils import uniform_smoothing, additive_smoothing, sign_smoothing
from utils.common import BERT_DIM

ACTIVATION_DICT = {'relu': F.relu, 'gelu': F.gelu, 'tanh': F.tanh, 'sigmoid': F.sigmoid}

class NCRFAE(nn.Module):
    def __init__(self, args):
        super(NCRFAE, self).__init__()
        self.embedding_dim = BERT_DIM[args.encode_method]
        self.hidden_dim = args.hidden

        self.em2h1 = nn.Linear(self.embedding_dim, args.hidden)
        self.em2h2 = nn.Linear(self.embedding_dim, args.hidden)
        self.activation = ACTIVATION_DICT[args.act_func]
        self.dropout = nn.Dropout(p=args.dropout)

        self.lstm = nn.LSTM(args.hidden, args.hidden // 2,
                            num_layers=args.layers, bidirectional=True)

        self.crie = torch.nn.CrossEntropyLoss(reduction='none')
        self.esiner = IncrementalEisnerDecoder()
        self.sampler = TreeSampler(['RB', 'RB', 'RB'])

        self.decoder_learn = args.decoder_learn

    def init(self):
        pass

    def real_forward(self, sentences):
        r_embed = self.get_batch_embedding(sentences)
        r_embed = torch.stack(r_embed, dim=0).to(self.multinomial.device)

        batch_size, length, _ = r_embed.size()
        # self.get_batch_seq_size(r_embed)

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

        return score_mat

    def init_forward(self, sentences, method='head_selection'):
        score_mat = self.real_forward(sentences)
        encode_p = torch.log_softmax(score_mat, dim=2)
        decode_p = torch.log_softmax(score_mat, dim=1)
        ae_p = (encode_p + decode_p).permute(0, 2, 1)
        loss = 0.0
        batch, length = score_mat.size()[:2]
        for score_mat, sentence in zip(ae_p, sentences):
            arcs = self.sampler.sample(sentence.raw.edu_ids,
                                       sentence.raw.edus,
                                       sentence.raw.edus_head,
                                       sentence.raw.sbnds,
                                       None,
                                       has_root=True)
            for arc in arcs:
                loss += score_mat[arc[0]][arc[1]]
        return -1.0 * loss / batch

    def forward(self, sentences, method='head_selection'):
        score_mat = self.real_forward(sentences)
        batch, length = score_mat.size()[:2]
        encode_p = torch.log_softmax(score_mat, dim=2)
        decode_p = torch.log_softmax(score_mat, dim=1)
        ae_p = (encode_p + decode_p).permute(0, 2, 1)
        loss = 0.0
        for score_mat, sentence in zip(ae_p, sentences):
            # loss += self.esiner.partition(arc_scores=score_mat, edu_ids=sentence.raw.edu_ids,
            #                               sbnds=sentence.raw.sbnds, pbnds=sentence.raw.pbnds,
            #                               use_sbnds=True, use_pbnds=False)
            unlabeled_arcs = self.esiner.global_decode(arc_scores=score_mat.cpu().detach().numpy(),
                                                       edu_ids=sentence.raw.edu_ids,
                                                       sbnds=sentence.raw.sbnds, pbnds=sentence.raw.pbnds,
                                                       use_sbnds=True, use_pbnds=False)
            for arc in unlabeled_arcs:
                loss += score_mat[arc[0]][arc[1]]
        return -1.0 * loss / batch

    def decoding(self, sentences):
        batch_score_mat = self.real_forward(sentences)
        best_tree = []
        batch_score_mat_np = batch_score_mat.permute(0, 2, 1).detach().cpu().numpy()
        for idx, (sentence, score_mat_np) in enumerate(zip(sentences, batch_score_mat_np)):
            unlabeled_arcs = self.esiner.global_decode(arc_scores=score_mat_np, edu_ids=sentence.raw.edu_ids,
                                                       sbnds=sentence.raw.sbnds, pbnds=sentence.raw.pbnds,
                                                       use_sbnds=True, use_pbnds=False)
            # pred_arcs = [arc[0] for arc in sorted(unlabeled_arcs, key=lambda x: x[1])]
            pred_arcs = [arc[0] for arc in sorted(unlabeled_arcs, key=lambda x: x[1])]
            best_tree.append([0] + pred_arcs)
        return best_tree

    @staticmethod
    def get_batch_embedding(sentences) -> List[torch.Tensor]:
        return [sentence.r_embed for sentence in sentences]



class KMeansNCRFAE(NCRFAE):

    def __init__(self, args):
        super(KMeansNCRFAE, self).__init__(args)
        self.args = args
        self.kcluster_embed = Embedding(self.args.kcluster, self.embedding_dim)
        self.kmeans = None
        self.multinomial = nn.Parameter(torch.ones(self.args.kcluster, self.args.kcluster))

        self.negativesampler = NegativeTreeSampler()

    @torch.no_grad()
    def calculate_kmeans(self, dataset):
        kmeans_path = './data/kmeans/' +self.args.corpus+'_'+self.args.encode_method+'_'+str(self.args.kcluster)+'.pkl'
        if os.path.exists(kmeans_path):
            self.kmeans = pickle.load(open(kmeans_path, 'rb'))
        else:
            self.eval()
            batch_encode = self.get_batch_embedding(dataset)
            batch_encode_np = torch.cat(batch_encode, dim=0).cpu().detach().numpy()
            self.kmeans = KMeans(n_clusters=self.args.kcluster, random_state=self.args.random_seed).fit(batch_encode_np)
            pickle.dump(self.kmeans, open(kmeans_path, 'wb'))

    @torch.no_grad()
    def kmeans_label_predict(self, dataset):
        self.eval()
        for sentence in dataset:
            # assert isinstance(sentence, SCIDTBSentence)
            edu_rep_np = sentence.r_embed.cpu().detach().numpy()
            labels = self.kmeans.predict(edu_rep_np)
            setattr(sentence, 'kmeans_labels', labels)

    def real_forward(self, sentences):
        if self.args.word_embed:
            r_embed = self.get_batch_embedding(sentences)
            r_embed = torch.stack(r_embed, dim=0).to(self.multinomial.device)
            batch_size, length, _ = r_embed.size()
        else:
            kmeans_labels = torch.tensor([getattr(sentence, 'kmeans_labels', None)
                                          for sentence in sentences]).long().to(self.multinomial.device)
            batch_size, length = kmeans_labels.size()
            r_embed = self.kcluster_embed(kmeans_labels)
        # batch_size, length, _ = r_embed.size()
        # self.get_batch_seq_size(r_embed)

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

        return score_mat

    # def init_forward(self, sentences):
    #     score_mat = self.real_forward(sentences)
    #     encoder_p = torch.log_softmax(score_mat, dim=2).permute(0, 2, 1).cpu()
    #     loss = 0.0
    #     # partitions = self.esiner.batch_summer(encoder_p, sentences[0].raw.edu_ids)
    #     for p_mat, sentence in zip(encoder_p, sentences):
    #         labels = getattr(sentence, 'kmeans_labels', None)
    #         assert labels is not None
    #         # rec_weight = torch.log_softmax(self.multinomial, dim=1)[labels[:, None], labels[None, :]]
    #         rec_weight = self.multinomial[labels[:, None], labels[None, :]].cpu()
    #         partition = self.esiner.partition(arc_scores=p_mat, edu_ids=sentence.raw.edu_ids,
    #                                           sbnds=sentence.raw.sbnds, pbnds=sentence.raw.pbnds,
    #                                           use_sbnds=True, use_pbnds=True)
    #
    #         joint_weight = p_mat + rec_weight
    #         best_tree = self.sampler.sample(sentence.raw.edu_ids,
    #                                         sentence.raw.edus,
    #                                         sentence.raw.edus_head,
    #                                         sentence.raw.sbnds,
    #                                         None,
    #                                         has_root=True)
    #         best_score = 0.0
    #         for arc in best_tree:
    #             best_score += joint_weight[arc[0]][arc[1]]
    #         loss += -(best_score - partition)

    #   return loss / len(sentences)

    def init_forward(self, sentences, method='head_selection'):
        return getattr(self, 'init_forward_' + method)(sentences)
        # return self.init_forward_head_selection(sentences)

    def forward(self, sentences, method='forward_head_selection'):
        return getattr(self, method)(sentences)

    def forward_partition(self, sentences):
        score_mat = self.real_forward(sentences)
        encoder_p = torch.log_softmax(score_mat, dim=-1).permute(0, 2, 1)
        supervised_loss = 0.0
        unsupervised_loss = 0.0
        supervised_cnt = 0
        #  partitions = self.esiner.batch_summer(encoder_p, sentences[0].raw.edu_ids)
        for p_mat, sentence in zip(encoder_p, sentences):
            labels = getattr(sentence, 'kmeans_labels', None)
            assert labels is not None
            # rec_weight = torch.log_softmax(self.multinomial, dim=1)[labels[:, None], labels[None, :]].permute(1, 0)
            rec_weight = self.multinomial[labels[:, None], labels[None, :]]# .cpu()
            partition = self.esiner.partition(arc_scores=p_mat, edu_ids=sentence.raw.edu_ids,
                                              sbnds=sentence.raw.sbnds, pbnds=sentence.raw.pbnds,
                                              use_sbnds=True, use_pbnds=False)

            joint_weight = p_mat + rec_weight
            if sentence.supervised:
                best_tree = sentence.raw.arcs
            else:
                best_tree = self.esiner.global_decode(arc_scores=joint_weight.detach().cpu().numpy(),
                                                      edu_ids=sentence.raw.edu_ids,
                                                      sbnds=sentence.raw.sbnds,
                                                      pbnds=sentence.raw.pbnds,
                                                      use_sbnds=True, use_pbnds=False)
            best_score = 0.0
            for arc in best_tree:
                best_score += joint_weight[arc[0]][arc[1]]
            if sentence.supervised:
                supervised_loss += -(best_score - partition)
                supervised_cnt += 1
            else:
                unsupervised_loss += -(best_score - partition)
        # supervised and unsupervised
        if supervised_cnt != 0 and supervised_cnt != len(sentences):
            loss = supervised_loss / supervised_cnt + \
                   self.args.unsupervised_weight * unsupervised_loss / (len(sentences) - supervised_cnt)
        # only supervised
        elif supervised_cnt != 0:
            loss = supervised_loss / supervised_cnt
        # only unsupervised
        else:
            loss = self.args.unsupervised_weight * unsupervised_loss / (len(sentences) - supervised_cnt)
        return loss

    def init_forward_partition(self, sentences):
        score_mat = self.real_forward(sentences)
        encoder_p = torch.log_softmax(score_mat, dim=2).permute(0, 2, 1)
        supervised_loss = 0.0
        unsupervised_loss = 0.0
        supervised_cnt = 0

        # partitions = self.esiner.batch_summer(encoder_p, sentences[0].raw.edu_ids)
        for p_mat, sentence in zip(encoder_p, sentences):
            labels = getattr(sentence, 'kmeans_labels', None)
            assert labels is not None
            # rec_weight = torch.log_softmax(self.multinomial, dim=1)[labels[:, None], labels[None, :]].permute(1, 0)
            rec_weight = self.multinomial[labels[:, None], labels[None, :]]# .cpu()
            partition = self.esiner.partition(arc_scores=p_mat, edu_ids=sentence.raw.edu_ids,
                                              sbnds=sentence.raw.sbnds, pbnds=sentence.raw.pbnds,
                                              use_sbnds=True, use_pbnds=True)

            joint_weight = p_mat + rec_weight
            if sentence.supervised:
                best_tree = sentence.raw.arcs
            else:
                best_tree = self.sampler.sample(sentence.raw.edu_ids,
                                                sentence.raw.edus,
                                                sentence.raw.edus_head,
                                                sentence.raw.sbnds,
                                                sentence.raw.pbnds,
                                                has_root=True)
            best_score = 0.0
            for arc in best_tree:
                best_score += joint_weight[arc[0]][arc[1]]
            if sentence.supervised:
                supervised_loss += -(best_score - partition)
                supervised_cnt += 1
            else:
                unsupervised_loss += -(best_score - partition)
        # supervised and unsupervised
        if supervised_cnt != 0 and supervised_cnt != len(sentences):
            loss = supervised_loss / supervised_cnt + \
                   self.args.unsupervised_weight * unsupervised_loss / (len(sentences) - supervised_cnt)
        # only supervised
        elif supervised_cnt != 0:
            loss = supervised_loss / supervised_cnt
        # only unsupervised
        else:
            loss = self.args.unsupervised_weight * unsupervised_loss / (len(sentences) - supervised_cnt)
        return loss

        # return self.forward_head_selection(sentences)
    def init_forward_head_selection(self, sentences):
        batch_encoder_mat = self.real_forward(sentences)
        batch_encoder_mat_prob = torch.log_softmax(batch_encoder_mat, dim=2)
        multinomial_prob = torch.log_softmax(self.multinomial, dim=-1)
        batch, length = batch_encoder_mat.size()[:2]
        all_score_mat = []
        gold = []
        weight = [self.args.unsupervised_weight if not sentence.supervised else 1.0 for sentence in sentences]
        for idx, sentence in enumerate(sentences):
            labels = getattr(sentence, 'kmeans_labels', None)
            if sentence.supervised:
                unlabeled_arcs = sentence.raw.arcs
            else:
                unlabeled_arcs = self.sampler.sample(sentence.raw.edu_ids,
                                                     sentence.raw.edus,
                                                     sentence.raw.edus_head,
                                                     sentence.raw.sbnds,
                                                     sentence.raw.pbnds,
                                                     has_root=True)
            pred_arcs = [arc[0] for arc in sorted(unlabeled_arcs, key=lambda x: x[1])]
            gold.append([0] + pred_arcs)
            all_score_mat.append(multinomial_prob[labels[:, None], labels[None, :]] + batch_encoder_mat_prob[idx])
        all_score_mat = torch.stack(all_score_mat, dim=0)
        gold = torch.from_numpy(np.array(gold, dtype=int)).long().to(all_score_mat.device)
        loss = self.crie(all_score_mat.view(-1, length), gold.view(-1)).view(batch, length)[:, 1:].mean(dim=1)
        return (torch.tensor(weight).to(loss.device) * loss).mean()

    def forward_head_selection(self, sentences):
        batch_encoder_mat = self.real_forward(sentences)
        batch_score_mat_prob = torch.log_softmax(batch_encoder_mat, dim=2)
        multinomial_prob = torch.log_softmax(self.multinomial, dim=-1)
        batch, length = batch_encoder_mat.size()[:2]
        all_score_mat = []
        gold = []
        # batch_score_mat_np = score_mat.permute(0, 2, 1).detach().cpu().numpy()
        weight = [self.args.unsupervised_weight if not sentence.supervised else 1.0 for sentence in sentences]
        for idx, sentence in enumerate(sentences):
            labels = getattr(sentence, 'kmeans_labels', None)
            all_score_mat.append(multinomial_prob[labels[:, None], labels[None, :]] + batch_score_mat_prob[idx])
            if sentence.supervised:
                unlabeled_arcs = sentence.raw.arcs
                pred_heads = torch.tensor([0] + [arc[0] for arc in sorted(unlabeled_arcs, key=lambda x: x[1])]).to(batch_encoder_mat.device)
            else:
                pred_heads = torch.argmax(all_score_mat[idx], dim=-1)
            gold.append(pred_heads)
        gold = torch.stack(gold, dim=0)
        all_score_mat = torch.stack(all_score_mat)
        loss = self.crie(all_score_mat.view(-1, length), gold.view(-1)).view(batch, length)[:, 1:].mean(dim=1)
        return (torch.tensor(weight).to(loss.device) * loss).mean()
    # def forward(self, sentences):
    #     score_mat = self.real_forward(sentences)
    #     encoder_p = torch.log_softmax(score_mat, dim=-1).permute(0, 2, 1).cpu()
    #     loss = 0.0
    #     #  partitions = self.esiner.batch_summer(encoder_p, sentences[0].raw.edu_ids)
    #     for p_mat, sentence in zip(encoder_p, sentences):
    #         labels = getattr(sentence, 'kmeans_labels', None)
    #         assert labels is not None
    #         # rec_weight = torch.log_softmax(self.multinomial, dim=1)[labels[:, None], labels[None, :]]
    #         rec_weight = self.multinomial[labels[:, None], labels[None, :]].cpu()
    #         partition = self.esiner.partition(arc_scores=p_mat, edu_ids=sentence.raw.edu_ids,
    #                                           sbnds=sentence.raw.sbnds, pbnds=sentence.raw.pbnds,
    #                                           use_sbnds=True, use_pbnds=False)
    #
    #         joint_weight = p_mat + rec_weight
    #         best_tree = self.esiner.global_decode(arc_scores=joint_weight.detach().cpu().numpy(),
    #                                               edu_ids=sentence.raw.edu_ids,
    #                                               sbnds=sentence.raw.sbnds,
    #                                               pbnds=sentence.raw.pbnds,
    #                                               use_sbnds=True, use_pbnds=False)
    #         best_score = 0.0
    #         for arc in best_tree:
    #             best_score += joint_weight[arc[0]][arc[1]]
    #         loss += -(best_score - partition)
    #     return loss / len(sentences)

    def init_embedding(self):
        if self.kmeans is None:
            return
        cluster_centers = self.kmeans.cluster_centers_
        self.kcluster_embed.weight.data = torch.from_numpy(cluster_centers).to(self.kcluster_embed.weight.device)
        self.kcluster_embed.requires_grad_(False)

    # def decoding(self, sentences):
    #     batch_score_mat = self.real_forward(sentences)
    #     best_tree = []
    #     batch_score_mat_np = batch_score_mat.permute(0, 2, 1).detach().cpu().numpy()
    #     for idx, (sentence, score_mat_np) in enumerate(zip(sentences, batch_score_mat_np)):
    #         unlabeled_arcs = self.esiner.global_decode(arc_scores=score_mat_np, edu_ids=sentence.raw.edu_ids,
    #                                                    sbnds=sentence.raw.sbnds, pbnds=sentence.raw.pbnds,
    #                                                    use_sbnds=True, use_pbnds=False)
    #         pred_arcs = [arc[0] for arc in sorted(unlabeled_arcs, key=lambda x: x[1])]
    #         best_tree.append([0] + pred_arcs)
    #     return best_tree

    def decoding(self, sentences):
        batch_encoder_mat = self.real_forward(sentences)
        best_tree = []
        multinomial_prob = torch.log_softmax(self.multinomial, dim=-1)
        batch_encoder_mat_np = batch_encoder_mat.permute(0, 2, 1).detach().cpu().numpy()
        multinomial_prob_np = multinomial_prob.detach().cpu().numpy()
        # multinomial_prob_np = multinomial_prob.detach().cpu().permute(0, 1).numpy()

        for idx, (sentence, encoder_mat_np) in enumerate(zip(sentences, batch_encoder_mat_np)):
            labels = getattr(sentence, 'kmeans_labels', None)
            all_score_np = encoder_mat_np + multinomial_prob_np[labels[:, None], labels[None, :]]
            unlabeled_arcs = self.esiner.global_decode(arc_scores=all_score_np, edu_ids=sentence.raw.edu_ids,
                                                       sbnds=sentence.raw.sbnds, pbnds=sentence.raw.pbnds,
                                                       use_sbnds=True, use_pbnds=False)
            pred_arcs = [arc[0] for arc in sorted(unlabeled_arcs, key=lambda x: x[1])]
            best_tree.append([0] + pred_arcs)
        return best_tree

    def set_prob(self, prob):
        # log and replace zero
        prob = np.log(prob)
        for i in range(self.args.kcluster):
            for j in range(self.args.kcluster):
                if np.abs(prob[i][j]) == np.inf or math.isnan(prob[i][j]):
                    prob[i][j] = -1e5
        self.multinomial.data = torch.from_numpy(prob).float().to(self.multinomial.device)
        self.multinomial.requires_grad = self.decoder_learn

    def get_rightbranching_prob_mat(self, dataset):
        prob_mat = [[0] * self.args.kcluster for _ in range(self.args.kcluster)]
        for inst in dataset:
            labels = np.array(inst.kmeans_labels).astype(int)
            arcs = self.sampler.sample(inst.raw.edu_ids, inst.raw.edus, inst.raw.edus_head,
                                       inst.raw.sbnds, inst.raw.pbnds, has_root=True)
            for arc in arcs:
                id1 = labels[arc[0]]
                id2 = labels[arc[1]]
                prob_mat[id1][id2] += 1
        # to prob and smooth
        for i in range(self.args.kcluster):
            i_all = float(sum(prob_mat[i]))
            if i_all == 0:
                # prob_mat[i] = additive_smoothing(prob_mat[i], alpha)
                continue
            else:
                if self.args.smooth == 'additive':
                    prob_mat[i] = additive_smoothing(prob_mat[i], self.args.alpha)
                elif self.args.smooth == 'sign':
                    prob_mat[i] = sign_smoothing(prob_mat[i], self.args.alpha)
        # uniform smoothing
        if self.args.smooth == 'uniform':
            prob_mat = uniform_smoothing(prob_mat, self.args.alpha)
        return prob_mat

    def get_gold_prob_mat(self, dataset):
        prob_mat = [[0] * self.args.kcluster for _ in range(self.args.kcluster)]
        for inst in dataset:
            labels = np.array(inst.kmeans_labels).astype(int)
            arcs = inst.raw.arcs
            for arc in arcs:
                id1 = labels[arc[0]]
                id2 = labels[arc[1]]
                prob_mat[id1][id2] += 1
        # to prob and smooth
        for i in range(self.args.kcluster):
            i_all = float(sum(prob_mat[i]))
            if i_all == 0:
                # prob_mat[i] = additive_smoothing(prob_mat[i], alpha)
                continue
            else:
                if self.args.smooth == 'additive':
                    prob_mat[i] = additive_smoothing(prob_mat[i], self.args.alpha)
                elif self.args.smooth == 'sign':
                    prob_mat[i] = sign_smoothing(prob_mat[i], self.args.alpha)
                # uniform smoothing
        if self.args.smooth == 'uniform':
            prob_mat = uniform_smoothing(prob_mat, self.args.alpha)
        return prob_mat

class KMeansBiaffineNCRFAE(nn.Module):

    def __init__(self, args):
        super(KMeansBiaffineNCRFAE, self).__init__()
        self.args = args
        self.embedding_dim = BERT_DIM[args.encode_method]
        self.kcluster_embed = Embedding(self.args.kcluster, self.embedding_dim)
        self.kmeans = None
        self.multinomial = nn.Parameter(torch.ones(self.args.kcluster, self.args.kcluster))
        self.sampler = TreeSampler(['RB', 'RB', 'RB'])

        self.embed_dropout = IndependentDropout(p=args.embed_dropout)

        # the word-lstm layer
        self.lstm = BiLSTM(input_size=BERT_DIM[args.encode_method],
                           hidden_size=args.n_lstm_hidden,
                           num_layers=args.n_lstm_layers,
                           dropout=args.lstm_dropout)

        self.lstm_dropout = SharedDropout(p=args.lstm_dropout)

        # the MLP layers
        self.mlp_arc_h = MLP(n_in=args.n_lstm_hidden * 2,
                             n_hidden=args.n_mlp_arc,
                             dropout=args.mlp_dropout)
        self.mlp_arc_d = MLP(n_in=args.n_lstm_hidden * 2,
                             n_hidden=args.n_mlp_arc,
                             dropout=args.mlp_dropout)

        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=args.n_mlp_arc,
                                 bias_x=False,
                                 bias_y=False,
                                 identity=self.args.identity_biaffine)

        self.esiner = IncrementalEisnerDecoder()
        self.crie = nn.CrossEntropyLoss(reduction='none')

        self.negativesampler = NegativeTreeSampler()

    @torch.no_grad()
    def calculate_kmeans(self, dataset):
        kmeans_path = './data/kmeans/' +self.args.corpus+'_'+self.args.encode_method+'_'+str(self.args.kcluster)+'.pkl'
        if os.path.exists(kmeans_path):
            self.kmeans = pickle.load(open(kmeans_path, 'rb'))
        else:
            self.eval()
            batch_encode = self.get_batch_embedding(dataset)
            batch_encode_np = torch.cat(batch_encode, dim=0).cpu().detach().numpy()
            self.kmeans = KMeans(n_clusters=self.args.kcluster, random_state=self.args.random_seed).fit(batch_encode_np)
            pickle.dump(self.kmeans, open(kmeans_path, 'wb'))

    @torch.no_grad()
    def kmeans_label_predict(self, dataset):
        self.eval()
        for sentence in dataset:
            # assert isinstance(sentence, SCIDTBSentence)
            edu_rep_np = sentence.r_embed.cpu().detach().numpy()
            labels = self.kmeans.predict(edu_rep_np)
            setattr(sentence, 'kmeans_labels', labels)

    def real_forward(self, sentences):
        if self.args.word_embed:
            embed = self.get_batch_embedding(sentences)
            embed = torch.stack(embed, dim=0).to(self.multinomial.device)
            batch_size, length, _ = embed.size()
        else:
            kmeans_labels = torch.tensor([getattr(sentence, 'kmeans_labels', None)
                                          for sentence in sentences]).long().to(self.multinomial.device)
            batch_size, length = kmeans_labels.size()
            embed = self.kcluster_embed(kmeans_labels)

        tmp_len = len(sentences[0].raw.edu_ids)

        lens = torch.tensor([tmp_len] * batch_size)

        x = nn.utils.rnn.pack_padded_sequence(embed, lens, True, False)
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, True, total_length=tmp_len)
        x = self.lstm_dropout(x)

        arc_h = self.mlp_arc_h(x)  # [batch_size, seq_len, d]
        arc_d = self.mlp_arc_d(x)

        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        # s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))
        return s_arc
    #
    # def init_forward_partition(self, sentences):
    #     score_mat = self.real_forward(sentences)
    #     encoder_p = torch.log_softmax(score_mat, dim=2).permute(0, 2, 1)
    #     loss = 0.0
    #     # partitions = self.esiner.batch_summer(encoder_p, sentences[0].raw.edu_ids)
    #     for p_mat, sentence in zip(encoder_p, sentences):
    #         labels = getattr(sentence, 'kmeans_labels', None)
    #         assert labels is not None
    #         rec_weight = torch.log_softmax(self.multinomial, dim=1)[labels[:, None], labels[None, :]].permute(1, 0)
    #         # rec_weight = self.multinomial[labels[:, None], labels[None, :]].cpu()
    #         partition = self.esiner.partition(arc_scores=p_mat, edu_ids=sentence.raw.edu_ids,
    #                                           sbnds=sentence.raw.sbnds, pbnds=sentence.raw.pbnds,
    #                                           use_sbnds=True, use_pbnds=True)
    #
    #         joint_weight = p_mat + rec_weight
    #         best_tree = self.sampler.sample(sentence.raw.edu_ids,
    #                                         sentence.raw.edus,
    #                                         sentence.raw.edus_head,
    #                                         sentence.raw.sbnds,
    #                                         sentence.raw.pbnds,
    #                                         has_root=True)
    #         best_score = 0.0
    #         for arc in best_tree:
    #             best_score += joint_weight[arc[0]][arc[1]]
    #         loss += -(best_score - partition)
    #
    #     return loss / len(sentences)
    #
    #
    # def forward_partition(self, sentences):
    #     score_mat = self.real_forward(sentences)
    #     encoder_p = torch.log_softmax(score_mat, dim=-1).permute(0, 2, 1)
    #     loss = 0.0
    #     #  partitions = self.esiner.batch_summer(encoder_p, sentences[0].raw.edu_ids)
    #     for p_mat, sentence in zip(encoder_p, sentences):
    #         labels = getattr(sentence, 'kmeans_labels', None)
    #         assert labels is not None
    #         rec_weight = torch.log_softmax(self.multinomial, dim=1)[labels[:, None], labels[None, :]].permute(1, 0)
    #         # rec_weight = self.multinomial[labels[:, None], labels[None, :]].cpu()
    #         partition = self.esiner.partition(arc_scores=p_mat, edu_ids=sentence.raw.edu_ids,
    #                                           sbnds=sentence.raw.sbnds, pbnds=sentence.raw.pbnds,
    #                                           use_sbnds=True, use_pbnds=False)
    #
    #         joint_weight = p_mat + rec_weight
    #         best_tree = self.esiner.global_decode(arc_scores=joint_weight.detach().cpu().numpy(),
    #                                               edu_ids=sentence.raw.edu_ids,
    #                                               sbnds=sentence.raw.sbnds,
    #                                               pbnds=sentence.raw.pbnds,
    #                                               use_sbnds=True, use_pbnds=False)
    #         best_score = 0.0
    #         for arc in best_tree:
    #             best_score += joint_weight[arc[0]][arc[1]]
    #         loss += -(best_score - partition)
    #     return loss / len(sentences)
    #
    #     # return self.init_forward_head_selection(sentences)


    def forward_partition(self, sentences):
        score_mat = self.real_forward(sentences)
        encoder_p = torch.log_softmax(score_mat, dim=-1).permute(0, 2, 1)
        supervised_loss = 0.0
        unsupervised_loss = 0.0
        supervised_cnt = 0
        #  partitions = self.esiner.batch_summer(encoder_p, sentences[0].raw.edu_ids)
        for p_mat, sentence in zip(encoder_p, sentences):
            labels = getattr(sentence, 'kmeans_labels', None)
            assert labels is not None
            # rec_weight = torch.log_softmax(self.multinomial, dim=1)[labels[:, None], labels[None, :]].permute(1, 0)
            rec_weight = self.multinomial[labels[:, None], labels[None, :]]# .cpu()
            partition = self.esiner.partition(arc_scores=p_mat, edu_ids=sentence.raw.edu_ids,
                                              sbnds=sentence.raw.sbnds, pbnds=sentence.raw.pbnds,
                                              use_sbnds=True, use_pbnds=False)

            joint_weight = p_mat + rec_weight

            if sentence.supervised:
                best_tree = sentence.raw.arcs
            else:
                best_tree = self.esiner.global_decode(arc_scores=joint_weight.detach().cpu().numpy(),
                                                      edu_ids=sentence.raw.edu_ids,
                                                      sbnds=sentence.raw.sbnds,
                                                      pbnds=sentence.raw.pbnds,
                                                      use_sbnds=True, use_pbnds=False)
            best_score = 0.0

            for arc in best_tree:
                best_score += joint_weight[arc[0]][arc[1]]
            if sentence.supervised:
                supervised_loss += -(best_score - partition)
                supervised_cnt += 1
            else:
                unsupervised_loss += -(best_score - partition)
        # supervised and unsupervised
        if supervised_cnt != 0 and supervised_cnt != len(sentences):
            loss = supervised_loss / supervised_cnt + \
                   self.args.unsupervised_weight * unsupervised_loss / (len(sentences) - supervised_cnt)
        # only supervised
        elif supervised_cnt != 0:
            loss = supervised_loss / supervised_cnt
        # only unsupervised
        else:
            loss = self.args.unsupervised_weight * unsupervised_loss / (len(sentences) - supervised_cnt)
        return loss

    def init_forward_partition(self, sentences):
        score_mat = self.real_forward(sentences)
        encoder_p = torch.log_softmax(score_mat, dim=2).permute(0, 2, 1)
        # partitions = self.esiner.batch_summer(encoder_p, sentences[0].raw.edu_ids)
        supervised_loss = 0.0
        unsupervised_loss = 0.0
        supervised_cnt = 0

        for p_mat, sentence in zip(encoder_p, sentences):
            labels = getattr(sentence, 'kmeans_labels', None)
            assert labels is not None
            # rec_weight = torch.log_softmax(self.multinomial, dim=1)[labels[:, None], labels[None, :]].permute(1, 0)
            rec_weight = self.multinomial[labels[:, None], labels[None, :]]# .cpu()
            partition = self.esiner.partition(arc_scores=p_mat, edu_ids=sentence.raw.edu_ids,
                                              sbnds=sentence.raw.sbnds, pbnds=sentence.raw.pbnds,
                                              use_sbnds=True, use_pbnds=True)

            joint_weight = p_mat + rec_weight
            if sentence.supervised:
                best_tree = sentence.raw.arcs
            else:
                best_tree = self.sampler.sample(sentence.raw.edu_ids,
                                                sentence.raw.edus,
                                                sentence.raw.edus_head,
                                                sentence.raw.sbnds,
                                                sentence.raw.pbnds,
                                                has_root=True)
            best_score = 0.0
            for arc in best_tree:
                best_score += joint_weight[arc[0]][arc[1]]

            if sentence.supervised:
                supervised_loss += -(best_score - partition)
                supervised_cnt += 1
            else:
                unsupervised_loss += -(best_score - partition)
        # supervised and unsupervised
        if supervised_cnt != 0 and supervised_cnt != len(sentences):
            loss = supervised_loss / supervised_cnt + \
                   self.args.unsupervised_weight * unsupervised_loss / (len(sentences) - supervised_cnt)
        # only supervised
        elif supervised_cnt != 0:
            loss = supervised_loss / supervised_cnt
        # only unsupervised
        else:
            loss = self.args.unsupervised_weight * unsupervised_loss / (len(sentences) - supervised_cnt)
        return loss / len(sentences)

    def forward(self, sentences, method='forward_head_selection'):
        return getattr(self, method)(sentences)

        # return self.forward_head_selection(sentences)
    def init_forward_head_selection(self, sentences):
        batch_encoder_mat = self.real_forward(sentences)
        batch_encoder_mat_prob = torch.log_softmax(batch_encoder_mat, dim=2)
        multinomial_prob = torch.log_softmax(self.multinomial, dim=-1)
        batch, length = batch_encoder_mat.size()[:2]
        all_score_mat = []
        gold = []
        weight = [self.args.unsupervised_weight if not sentence.supervised else 1.0 for sentence in sentences]

        for idx, sentence in enumerate(sentences):
            labels = getattr(sentence, 'kmeans_labels', None)
            if sentence.supervised:
                unlabeled_arcs = sentence.raw.arcs
            else:
                unlabeled_arcs = self.sampler.sample(sentence.raw.edu_ids,
                                                     sentence.raw.edus,
                                                     sentence.raw.edus_head,
                                                     sentence.raw.sbnds,
                                                     sentence.raw.pbnds,
                                                     has_root=True)
            pred_arcs = [arc[0] for arc in sorted(unlabeled_arcs, key=lambda x: x[1])]
            gold.append([0] + pred_arcs)
            all_score_mat.append(multinomial_prob[labels[:, None], labels[None, :]] + batch_encoder_mat_prob[idx])
        all_score_mat = torch.stack(all_score_mat, dim=0)
        gold = torch.from_numpy(np.array(gold, dtype=int)).long().to(all_score_mat.device)
        loss = self.crie(all_score_mat.view(-1, length), gold.view(-1)).view(batch, length)[:, 1:].mean(dim=1)
        return (torch.tensor(weight).to(loss.device) * loss).mean()

    def forward_head_selection(self, sentences):
        batch_encoder_mat = self.real_forward(sentences)
        batch_score_mat_prob = torch.log_softmax(batch_encoder_mat, dim=2)
        multinomial_prob = torch.log_softmax(self.multinomial, dim=-1)
        batch, length = batch_encoder_mat.size()[:2]
        all_score_mat = []
        gold = []
        # batch_score_mat_np = score_mat.permute(0, 2, 1).detach().cpu().numpy()
        weight = [self.args.unsupervised_weight if not sentence.supervised else 1.0 for sentence in sentences]
        for idx, sentence in enumerate(sentences):
            labels = getattr(sentence, 'kmeans_labels', None)
            all_score_mat.append(multinomial_prob[labels[:, None], labels[None, :]] + batch_score_mat_prob[idx])
            if sentence.supervised:
                unlabeled_arcs = sentence.raw.arcs
                pred_heads = torch.tensor([0] + [arc[0] for arc in sorted(unlabeled_arcs, key=lambda x: x[1])]).to(batch_encoder_mat.device)
            else:
                pred_heads = torch.argmax(all_score_mat[idx], dim=-1)
            gold.append(pred_heads)
        gold = torch.stack(gold, dim=0)
        all_score_mat = torch.stack(all_score_mat)

        loss = self.crie(all_score_mat.view(-1, length), gold.view(-1)).view(batch, length)[:, 1:].mean(dim=1)
        return (torch.tensor(weight).to(loss.device) * loss).mean()

    # init for tree restriction method is same with head selection
    def init_forward_tree_restriction(self, sentences):
        batch_encoder_mat = self.real_forward(sentences)
        if self.args.encode_prob:
            batch_encoder_mat = torch.log_softmax(batch_encoder_mat, dim=2)
        multinomial_prob = torch.log_softmax(self.multinomial, dim=-1)
        batch, length = batch_encoder_mat.size()[:2]
        all_score_mat = []
        gold = []
        weight = [self.args.unsupervised_weight if not sentence.supervised else 1.0 for sentence in sentences]
        for idx, sentence in enumerate(sentences):
            labels = getattr(sentence, 'kmeans_labels', None)
            if sentence.supervised:
                unlabeled_arcs = sentence.raw.arcs
            else:
                unlabeled_arcs = self.sampler.sample(sentence.raw.edu_ids,
                                                     sentence.raw.edus,
                                                     sentence.raw.edus_head,
                                                     sentence.raw.sbnds,
                                                     sentence.raw.pbnds,
                                                     has_root=True)
            pred_arcs = [arc[0] for arc in sorted(unlabeled_arcs, key=lambda x: x[1])]
            gold.append([0] + pred_arcs)
            all_score_mat.append(multinomial_prob[labels[:, None], labels[None, :]] + batch_encoder_mat[idx])
        all_score_mat = torch.stack(all_score_mat, dim=0)
        gold = torch.from_numpy(np.array(gold, dtype=int)).long().to(all_score_mat.device)
        loss = self.crie(all_score_mat.view(-1, length), gold.view(-1)).view(batch, length)[:, 1:].mean(dim=1)
        return (torch.tensor(weight).to(loss.device) * loss).mean()

    def forward_tree_restriction(self, sentences):
        batch_encoder_mat = self.real_forward(sentences)
        if self.args.encode_prob:
            batch_encoder_mat = torch.log_softmax(batch_encoder_mat, dim=2)
        multinomial_prob = torch.log_softmax(self.multinomial, dim=-1)
        batch, length = batch_encoder_mat.size()[:2]
        gold = []
        all_score_mat = []
        batch_encoder_mat_np = batch_encoder_mat.permute(0, 2, 1).detach().cpu().numpy()
        multinomial_prob_np = multinomial_prob.detach().cpu().numpy()
        weight = [self.args.unsupervised_weight if not sentence.supervised else 1.0 for sentence in sentences]

        for idx, sentence in enumerate(sentences):
            labels = getattr(sentence, 'kmeans_labels', None)
            all_score_mat_np = multinomial_prob_np[labels[:, None], labels[None, :]] + batch_encoder_mat_np[idx]
            if sentence.supervised:
                pred_tree = sentence.raw.arcs
            else:
                pred_tree = self.esiner.global_decode(arc_scores=all_score_mat_np,
                                                      edu_ids=sentence.raw.edu_ids,
                                                      sbnds=sentence.raw.sbnds,
                                                      pbnds=sentence.raw.pbnds,
                                                      use_sbnds=True, use_pbnds=True)
            pred_arcs = [arc[0] for arc in sorted(pred_tree, key=lambda x: x[1])]
            gold.append([0] + pred_arcs)
            all_score_mat.append(multinomial_prob[labels[:, None], labels[None, :]] + batch_encoder_mat[idx])
        all_score_mat = torch.log_softmax(torch.stack(all_score_mat, dim=0), dim=1)
        gold = torch.from_numpy(np.array(gold, dtype=int)).long().to(all_score_mat.device)
        loss = self.crie(all_score_mat.view(-1, length), gold.view(-1)).view(batch, length)[:, 1:].mean(dim=1)
        return (torch.tensor(weight).to(loss.device) * loss).mean()

    def init_forward_edge_selection(self, sentences):
        raise NotImplementedError
        batch_encoder_mat = self.real_forward(sentences)
        batch_encoder_mat_prob = torch.log_softmax(batch_encoder_mat, dim=2)
        multinomial_prob = torch.log_softmax(self.multinomial, dim=-1)
        batch, length = batch_encoder_mat.size()[:2]
        all_score_mat = []
        gold = []
        for idx, sentence in enumerate(sentences):
            labels = getattr(sentence, 'kmeans_labels', None)
            unlabeled_arcs = self.sampler.sample(sentence.raw.edu_ids,
                                                 sentence.raw.edus,
                                                 sentence.raw.edus_head,
                                                 sentence.raw.sbnds,
                                                 sentence.raw.pbnds,
                                                 has_root=True)
            pred_arcs = [arc[0] for arc in sorted(unlabeled_arcs, key=lambda x: x[1])]
            gold.append([0] + pred_arcs)
            all_score_mat.append(multinomial_prob[labels[:, None], labels[None, :]] + batch_encoder_mat_prob[idx])
        all_score_mat = torch.stack(all_score_mat, dim=0)
        gold = torch.from_numpy(np.array(gold, dtype=int)).long().to(all_score_mat.device)
        return self.crie(all_score_mat.view(-1, length), gold.view(-1)).view(batch, length)[:, 1:].mean()

    def forward_edge_selection(self, sentences):
        raise NotImplementedError
        batch_encoder_mat = self.real_forward(sentences)
        batch_score_mat_prob = torch.log_softmax(batch_encoder_mat, dim=2)
        multinomial_prob = torch.log_softmax(self.multinomial, dim=-1)
        batch, length = batch_encoder_mat.size()[:2]
        all_score_mat = []
        neg_score_mat = []
        # batch_score_mat_np = score_mat.permute(0, 2, 1).detach().cpu().numpy()
        for idx, sentence in enumerate(sentences):
            labels = getattr(sentence, 'kmeans_labels', None)
            all_score_mat.append(multinomial_prob[labels[:, None], labels[None, :]] + batch_score_mat_prob[idx])
            neg_score_mat.append(torch.log(1.0 - torch.exp(multinomial_prob[labels[:, None], labels[None, :]])) +
                                 torch.log(1.0 -torch.exp(batch_score_mat_prob[idx])))
        all_score_mat = torch.stack(all_score_mat)
        loss = torch.logsumexp(all_score_mat.view(-1), dim=0) / (length * batch)

    def init_embedding(self):
        if self.kmeans is None:
            return
        cluster_centers = self.kmeans.cluster_centers_
        self.kcluster_embed.weight.data = torch.from_numpy(cluster_centers).to(self.kcluster_embed.weight.device)
        self.kcluster_embed.requires_grad_(False)

    # def decoding(self, sentences):
    #     batch_encoder_mat = self.real_forward(sentences)
    #     best_tree = []
    #     multinomial_prob = torch.log_softmax(self.multinomial, dim=-1)
    #     batch_encoder_mat_np = batch_encoder_mat.permute(0, 2, 1).detach().cpu().numpy()
    #     multinomial_prob_np = multinomial_prob.detach().cpu().permute(0, 1).numpy()
    #
    #     for idx, (sentence, encoder_mat_np) in enumerate(zip(sentences, batch_encoder_mat_np)):
    #         labels = getattr(sentence, 'kmeans_labels', None)
    #         all_score_np = encoder_mat_np + multinomial_prob_np[labels[:, None], labels[None, :]]
    #         unlabeled_arcs = self.esiner.global_decode(arc_scores=all_score_np, edu_ids=sentence.raw.edu_ids,
    #                                                    sbnds=sentence.raw.sbnds, pbnds=sentence.raw.pbnds,
    #                                                    use_sbnds=True, use_pbnds=False)
    #         pred_arcs = [arc[0] for arc in sorted(unlabeled_arcs, key=lambda x: x[1])]
    #         best_tree.append([0] + pred_arcs)
    #     return best_tree

    def decoding(self, sentences):
        batch_encoder_mat = self.real_forward(sentences)
        best_tree = []
        multinomial_prob = torch.log_softmax(self.multinomial, dim=-1)
        batch_encoder_mat_np = batch_encoder_mat.permute(0, 2, 1).detach().cpu().numpy()
        multinomial_prob_np = multinomial_prob.detach().cpu().numpy()
        # multinomial_prob_np = multinomial_prob.detach().cpu().permute(0, 1).numpy()

        for idx, (sentence, encoder_mat_np) in enumerate(zip(sentences, batch_encoder_mat_np)):
            labels = getattr(sentence, 'kmeans_labels', None)
            all_score_np = encoder_mat_np + multinomial_prob_np[labels[:, None], labels[None, :]]
            unlabeled_arcs = self.esiner.global_decode(arc_scores=all_score_np, edu_ids=sentence.raw.edu_ids,
                                                       sbnds=sentence.raw.sbnds, pbnds=sentence.raw.pbnds,
                                                       use_sbnds=True, use_pbnds=False)
            pred_arcs = [arc[0] for arc in sorted(unlabeled_arcs, key=lambda x: x[1])]
            best_tree.append([0] + pred_arcs)
        return best_tree


    def set_prob(self, prob):
        # log and replace zero
        prob = np.log(prob)
        for i in range(self.args.kcluster):
            for j in range(self.args.kcluster):
                if np.abs(prob[i][j]) == np.inf or math.isnan(prob[i][j]):
                    prob[i][j] = -1e5
        self.multinomial.data = torch.from_numpy(prob).float().to(self.multinomial.device)
        self.multinomial.requires_grad = self.args.decoder_learn

    def get_rightbranching_prob_mat(self, dataset):
        prob_mat = [[0] * self.args.kcluster for _ in range(self.args.kcluster)]
        for inst in dataset:
            labels = np.array(inst.kmeans_labels).astype(int)
            arcs = self.sampler.sample(inst.raw.edu_ids, inst.raw.edus, inst.raw.edus_head,
                                       inst.raw.sbnds, inst.raw.pbnds, has_root=True)
            for arc in arcs:
                id1 = labels[arc[0]]
                id2 = labels[arc[1]]
                prob_mat[id1][id2] += 1
        # to prob and smooth
        for i in range(self.args.kcluster):
            i_all = float(sum(prob_mat[i]))
            if i_all == 0:
                # prob_mat[i] = additive_smoothing(prob_mat[i], alpha)
                continue
            else:
                if self.args.smooth == 'additive':
                    prob_mat[i] = additive_smoothing(prob_mat[i], self.args.alpha)
                elif self.args.smooth == 'sign':
                    prob_mat[i] = sign_smoothing(prob_mat[i], self.args.alpha)
        # uniform smoothing
        if self.args.smooth == 'uniform':
            prob_mat = uniform_smoothing(prob_mat, self.args.alpha)
        return prob_mat

    def get_gold_prob_mat(self, dataset):
        prob_mat = [[0] * self.args.kcluster for _ in range(self.args.kcluster)]
        for inst in dataset:
            labels = np.array(inst.kmeans_labels).astype(int)
            arcs = inst.raw.arcs
            for arc in arcs:
                id1 = labels[arc[0]]
                id2 = labels[arc[1]]
                prob_mat[id1][id2] += 1
        # to prob and smooth
        for i in range(self.args.kcluster):
            i_all = float(sum(prob_mat[i]))
            if i_all == 0:
                # prob_mat[i] = additive_smoothing(prob_mat[i], alpha)
                continue
            else:
                if self.args.smooth == 'additive':
                    prob_mat[i] = additive_smoothing(prob_mat[i], self.args.alpha)
                elif self.args.smooth == 'sign':
                    prob_mat[i] = sign_smoothing(prob_mat[i], self.args.alpha)
                # uniform smoothing
        if self.args.smooth == 'uniform':
            prob_mat = uniform_smoothing(prob_mat, self.args.alpha)
        return prob_mat

    @staticmethod
    def get_batch_embedding(sentences) -> List[torch.Tensor]:
        return [sentence.r_embed for sentence in sentences]

    def partition_calculation(self, sentences):
        import time
        score_mat = self.real_forward(sentences)
        encoder_p = torch.log_softmax(score_mat, dim=2).permute(0, 2, 1)
        times = []
        from model.eisner import IncrementalEisnerDecoder
        eisner = IncrementalEisnerDecoder()
        for p_mat, sentence in zip(encoder_p, sentences):
            labels = getattr(sentence, 'kmeans_labels', None)
            assert labels is not None
            start_time = time.time()
            # rec_weight = torch.log_softmax(self.multinomial, dim=1)[labels[:, None], labels[None, :]].permute(1, 0)
            # partition = self.esiner.partition(arc_scores=p_mat, edu_ids=sentence.raw.edu_ids,
            #                                   sbnds=sentence.raw.sbnds, pbnds=sentence.raw.pbnds,
            #                                   use_sbnds=True, use_pbnds=True)
            partition = eisner.partition(arc_scores=p_mat, edu_ids=sentence.raw.edu_ids)

            times.append(time.time() - start_time)
            del partition
        avg_time = sum(times) / len(times)
        print(str(len(sentences[0].raw.edu_ids)) + '\t' + str(avg_time))

