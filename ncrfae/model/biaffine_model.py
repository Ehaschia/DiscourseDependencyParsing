# -*- coding: utf-8 -*-
import os
import pickle
from sklearn.cluster import KMeans

from model.module import CHAR_LSTM, MLP, Biaffine, BiLSTM
from model.module.dropout import IndependentDropout, SharedDropout

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)

# from parser.modules import crfae
from model.eisner_v2 import IncrementalEisnerDecoder
from utils.common import BERT_DIM
# from parser.modules import NICETrans
import numpy as np
import math
from utils import uniform_smoothing, additive_smoothing, sign_smoothing


class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args
        # the embedding layer
        # if args.bert is False:
        #     self.word_embed = nn.Embedding(num_embeddings=args.n_words,
        #                                    embedding_dim=args.word_embed)
        #     if args.freeze_word_emb:
        #         self.word_embed.weight.requires_grad = False
        # else:
        #     self.word_embed = BertEmbedding(model=args.bert_model,
        #                                     n_layers=args.n_bert_layers,
        #                                     n_out=args.word_embed)

        # if args.feat == 'char':
        #     self.feat_embed = CHAR_LSTM(n_chars=args.n_feats,
        #                                 n_embed=args.n_char_embed,
        #                                 n_out=args.n_embed)
        # elif args.feat == 'bert':
        #     self.feat_embed = BertEmbedding(model=args.bert_model,
        #                                     n_layers=args.n_bert_layers,
        #                                     n_out=args.n_embed)
        # else:
        # self.feat_embed = nn.Embedding(num_embeddings=args.n_feats,
        #                                embedding_dim=args.n_embed)

        # if args.freeze_feat_emb:
        #     self.feat_embed.weight.requires_grad = False

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
        self.mlp_rel_d = MLP(n_in=args.n_lstm_hidden*2,
                             n_hidden=args.n_mlp_rel,
                             dropout=args.mlp_dropout)
        self.mlp_rel_h = MLP(n_in=args.n_lstm_hidden*2,
                             n_hidden=args.n_mlp_rel,
                             dropout=args.mlp_dropout)

        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=args.n_mlp_arc,
                                 bias_x=True,
                                 bias_y=False)
        self.rel_attn = Biaffine(n_in=args.n_mlp_rel,
                                 n_out=args.n_rels,
                                 bias_x=True,
                                 bias_y=True)

        # self.pad_index = args.pad_index
        # self.unk_index = args.unk_index

        # decoder
        # self.multinomial = nn.Parameter(torch.ones(args.n_feats, args.n_feats) / args.n_feats)
        # self.multinomial.requires_grad = False
        # self.multinomial = nn.Parameter(torch.ones(args.n_feats, args.n_feats))
        self.esiner = IncrementalEisnerDecoder()
        self.crie = nn.CrossEntropyLoss(reduction='none')

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            nn.init.zeros_(self.word_embed.weight)

        return self

    def W_Reg(self):
        diff = 0.
        for name, param in self.named_parameters():
            if name == 'pretrained.weight':
                continue
            diff += ((self.init_params[name] - param) ** 2).sum()

        return 0.5 * diff * self.args.W_beta

    def E_Reg(self, words, bert, feats, source_model, tar_score):
        source_model.eval()
        with torch.no_grad():
            source_score = source_model(words, bert, feats)
            source_score = source_model.decoder(source_score, feats)
        diff = ((source_score - tar_score) ** 2).sum()
        return 0.5 * diff * self.args.E_beta

    def T_Reg(self, words, bert, feats, source_model):
        source_model.eval()
        with torch.no_grad():
            source_score = source_model(words, bert, feats)
            source_score = source_model.decoder(source_score, feats)
        return source_score

    def real_forward(self, sentences):
        self.batch_size = len(sentences)
        tmp_len = []
        for sentence in sentences:
            tmp_len.append(len(sentence.raw.edu_ids))
        self.seq_len = max(tmp_len)
        # self.batch_size, self.seq_len = words.shape
        # mask = words.ne(self.pad_index)
        # lens = mask.sum(dim=1)


        # if self.args.bert is False:
        #     ext_mask = words.ge(self.word_embed.num_embeddings)
        #     ext_words = words.masked_fill(ext_mask, self.unk_index)
        #     word_embed = self.word_embed(ext_words)
        # else:
        #     word_embed = self.word_embed(*bert)
        #
        # if hasattr(self, 'pretrained'):
        #     word_embed += self.pretrained(words)
        # if self.args.feat == 'char':
        #     feat_embed = self.feat_embed(feats[mask])
        #     feat_embed = pad_sequence(feat_embed.split(lens.tolist()), True)
        # elif self.args.feat == 'bert':
        #     feat_embed = self.feat_embed(*feats)
        # else:
        #     feat_embed = self.feat_embed(feats)
        # word_embed, feat_embed = self.embed_dropout(word_embed, feat_embed)
        # embed = torch.cat((word_embed, feat_embed), dim=-1)
        lens = torch.tensor(tmp_len)
        mask = (torch.arange(self.seq_len)[None, :] < lens[:, None]).to(self.mlp_arc_d.linear.weight.device)
        embed = self.get_batch_embedding(sentences)
        embed = pad_sequence(embed, batch_first=True, padding_value=0.0).to(self.mlp_arc_d.linear.weight.device)

        x = pack_padded_sequence(embed, lens, True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=self.seq_len)
        x = self.lstm_dropout(x)

        arc_h = self.mlp_arc_h(x)  # [batch_size, seq_len, d]
        arc_d = self.mlp_arc_d(x)
        rel_d = self.mlp_rel_d(x)
        rel_h = self.mlp_rel_h(x)

        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        # s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))
        s_arc.masked_fill_(~mask.unsqueeze(1), -1e5)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)

        return s_arc.squeeze(0), s_rel.squeeze(0), mask

    def get_batch_embedding(self, sentences):
        return [sentence.r_embed for sentence in sentences]

    def forward(self, sentences):
        arc_mat, rel_mat, mask = self.real_forward(sentences)
        mask[:, 0] = 0
        total = torch.sum(mask)
        # arc_mat = torch.log_softmax(arc_mat, dim=2)
        batch, length = arc_mat.size()[:2]

        arc_gold = pad_sequence([torch.tensor(sent.heads) for sent in sentences], batch_first=True, padding_value=0).long().to(arc_mat.device)
        arc_loss = torch.sum(self.crie(arc_mat.view(-1, length), arc_gold.view(-1)).reshape_as(arc_gold) * mask)

        rel_gold = torch.cat([torch.tensor(sent.rels) for sent in sentences], dim=0).to(rel_mat.device)
        rel_mat = rel_mat[mask]
        rels_scores = rel_mat[torch.arange(rel_mat.shape[0]), arc_gold[mask]]
        rel_loss = torch.sum(self.crie(rels_scores, rel_gold.view(-1)))
        loss = arc_loss + rel_loss / total
        return loss

    def decoding(self, sentences):
        batch_arc_mat, batch_rel_mat, _ = self.real_forward(sentences)
        # batch_score_mat = torch.log_softmax(batch_score_mat, dim=2)
        best_arcs = []
        best_rels = []
        batch_arc_mat_np = batch_arc_mat.permute(0, 2, 1).detach().cpu().numpy()
        batch_rel_idx_mat = torch.argmax(batch_rel_mat, dim=-1).detach().cpu()
        for idx, (sentence, arc_mat_np) in enumerate(zip(sentences, batch_arc_mat_np)):
            inst_len = len(sentence.raw.edu_ids)
            unlabeled_arcs = self.esiner.global_decode(arc_scores=arc_mat_np[:inst_len, :inst_len],
                                                       edu_ids=sentence.raw.edu_ids,
                                                       sbnds=sentence.raw.sbnds, pbnds=sentence.raw.pbnds,
                                                       use_sbnds=True, use_pbnds=True)
            pred_arcs = [0] + [arc[0] for arc in sorted(unlabeled_arcs, key=lambda x: x[1])]
            pred_rels = batch_rel_idx_mat[idx, :inst_len, :inst_len].gather(-1, torch.tensor(pred_arcs).unsqueeze(-1)).squeeze(-1).numpy()
            best_arcs.append(pred_arcs)
            best_rels.append(pred_rels)
        return best_arcs, best_rels

    def eisner_decoding(self, sentences):
        batch_arc_mat, batch_rel_mat, _ = self.real_forward(sentences)
        # batch_score_mat = torch.log_softmax(batch_score_mat, dim=2)
        best_arcs = []
        best_rels = []
        batch_arc_mat_np = batch_arc_mat.permute(0, 2, 1).detach().cpu().numpy()
        batch_rel_idx_mat = torch.argmax(batch_rel_mat, dim=-1).detach().cpu()
        for idx, (sentence, arc_mat_np) in enumerate(zip(sentences, batch_arc_mat_np)):
            inst_len = len(sentence.raw.edu_ids)
            unlabeled_arcs, _ = self.esiner.decoder.decode(arc_scores=arc_mat_np[:inst_len, :inst_len],
                                                       edu_ids=sentence.raw.edu_ids)
            pred_arcs = [0] + [arc[0] for arc in sorted(unlabeled_arcs, key=lambda x: x[1])]
            pred_rels = batch_rel_idx_mat[idx, :inst_len, :inst_len].gather(-1, torch.tensor(pred_arcs).unsqueeze(-1)).squeeze(-1).numpy()
            best_arcs.append(pred_arcs)
            best_rels.append(pred_rels)
        return best_arcs, best_rels

    def decoder(self, crf_weight, feats):
        m = nn.Softmax(dim=1)(self.multinomial)
        recons_weight = torch.log(m[feats[:, :, None], feats[:, None, :]])  # B * N * N
        joint_weights = crf_weight + recons_weight
        return joint_weights

    @classmethod
    def load(cls, path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state = torch.load(path, map_location=device)
        model = cls(state['args'])
        model.load_pretrained(state['pretrained'])
        model.load_state_dict(state['state_dict'], False)
        model.to(device)

        return model

    def save(self, path):
        state_dict, pretrained = self.state_dict(), None
        if hasattr(self, 'pretrained'):
            pretrained = state_dict.pop('pretrained.weight')
        state = {
            'args': self.args,
            'state_dict': state_dict,
            'pretrained': pretrained
        }
        torch.save(state, path)



class BiaffineAE(nn.Module):

    def __init__(self, args):
        super(BiaffineAE, self).__init__()

        self.args = args
        # the embedding layer
        # if args.bert is False:
        #     self.word_embed = nn.Embedding(num_embeddings=args.n_words,
        #                                    embedding_dim=args.word_embed)
        #     if args.freeze_word_emb:
        #         self.word_embed.weight.requires_grad = False
        # else:
        #     self.word_embed = BertEmbedding(model=args.bert_model,
        #                                     n_layers=args.n_bert_layers,
        #                                     n_out=args.word_embed)

        # if args.feat == 'char':
        #     self.feat_embed = CHAR_LSTM(n_chars=args.n_feats,
        #                                 n_embed=args.n_char_embed,
        #                                 n_out=args.n_embed)
        # elif args.feat == 'bert':
        #     self.feat_embed = BertEmbedding(model=args.bert_model,
        #                                     n_layers=args.n_bert_layers,
        #                                     n_out=args.n_embed)
        # else:
        # self.feat_embed = nn.Embedding(num_embeddings=args.n_feats,
        #                                embedding_dim=args.n_embed)

        # if args.freeze_feat_emb:
        #     self.feat_embed.weight.requires_grad = False

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
        self.mlp_rel_d = MLP(n_in=args.n_lstm_hidden*2,
                             n_hidden=args.n_mlp_rel,
                             dropout=args.mlp_dropout)
        self.mlp_rel_h = MLP(n_in=args.n_lstm_hidden*2,
                             n_hidden=args.n_mlp_rel,
                             dropout=args.mlp_dropout)

        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=args.n_mlp_arc,
                                 bias_x=True,
                                 bias_y=False)
        self.rel_attn = Biaffine(n_in=args.n_mlp_rel,
                                 n_out=args.n_rels,
                                 bias_x=True,
                                 bias_y=True)

        # self.pad_index = args.pad_index
        # self.unk_index = args.unk_index

        # decoder
        self.multinomial = nn.Parameter(torch.ones(self.args.kcluster, self.args.kcluster))
        self.kmeans = None
        # self.multinomial.requires_grad = False
        # self.multinomial = nn.Parameter(torch.ones(args.n_feats, args.n_feats))
        self.esiner = IncrementalEisnerDecoder()
        self.crie = nn.CrossEntropyLoss(reduction='none')

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            nn.init.zeros_(self.word_embed.weight)

        return self

    def W_Reg(self):
        diff = 0.
        for name, param in self.named_parameters():
            if name == 'pretrained.weight':
                continue
            diff += ((self.init_params[name] - param) ** 2).sum()

        return 0.5 * diff * self.args.W_beta

    def E_Reg(self, words, bert, feats, source_model, tar_score):
        source_model.eval()
        with torch.no_grad():
            source_score = source_model(words, bert, feats)
            source_score = source_model.decoder(source_score, feats)
        diff = ((source_score - tar_score) ** 2).sum()
        return 0.5 * diff * self.args.E_beta

    def T_Reg(self, words, bert, feats, source_model):
        source_model.eval()
        with torch.no_grad():
            source_score = source_model(words, bert, feats)
            source_score = source_model.decoder(source_score, feats)
        return source_score

    def real_forward(self, sentences):
        self.batch_size = len(sentences)
        tmp_len = []
        for sentence in sentences:
            tmp_len.append(len(sentence.raw.edu_ids))
        self.seq_len = max(tmp_len)
        # self.batch_size, self.seq_len = words.shape
        # mask = words.ne(self.pad_index)
        # lens = mask.sum(dim=1)


        # if self.args.bert is False:
        #     ext_mask = words.ge(self.word_embed.num_embeddings)
        #     ext_words = words.masked_fill(ext_mask, self.unk_index)
        #     word_embed = self.word_embed(ext_words)
        # else:
        #     word_embed = self.word_embed(*bert)
        #
        # if hasattr(self, 'pretrained'):
        #     word_embed += self.pretrained(words)
        # if self.args.feat == 'char':
        #     feat_embed = self.feat_embed(feats[mask])
        #     feat_embed = pad_sequence(feat_embed.split(lens.tolist()), True)
        # elif self.args.feat == 'bert':
        #     feat_embed = self.feat_embed(*feats)
        # else:
        #     feat_embed = self.feat_embed(feats)
        # word_embed, feat_embed = self.embed_dropout(word_embed, feat_embed)
        # embed = torch.cat((word_embed, feat_embed), dim=-1)
        lens = torch.tensor(tmp_len)
        mask = (torch.arange(self.seq_len)[None, :] < lens[:, None]).to(self.mlp_arc_d.linear.weight.device)
        embed = self.get_batch_embedding(sentences)
        embed = pad_sequence(embed, batch_first=True, padding_value=0.0).to(self.mlp_arc_d.linear.weight.device)

        x = pack_padded_sequence(embed, lens, True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=self.seq_len)
        x = self.lstm_dropout(x)

        arc_h = self.mlp_arc_h(x)  # [batch_size, seq_len, d]
        arc_d = self.mlp_arc_d(x)
        rel_d = self.mlp_rel_d(x)
        rel_h = self.mlp_rel_h(x)


        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        s_arc.masked_fill_(~mask.unsqueeze(1), -1e5)
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)

        return s_arc, s_rel, mask
    def get_batch_embedding(self, sentences):
        return [sentence.r_embed for sentence in sentences]

    def forward(self, sentences):
        batch_arcs_mat, batch_rels_mat, batch_mask = self.real_forward(sentences)
        batch_mask[:, 0] = 0
        total = torch.sum(batch_mask)
        batch_arcs_mat_prob = torch.log_softmax(batch_arcs_mat, dim=2)
        multinomial_prob = torch.log_softmax(self.multinomial, dim=-1)
        batch, length = batch_arcs_mat.size()[:2]
        # batch_score_mat_np = score_mat.permute(0, 2, 1).detach().cpu().numpy()
        batch_cluster_labels = []
        for idx, sentence in enumerate(sentences):
            labels = torch.tensor(getattr(sentence, 'kmeans_labels', None))
            batch_cluster_labels.append(labels)

        batch_cluster_labels = pad_sequence(batch_cluster_labels, batch_first=True, padding_value=0).long().to(batch_arcs_mat.device)
        all_arcs_mat = multinomial_prob[batch_cluster_labels[:,: , None], batch_cluster_labels[:, None, :]] + batch_arcs_mat_prob
        gold_arcs = pad_sequence([torch.tensor(sent.heads) for sent in sentences], batch_first=True, padding_value=0).long().to(batch_arcs_mat.device)
        # gold = torch.tensor([sent.heads for sent in sentences]).long().to(batch_encoder_mat.device)
        # all_score_mat = torch.stack(all_score_mat)
        arcs_loss = torch.sum(self.crie(all_arcs_mat.permute(0, 2, 1).contiguous().view(-1, length), gold_arcs.view(-1)).view(batch, length) * batch_mask)

        rel_gold = torch.cat([torch.tensor(sent.rels) for sent in sentences], dim=0).to(batch_rels_mat.device)
        batch_rels_mat = batch_rels_mat[batch_mask]
        batch_rels_scores = batch_rels_mat[torch.arange(batch_rels_mat.shape[0]), gold_arcs[batch_mask]]
        rel_loss = torch.sum(self.crie(batch_rels_scores, rel_gold.view(-1)))
        loss = arcs_loss + rel_loss / total

        return loss

    def decoding(self, sentences):
        batch_arc_mat, batch_rel_mat, _ = self.real_forward(sentences)
        best_arcs = []
        best_rels = []
        batch_arc_mat_prob = torch.log_softmax(batch_arc_mat, dim=2)
        multinomial_prob = torch.log_softmax(self.multinomial, dim=-1)
        batch_arc_mat_np = batch_arc_mat_prob.detach().cpu().numpy()
        # batch_encoder_mat_np = batch_encoder_mat_prob.permute(0, 2, 1).detach().cpu().numpy()
        multinomial_prob_np = multinomial_prob.detach().cpu().numpy()
        # multinomial_prob_np = multinomial_prob.permute(1, 0).detach().cpu().numpy()
        batch_rel_idx_mat = torch.argmax(batch_rel_mat, dim=-1).detach().cpu()

        for idx, (sentence, arc_mat_np) in enumerate(zip(sentences, batch_arc_mat_np)):
            labels = getattr(sentence, 'kmeans_labels', None)
            inst_len = len(sentence.raw.edu_ids)
            all_score_np =arc_mat_np[:inst_len, :inst_len] + multinomial_prob_np[labels[:, None], labels[None, :]]
            unlabeled_arcs = self.esiner.global_decode(arc_scores=all_score_np, edu_ids=sentence.raw.edu_ids,
                                                       sbnds=sentence.raw.sbnds, pbnds=sentence.raw.pbnds,
                                                       use_sbnds=True, use_pbnds=False)
            pred_arcs = [0] + [arc[0] for arc in sorted(unlabeled_arcs, key=lambda x: x[1])]
            pred_rels = batch_rel_idx_mat[idx, :inst_len, :inst_len].gather(-1, torch.tensor(pred_arcs).unsqueeze(-1)).squeeze(-1).numpy()
            best_arcs.append(pred_arcs)
            best_rels.append(pred_rels)
        return best_arcs, best_rels


    def eisner_decoding(self, sentences):
        batch_arc_mat, batch_rel_mat, _ = self.real_forward(sentences)
        best_arcs = []
        best_rels = []
        batch_arc_mat_prob = torch.log_softmax(batch_arc_mat, dim=2)
        multinomial_prob = torch.log_softmax(self.multinomial, dim=-1)
        batch_arc_mat_np = batch_arc_mat_prob.detach().cpu().numpy()
        # batch_encoder_mat_np = batch_encoder_mat_prob.permute(0, 2, 1).detach().cpu().numpy()
        multinomial_prob_np = multinomial_prob.detach().cpu().numpy()
        # multinomial_prob_np = multinomial_prob.permute(1, 0).detach().cpu().numpy()
        batch_rel_idx_mat = torch.argmax(batch_rel_mat, dim=-1).detach().cpu()

        for idx, (sentence, arc_mat_np) in enumerate(zip(sentences, batch_arc_mat_np)):
            labels = getattr(sentence, 'kmeans_labels', None)
            inst_len = len(sentence.raw.edu_ids)
            all_score_np =arc_mat_np[:inst_len, :inst_len] + multinomial_prob_np[labels[:, None], labels[None, :]]
            unlabeled_arcs, _ = self.esiner.decoder.decode(arc_scores=all_score_np, edu_ids=sentence.raw.edu_ids)
            pred_arcs = [0] + [arc[0] for arc in sorted(unlabeled_arcs, key=lambda x: x[1])]
            pred_rels = batch_rel_idx_mat[idx, :inst_len, :inst_len].gather(-1, torch.tensor(pred_arcs).unsqueeze(-1)).squeeze(-1).numpy()
            best_arcs.append(pred_arcs)
            best_rels.append(pred_rels)
        return best_arcs, best_rels


    def decoder(self, crf_weight, feats):
        m = nn.Softmax(dim=1)(self.multinomial)
        recons_weight = torch.log(m[feats[:, :, None], feats[:, None, :]])  # B * N * N
        joint_weights = crf_weight + recons_weight
        return joint_weights

    def set_prob(self, prob):
        # log and replace zero
        prob = np.log(prob)
        for i in range(self.args.kcluster):
            for j in range(self.args.kcluster):
                if np.abs(prob[i][j]) == np.inf or math.isnan(prob[i][j]):
                    prob[i][j] = -1e5
        self.multinomial.data = torch.from_numpy(prob).float().to(self.multinomial.device)
        self.multinomial.requires_grad = self.args.decoder_learn

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

    @classmethod
    def load(cls, path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state = torch.load(path, map_location=device)
        model = cls(state['args'])
        model.load_pretrained(state['pretrained'])
        model.load_state_dict(state['state_dict'], False)
        model.to(device)

        return model

    def save(self, path):
        state_dict, pretrained = self.state_dict(), None
        if hasattr(self, 'pretrained'):
            pretrained = state_dict.pop('pretrained.weight')
        state = {
            'args': self.args,
            'state_dict': state_dict,
            'pretrained': pretrained
        }
        torch.save(state, path)
