from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from model.dep_dataset import SCIDTBSentence
from model.context_sensitive_encoder import CSEncoder
from model.eisner_v2 import IncrementalEisnerDecoder
# from model.eisner import IncrementalEisnerDecoder
from treesamplers import TreeSampler

ACTIVATION_DICT = {'relu': F.relu, 'gelu': F.gelu, 'tanh': F.tanh, 'sigmoid': F.sigmoid}
EDU_REPRESENTATION_DIM = {'encode_edu': 768, 'encode_minus': 768,
                          'encode_avg_pooling': 768,
                          'encode_max_pooling': 768, 'encode_mean': 768,
                          'encode_endpoint': 1536, 'encode_diffsum': 1536,
                          'encode_coherent': 768, 'encode_attention': 768}

class NCRF(nn.Module):
    def __init__(self, args):
        super(NCRF, self).__init__()
        edu_repre = 'encode_' + args.encode_method
        self.embedding_dim = EDU_REPRESENTATION_DIM[edu_repre]
        self.hidden_dim = args.hidden
        self.process_batch = getattr(self, edu_repre)

        if edu_repre == 'encode_attention':
            self.attention_key = Parameter(torch.Tensor(EDU_REPRESENTATION_DIM[edu_repre]))
            nn.init.uniform_(self.attention_key.data)

        self.em2h1 = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.em2h2 = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.activation = ACTIVATION_DICT[args.activation_function]
        self.dropout = nn.Dropout(p=args.dropout)

        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim // 2,
                            num_layers=args.layers, bidirectional=True)

        self.batch_size = 1  # temp
        self.seq_length = 1  # temp
        self.crie = torch.nn.CrossEntropyLoss(reduction='none')
        self.esiner = IncrementalEisnerDecoder()
        self.cs_encoder = CSEncoder('bert', gpu=True, finetune=args.finetune)
        self.finetune = args.finetune

    def init(self):
        pass

    # process context sensitive batch
    def encode_edu(self, sents: List[SCIDTBSentence]) -> List[torch.Tensor]:
        batch_p_embed = []
        batch_r_embed = []
        for sent in sents:
            if sent.r_embed is None or sent.r_embed is None:
                embed1, _ = self.cs_encoder.encode(sent.raw)
                sent.r_embed = embed1 if not self.finetune else None
                # sent.p_embed = embed2 if not self.finetune else None
            else:
                embed1, embed2 = sent.r_embed, sent.p_embed
            batch_r_embed.append(embed1)
            # batch_p_embed.append(embed2)

        return batch_r_embed
    #
    # # process sentence sensitive edu
    # def encode_concat_sent(self, discourses: List[SCIDTBSentence]) -> Tuple[torch.Tensor, torch.Tensor]:
    #     batch_p_embed = []
    #     batch_r_embed = []
    #     batch_s_embed = []
    #     for discourse in discourses:
    #         if discourse.r_embed is None:
    #             embed1, embed2 = self.cs_encoder.encode(discourse.raw)
    #             discourse.r_embed = embed1 if not self.finetune else None
    #             discourse.p_embed = embed2 if not self.finetune else None
    #
    #             s_embed = self.cs_encoder.encode_sentence(discourse.raw)
    #
    #             # sent merge
    #             rep_list = torch.split(s_embed[:, 0], 1, dim=0)
    #             final = [rep_list[0]]
    #             _, dim = rep_list[0].size()
    #             for span, one_represent in zip(discourse.sbnds, rep_list[1:]):
    #                 final.append(one_represent.expand(span[1] - span[0] + 1, dim))
    #             final = torch.cat(final, dim=0).detach()
    #             discourse.s_embed = final if not self.finetune else None
    #         else:
    #             embed1 = discourse.r_embed
    #             embed2 = discourse.p_embed
    #             final = discourse.s_embed
    #         batch_r_embed.append(embed1)
    #         batch_p_embed.append(embed2)
    #         batch_s_embed.append(final)
    #     concat_r_embed = torch.cat([torch.stack(batch_r_embed, dim=0), torch.stack(batch_s_embed, dim=0)], dim=2)
    #     # concat_p_embed = torch.cat([torch.stack(batch_p_embed, dim=0), torch.stack(batch_s_embed, dim=0)], dim=2)
    #     return concat_r_embed # , concat_p_embed

    # sentence edu minus representation
    # process sentence sensitive edu
    def encode_minus(self, sents: List[SCIDTBSentence]) -> List[torch.Tensor]:
        batch_r_embed = []
        for sent in sents:
            if sent.r_embed is None:
                edus_representation = self.cs_encoder.sent_sensitive_encode(sent.raw)
                final = []
                for edu_representation in edus_representation:
                    if edu_representation.shape[0] == 0:
                        print("Bug here")
                    final.append(edu_representation[-1] - edu_representation[0])
                final = torch.stack(final, dim=0)
                sent.r_embed = final if not self.finetune else None
            else:
                final = sent.r_embed
            batch_r_embed.append(final)
        return batch_r_embed

    # avg pooling the result
    def encode_avg_pooling(self, sents: List[SCIDTBSentence]) -> List[torch.Tensor]:
        batch_r_embed = []
        for sent in sents:
            if sent.r_embed is None:
                edus_representation = self.cs_encoder.sent_sensitive_encode(sent.raw)
                final = []
                for edu_representation in edus_representation:
                    edu_size = edu_representation.shape[0]
                    edu_representation = torch.avg_pool1d(edu_representation.unsqueeze(0).permute(0, 2, 1),
                                                          kernel_size=edu_size, stride=edu_size).squeeze()
                    final.append(edu_representation)
                final = torch.stack(final, dim=0)
                sent.r_embed = final if not self.finetune else None
            else:
                final = sent.r_embed
            batch_r_embed.append(final)
        return batch_r_embed

    # max pooling the result
    def encode_max_pooling(self, sents: List[SCIDTBSentence]) -> List[torch.Tensor]:
        batch_r_embed = []
        for sent in sents:
            if sent.r_embed is None:
                edus_representation = self.cs_encoder.sent_sensitive_encode(sent.raw)
                final = []
                for edu_representation in edus_representation:
                    edu_size = edu_representation.shape[0]
                    edu_representation = torch.nn.functional.max_pool1d(
                        edu_representation.unsqueeze(0).permute(0, 2, 1),
                        kernel_size=edu_size, stride=edu_size).squeeze()
                    final.append(edu_representation)
                final = torch.stack(final, dim=0)
                sent.r_embed = final if not self.finetune else None
            else:
                final = sent.r_embed
            batch_r_embed.append(final)
        return batch_r_embed

    # max pooling the result
    def encode_mean(self, sents: List[SCIDTBSentence]) -> List[torch.Tensor]:
        batch_r_embed = []
        for sent in sents:
            if sent.r_embed is None:
                edus_representation = self.cs_encoder.sent_sensitive_encode(sent.raw)
                final = []
                for edu_representation in edus_representation:
                    edu_representation = edu_representation.mean(dim=0)
                    final.append(edu_representation)
                final = torch.stack(final, dim=0)
                sent.r_embed = final if not self.finetune else None
            else:
                final = sent.r_embed
            batch_r_embed.append(final)
        return batch_r_embed

    # concat begin and end [e_i, e_j]
    def encode_endpoint(self, sents: List[SCIDTBSentence]) -> List[torch.Tensor]:
        batch_r_embed = []
        for sent in sents:
            if sent.r_embed is None:
                edus_representation = self.cs_encoder.sent_sensitive_encode(sent.raw)
                final = []
                for edu_representation in edus_representation:
                    edu_representation = torch.cat([edu_representation[0], edu_representation[-1]], dim=-1)
                    final.append(edu_representation)
                final = torch.stack(final, dim=0)
                sent.r_embed = final if not self.finetune else None
            else:
                final = sent.r_embed
            batch_r_embed.append(final)
        return batch_r_embed

    # concat [e_i+e_j, e_j-e_i]
    def encode_diffsum(self, sents: List[SCIDTBSentence]) -> List[torch.Tensor]:
        batch_r_embed = []
        for sent in sents:
            if sent.r_embed is None:
                edus_representation = self.cs_encoder.sent_sensitive_encode(sent.raw)
                final = []
                for edu_representation in edus_representation:
                    edu_representation = torch.cat([edu_representation[-1]+edu_representation[0],
                                                    edu_representation[-1]-edu_representation[0]], dim=-1)
                    final.append(edu_representation)
                final = torch.stack(final, dim=0)
                sent.r_embed = final if not self.finetune else None
            else:
                final = sent.r_embed
            batch_r_embed.append(final)
        return batch_r_embed

    # [e_i^1, e_j^2, e_i^3, e_j^4]
    def encode_coherent(self, sents: List[SCIDTBSentence]) -> List[torch.Tensor]:
        batch_r_embed = []
        quat_dim = EDU_REPRESENTATION_DIM['encode_coherent'] // 4
        for sent in sents:
            if sent.r_embed is None:
                edus_representation = self.cs_encoder.sent_sensitive_encode(sent.raw)
                final = []
                for edu_representation in edus_representation:
                    edu_representation = torch.cat([edu_representation[0][0: quat_dim],
                                                    edu_representation[-1][quat_dim: 2*quat_dim],
                                                    edu_representation[0][2*quat_dim: 3*quat_dim],
                                                    edu_representation[-1][3*quat_dim:]], dim=-1)
                    final.append(edu_representation)
                final = torch.stack(final, dim=0)
                sent.r_embed = final if not self.finetune else None
            else:
                final = sent.r_embed
            batch_r_embed.append(final)
        return batch_r_embed

    # attention encode
    def encode_attention(self, sents: List[SCIDTBSentence]) -> List[torch.Tensor]:
        batch_r_embed = []
        for sent in sents:
            if sent.r_embed is None:
                edus_representation = self.cs_encoder.sent_sensitive_encode(sent.raw)
                final = []
                for edu_representation in edus_representation:
                    alpha = torch.softmax(torch.matmul(edu_representation, self.attention_key), dim=0)

                    edu_representation = torch.sum(alpha.unsqueeze(1) * edu_representation, dim=0)
                    final.append(edu_representation)
                final = torch.stack(final, dim=0)
                sent.r_embed = final if not self.finetune else None
            else:
                final = sent.r_embed
            batch_r_embed.append(final)
        return batch_r_embed

    def real_forward(self, sentences):
        r_embed = self.process_batch(sentences)
        r_embed = torch.stack(r_embed, dim=0)
        std = getattr(self, 'std', None)
        if std is not None:
            r_embed = r_embed / std

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

    def forward(self, sentences):
        score_mat = self.real_forward(sentences)
        score_mat = torch.log_softmax(score_mat, dim=2)
        # batch, length = score_mat.size()[:2]
        # gold = torch.tensor([sent.heads for sent in sentences]).long().to(score_mat.device)
        ll = 0.0
        for idx, sentence in enumerate(sentences):
            partition = self.esiner.partition(arc_scores=score_mat[idx], edu_ids=sentence.raw.edu_ids,
                                              sbnds=sentence.raw.sbnds, pbnds=sentence.raw.pbnds,
                                              use_sbnds=True, use_pbnds=False)
            prob = 0.0
            for arc in sentence.raw.arcs:
                prob += score_mat[idx][arc[0]][arc[1]]
            ll += (prob - partition)
        # ll = self.crie(score_mat.view(-1, length), gold.view(-1)).reshape(batch, length)
        return ll / len(sentences)

    def decoding(self, sentences):
        batch_score_mat = self.real_forward(sentences)
        batch_score_mat = torch.log_softmax(batch_score_mat, dim=2)
        best_tree = []
        batch_score_mat_np = batch_score_mat.permute(0, 2, 1).detach().cpu().numpy()
        for idx, (sentence, score_mat_np) in enumerate(zip(sentences, batch_score_mat_np)):
            unlabeled_arcs = self.esiner.global_decode(arc_scores=score_mat_np, edu_ids=sentence.raw.edu_ids,
                                                       sbnds=sentence.raw.sbnds, pbnds=sentence.raw.pbnds,
                                                       use_sbnds=True, use_pbnds=False)
            pred_arcs = [arc[0] for arc in sorted(unlabeled_arcs, key=lambda x: x[1])]
            best_tree.append([0] + pred_arcs)
        return best_tree

    @torch.no_grad()
    def prepare_norm(self, dataset):
        self.eval()
        batch_encode = self.process_batch(dataset)
        std = torch.std(torch.cat(batch_encode, dim=0))
        setattr(self, 'std', std)


