import functools
from typing import List, Tuple

import torch

# from model.context_sensitive_encoder import CSEncoder
from model.definition import ROOT


class PascalSentence():
    def __init__(self, words, ctags, ftags, utags, heads=None):
        self.words = [ROOT] + words
        self.ctags = [ROOT] + ctags
        self.ftags = [ROOT] + ftags
        self.utags = [ROOT] + utags
        self.heads = [-1] + heads
        self.length = len(self.words)

    def prior_weight(self):
        ret = torch.zeros(self.length, self.length)
        for i in range(self.length):
            for j in range(self.length):
                ret[i, j] = 1.0 if PascalSentence.utags_linguistics_rule(self.utags[i], self.utags[j]) else 0.0
        return ret

    @staticmethod
    def utags_linguistics_rule(head, child):
        rules = [("VERB", "VERB"), ("VERB", "NOUN"), ("VERB", "PRON"), ("VERB", "ADV"), ("VERB", "ADP"),
                 ("NOUN", "NOUN"), ("NOUN", "ADJ"), ("NOUN", "DET"), ("NOUN", "NUM"), ("NOUN", "CONJ"),
                 ("ADJ", "ADV"), ("ADP", "NOUN")]
        return (head.upper(), child.upper()) in rules

    def __str__(self):
        return ("words={}, ctags={}, ftags={}, utags={}, heads={}"
                .format(self.words, self.ctags, self.ftags, self.utags, self.heads))

    def __len__(self):
        return self.length


class WSJSentence():
    def __init__(self, words, ftags, heads=None):
        self.words = [ROOT] + words
        self.ftags = [ROOT] + ftags
        self.heads = [-1] + heads
        self.length = len(self.words)

    def prior_weight(self):
        ret = torch.zeros(self.length, self.length)
        for i in range(self.length):
            for j in range(self.length):
                ret[i, j] = 1.0 if WSJSentence.ftags_linguistics_rule(self.ftags[i], self.ftags[j]) else 0.0
        return ret

    @staticmethod
    def ftags_linguistics_rule(head, child):
        rules = [(ROOT, "MD"), (ROOT, "VB"),
                 ("VB", "NN"), ("VB", "WP"), ("VB", "PR"), ("VB", "RB"), ("VB", "VB"),
                 ("NN", "JJ"), ("NN", "NN"), ("NN", "CD"), ("IN", "NN"),
                 ("JJ", "RB"), ("MD", "VB")]

        matches = [head.upper().startswith(a) and child.upper().startswith(b) for (a, b) in rules]
        return any(matches)

    def __len__(self):
        return self.length

    def __str__(self):
        return "words={}, ftags={}, heads={}".format(self.words, self.ftags, self.heads)



class SCIDTBSentence():
    def __init__(self, words, ftags, heads=None, sbnds=None, raw=None, rels=None):
        self.words = [[ROOT]] + words
        self.ftags = [ROOT] + ftags
        self.heads = [0] + heads
        self.sbnds = sbnds
        self.length = len(self.words)
        self.r_embed = None
        self.raw = raw
        self.supervised = False
        self.rels = rels

    def prior_weight(self):
        return self.make_same_sent_map(self.length, self.sbnds)

    @staticmethod
    def make_same_sent_map(length, sbnds):
        """
        :type length: int
        :type sbnds: list of (int, int)
        :rtype: numpy.ndarray(shape=(n_edus, n_edus), dtype=np.int32)
        """
        # NOTE: Indices (b, e) \in sbnds are shifted by -1 compared to EDU IDs which include ROOT.
        #       For example, (b, e) \in sbnds indicates that EDUs[b+1:e+1+1] belongs to one sentence.
        same_sent_map = torch.zeros((length, length))
        for begin_i, end_i in sbnds:
            same_sent_map[begin_i + 1:end_i + 1 + 1, begin_i + 1:end_i + 1 + 1] = 1.0
        return same_sent_map

    def __len__(self):
        return self.length

    def __str__(self):
        return "words={}, ftags={}, heads={}".format(self.words, self.ftags, self.heads)


class SentencePreprocessor():
    def __init__(self, token_extract_method, train, dev, test, use_gpu):
        self.use_gpu = use_gpu
        self.token_extract_method = token_extract_method

        train_sents_tokens = list(map(token_extract_method, train))
        dev_sents_tokens = list(map(token_extract_method, dev))
        test_sents_tokens = list(map(token_extract_method, test))

        train_sents_tokens = functools.reduce(lambda x, y: x+y, train_sents_tokens)
        dev_sents_tokens = functools.reduce(lambda x, y: x+y, dev_sents_tokens)
        test_sents_tokens = functools.reduce(lambda x, y: x+y, test_sents_tokens)

        token_set = functools.reduce(lambda x, y: x | y,
                                     map(lambda t: set(t), train_sents_tokens + dev_sents_tokens + test_sents_tokens),
                                     set())
        self.token_id_map = {t: i for i, t in enumerate(sorted(list(token_set)))}
        if '<_PAD>' not in self.token_id_map:
            self.token_id_map['<_PAD>'] = len(self.token_id_map)
        self.token_num = len(self.token_id_map)

    def process_batch(self, sents: List[PascalSentence]) -> torch.autograd.Variable:
        sents_tokens = list(map(self.token_extract_method, sents))
        sents_token_ids = list(map(lambda t: list(map(lambda m: self.token_id_map[m], t)), sents_tokens))

        input = torch.LongTensor(sents_token_ids)

        if self.use_gpu:
            feat_v = torch.autograd.Variable(input.cuda(), requires_grad=False)
        else:
            feat_v = torch.autograd.Variable(input, requires_grad=False)
        return feat_v

    def process_discourse_batch(self, sents: List[PascalSentence]) -> Tuple[torch.Tensor, torch.LongTensor]:
        sents_tokens = list(map(self.token_extract_method, sents))
        sents_token_ids = list(map(lambda t: list(map(lambda k: list(map(lambda m: self.token_id_map[m], k)), t)), sents_tokens))

        input, mask = self.padding_sent(sents_token_ids)

        if self.use_gpu:
            feat_v = input.cuda()
            feat_v.requires_grad = False
            mask = mask.cuda()
            mask.requires_grad = False
        else:
            feat_v = input
            feat_v.requires_grad = False
            mask.requires_grad = False
        return feat_v, mask

    def padding_sent(self, sents: List[List[int]]) -> Tuple[torch.LongTensor, torch.LongTensor]:
        max_edu_len = 0
        masks = []
        for sent in sents:
            for edu in sent:
                if max_edu_len < len(edu):
                    max_edu_len = len(edu)
        for sent in sents:
            mask = []
            for idx, edu in enumerate(sent):

                mask.append([1] * len(edu) + [0] * (max_edu_len - len(edu)))
                sent[idx] = edu + [self.token_id_map['<_PAD>']] * (max_edu_len - len(edu))
            masks.append(mask)
        return torch.LongTensor(sents), torch.LongTensor(masks)


class DiscoursePreprocessor:
    def __init__(self, token_extract_method, cs_encoder, train, dev, test, use_gpu):
        self.use_gpu = use_gpu
        self.token_extract_method = token_extract_method
        self.cs_encoder = None # CSEncoder(cs_encoder)
        train_sents_tokens = list(map(token_extract_method, train))
        dev_sents_tokens = list(map(token_extract_method, dev))
        test_sents_tokens = list(map(token_extract_method, test))

        train_sents_tokens = functools.reduce(lambda x, y: x+y, train_sents_tokens)
        dev_sents_tokens = functools.reduce(lambda x, y: x+y, dev_sents_tokens)
        test_sents_tokens = functools.reduce(lambda x, y: x+y, test_sents_tokens)

        token_set = functools.reduce(lambda x, y: x | y,
                                     map(lambda t: set(t), train_sents_tokens + dev_sents_tokens + test_sents_tokens),
                                     set())
        self.token_id_map = {t: i for i, t in enumerate(sorted(list(token_set)))}
        self.token_num = len(self.token_id_map)
        PROCESSMETHOD = {'minus': self.process_sm_batch,
                         'edu': self.process_cs_batch,
                         'sent': self.process_ss_batch,
                         'avg_pooling': self.process_map_batch,
                         'max_pooling': self.process_map_batch,
                         'mean': self.process_mean_batch}

    def process_discourse_batch(self, sents: List[SCIDTBSentence]) -> Tuple[torch.Tensor, torch.LongTensor]:
        sents_tokens = list(map(self.token_extract_method, sents))
        sents_token_ids = list(map(lambda t: list(map(lambda k: list(map(lambda m: self.token_id_map[m], k)), t)), sents_tokens))

        input, mask = self.padding_sent(sents_token_ids)

        if self.use_gpu:
            feat_v = input.cuda()
            feat_v.requires_grad = False
            mask = mask.cuda()
            mask.requires_grad = False
        else:
            feat_v = input
            feat_v.requires_grad = False
            mask.requires_grad = False
        return feat_v, mask

    def padding_sent(self, sents: List[List[int]]) -> Tuple[torch.LongTensor, torch.LongTensor]:
        max_edu_len = 0
        masks = []
        for sent in sents:
            for edu in sent:
                if max_edu_len < len(edu):
                    max_edu_len = len(edu)
        for sent in sents:
            mask = []
            for idx, edu in enumerate(sent):

                mask.append([1] * len(edu) + [0] * (max_edu_len - len(edu)))
                sent[idx] = edu + [self.token_id_map['<_PAD>']] * (max_edu_len - len(edu))
            masks.append(mask)
        return torch.LongTensor(sents), torch.LongTensor(masks)

    # process context sensitive batch
    def process_cs_batch(self, sents: List[SCIDTBSentence]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_p_embed = []
        batch_r_embed = []
        for sent in sents:
            if sent.r_embed is None or sent.r_embed is None:
                embed1, embed2 = self.cs_encoder.encode(sent.raw)
                sent.r_embed = embed1
                sent.p_embed = embed2
            batch_r_embed.append(sent.r_embed)
            batch_p_embed.append(sent.p_embed)
        return torch.stack(batch_r_embed, dim=0),torch.stack(batch_p_embed, dim=0)

    # process sentence sensitive edu
    def process_ss_batch(self, discourses: List[SCIDTBSentence]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_p_embed = []
        batch_r_embed = []
        batch_s_embed = []
        for discourse in discourses:
            if discourse.r_embed is None:
                embed1, embed2 = self.cs_encoder.encode(discourse.raw)
                discourse.r_embed = embed1
                discourse.p_embed = embed2

                s_embed = self.cs_encoder.encode_sentence(discourse.raw)

                # sent merge
                rep_list = torch.split(s_embed[:, 0], 1, dim=0)
                final = [rep_list[0]]
                _, dim = rep_list[0].size()
                for span, one_represent in zip(discourse.sbnds, rep_list[1:]):
                    final.append(one_represent.expand(span[1] - span[0] + 1, dim))

                discourse.s_embed = torch.cat(final, dim=0).detach()
            batch_r_embed.append(discourse.r_embed)
            batch_p_embed.append(discourse.p_embed)
            batch_s_embed.append(discourse.s_embed)
        concat_r_embed = torch.cat([torch.stack(batch_r_embed, dim=0), torch.stack(batch_s_embed, dim=0)], dim=2)
        concat_p_embed = torch.cat([torch.stack(batch_p_embed, dim=0), torch.stack(batch_s_embed, dim=0)], dim=2)
        return concat_r_embed, concat_p_embed

    # sentence edu minus representation
    # process sentence sensitive edu
    def process_sm_batch(self, sents: List[SCIDTBSentence]) -> torch.Tensor:
        batch_r_embed = []
        for sent in sents:
            if sent.r_embed is None:
                embed1 = self.cs_encoder.sent_minus_encode(sent.raw)
                sent.r_embed = embed1
            batch_r_embed.append(sent.r_embed)

        concat_r_embed = torch.stack(batch_r_embed, dim=0)
        return concat_r_embed / torch.std(concat_r_embed)

    # avg pooling the result
    def process_sap_batch(self, sents: List[SCIDTBSentence]) -> torch.Tensor:
        batch_r_embed = []
        for sent in sents:
            if sent.r_embed is None:
                embed1 = self.cs_encoder.sent_avg_pooling_encode(sent.raw)
                sent.r_embed = embed1
            batch_r_embed.append(sent.r_embed)
        concat_r_embed = torch.stack(batch_r_embed, dim=0)
        return concat_r_embed

    # max pooling the result
    def process_map_batch(self, sents: List[SCIDTBSentence]) -> torch.Tensor:
        batch_r_embed = []
        for sent in sents:
            if sent.r_embed is None:
                embed1 = self.cs_encoder.sent_max_pooling_encode(sent.raw)
                sent.r_embed = embed1
            batch_r_embed.append(sent.r_embed)
        concat_r_embed = torch.stack(batch_r_embed, dim=0)
        return concat_r_embed

    # max pooling the result
    def process_mean_batch(self, sents: List[SCIDTBSentence]) -> torch.Tensor:
        batch_r_embed = []
        for sent in sents:
            if sent.r_embed is None:
                embed1 = self.cs_encoder.sent_mean_encode(sent.raw)
                sent.r_embed = embed1
            batch_r_embed.append(sent.r_embed)
        concat_r_embed = torch.stack(batch_r_embed, dim=0)
        return concat_r_embed

