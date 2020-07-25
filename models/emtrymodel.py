import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import utils


class EmTryModel(nn.Module):

    def __init__(self,
                 vocab_word,
                 vocab_postag,
                 vocab_deprel,
                 vocab_relation,
                 word_dim,
                 postag_dim,
                 deprel_dim,
                 lstm_dim,
                 mlp_dim,
                 initialW,
                 template_feature_extractor1,
                 template_feature_extractor2,
                 device):
        """
        :type vocab_word: {str: int}
        :type vocab_postag: {str: int}
        :type vocab_deprel: {str: int}
        :type vocab_relation: {str: int}
        :type word_dim: int
        :type postag_dim: int
        :type deprel_dim: int
        :type lstm_dim: int
        :type mlp_dim: int
        :type initialW: numpy.ndarray(shape=(|V|, word_dim), dtype=np.float32)
        :type template_feature_extractor1: TemplateFeatureExtractor1
        :type template_feature_extractor2: TemplateFeatureExtractor2
        """
        super(EmTryModel, self).__init__()
        assert "<unk>" in vocab_word
        assert "<unk>" in vocab_deprel
        assert "<root>" in vocab_word
        assert "<root>" in vocab_postag
        assert "<root>" in vocab_deprel

        self.device = device

        self.vocab_word = vocab_word
        self.vocab_postag = vocab_postag
        self.vocab_deprel = vocab_deprel
        self.vocab_relation = vocab_relation
        self.ivocab_relation = {i: l for l, i in self.vocab_relation.items()}

        # Word embedding
        self.word_dim = word_dim
        self.postag_dim = postag_dim
        self.deprel_dim = deprel_dim

        # BiLSTM over EDUs
        self.lstm_dim = lstm_dim
        self.bilstm_dim = lstm_dim + lstm_dim

        # Template features
        self.template_feature_extractor1 = template_feature_extractor1
        self.template_feature_extractor2 = template_feature_extractor2
        self.tempfeat1_dim = self.template_feature_extractor1.feature_size
        self.tempfeat2_dim = self.template_feature_extractor2.feature_size

        # MLP
        self.mlp_dim = mlp_dim
        self.n_relations = len(self.vocab_relation)

        self.unk_word_id = self.vocab_word["<unk>"]
        self.unk_deprel_id = self.vocab_deprel["<unk>"]

        # links = {}
        # EDU embedding
        # links["embed_word"] = L.EmbedID(len(self.vocab_word),
        #                                 self.word_dim,
        #                                 ignore_label=-1,
        #                                 initialW=initialW)

        self.embed_word = nn.Embedding(len(self.vocab_word),
                                       self.word_dim)
        self.embed_word.weight.data = torch.tensor(initialW)
        # links["embed_postag"] = L.EmbedID(len(self.vocab_postag),
        #                                   self.postag_dim,
        #                                   ignore_label=-1,
        #                                   initialW=None)
        self.embed_postag = nn.Embedding(len(self.vocab_postag),
                                         self.postag_dim)

        # links["embed_deprel"] = L.EmbedID(len(self.vocab_deprel),
        #                                   self.deprel_dim,
        #                                   ignore_label=-1,
        #                                   initialW=None)
        self.embed_deprel = nn.Embedding(len(self.vocab_deprel),
                                         self.deprel_dim)
        # links["W_edu"] = L.Linear(self.word_dim + self.postag_dim +
        #                           self.word_dim + self.postag_dim +
        #                           self.word_dim + self.postag_dim + self.deprel_dim,
        #                           self.word_dim)
        self.W_edu = nn.Linear(self.word_dim + self.postag_dim +
                               self.word_dim + self.postag_dim +
                               self.word_dim + self.postag_dim + self.deprel_dim,
                               self.word_dim)
        # BiLSTM
        # links["bilstm"] = L.NStepBiLSTM(n_layers=1,
        #                                 in_size=self.word_dim,
        #                                 out_size=self.lstm_dim,
        #                                 dropout=0.0)
        self.bilstm = nn.LSTM(num_layers=1,
                              input_size=self.word_dim,
                              hidden_size=self.lstm_dim,
                              bidirectional=True,
                              dropout=0.0)
        # MLPs
        # links["W1_a"] = L.Linear(self.bilstm_dim + self.tempfeat1_dim +
        #                          self.bilstm_dim + self.tempfeat1_dim +
        #                          self.tempfeat2_dim,
        #                          self.mlp_dim)
        self.W1_a = nn.Linear(self.bilstm_dim + self.tempfeat1_dim +
                              self.bilstm_dim + self.tempfeat1_dim +
                              self.tempfeat2_dim,
                              self.mlp_dim)
        # links["W2_a"] = L.Linear(self.mlp_dim, 1)
        self.W2_a = nn.Linear(self.mlp_dim, 1)
        # links["W1_r"] = L.Linear(self.bilstm_dim + self.tempfeat1_dim +
        #                          self.bilstm_dim + self.tempfeat1_dim +
        #                          self.tempfeat2_dim,
        #                          self.mlp_dim)
        self.W1_r = nn.Linear(self.bilstm_dim + self.tempfeat1_dim +
                              self.bilstm_dim + self.tempfeat1_dim +
                              self.tempfeat2_dim,
                              self.mlp_dim)
        # links["W2_r"] = L.Linear(self.mlp_dim, self.n_relations)
        self.W2_r = nn.Linear(self.mlp_dim, self.n_relations)

    #########################
    # Forwarding

    def forward_edus(self, edus, edus_postag, edus_head):
        """
        :type edus: list of list of str
        :type edus_postag: list of list of str
        :type edus_head: list of (str, str, str)
        :rtype: Variable(shape=(n_edus, bilstm_dim + tempfeat1_dim), dtype=np.float32)
        """
        assert len(edus[0]) == 1 # NOTE
        assert edus[0][0] == "<root>" # NOTE

        assert len(edus_postag[0]) == 1 # NOTE
        assert edus_postag[0][0] == "<root>" # NOTE

        assert len(edus_head[0]) == 3 # NOTE
        assert edus_head[0] == ("<root>", "<root>", "<root>") # NOTE


        # Beginning-word embedding
        begin_word_ids = [self.vocab_word.get(edu[0], self.unk_word_id) for edu in edus] # n_edus * int
        begin_word_ids = np.asarray(begin_word_ids, dtype=np.int32) # (n_edus,)
        begin_word_ids = torch.tensor(begin_word_ids).long().to(self.device) # (n_edus,)
        begin_word_vectors = F.dropout(self.embed_word(begin_word_ids), p=0.2) # (n_edus, word_dim)

        # End-word embedding
        end_word_ids = [self.vocab_word.get(edu[-1], self.unk_word_id) for edu in edus] # n_edus * int
        end_word_ids = np.asarray(end_word_ids, dtype=np.int32) # (n_edus,)
        end_word_ids = torch.tensor(end_word_ids).long().to(self.device) # (n_edus,)
        end_word_vectors = F.dropout(self.embed_word(end_word_ids), p=0.2) # (n_edus, word_dim)

        # Head-word embedding
        head_word_ids = [self.vocab_word.get(head_word, self.unk_word_id)
                         for (head_word, head_postag, head_deprel) in edus_head] # n_edus * int
        head_word_ids = np.asarray(head_word_ids, dtype=np.int32) # (n_edus,)
        head_word_ids = torch.tensor(head_word_ids).long().to(self.device) # (n_edus,)
        head_word_vectors = F.dropout(self.embed_word(head_word_ids), p=0.2) # (n_edus, word_dim)

        # Beginning-postag embedding
        begin_postag_ids = [self.vocab_postag[edu_postag[0]] for edu_postag in edus_postag] # n_edus * int
        begin_postag_ids = np.asarray(begin_postag_ids, dtype=np.int32) # (n_edus,)
        begin_postag_ids = torch.tensor(begin_postag_ids).long().to(self.device) # (n_edus,)
        begin_postag_vectors = F.dropout(self.embed_postag(begin_postag_ids), p=0.2) # (n_edus, postag_dim)

        # End-postag embedding
        end_postag_ids = [self.vocab_postag[edu_postag[-1]] for edu_postag in edus_postag] # n_edus * int
        end_postag_ids = np.asarray(end_postag_ids, dtype=np.int32) # (n_edus,)
        end_postag_ids = torch.tensor(end_postag_ids).long().to(self.device) # (n_edus,)
        end_postag_vectors = F.dropout(self.embed_postag(end_postag_ids), p=0.2) # (n_edus, postag_dim)

        # Head-postag embedding
        head_postag_ids = [self.vocab_postag[head_postag]
                         for (head_word, head_postag, head_deprel) in edus_head] # n_edus * int
        head_postag_ids = np.asarray(head_postag_ids, dtype=np.int32) # (n_edus,)
        head_postag_ids = torch.tensor(head_postag_ids).long().to(self.device) # (n_edus,)
        head_postag_vectors = F.dropout(self.embed_postag(head_postag_ids), p=0.2) # (n_edus, postag_dim)

        # Head-deprel embedding
        head_deprel_ids = [self.vocab_deprel.get(head_deprel, self.unk_deprel_id)
                         for (head_word, head_postag, head_deprel) in edus_head] # n_edus * int
        head_deprel_ids = np.asarray(head_deprel_ids, dtype=np.int32) # (n_edus,)
        head_deprel_ids = torch.tensor(head_deprel_ids).long().to(self.device) # (n_edus,)
        head_deprel_vectors = F.dropout(self.embed_deprel(head_deprel_ids), p=0.2) # (n_edus, deprel_dim)

        # Concat
        edu_vectors = torch.cat([begin_word_vectors,
                                end_word_vectors,
                                head_word_vectors,
                                begin_postag_vectors,
                                end_postag_vectors,
                                head_postag_vectors,
                                head_deprel_vectors],
                                dim=1) # (n_edus, 3 * word_dim + 3 * postag_dim + deprel_dim)
        edu_vectors = F.relu(self.W_edu(edu_vectors)) # (n_edus, word_dim)

        # BiLSTM
        packed_edu = nn.utils.rnn.pack_padded_sequence(edu_vectors.unsqueeze(0),
                                                       torch.tensor([edu_vectors.size(0)]),
                                                       batch_first=True,
                                                       enforce_sorted=False)
        packed_output, hidden = self.bilstm(packed_edu)
        states, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # h_init, c_init = None, None
        # _, _, states = self.bilstm(hx=h_init, cx=c_init, xs=[edu_vectors]) # (1, n_edus, bilstm_dim)
        edu_vectors = states.squeeze(0) # (n_edus, bilstm_dim)

        # Template features
        tempfeat1_vectors = self.template_feature_extractor1.extract_batch_features(
                                    edus=edus,
                                    edus_postag=edus_postag,
                                    edus_head=edus_head) # (n_edus, tempfeat1_dim)
        tempfeat1_vectors = torch.tensor(tempfeat1_vectors).to(self.device) # (n_edus, tempfeat1_dim)

        # Concat
        edu_vectors = torch.cat([edu_vectors, tempfeat1_vectors], dim=1) # (n_edus, bilstm_dim + tempfeat1_dim)

        return edu_vectors

    def forward_arcs_for_attachment(
                    self,
                    edu_vectors,
                    same_sent_map,
                    batch_arcs,
                    aggregate=True):
        """
        :type edu_vectors: Variable(shape=(n_edus, bilstm_dim + tempfeat1_dim), dtype=np.float32) # XXX
        :type same_sent_map: numpy.ndarray(shape=(n_edus, n_edus), dtype=np.int32)
        :type batch_arcs: list of list of (int, int)
        :type aggregate: bool
        :rtype: Variable(shape=(batch_size,1)/(batch_size,n_arcs,1), dtype=np.float32)
        """
        batch_size = len(batch_arcs)
        n_arcs = len(batch_arcs[0])
        # total_arcs = batch_size * n_arcs
        # for arcs in batch_arcs:
        #     assert len(arcs) == n_arcs

        # Reshape
        flatten_batch_arcs = utils.flatten_lists(batch_arcs) # total_arcs * (int, int)
        batch_head, batch_dep = zip(*flatten_batch_arcs)
        batch_head = list(batch_head) # total_arcs * int
        batch_dep = list(batch_dep) # total_arcs * int

        # Feature extraction
        batch_head_vectors = torch.embedding(edu_vectors, torch.tensor(batch_head).to(self.device)) # (total_arcs, bilstm_dim + tempfeat1_dim)
        batch_dep_vectors = torch.embedding(edu_vectors, torch.tensor(batch_dep).to(self.device)) # (total_arcs, bilstm_dim + tempfeat1_dim)
        tempfeat2_vectors = self.template_feature_extractor2.extract_batch_features(
                                    batch_head,
                                    batch_dep,
                                    same_sent_map) # (total_arcs, tempfeat2_dim)

        tempfeat2_vectors = torch.tensor(tempfeat2_vectors).to(self.device) # (total_arcs, tempfeat2_dim)
        batch_arc_vectors = torch.cat([batch_head_vectors,
                                      batch_dep_vectors,
                                      tempfeat2_vectors],
                                      dim=1) # (total_arcs, bilstm_dim + tempfeat1_dim + bilstm_dim + tempfeat1_dim + tempfeat2_dim)

        # MLP (Attachment Scoring)
        arc_scores = self.W2_a(F.dropout(F.relu(self.W1_a(batch_arc_vectors)), p=0.2)) # (total_arcs, 1)
        arc_scores = torch.reshape(arc_scores, (batch_size, n_arcs, 1)) # (batch_size, n_arcs, 1)

        # Aggregate
        if aggregate:
            tree_scores = torch.sum(arc_scores, dim=1) # (batch_size, 1)
            return tree_scores
        else:
            return arc_scores # (batch_size, n_arcs, 1)

    def forward_arcs_for_labeling(
                    self,
                    edu_vectors,
                    same_sent_map,
                    batch_arcs):
        """
        :type edu_vectors: Variable(shape=(n_edus, bilstm_dim + tempfeat1_dim), dtype=np.float32)
        :type same_sent_map: numpy.ndarray(shape=(n_edus,n_edus), dtype=np.int32)
        :type batch_arcs: list of list of (int, int)
        :rtype: Variable(shape=(batch_size,n_arcs,n_relations), dtype=np.float32)
        """
        batch_size = len(batch_arcs)
        n_arcs = len(batch_arcs[0])
        # total_arcs = batch_size * n_arcs
        # for arcs in batch_arcs:
        #     assert len(arcs) == n_arcs

        # Reshape
        flatten_batch_arcs = utils.flatten_lists(batch_arcs)  # total_arcs * (int, int)
        batch_head, batch_dep = zip(*flatten_batch_arcs)
        batch_head = list(batch_head)  # total_arcs * int
        batch_dep = list(batch_dep)  # total_arcs * int

        # Feature extraction
        batch_head_vectors = torch.embedding(edu_vectors, torch.tensor(batch_head).to(self.device))  # (total_arcs, bilstm_dim + tempfeat1_dim)
        batch_dep_vectors = torch.embedding(edu_vectors, torch.tensor(batch_dep).to(self.device))  # (total_arcs, bilstm_dim + tempfeat1_dim)
        tempfeat2_vectors = self.template_feature_extractor2.extract_batch_features(
                                    batch_head,
                                    batch_dep,
                                    same_sent_map) # (total_arcs, tempfeat2_dim)
        tempfeat2_vectors = torch.tensor(tempfeat2_vectors).to(self.device) # (total_arcs, tempfeat2_dim)
        batch_arc_vectors = torch.cat([batch_head_vectors,
                                      batch_dep_vectors,
                                      tempfeat2_vectors],
                                      dim=1)  # (total_arcs, bilstm_dim + tempfeat1_dim + bilstm_dim + tempfeat1_dim + tempfeat2_dim)

        # MLP (labeling)
        logits = self.W2_r(F.dropout(F.relu(self.W1_r(batch_arc_vectors)), p=0.2))  # (total_arcs, n_relations)
        logits = torch.reshape(logits, (batch_size, n_arcs, self.n_relations))  # (batch_size, n_arcs, n_relations)

        return logits
