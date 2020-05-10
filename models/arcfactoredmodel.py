import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

import utils

class ArcFactoredModel(chainer.Chain):

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
                 template_feature_extractor2):
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
        assert "<unk>" in vocab_word
        assert "<unk>" in vocab_deprel
        assert "<root>" in vocab_word
        assert "<root>" in vocab_postag
        assert "<root>" in vocab_deprel

        self.vocab_word = vocab_word
        self.vocab_postag = vocab_postag
        self.vocab_deprel = vocab_deprel
        self.vocab_relation = vocab_relation
        self.ivocab_relation = {i:l for l,i in self.vocab_relation.items()}

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

        links = {}
        # EDU embedding
        links["embed_word"] = L.EmbedID(len(self.vocab_word),
                                        self.word_dim,
                                        ignore_label=-1,
                                        initialW=initialW)
        links["embed_postag"] = L.EmbedID(len(self.vocab_postag),
                                          self.postag_dim,
                                          ignore_label=-1,
                                          initialW=None)
        links["embed_deprel"] = L.EmbedID(len(self.vocab_deprel),
                                          self.deprel_dim,
                                          ignore_label=-1,
                                          initialW=None)
        links["W_edu"] = L.Linear(self.word_dim + self.postag_dim +
                                  self.word_dim + self.postag_dim +
                                  self.word_dim + self.postag_dim + self.deprel_dim,
                                  self.word_dim)
        # BiLSTM
        links["bilstm"] = L.NStepBiLSTM(n_layers=1,
                                        in_size=self.word_dim,
                                        out_size=self.lstm_dim,
                                        dropout=0.0)
        # MLPs
        links["W1_a"] = L.Linear(self.bilstm_dim + self.tempfeat1_dim +
                                 self.bilstm_dim + self.tempfeat1_dim +
                                 self.tempfeat2_dim,
                                 self.mlp_dim)
        links["W2_a"] = L.Linear(self.mlp_dim, 1)
        links["W1_r"] = L.Linear(self.bilstm_dim + self.tempfeat1_dim +
                                 self.bilstm_dim + self.tempfeat1_dim +
                                 self.tempfeat2_dim,
                                 self.mlp_dim)
        links["W2_r"] = L.Linear(self.mlp_dim, self.n_relations)
        super(ArcFactoredModel, self).__init__(**links)

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

        #################
        # TODO?
        # Bag-of-word embeddings
        # word_ids = [[self.vocab_word.get(w, self.unk_word_id) for w in edu]
        #             for edu in edus] # n_edus * length * int
        # word_ids, mask = utils.padding(word_ids, head=True, with_mask=True) # (n_edus, max_length), (n_edus, max_length)
        # n_edus, max_length = word_ids.shape
        # word_ids = utils.convert_ndarray_to_variable(word_ids, seq=False) # (n_edus, max_length)
        # mask = utils.convert_ndarray_to_variable(mask, seq=False) # (n_edus, max_length)
        # word_ids = F.reshape(word_ids, (n_edus * max_length,)) # (n_edus * max_length,)
        # word_vectors = F.dropout(self.embed(word_ids), ratio=0.2) # (n_edus * max_length, word_dim)
        # word_vectors = F.reshape(word_vectors, (n_edus, max_length, self.word_dim)) # (n_edus, max_length, word_dim)
        # mask = F.broadcast_to(mask[:,:,None], (n_edus, max_length, self.word_dim)) # (n_edus, max_length, word_dim)
        # word_vectors = word_vectors * mask # (n_edus, max_length, word_dim)
        # bow_vectors = F.sum(word_vectors, axis=1) # (n_edus, word_dim)
        #################

        # Beginning-word embedding
        begin_word_ids = [self.vocab_word.get(edu[0], self.unk_word_id) for edu in edus] # n_edus * int
        begin_word_ids = np.asarray(begin_word_ids, dtype=np.int32) # (n_edus,)
        begin_word_ids = utils.convert_ndarray_to_variable(begin_word_ids, seq=False) # (n_edus,)
        begin_word_vectors = F.dropout(self.embed_word(begin_word_ids), ratio=0.2) # (n_edus, word_dim)

        # End-word embedding
        end_word_ids = [self.vocab_word.get(edu[-1], self.unk_word_id) for edu in edus] # n_edus * int
        end_word_ids = np.asarray(end_word_ids, dtype=np.int32) # (n_edus,)
        end_word_ids = utils.convert_ndarray_to_variable(end_word_ids, seq=False) # (n_edus,)
        end_word_vectors = F.dropout(self.embed_word(end_word_ids), ratio=0.2) # (n_edus, word_dim)

        # Head-word embedding
        head_word_ids = [self.vocab_word.get(head_word, self.unk_word_id)
                         for (head_word, head_postag, head_deprel) in edus_head] # n_edus * int
        head_word_ids = np.asarray(head_word_ids, dtype=np.int32) # (n_edus,)
        head_word_ids = utils.convert_ndarray_to_variable(head_word_ids, seq=False) # (n_edus,)
        head_word_vectors = F.dropout(self.embed_word(head_word_ids), ratio=0.2) # (n_edus, word_dim)

        # Beginning-postag embedding
        begin_postag_ids = [self.vocab_postag[edu_postag[0]] for edu_postag in edus_postag] # n_edus * int
        begin_postag_ids = np.asarray(begin_postag_ids, dtype=np.int32) # (n_edus,)
        begin_postag_ids = utils.convert_ndarray_to_variable(begin_postag_ids, seq=False) # (n_edus,)
        begin_postag_vectors = F.dropout(self.embed_postag(begin_postag_ids), ratio=0.2) # (n_edus, postag_dim)

        # End-postag embedding
        end_postag_ids = [self.vocab_postag[edu_postag[-1]] for edu_postag in edus_postag] # n_edus * int
        end_postag_ids = np.asarray(end_postag_ids, dtype=np.int32) # (n_edus,)
        end_postag_ids = utils.convert_ndarray_to_variable(end_postag_ids, seq=False) # (n_edus,)
        end_postag_vectors = F.dropout(self.embed_postag(end_postag_ids), ratio=0.2) # (n_edus, postag_dim)

        # Head-postag embedding
        head_postag_ids = [self.vocab_postag[head_postag]
                         for (head_word, head_postag, head_deprel) in edus_head] # n_edus * int
        head_postag_ids = np.asarray(head_postag_ids, dtype=np.int32) # (n_edus,)
        head_postag_ids = utils.convert_ndarray_to_variable(head_postag_ids, seq=False) # (n_edus,)
        head_postag_vectors = F.dropout(self.embed_postag(head_postag_ids), ratio=0.2) # (n_edus, postag_dim)

        # Head-deprel embedding
        head_deprel_ids = [self.vocab_deprel.get(head_deprel, self.unk_deprel_id)
                         for (head_word, head_postag, head_deprel) in edus_head] # n_edus * int
        head_deprel_ids = np.asarray(head_deprel_ids, dtype=np.int32) # (n_edus,)
        head_deprel_ids = utils.convert_ndarray_to_variable(head_deprel_ids, seq=False) # (n_edus,)
        head_deprel_vectors = F.dropout(self.embed_deprel(head_deprel_ids), ratio=0.2) # (n_edus, deprel_dim)

        # Concat
        edu_vectors = F.concat([begin_word_vectors,
                                end_word_vectors,
                                head_word_vectors,
                                begin_postag_vectors,
                                end_postag_vectors,
                                head_postag_vectors,
                                head_deprel_vectors],
                                axis=1) # (n_edus, 3 * word_dim + 3 * postag_dim + deprel_dim)
        edu_vectors = F.relu(self.W_edu(edu_vectors)) # (n_edus, word_dim)

        # BiLSTM
        h_init, c_init = None, None
        _, _, states = self.bilstm(hx=h_init, cx=c_init, xs=[edu_vectors]) # (1, n_edus, bilstm_dim)
        edu_vectors = states[0] # (n_edus, bilstm_dim)

        # Template features
        tempfeat1_vectors = self.template_feature_extractor1.extract_batch_features(
                                    edus=edus,
                                    edus_postag=edus_postag,
                                    edus_head=edus_head) # (n_edus, tempfeat1_dim)
        tempfeat1_vectors = utils.convert_ndarray_to_variable(tempfeat1_vectors, seq=False) # (n_edus, tempfeat1_dim)

        # Concat
        edu_vectors = F.concat([edu_vectors, tempfeat1_vectors], axis=1) # (n_edus, bilstm_dim + tempfeat1_dim)

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
        batch_head_vectors = F.get_item(edu_vectors, batch_head) # (total_arcs, bilstm_dim + tempfeat1_dim)
        batch_dep_vectors = F.get_item(edu_vectors, batch_dep) # (total_arcs, bilstm_dim + tempfeat1_dim)
        tempfeat2_vectors = self.template_feature_extractor2.extract_batch_features(
                                    batch_head,
                                    batch_dep,
                                    same_sent_map) # (total_arcs, tempfeat2_dim)
        tempfeat2_vectors = utils.convert_ndarray_to_variable(tempfeat2_vectors, seq=False) # (total_arcs, tempfeat2_dim)
        batch_arc_vectors = F.concat([batch_head_vectors,
                                      batch_dep_vectors,
                                      tempfeat2_vectors],
                                     axis=1) # (total_arcs, bilstm_dim + tempfeat1_dim + bilstm_dim + tempfeat1_dim + tempfeat2_dim)

        # MLP (Attachment Scoring)
        arc_scores = self.W2_a(F.dropout(F.relu(self.W1_a(batch_arc_vectors)), ratio=0.2)) # (total_arcs, 1)
        arc_scores = F.reshape(arc_scores, (batch_size, n_arcs, 1)) # (batch_size, n_arcs, 1)

        # Aggregate
        if aggregate:
            tree_scores = F.sum(arc_scores, axis=1) # (batch_size, 1)
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
        flatten_batch_arcs = utils.flatten_lists(batch_arcs) # total_arcs * (int, int)
        batch_head, batch_dep = zip(*flatten_batch_arcs)
        batch_head = list(batch_head) # total_arcs * int
        batch_dep = list(batch_dep) # total_arcs * int

        # Feature extraction
        batch_head_vectors = F.get_item(edu_vectors, batch_head) # (total_arcs, bilstm_dim + tempfeat1_dim)
        batch_dep_vectors = F.get_item(edu_vectors, batch_dep) # (total_arcs, bilstm_dim + tempfeat1_dim)
        tempfeat2_vectors = self.template_feature_extractor2.extract_batch_features(
                                    batch_head,
                                    batch_dep,
                                    same_sent_map) # (total_arcs, tempfeat2_dim)
        tempfeat2_vectors = utils.convert_ndarray_to_variable(tempfeat2_vectors, seq=False) # (total_arcs, tempfeat2_dim)
        batch_arc_vectors = F.concat([batch_head_vectors,
                                      batch_dep_vectors,
                                      tempfeat2_vectors],
                                      axis=1) # (total_arcs, bilstm_dim + tempfeat1_dim + bilstm_dim + tempfeat1_dim + tempfeat2_dim)

        # MLP (labeling)
        logits = self.W2_r(F.dropout(F.relu(self.W1_r(batch_arc_vectors)), ratio=0.2)) # (total_arcs, n_relations)
        logits = F.reshape(logits, (batch_size, n_arcs, self.n_relations)) # (batch_size, n_arcs, n_relations)

        return logits

