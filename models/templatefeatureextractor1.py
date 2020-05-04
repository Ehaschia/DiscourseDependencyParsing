from collections import Counter

import numpy as np
import pyprind

import utils
from utils import TemplateFeatureExtractor

class TemplateFeatureExtractor1(TemplateFeatureExtractor):

    def __init__(self, databatch, vocab_relation):
        """
        :type databatch: DataBatch
        :type vocab_relation: {str: int}
        """
        super().__init__()
        self.LENGTH_SPANS = [(1,2), (3,5), (6,10), (11,20), (21,np.inf)]
        self.DISTANCE_SPANS = [(0,0), (1,2), (3,5), (6,np.inf)]
        self.aggregate_templates(databatch=databatch, vocab_relation=vocab_relation)
        self.prepare()

    ##########################
    def build_ngrams(self, batch_edus, batch_arcs, vocab_relation, threshold):
        """
        :type batch_edus: list of list of list of str
        :type batch_arcs: list of list of (int, int, str)
        :type vocab_relation: {str: int}
        :type threshold: float
        :rtype: list of (str, float), list of (str, float), {(str, str): int}, {(str, str): int}
        """
        # Counting
        counter_begin = Counter()
        counter_end = Counter()
        prog_bar = pyprind.ProgBar(len(batch_edus))
        for edus, arcs in zip(batch_edus, batch_arcs):
            ngrams_begin = [] # list of (str, str)
            ngrams_end = [] # list of (str, str)
            for h, d, l in arcs:
                edu_h = edus[h]
                edu_d = edus[d]
                # N-grams w/ relations (head-side)
                part_ngrams_begin = self.extract_ngrams(edu_h, position="begin") # list of str
                part_ngrams_end = self.extract_ngrams(edu_h, position="end") # list of str
                part_ngrams_begin = [(ngram, l) for ngram in part_ngrams_begin] # list of (str, str)
                part_ngrams_end = [(ngram, l) for ngram in part_ngrams_end] # list of (str, str)
                ngrams_begin.extend(part_ngrams_begin)
                ngrams_end.extend(part_ngrams_end)
                # N-grams w/ relations (dependent-side)
                part_ngrams_begin = self.extract_ngrams(edu_d, position="begin") # list of str
                part_ngrams_end = self.extract_ngrams(edu_d, position="end") # list of str
                part_ngrams_begin = [(ngram, l) for ngram in part_ngrams_begin] # list of (str, str)
                part_ngrams_end = [(ngram, l) for ngram in part_ngrams_end] # list of (str, str)
                ngrams_begin.extend(part_ngrams_begin)
                ngrams_end.extend(part_ngrams_end)
            counter_begin.update(ngrams_begin)
            counter_end.update(ngrams_end)
            prog_bar.update()

        ngram_list_begin = list(set([ngram for (ngram, relation) in counter_begin.keys()])) # list of str
        ngram_list_end = list(set([ngram for (ngram, relation) in counter_end.keys()])) # list of str

        # Weighting
        weights_begin = self.compute_conditional_entropy(
                                ngram_list_begin,
                                counter_begin,
                                vocab_relation) # {str: float}
        weights_end = self.compute_conditional_entropy(
                                ngram_list_end,
                                counter_end,
                                vocab_relation) # {str: float}

        # Filtering
        weights_begin = [(ngram,ent) for ngram,ent in weights_begin.items() if ent <= threshold]
        weights_begin = sorted(weights_begin, key=lambda x: (x[1], x[0]))
        weights_end = [(ngram,ent) for ngram,ent in weights_end.items() if ent <= threshold]
        weights_end = sorted(weights_end, key=lambda x: (x[1], x[0]))

        return weights_begin, weights_end, counter_begin, counter_end

    def extract_ngrams(self, edu, position):
        """
        :type edu: list of str
        :type position: str
        :rtype: list of str
        """
        N = 2
        ngrams = []
        if position == "begin":
            for i in range(0, min(N, len(edu))):
                ngrams.append(" ".join(edu[:i+1]))
        elif position == "end":
            for i in range(0, min(N, len(edu))):
                ngrams.append(" ".join(edu[-1-i:]))
        else:
            raise ValueError("Invalid position=%s" % position)

        return ngrams

    def compute_conditional_entropy(self, symbols, counter, vocab_relation):
        """
        :type symbols: list of str
        :type counter: {(str, str): int}
        :type vocab_relation: {str: int}
        :rtype: {str: float}
        """
        C = 1.0
        weights = {} # {str: float}
        for symbol in symbols:
            counts = [counter[symbol, relation] for relation in vocab_relation.keys()]
            counts = np.asarray(counts, dtype=np.float64)
            counts += C # Smoothing; to avoid NaN in log(prob)
            prob = counts / np.sum(counts, axis=0)
            mi = -np.sum(prob * np.log(prob))
            weights[symbol] = mi
        return weights

    def build_heads(self, batch_edus_head, batch_arcs, vocab_relation, threshold):
        """
        :type batch_edus_head: list of list of (str, str, str)
        :type batch_arcs: list of list of (int, int, str)
        :type vocab_relation: {str: int}
        :type threshold: float
        :rtype: list of (str, float), list of (str, float), list of (str, float), {(str, str): int}, {(str, str): int}, {(str, str): int}
        """
        # Counting
        counter_hw = Counter()
        counter_hp = Counter()
        counter_hr = Counter()
        prog_bar = pyprind.ProgBar(len(batch_edus_head))
        for edus_head, arcs in zip(batch_edus_head, batch_arcs):
            head_words = [] # list of (str, str)
            head_postags = [] # list of (str, str)
            head_relations = [] # list of (str, str)
            for h, d, l in arcs:
                edu_head_h = edus_head[h]
                edu_head_d = edus_head[d]
                # Head word, head postag, head ralation w/ relations (head-side)
                head_word, head_postag, head_relation = edu_head_h
                head_words.append((head_word, l))
                head_postags.append((head_postag, l))
                head_relations.append((head_relation, l))
                # Head word, head postag, head ralation w/ relations (dependent-side)
                head_word, head_postag, head_relation = edu_head_d
                head_words.append((head_word, l))
                head_postags.append((head_postag, l))
                head_relations.append((head_relation, l))
            counter_hw.update(head_words)
            counter_hp.update(head_postags)
            counter_hr.update(head_relations)
            prog_bar.update()

        head_word_list = list(set([head_word for (head_word, relation) in counter_hw.keys()])) # list of str
        head_postag_list = list(set([head_postag for (head_postag, relation) in counter_hp.keys()])) # list of str
        head_relation_list = list(set([head_relation for (head_relation, relation) in counter_hr.keys()])) # list of str

        # Weighting
        weights_hw = self.compute_conditional_entropy(
                                head_word_list,
                                counter_hw,
                                vocab_relation) # {str: float}
        weights_hp = self.compute_conditional_entropy(
                                head_postag_list,
                                counter_hp,
                                vocab_relation) # {str: float}
        weights_hr = self.compute_conditional_entropy(
                                head_relation_list,
                                counter_hr,
                                vocab_relation) # {str: float}

        # Filtering
        weights_hw = [(head_word,ent) for head_word,ent in weights_hw.items() if ent <= threshold]
        weights_hw = sorted(weights_hw, key=lambda x: (x[1], x[0]))
        weights_hp = [(head_postag,ent) for head_postag,ent in weights_hp.items()]
        weights_hp = sorted(weights_hp, key=lambda x: (x[1], x[0]))
        weights_hr = [(head_relation,ent) for head_relation,ent in weights_hr.items()]
        weights_hr = sorted(weights_hr, key=lambda x: (x[1], x[0]))

        return weights_hw, weights_hp, weights_hr, counter_hw, counter_hp, counter_hr
    ##########################

    ##########################
    def aggregate_templates(self, databatch, vocab_relation):
        """
        :type databatch: DataBatch
        :type vocab_relation: {str: int}
        :rtype: None
        """
        ########################
        # lex_ngrams_begin, lex_ngrams_end, lex_counter_begin, lex_counter_end = \
        #         self.build_ngrams(
        #                 batch_edus=databatch.batch_edus,
        #                 batch_arcs=databatch.batch_arcs,
        #                 vocab_relation=vocab_relation,
        #                 threshold=2.85)
        # pos_ngrams_begin, pos_ngrams_end, pos_counter_begin, pos_counter_end = \
        #         self.build_ngrams(
        #                 batch_edus=databatch.batch_edus_postag,
        #                 batch_arcs=databatch.batch_arcs,
        #                 vocab_relation=vocab_relation,
        #                 threshold=2.85)
        head_words, head_postags, head_relations, counter_hw, counter_hp, counter_hr = \
                self.build_heads(
                        batch_edus_head=databatch.batch_edus_head,
                        batch_arcs=databatch.batch_arcs,
                        vocab_relation=vocab_relation,
                        threshold=2.85)
        ########################

        ########################
        # Features on N-grams
        # for ngram, ent in lex_ngrams_begin:
        #     self.add_template(lex_ngram_begin=ngram)
        # for ngram, ent in lex_ngrams_end:
        #     self.add_template(lex_ngram_end=ngram)
        # for ngram, ent in pos_ngrams_begin:
        #     self.add_template(pos_ngram_begin=ngram)
        # for ngram, ent in pos_ngrams_end:
        #     self.add_template(pos_ngram_end=ngram)
        ########################

        # Features on size
        for span in self.LENGTH_SPANS:
            self.add_template(length="%s~%s" % span)

        # Features on position
        for span in self.DISTANCE_SPANS:
            self.add_template(dist_from_begin="%s~%s" % span)
        for span in self.DISTANCE_SPANS:
            self.add_template(dist_from_end="%s~%s" % span)

        # Features on syntax
        for head_word, ent in head_words:
            self.add_template(dep_head_word=head_word)
        for head_postag, ent in head_postags:
            self.add_template(dep_head_postag=head_postag)
        for head_relation, ent in head_relations:
            self.add_template(dep_head_relation=head_relation)

        assert len(self.templates) == len(set(self.templates))
    ##########################

    ##########################
    def extract_features(self, edu, edu_postag, edu_head, edu_index, n_edus):
        """
        :type edu: list of str
        :type edu_postag: list of str
        :type edu_head: (str, str)
        :type edu_index: int
        :type n_edus: int
        :rtype: numpy.ndarray(shape=(1, feature_size), dtype=np.float32)
        """
        templates = self.generate_templates(edu=edu,
                                            edu_postag=edu_postag,
                                            edu_head=edu_head,
                                            edu_index=edu_index,
                                            n_edus=n_edus) # list of str
        template_dims = [self.template2dim.get(t, self.UNK_TEMPLATE_DIM) for t in templates] # list of int
        vector = utils.make_multihot_vectors(self.feature_size+1, [template_dims]) # (1, feature_size+1)
        vector = vector[:,:-1] # (1, feature_size)
        return vector

    def extract_batch_features(self, edus, edus_postag, edus_head):
        """
        :type edus: list of list of str
        :type edus_postag: list of list of str
        :type edus_head: list of (str, str)
        :rtype: numpy.ndarray(shape=(n_edus, feature_size), dtype=np.float32)
        """
        fire = [] # list of list of int
        n_edus = len(edus)
        for edu_index, (edu, edu_postag, edu_head) in enumerate(zip(edus, edus_postag, edus_head)):
            templates = self.generate_templates(edu=edu,
                                                edu_postag=edu_postag,
                                                edu_head=edu_head,
                                                edu_index=edu_index,
                                                n_edus=n_edus) # list of str
            template_dims = [self.template2dim.get(t, self.UNK_TEMPLATE_DIM) for t in templates] # list of int
            fire.append(template_dims)
        vectors = utils.make_multihot_vectors(self.feature_size+1, fire) # (n_edus, feature_size+1)
        vectors = vectors[:,:-1] # (n_edus, feature_size)
        return vectors

    def generate_templates(self, edu, edu_postag, edu_head, edu_index, n_edus):
        """
        :type edu: list of str
        :type edu_postag: list of str
        :type edu_head: (str, str)
        :type edu_index: int
        :type n_edus: int
        :rtype: list of str
        """
        templates = []

        ########################
        # Features on N-grams
        # ngrams = self.extract_ngrams(edu, position="begin") # list of str
        # part_templates = [self.convert_to_template(lex_ngram_begin=ngram) for ngram in ngrams]
        # templates.extend(part_templates)
        #
        # ngrams = self.extract_ngrams(edu, position="end") # list of str
        # part_templates = [self.convert_to_template(lex_ngram_end=ngram) for ngram in ngrams]
        # templates.extend(part_templates)
        #
        # ngrams = self.extract_ngrams(edu_postag, position="begin") # list of str
        # part_templates = [self.convert_to_template(pos_ngram_begin=ngram) for ngram in ngrams]
        # templates.extend(part_templates)
        #
        # ngrams = self.extract_ngrams(edu_postag, position="end") # list of str
        # part_templates = [self.convert_to_template(pos_ngram_end=ngram) for ngram in ngrams]
        # templates.extend(part_templates)
        ########################

        # Features on size
        length_span = self.get_length_span(len(edu))
        template = self.convert_to_template(length="%s~%s" % length_span)
        templates.append(template)

        # Features on position
        distance_span = self.get_distance_span(edu_index)
        template = self.convert_to_template(dist_from_begin="%s~%s" % distance_span)
        templates.append(template)

        distance_span = self.get_distance_span(n_edus - edu_index - 1)
        template = self.convert_to_template(dist_from_end="%s~%s" % distance_span)
        templates.append(template)

        # Features on syntax
        head_word, head_postag, head_relation = edu_head

        template = self.convert_to_template(dep_head_word=head_word)
        templates.append(template)

        template = self.convert_to_template(dep_head_postag=head_postag)
        templates.append(template)

        template = self.convert_to_template(dep_head_relation=head_relation)
        templates.append(template)

        return templates

    def get_length_span(self, length):
        """
        :type length: int
        :rtype: (int, int)
        """
        for span_min, span_max in self.LENGTH_SPANS:
            if span_min <= length <= span_max:
                return (span_min, span_max)
        raise ValueError("Should never happen: length=%d" % length)

    def get_distance_span(self, distance):
        """
        :type distance: int
        :rtype: (int, int)
        """
        for span_min, span_max in self.DISTANCE_SPANS:
            if span_min <= distance <= span_max:
                return (span_min, span_max)
        raise ValueError("Should never happen: distance=%d" % distance)

