import numpy as np

import utils
from utils import TemplateFeatureExtractor

class TemplateFeatureExtractor2(TemplateFeatureExtractor):

    def __init__(self):
        super().__init__()
        self.DISTANCE_SPANS = [(-np.inf,-6), (-5,-3), (-2,-1), (1,2), (3,5), (6,np.inf)]
        self.aggregate_templates()
        self.prepare()

    ############################
    def aggregate_templates(self):
        """
        :rtype: None
        """
        for span in self.DISTANCE_SPANS:
            self.add_template(dist_between="%s~%s" % span)

        for same_sent in [0, 1]:
            self.add_template(same_sent=str(same_sent))

        assert len(self.templates) == len(set(self.templates))
    ############################

    ############################
    def extract_features(self, edu1_index, edu2_index, same_sent_map):
        """
        :type edu1_index: int
        :type edu2_index: int
        :type same_sent_map: numpy.ndarray(shape=(N,N), dtype=np.int32)
        :rtype: numpy.ndarray(shape=(1, feature_size), dtype=np.float32)
        """
        templates = self.generate_templates(edu1_index=edu1_index,
                                          edu2_index=edu2_index,
                                          same_sent=same_sent_map[edu1_index, edu2_index]) # list of str
        template_dims = [self.template2dim[t] for t in templates] # list of int
        vector = utils.make_multihot_vectors(self.feature_size, [template_dims]) # (1, feature_size)
        return vector

    def extract_batch_features(self, edu1_indices, edu2_indices, same_sent_map):
        """
        :type edu1_indices: list of int
        :type edu2_indices: list of int
        :type same_sent_map: numpy.ndarray(shape=(N,N), dtype=np.int32)
        :rtype: numpy.ndarray(shape=(N, feature_size), dtype=np.float32)
        """
        fire = [] # list of list of int
        for edu1_index, edu2_index in zip(edu1_indices, edu2_indices):
            templates = self.generate_templates(edu1_index=edu1_index,
                                              edu2_index=edu2_index,
                                              same_sent=same_sent_map[edu1_index, edu2_index]) # list of str
            template_dims = [self.template2dim[t] for t in templates] # list of int
            fire.append(template_dims)
        vectors = utils.make_multihot_vectors(self.feature_size, fire) # (N, feature_size)
        return vectors

    def generate_templates(self, edu1_index, edu2_index, same_sent):
        """
        :type edu1_index: int
        :type edu2_index: int
        :type same_sent: int
        :rtype: list of str
        """
        templates = []

        distance_span = self.get_distance_span(edu2_index - edu1_index)
        template = self.convert_to_template(dist_between="%s~%s" % distance_span)
        templates.append(template)

        template = self.convert_to_template(same_sent=str(same_sent))
        templates.append(template)

        return templates

    def get_distance_span(self, distance):
        """
        :type distance: int
        :rtype: (int, int)
        """
        for span_min, span_max in self.DISTANCE_SPANS:
            if span_min <= distance <= span_max:
                return (span_min, span_max)
        raise ValueError("Should never happen: distance=%d" % distance)

