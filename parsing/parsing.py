import numpy as np
from chainer import cuda
import pyprind

import treetk

import models

def parse(model, decoder, databatch, path_pred):
    """
    :type model: Model
    :type decoder: IncrementalEisnerDecoder
    :type databatch: DataBatch
    :type path_pred: str
    :rtype: None
    """
    with open(path_pred, "w") as f:
        prog_bar = pyprind.ProgBar(len(databatch))

        for edu_ids, edus, edus_postag, edus_head, sbnds, pbnds \
                in zip(databatch.batch_edu_ids,
                       databatch.batch_edus,
                       databatch.batch_edus_postag,
                       databatch.batch_edus_head,
                       databatch.batch_sbnds,
                       databatch.batch_pbnds):

            # Feature extraction
            edu_vectors = model.forward_edus(edus, edus_postag, edus_head) # (n_edus, bilstm_dim + tempfeat1_dim)
            same_sent_map = models.make_same_sent_map(edus=edus, sbnds=sbnds) # (n_edus, n_edus)

            # Parsing (attachment)
            unlabeled_arcs = decoder.decode(
                                model=model,
                                edu_ids=edu_ids,
                                edu_vectors=edu_vectors,
                                same_sent_map=same_sent_map,
                                sbnds=sbnds,
                                pbnds=pbnds,
                                use_sbnds=True,
                                use_pbnds=True) # list of (int, int)

            # Parsing (labeling)
            logits_rel = model.forward_arcs_for_labeling(
                                    edu_vectors=edu_vectors,
                                    same_sent_map=same_sent_map,
                                    batch_arcs=[unlabeled_arcs]) # (1, n_arcs, n_labels)
            logits_rel = cuda.to_cpu(logits_rel.data)[0] # (n_spans, n_relations)
            relations = np.argmax(logits_rel, axis=1) # (n_spans,)
            relations = [model.ivocab_relation[r] for r in relations] # list of str
            labeled_arcs = [(h,d,r) for (h,d),r in zip(unlabeled_arcs, relations)] # list of (int, int, str)

            dtree = treetk.arcs2dtree(arcs=labeled_arcs)
            labeled_arcs = ["%s-%s-%s" % (x[0],x[1],x[2]) for x in dtree.tolist()]
            f.write("%s\n" % " ".join(labeled_arcs))

            prog_bar.update()

