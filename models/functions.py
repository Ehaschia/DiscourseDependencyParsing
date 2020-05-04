import numpy as np

def make_same_sent_map(edus, sbnds):
    """
    :type edus: list of list of str
    :type sbnds: list of (int, int)
    :rtype: numpy.ndarray(shape=(n_edus, n_edus), dtype=np.int32)
    """
    # NOTE: Indices (b, e) \in sbnds are shifted by -1 compared to EDU IDs which include ROOT.
    #       For example, (b, e) \in sbnds indicates that EDUs[b+1:e+1+1] belongs to one sentence.
    n_edus = len(edus)
    same_sent_map = np.zeros((n_edus, n_edus), dtype=np.int32)
    for begin_i, end_i in sbnds:
        same_sent_map[begin_i+1:end_i+1+1, begin_i+1:end_i+1+1] = 1.0
    return same_sent_map

