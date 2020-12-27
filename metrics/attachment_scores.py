import utils
import treetk

def attachment_scores(pred_path, gold_path):
    """
    :type pred_path: str
    :type gold_path: str
    :rtype: {str: Any}
    """
    preds = read_arcs(pred_path)
    golds = read_arcs(gold_path)
    scores = _attachment_scores(preds, golds)
    return scores

def read_arcs(path):
    """
    :type path: str
    :rtype: list of list of (int, int, str)
    """
    hyphens = utils.read_lines(path, process=lambda line: line.split()) # list of list of str
    arcs = [treetk.hyphens2arcs(h) for h in hyphens] # list of list of (int, int, str)
    return arcs

def _attachment_scores(preds, golds):
    """
    :type preds: list of list of (int, int, str)
    :type golds: list of list of (int, int, str)
    """
    assert len(preds) == len(golds)

    scores = {} # {str: Any}

    total_ok_unlabeled = 0.0
    total_ok_labeled = 0.0
    total_arcs = 0.0
    for pred_arcs, gold_arcs in zip(preds, golds):
        assert len(pred_arcs) == len(gold_arcs)

        n_ok_unlabeled = 0.0
        n_ok_labeled = 0.0
        n_arcs = 0.0

        pred_dtree = treetk.arcs2dtree(pred_arcs)
        gold_dtree = treetk.arcs2dtree(gold_arcs)

        for d in range(len(pred_dtree.tokens)):
            if d == 0:
                continue # Ignore ROOT
            pred_h, pred_l = pred_dtree.get_head(d)
            gold_h, gold_l = gold_dtree.get_head(d)
            n_arcs += 1.0
            if pred_h == gold_h:
                n_ok_unlabeled += 1.0
            if pred_h == gold_h and pred_l == gold_l:
                n_ok_labeled += 1.0

        total_ok_unlabeled += n_ok_unlabeled
        total_ok_labeled += n_ok_labeled
        total_arcs += n_arcs

    uas = total_ok_unlabeled / total_arcs
    las = total_ok_labeled / total_arcs
    uas_info = "%d/%d" % (total_ok_unlabeled, total_arcs)
    las_info = "%d/%d" % (total_ok_labeled, total_arcs)
    scores = {"UAS": uas,
              "LAS": las,
              "UAS_info": uas_info,
              "LAS_info": las_info}
    return scores

def attachment_scores_v2(preds, golds):
    scores = _attachment_scores(preds, golds)
    return scores