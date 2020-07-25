# load baseline file and evaluate
import os

import dataloader
import utils
import treetk

import metrics

def load_baseline():

    train_dataset = dataloader.read_scidtb("train", "", relation_level="coarse-grained")
    test_dataset = dataloader.read_scidtb("test", "gold", relation_level="coarse-grained")
    dev_dataset = dataloader.read_scidtb("dev", "gold", relation_level="coarse-grained")

    root = "/home/zhanglw/code/DiscourseConstituencyInduction-ViterbiEM/data/scidtb/preprocessed"
    train_path = "/train/"
    dev_path = "/dev/gold"
    test_path = "/test/gold"

    processed_train = []
    for data in train_dataset:
        filename = data.name

        hyphens = utils.read_lines(os.path.join(root, "train", filename.replace(".edus.tokens", ".baseline.arcs")),
                                   process=lambda line: line.split())
        assert len(hyphens) == 1
        hyphens = hyphens[0] # list of str
        arcs = treetk.hyphens2arcs(hyphens) # list of (int, int, str)
        processed_train.append((data, arcs))

        processed_dev = []
        for data in dev_dataset:
            filename = data.name

            hyphens = utils.read_lines(os.path.join(root, "dev/gold", filename.replace(".edus.tokens", ".baseline.arcs")),
                                       process=lambda line: line.split())
            assert len(hyphens) == 1
            hyphens = hyphens[0]  # list of str
            arcs = treetk.hyphens2arcs(hyphens)  # list of (int, int, str)
            processed_dev.append((data, arcs))

        processed_test = []
        for data in test_dataset:
            filename = data.name

            hyphens = utils.read_lines(os.path.join(root, "test/gold", filename.replace(".edus.tokens", ".baseline.arcs")),
                                       process=lambda line: line.split())
            assert len(hyphens) == 1
            hyphens = hyphens[0]  # list of str
            arcs = treetk.hyphens2arcs(hyphens)  # list of (int, int, str)
            processed_test.append((data, arcs))
        return processed_train, processed_dev, processed_test


def eval(dataset):

    with open("./tmp/gold_arcs", "w") as gf:
        with open("./tmp/pred_arcs", "w") as pf:
            for data, base_arcs in dataset:
                dtree = treetk.arcs2dtree(arcs=data.arcs)
                labeled_arcs = ["%s-%s-%s" % (x[0], x[1], x[2]) for x in dtree.tolist()]
                gf.write("%s\n" % " ".join(labeled_arcs))

                dtree = treetk.arcs2dtree(arcs=base_arcs)
                labeled_arcs = ["%s-%s-%s" % (x[0], x[1], x[2]) for x in dtree.tolist()]
                pf.write("%s\n" % " ".join(labeled_arcs))

    # save to tmp to call

    res = metrics.attachment_scores("./tmp/pred_arcs", "./tmp/gold_arcs")
    print(res)

def main():
    train, dev, test = load_baseline()
    eval(test)

if __name__ == '__main__':
    main()