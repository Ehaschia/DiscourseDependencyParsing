from typing import List, Set, Tuple, Dict

import dataloader

from utils import DataInstance

# load dataset
def load_scidtb(fine_relation: bool = False):
    if fine_relation:
        train_dataset = dataloader.read_scidtb("train", "", relation_level="fine-grained")
    else:
        train_dataset = dataloader.read_scidtb("train", "", relation_level="coarse-grained")
    return train_dataset


from collections import Counter
def main():
    dataset = load_scidtb()
    head_counter = Counter()
    total_counter = Counter()

    for data in dataset:
        head_pos = Counter([ele[1] for ele in data.edus_head])
        head_counter += head_pos
        pos_list = data.edus_postag
        em = []
        for i in pos_list:
            em += i
        total_counter += Counter(em)
    print(head_counter)
    print(total_counter)

if __name__ == '__main__':
    main()