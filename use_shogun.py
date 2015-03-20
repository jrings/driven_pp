from modshogun import RealFeatures, MultilabelSOLabels, MultilabelModel

import os
import inspect
import sys

import numpy as np
import pandas as pd

cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

from simple_model import prepare_data

def main():

    train, test, labels, ids, test_ids = prepare_data()

    # Build Multilabel object
    labels = labels.set_index("id")
    labels_so = MultilabelSOLabels(labels.shape[0], labels.shape[1])
    for i, (_, row) in enumerate(labels.iterrows()):
        labels_so.set_sparse_label(int(i), np.array(row, dtype=np.int32))

    # Build train object
    train_features = RealFeatures(np.c_[np.array(train), np.ones(train.shape[0])].T)
    test_features = RealFeatures(np.c_[np.array(train), np.ones(train.shape[0])].T)
        
    

if __name__ == "__main__":
    main()
