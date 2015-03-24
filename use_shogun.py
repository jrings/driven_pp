from modshogun import RealFeatures, MultilabelSOLabels, MultilabelModel
from modshogun import StochasticSOSVM, DualLibQPBMSOSVM, StructuredAccuracy, LabelsFactory

import numpy as np
import pandas as pd

from simple_model import read_data

def main():

    train, test, labels, ids, test_ids = read_data()

    # Build Multilabel object
    labels = labels.set_index("id")
    labels_so = MultilabelSOLabels(labels.shape[0], labels.shape[1])
    for i, (_, row) in enumerate(labels.iterrows()):
        labels_so.set_sparse_label(int(i), np.array(row, dtype=np.int32))

    # Build train object
    train_features = RealFeatures(np.c_(np.array(train), np.ones(train.shape[0])))
    test_features = RealFeatures(np.c_(np.array(train), np.ones(train.shape[0])))
        
    

if __name__ == "__main__":
    main()