import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RandomizedLasso
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.multiclass import OneVsAllClassifier
import sys

def prepare_data():
    labels = pd.read_csv("../train_labels.csv")
    train = pd.read_csv("../train_values.csv", low_memory=False)
    test = pd.read_csv("../test_values.csv", low_memory=False)

    ids = train.pop("id")
    test_ids = test.pop("id")

    combined = pd.concat((train, test))

    #Drop columns with less than 5% coverage
    for col in combined.columns:
        if combined[col].isnull().sum()/combined.shape[0] >= 0.95:
            _ = combined.pop(col)

    #Expand ordinals and categories and impute missing numerical values
    combined = pd.get_dummies(
        combined, columns=[col for col in combined.columns if col.startswith("o_")
                           or col.startswith("c_") or col=="release"], dummy_na=True)
    for col in combined:
        if col.startswith("n_"):
            combined[col + "_nan"] = [1 if np.isnan(x) else 0 for x in combined[col]]
            filler = np.nanmean(combined[col])
            combined[col].fillna(filler, inplace=True)

    #Split up again
    train = combined.iloc[:len(ids)]
    test = combined.iloc[len(ids):]

    print(train.shape)
    return train, test, labels, ids, test_ids

def main():
    train, test, labels, ids, test_ids = prepare_data()
    model = ExtraTreesClassifier(
        n_estimators=400, min_samples_split=7, max_features="log2", random_state=0,
        n_jobs=4)
    multi = OneVsAllClassifier(model)
    X = np.array(train)
    X_test = np.array(test)
    print(X_test.shape)
    Y = np.array(labels)
    print(Y.shape, labels.columns)

    multi.train(X, Y)
    preds = multi.predict_proba(X_test)
    P = pd.DataFrame(preds)
    P["id"] = test_ids
    P = P[sorted(P.columns)]
    P.to_csv("submit_multi.csv", index=False)
#    print(np.mean(cross_val_score(model, X, y, verbose=True,
#                              cv=4, scoring="accuracy")))
    

    
if __name__ == "__main__":
    main()
