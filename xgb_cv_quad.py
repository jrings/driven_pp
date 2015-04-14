import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RandomizedLasso
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import log_loss
import xgboost as xgb
import sys


def prepare_data():
    labels = pd.read_csv("../train_labels.csv")
    train = pd.read_csv("../train_values.csv", low_memory=False)
    test = pd.read_csv("../test_values.csv", low_memory=False)

    interactions = open("quadrats.txt").readline().strip()[1:-1].split()
    interactions = [x.replace("'", "").split("||") for x in interactions][:100]

    ids = train.pop("id")
    test_ids = test.pop("id")

    combined = pd.concat((train, test))

    #Drop columns with less than 5% coverage
    for col in combined.columns:
        if combined[col].isnull().sum()/combined.shape[0] >= 0.95:
            _ = combined.pop(col)

    #Expand ordinals and categories and impute missing numerical values
    dummify_cols = ["release"] + [col for col in combined.columns if col.startswith("_o")
                                  or col.startswith("c_")]

    combined = pd.get_dummies(
        combined, columns=dummify_cols, dummy_na=True)
    
    for col in combined:
        if col not in dummify_cols:
            combined[col + "_nan"] = [1 if np.isnan(x) else 0 for x in combined[col]]
            filler = np.nanmedian(combined[col])
            combined[col].fillna(filler, inplace=True)

    for col_a, col_b in interactions:
        col_a = col_a.strip()
        col_b = col_b.strip().replace(",", "")
        if col_a not in combined.columns.tolist():
            print("xxx{}xxx".format(col_a), len(col_a), "not in combined columns")
            continue
        if col_b not in combined.columns.tolist():
            print("xxx{}xxx".format(col_b), len(col_b), "not in combined columns")
            continue
        combined["{}_x_{}".format(col_a, col_b)] = [a*b for a, b in zip(combined[col_a], combined[col_b])]
            
    #Split up again
    train = combined.iloc[:len(ids)]
    test = combined.iloc[len(ids):]

    print(train.shape)
    return train, test, labels, ids, test_ids

def main():
    train, test, labels, ids, test_ids = prepare_data()
    
    X = np.array(train)
    X_test = np.array(test)

    param = {'max_depth': 2, 'eta': 0.5, 'silent':1, 'objective':'binary:logistic', 
             'nthread': 8, 'eval_metric': 'logloss', 'seed': 1979 }
    best = pickle.load(open("best_params_quad.pkl", "rb"))
    all_preds = {}
    cvs = {}
    for i, col in enumerate("abcdefghijklmn"):
        ((num_round, md, eta), _) = best[col]
        param.update({"max_depth": md, "eta": eta})
        print("service_{}: {} {} rounds".format(col, param, num_round))
        y = np.array(labels["service_{}".format(col)])

        cross_values = []
        for train_idx, test_idx in KFold(X.shape[0], shuffle=True, random_state=1979):
            dtrain = xgb.DMatrix(X[train_idx, :], label=y[train_idx])
            dtest = xgb.DMatrix(X[test_idx, :])
            bst = xgb.train(param, dtrain, num_round)
            preds = bst.predict(dtest)
            
            cross_values.append(log_loss(y[test_idx], preds))
        cvs[col] = np.mean(cross_values)
        print("{} finished with {}".format(col, cvs[col]))
                    
    print("Overall: {}".format(np.mean(list(cvs.values()))))


    
if __name__ == "__main__":
    main()
