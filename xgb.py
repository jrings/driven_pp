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
import pickle
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
    
    X = np.array(train)
    X_test = np.array(test)
    param = {'max_depth': 2, 'eta': 0.5, 'silent':1, 'objective':'binary:logistic',
             'nthread': 8, 'eval_metric': 'logloss', 'seed': 1979 }
    best = pickle.load(open("best_params.pkl", "rb"))
    all_preds = {}
    for i, col in enumerate("abcdefghijklmn"):
        ((num_round, md, eta), _) = best[col]
        param.update({"max_depth": md, "eta": eta})
        
        print("service_{}: {}".format(col, param))
        y = np.array(labels["service_{}".format(col)])
        dtrain = xgb.DMatrix(X, label=y)
        dtest = xgb.DMatrix(X_test)
        bst = xgb.train(param, dtrain, num_round)
        preds = bst.predict(dtest)
            
        all_preds[col] = preds

    P = pd.DataFrame({"service_{}".format(col): arr for col, arr in all_preds.items()})
    P["id"] = test_ids
    P = P[sorted(P.columns)]
    P.to_csv("submit_xgb_gridded.csv", index=False)
    
if __name__ == "__main__":
    main()
