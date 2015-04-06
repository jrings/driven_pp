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
    combined = pd.get_dummies(
        combined, columns=[col for col in combined.columns if col.startswith("o_")
                           or col.startswith("c_") or col=="release"], dummy_na=True)

    for col in combined:
        if col.startswith("n_"):
            combined[col + "_nan"] = [1 if np.isnan(x) else 0 for x in combined[col]]
            filler = np.nanmean(combined[col])
            combined[col].fillna(filler, inplace=True)

    # Add the interaction terms                                                                                                                                                        
    for col_a, col_b in interactions:
        col_a = col_a.strip()
        col_b = col_b.strip().replace(",", "")

        if col_a.startswith("o_"):
            for col in combined.columns:
                if not col.startswith(col_a):
                    continue
                else:
                    combined["{}_x_{}".format(col, col_b)] = [a*b for a, b in zip(combined[col], combined[col_b])]
        if col_b.startswith("o_"):
            for col in combined.columns:
                if not col.startswith(col_b):
                    continue
                else:
                    combined["{}_x_{}".format(col_a, col)] = [a*b for a, b in zip(combined[col_a], combined[col])]
        if not col_a.startswith("o_") and not col_b.startswith("o_"):
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
    param = {'silent':1, 'objective':'binary:logistic',
             'nthread': 8, 'eval_metric': 'logloss', 'seed': 1979 }

    best = pickle.load(open("best_params_quad.pkl", "rb"))

    min_childs = [0.5, 0.75, 1, 1.25]
    subsamples = [.5, .75, 1]
    colsamples = [.7, .85, 1]

    best_params = {}
    # here comes the super-nested if. Don't do this at home!
    for i, col in enumerate("abcdefghijklmn"):
        all_preds = {}
        
        ((num_round, md, eta), _) = best[col]
        param.update({"max_depth": md, "eta": eta})

        for mc in min_childs:
            for ss in subsamples:
                for cs in colsamples:
                    key = (mc, ss, cs)
                
                    param.update({"min_child_weight": mc, "subsample": ss,
                                  "colsample_bytree": cs})

                    print("service_{} {}".format(col, key), end = "")
                    y = np.array(labels["service_{}".format(col)])

                    cvs = []
                    for train_idx, test_idx in KFold(X.shape[0], shuffle=True, random_state=1979):
                        dtrain = xgb.DMatrix(X[train_idx, :], label=y[train_idx])
                        dtest = xgb.DMatrix(X[test_idx, :])
                        bst = xgb.train(param, dtrain, n_round)
                        preds = bst.predict(dtest)
                        
                        cvs.append(log_loss(y[test_idx], preds))
                    n = np.mean(cvs)
                    if not np.isnan(n):
                        all_preds[key] = np.mean(cvs)
                    print(all_preds[key])
        best_params[col] = sorted(all_preds.items(), key = lambda x: x[1])[0]
        print(col, best_params[col])
    import pickle
    pickle.dump(best_params, open("best_params_stage_two.pkl", 'wb'))




    
if __name__ == "__main__":
    main()
