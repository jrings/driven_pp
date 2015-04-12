import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RandomizedLasso
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesRegressor
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
    cat_X = np.array(combined[[col for col in combined.columns if col.startswith("c") 
                               or col.startswith("o")]])

    for col in combined:
        if col.startswith("n_"):
            combined[col + "_nan"] = [1 if np.isnan(x) else 0 for x in combined[col]]
            print("Imputing {}".format(col))

            cat_y = np.array(combined[col])
            not_null = np.array([i for i in range(len(cat_y)) if np.isfinite(cat_y[i])])
            model = ExtraTreesRegressor(
                n_estimators=100, random_state=1979, min_samples_split=5, max_features="sqrt")
            model.fit(cat_X[not_null, :], cat_y[not_null])

            cat_pred = model.predict(cat_X).ravel()

            combined[col] = [x if np.isfinite(x) else cat_pred[i] 
                             for i, x in enumerate(combined[col].tolist())]


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
    P.to_csv("submit_xgb_early.csv", index=False)
    
if __name__ == "__main__":
    main()
