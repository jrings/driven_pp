import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RandomizedLasso
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
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
    model = GradientBoostingClassifier(
        n_estimators=400, min_samples_split=7, max_features="log2", random_state=0, 
        )
    
    gscv = GridSearchCV(model, n_jobs=8, param_grid={"min_samples_split": [3,7,11],
                                                      "max_depth": [3,5, None],
                                                      "max_features": ["log2", "sqrt", None]},
                        scoring="log_loss", verbose=2)
    X = np.array(train)
    X_test = np.array(test)
    print(X_test.shape)
    preds = {}
    for i, col in enumerate("abcdefghijklmn"):
        y = np.array(labels["service_{}".format(col)])
        gscv.fit(X, y)
        preds[col] = gscv.predict_proba(X_test)
        print(col, gscv.best_params_, gscv.best_score_)
    P = pd.DataFrame({"service_{}".format(letter): arr for col, arr in preds.items()})
    P["id"] = test_ids
    P = P[sorted(P.columns)]
    P.to_csv("submit_gscv.csv", index=False)


    
if __name__ == "__main__":
    main()
