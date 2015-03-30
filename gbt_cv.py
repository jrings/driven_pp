import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RandomizedLasso
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
import sys

grid_param = {"a":{'max_depth': 5, 'max_features': 'sqrt', 'min_samples_split': 7}, 
"b":{'max_depth': 3, 'max_features': 'sqrt', 'min_samples_split': 7}, 
"c":{'max_depth': 3, 'max_features': 'sqrt', 'min_samples_split': 3}, 
"d":{'max_depth': 3, 'max_features': 'log2', 'min_samples_split': 11},
"e":{'max_depth': 3, 'max_features': 'sqrt', 'min_samples_split': 11},
"f":{'max_depth': 3, 'max_features': 'sqrt', 'min_samples_split': 11},
"g":{'max_depth': 3, 'max_features': 'log2', 'min_samples_split': 7}, 
"h":{'max_depth': 3, 'max_features': 'sqrt', 'min_samples_split': 3}, 
"i":{'max_depth': 3, 'max_features': 'sqrt', 'min_samples_split': 11},
"j":{'max_depth': 3, 'max_features': 'sqrt', 'min_samples_split': 11},
"k":{'max_depth': 3, 'max_features': 'sqrt', 'min_samples_split': 11},
"l":{'max_depth': 5, 'max_features': 'sqrt', 'min_samples_split': 7}, 
"m":{'max_depth': 3, 'max_features': 'sqrt', 'min_samples_split': 11},
"n":{'max_depth': 3, 'max_features': 'sqrt', 'min_samples_split': 3}} 

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

    cvs = {}
    for i, col in enumerate("abcdefghijklmn"):
        model = GradientBoostingClassifier(
            n_estimators=400, random_state=0, **grid_param[col]
        )

        y = np.array(labels["service_{}".format(col)])
        cvs[col] = np.mean(cross_val_score(model, X, y, n_jobs=4))
        print("{} finished with {}".format(col, cvs[col]))
    print("Overall: {}".format(np.mean(list(cvs.values()))))


    
if __name__ == "__main__":
    main()
