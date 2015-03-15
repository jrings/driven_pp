import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score

labels = pd.read_csv("../train_labels.csv")
train = pd.read_csv("../train_values.csv")
test = pd.read_csv("../test_values.csv")


model = LogisticRegression()
ids = train.pop("id")
test_ids = test.pop("id")


combined = pd.concat((train, test))

#Expand categories and impute missing numerical values
combined = pd.get_dummies(combined)
for col in combined:
    if col.startswith("n_"):
        filler = np.nanmean(combined[col])
        combined[col].fillna(filler, inplace=True)

#Split up again
train = combined.iloc[:len(ids)]
test = combined.iloc[len(ids):]

