audit = {}

# Assumes you ran vowpal with quadratic terms and audit the prediction into ../audit.txt

for line in open("../audit.txt"):
    pp = line.split()
    for p in pp:
        if not p.startswith("n"):
            continue
        p = p.split(":")
        weight = float(p[-1])
        if weight == 0:
            continue
        x = p[0].split("^")
        if len(x) == 4:
            cols = (int(x[1]), int(x[3]))
            audit[cols] = audit.get(cols, 0) + 1

import pandas as pd
df = pd.read_csv("../train_a.csv") # Train values with labels from service_a as last column and label y
df = df[[col for col in df.columns if col != "y"]]
test = pd.read_csv("../test_values.csv")

combined = pd.concat((df, test))                               
not_covered = []                                                                                                       
for col in combined.columns:
    if combined[col].isnull().sum()/combined.shape[0] >= 0.95:
        not_covered.append(col)

col_lk = {i+1: df.columns.tolist()[i] for i in range(df.shape[1]-1)}

aud = {(col_lk[x[0]], col_lk[x[1]]): v for x, v in audit.items() if col_lk[x[0]] not in not_covered and col_lk[x[1]] not in not_covered}

print(["{}||{}".format(k[0][0], k[0][1]) for k in sorted(aud.items(), key = lambda x: x[1])[-250:]])
