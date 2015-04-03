audit = {}

for line in open("audit.txt"):
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
            audit[cols] = weight

import pandas as pd
df = pd.read_csv("train_a.csv")
df = df[[col for col in df.columns if col != "y"]]

col_lk = {i+1: df.columns.tolist()[i] for i in range(df.shape[1]-1)}

aud = {(col_lk[x[0]], col_lk[x[1]]): v for x, v in audit.items()}
print(sorted(aud.items(), key = lambda x: x[1])[-100:])
