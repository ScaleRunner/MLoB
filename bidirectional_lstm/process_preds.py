import numpy as np
import pandas as pd

print("Loading Data")

predictions = np.loadtxt("predictions.csv", delimiter=',')
df = pd.read_csv('../data/test.csv', encoding='utf-8')
ids = df['id']

print("Loading Data done!")

with open("predictions_submission.csv", "w") as f:
    f.write("id,toxic,severe_toxic,obscene,threat,insult,identity_hate\n")
    for i, preds in enumerate(predictions):
        f.write("{},{},{},{},{},{},{}\n".format(ids[i], *preds))

print("Writing Predictions Done!")
