from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle

df = pd.read_csv("Data/leaf_disease_features.csv")

X = df.drop("label", axis=1)
y = df["label"]

model = RandomForestClassifier()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))