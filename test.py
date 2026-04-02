import pandas as pd

df = pd.read_csv("labeled_data.csv")

import tqdm

print("buy:", df["buy_label"].sum())
print("sell:", df["sell_label"].sum())
print("idk:", df["idk_label"].sum())