import pandas as pd

df = pd.read_csv("labeled_data.csv")

print(f"buy: {df["buy_label"].sum()}/{len(df)} ({df["buy_label"].sum() / len(df)})")
print(f"sell: {df["sell_label"].sum()}/{len(df)} ({df["sell_label"].sum() / len(df)})")
print(f"idk: {df["idk_label"].sum()}/{len(df)} ({df["idk_label"].sum() / len(df)})")