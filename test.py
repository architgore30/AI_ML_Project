import pandas as pd

DATA_POINTS = 500
df = pd.read_csv("labeled_data_v2.csv", nrows=DATA_POINTS)
print("data imported")

import numpy as np
price = df["Close"].values
no_trade = [price[i] + 1 if df["no_trade_label"].iloc[i] == 1 else np.nan for i in range(len(df))]
trend_up = [price[i] + 2 if df["trend_up_label"].iloc[i] == 1 else np.nan for i in range(len(df))]
trend_down = [price[i] + -2 if df["trend_down_label"].iloc[i] == 1 else np.nan for i in range(len(df))]
reversal_up = [price[i] + 4 if df["reversal_up_label"].iloc[i] == 1 else np.nan for i in range(len(df))]
reversal_down = [price[i] - 4 if df["reversal_down_label"].iloc[i] == 1 else np.nan for i in range(len(df))]
print("masks applied")

import matplotlib.pyplot as plt

print("plotting")
plt.plot(price, color="blue", label="Price")
plt.plot(no_trade, color="orange", label="No Trade")
plt.plot(trend_up, color="green", label="Trend Up")
plt.plot(trend_down, color="red", label="Trend Down")
plt.plot(reversal_up, color="purple", label="Reversal Up")
plt.plot(reversal_down, color="brown", label="Reversal Down")
plt.legend()
plt.show()