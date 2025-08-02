import pandas as pd

df = pd.read_csv("dataset/train_val.csv")

df_sorted = df.sort_values(by="year")

val_size = int(0.10 * len(df_sorted)) # 10% of the training data

val_df = df_sorted.iloc[:val_size]
train_df = df_sorted.iloc[val_size:]

val_df.to_csv("dataset/val.csv", index=False)
train_df.to_csv("dataset/train.csv", index=False)

print(f"Validation set: {len(val_df)} samples")
print(f"Training set: {len(train_df)} samples")
