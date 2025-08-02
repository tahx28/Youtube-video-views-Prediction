import pandas as pd

train_df = pd.read_csv('dataset/train.csv')
val_df = pd.read_csv('dataset/val.csv')

train_df['views'] = train_df['views'] / 1000
val_df['views'] = val_df['views'] / 1000

train_df.to_csv('dataset/train_scaled.csv', index=False)
val_df.to_csv('dataset/val_scaled.csv', index=False)
