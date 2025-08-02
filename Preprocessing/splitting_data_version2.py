import pandas as pd

df = pd.read_csv('dataset/train_val.csv')
df['date'] = pd.to_datetime(df['date'], errors='coerce')

bins = [0, 5_000, 10_000, 25_000, 50_000, 100_000, 500_000, float('inf')] # Classes for classification
labels = [0, 1, 2, 3, 4, 5, 6]
df['view_class'] = pd.cut(df['views'], bins=bins, labels=labels, right=False)

df_2023 = df[df['date'].dt.year == 2023]

k = 100
df_val_list = []

for label in labels:
    class_subset = df_2023[df_2023['view_class'] == label]
    if len(class_subset) < k:
        print(f"⚠️ Class {label}: only {len(class_subset)} videos are available !")
    sampled = class_subset.sample(n=min(k, len(class_subset)), random_state=42)
    df_val_list.append(sampled)

df_val = pd.concat(df_val_list)
val_ids = set(df_val['id'])  

df_train = df[~df['id'].isin(val_ids)]

print(f"Validation set: {len(df_val)} vidéos")
print(f"Training set: {len(df_train)} vidéos")

df_train.to_csv("dataset/train.csv", index=False)
df_val.to_csv("dataset/val.csv", index=False)
