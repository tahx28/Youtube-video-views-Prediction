import pandas as pd

# Code for associating each video in train/val data to its corresponding class/category

def assign_category(df):
    df["views"] = pd.to_numeric(df["views"], errors='coerce')
    
    bins = [0, 5_000, 10_000, 25_000, 50_000, 100_000, 500_000, float('inf')]  # Classes for classification
    labels = [0, 1, 2, 3, 4, 5, 6]

    df["category"] = pd.cut(df["views"], bins=bins, labels=labels, right=False)
    df["category"] = df["category"].astype(int)

    return df

train_df = pd.read_csv("dataset/train_fe.csv") 
val_df = pd.read_csv("dataset/val_fe.csv")

train_df = assign_category(train_df)
val_df = assign_category(val_df)

train_df.to_csv("dataset/train_categorized.csv", index=False)
val_df.to_csv("dataset/val_categorized.csv", index=False)


print(train_df[["views", "category"]].head(10))
print(val_df[["views", "category"]].head(10))