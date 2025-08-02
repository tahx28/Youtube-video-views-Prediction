import torch
import pandas as pd
from PIL import Image
from transformers import BertTokenizer



class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, transforms, classifier=False, subchannel=None, metadata=["title"], tokenize=False, max_length=30):
        self.dataset_path = dataset_path
        self.split = split
        self.classifier = classifier
        self.subchannel = subchannel
        self.NON_NUMERIC = {'Unnamed: 0', "id", "title", "description", "meta", "channel", "date", "views", "year", "category", "channel_known", "view_class", "tags"}

        # - read the info csvs
        self.suffix = 'categorized' if self.classifier else 'fe'
        info = pd.read_csv(f"{dataset_path}/{split}_{self.suffix}.csv")
        if subchannel is not None:
            info = info[info['channel'] == self.subchannel]

        info["description"] = info["description"].fillna("")
        info["meta"] = info[metadata].agg(" + ".join, axis=1)

        if self.classifier:
            if "category" in info.columns:
                self.targets = info["category"].values
        else:
            if "views" in info.columns:
                self.targets = info["views"].values

        # - ids
        self.ids = info["id"].values

        # - text and tags
        self.text = info["meta"].values
        self.tags = info["tags"].values  

        # - features
        self.feature_cols = [c for c in info.columns if c not in self.NON_NUMERIC]
        self.feats = info[self.feature_cols].astype("float32").values

        self.transforms = transforms
        self.tokenize = tokenize
        self.max_length = max_length

        if self.tokenize:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, idx):
        # - load the image
        if self.split != 'test':
            self.path = 'train_val'
        else:
            self.path = self.split

        image = Image.open(f"{self.dataset_path}/{self.path}/{self.ids[idx]}.jpg").convert("RGB")
        image = self.transforms(image)

        # encode meta text
        if self.tokenize:
            encoded_text = self.tokenizer(
                self.text[idx],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            text_tensor = encoded_text["input_ids"].squeeze(0)  # [seq_len]
        else:
            text_tensor = self.text[idx]

        feat = torch.tensor(self.feats[idx], dtype=torch.float32)

        value = {
            "id": self.ids[idx],
            "image": image,
            "text": text_tensor,
            "features": feat,
            "tags": self.tags[idx]  
        }

        if hasattr(self, "targets"):
            dtype = torch.long if self.classifier else torch.float32
            value["target"] = torch.tensor(self.targets[idx], dtype=dtype)

        return value
