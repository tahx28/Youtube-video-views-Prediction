from torch.utils.data import DataLoader, random_split
import torch

from data.dataset import Dataset

from torch.nn.utils.rnn import pad_sequence

from open_clip import get_tokenizer

tokenizer = get_tokenizer("ViT-L-14")


def multimodal_collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    texts = [item["text"] for item in batch]
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    targets = torch.tensor([item["target"] for item in batch], dtype=torch.float32)

    return {
        "image": images,
        "text": texts_padded,
        "target" : targets
    }


def clip_collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    texts = [item["text"] for item in batch]  
    text_tokens = tokenizer(texts)  
    
    out = {
        "image": images,
        "text": text_tokens
    }

    if "target" in batch[0]:
        out["target"] = torch.stack([item["target"] for item in batch])
    if "id" in batch[0]:
        out["id"] = [item["id"] for item in batch]
    
    return out

def multimodal_collate_fn_sbert(batch, classifier=False):
    dtype = torch.long if classifier else torch.float32
    return {
        "image":    torch.stack([item["image"] for item in batch]),
        "text":     [item["text"] for item in batch],
        "features": torch.stack([item["features"] for item in batch]),
        "tags":     [item["tags"] for item in batch],  
        "target":   torch.tensor([item["target"] for item in batch], dtype=dtype),
        "id":       [item["id"] for item in batch],
    }




class DataModule:
    def __init__(
        self,
        dataset_path,
        train_transform,
        test_transform,
        batch_size,
        num_workers,
        metadata=["title"],
        tokenize = False, 
        classifier = False,
        subchannel = None
    ):
        self.dataset_path = dataset_path
        self.train_transform = train_transform  
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.metadata = metadata
        self.tokenize = tokenize
        self.classifier = classifier
        self.subchannel = subchannel
        self._train_val_split_done = False 


    def train_dataloader(self):
        """Train dataloader."""
        train_set = Dataset(
            self.dataset_path,
            "train",
            transforms=self.train_transform,
            metadata=self.metadata,
            tokenize = self.tokenize,
            classifier = self.classifier,
            subchannel = self.subchannel

        )
        return DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn= lambda batch: multimodal_collate_fn_sbert(batch, classifier=self.classifier)

        )

    def val_dataloader(self):
        """TODO: 
        Implement a strategy to create a validation set from the train set.
        """
        val_set = Dataset(
            self.dataset_path,
            "val",
            transforms=self.train_transform,
            metadata=self.metadata,
            tokenize = self.tokenize,
            classifier = self.classifier,
            subchannel = self.subchannel
        )
        return DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn= lambda batch: multimodal_collate_fn_sbert(batch, classifier=self.classifier)
        )
    
    def test_dataloader(self):
        """Test dataloader."""
        dataset = Dataset(
            self.dataset_path,
            "test",
            transforms=self.test_transform,
            metadata=self.metadata,
            classifier = self.classifier,
            tokenize = self.tokenize
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn= lambda batch: multimodal_collate_fn_sbert(batch, classifier=self.classifier)
        )