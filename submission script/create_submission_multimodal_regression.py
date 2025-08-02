import hydra
from torch.utils.data import DataLoader
import pandas as pd
import torch

from data.dataset import Dataset


from torch.nn.utils.rnn import pad_sequence

from open_clip import get_tokenizer

tokenizer = get_tokenizer("ViT-L-14")

def multimodal_collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    texts = [item["text"] for item in batch]
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    ids = [item["id"] for item in batch]

    return {
        "id": ids,
        "image": images,
        "text": texts_padded
    }

def clip_collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    texts = [item["text"] for item in batch]  # list of strings
    text_tokens = tokenizer(texts)  # returns tensor [B, 77]
    
    out = {
        "image": images,
        "text": text_tokens
    }

    if "target" in batch[0]:
        out["target"] = torch.stack([item["target"] for item in batch])
    if "id" in batch[0]:
        out["id"] = [item["id"] for item in batch]
    
    return out

def multimodal_collate_fn_sbert(batch):
    return {
        "image":    torch.stack([item["image"] for item in batch]),             
        "text":    [item["text"] for item in batch],                          
        "tags":     [item["tags"] for item in batch],
        "features": torch.stack([item["features"] for item in batch]),          
        "id":       [item["id"] for item in batch],                             
    }


@hydra.main(config_path="configs", config_name="trainMultimodalV4Regressor") # V4 must be changed to V3 or V2 or V1
def create_submission(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = DataLoader(
        Dataset(
            cfg.datamodule.dataset_path,
            "test",
            transforms=hydra.utils.instantiate(cfg.datamodule.test_transform),
            metadata=cfg.datamodule.metadata
        ),
        batch_size=cfg.datamodule.batch_size,
        shuffle=False,
        num_workers=cfg.datamodule.num_workers,
        collate_fn=multimodal_collate_fn_sbert
    )
    # - Load model and checkpoint
    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    checkpoint = torch.load(cfg.checkpoint_path)
    print(f"Loading model from checkpoint: {cfg.checkpoint_path}")
    model.load_state_dict(checkpoint)
    print("Model loaded")

    # - Create submission.csv
    submission = pd.DataFrame(columns=["ID", "views"])

    for i, batch in enumerate(test_loader):
        batch["image"] = batch["image"].to(device)
        if isinstance(batch["text"], torch.Tensor):
            batch["text"] = batch["text"].to(device)
        batch["features"] = batch["features"].to(device)

        with torch.no_grad():
            preds = model(batch).squeeze().cpu().numpy()
        submission = pd.concat(
            [
                submission,
                pd.DataFrame({"ID": batch["id"], "views": preds}),
            ]
        )
    submission.to_csv(f"{cfg.root_dir}/submission.csv", index=False)


if __name__ == "__main__":
    create_submission()
