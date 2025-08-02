import hydra
from torch.utils.data import DataLoader
import pandas as pd
import torch

from data.dataset import Dataset


def multimodal_collate_fn_sbert(batch):
    return {
        "image":    torch.stack([item["image"] for item in batch]),
        "text":     [item["text"] for item in batch],             
        "features": torch.stack([item["features"] for item in batch]),
        "id":       [item["id"] for item in batch],
    }

CLASS_INTERVALS = {
    0: (0, 7_000),
    1: (7_000, 17_000),
    2: (17_000, 53_000),
    3: (53_000, 205_000),
    4: (205_000, None),  
}

CLASS_REPR = {
    0: 3_500,
    1: 12_000,
    2: 35_000,
    3: 120_000,
    4: 400_000,
}

@hydra.main(config_path="configs", config_name="trainMultimodalV4Classifier") # V4 must be changed to V3 or V2
def create_submission(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = DataLoader(
        Dataset(
            cfg.datamodule.dataset_path,
            "test",
            transforms=hydra.utils.instantiate(cfg.datamodule.test_transform),
            metadata=cfg.datamodule.metadata,
        ),
        batch_size=cfg.datamodule.batch_size,
        shuffle=False,
        num_workers=cfg.datamodule.num_workers,
        collate_fn=multimodal_collate_fn_sbert,
    )

    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    ckpt_path = cfg.checkpoint_path
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f" Model loaded from : {ckpt_path}")

    rows = []

    with torch.no_grad():
        for batch in test_loader:
            batch["image"]    = batch["image"].to(device)
            batch["features"] = batch["features"].to(device)

            logits = model(batch)                    
            logits_np = logits.cpu().numpy()         

            for idx, vid in enumerate(batch["id"]):
                logit_vector = logits_np[idx].tolist()  
                rows.append([vid] + logit_vector)

    columns = ["ID"] + [f"logit_class_{i}" for i in range(logits.shape[1])]
    submission = pd.DataFrame(rows, columns=columns)
    submission.to_csv(f"{cfg.root_dir}/submissionLogits.csv", index=False)


if __name__ == "__main__":
    create_submission()
