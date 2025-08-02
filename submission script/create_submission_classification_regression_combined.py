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
    0: (0, 5_000),
    1: (5_000, 10_000),
    2: (10_000, 25_000),
    3: (25_000, 50_000),
    4: (50_000, 100_000),
    5: (100_000, 500_000),
    6: (500_000, None), 
}

CLASS_REPR = {
    0: 2_500,
    1: 7_500,
    2: 17_500,
    3: 37_500,
    4: 75_000,
    5: 300_000,
    6: 750_000,
}

def class_from_views(views: float) -> int:
    if views < 5_000: return 0
    if views < 10_000: return 1
    if views < 25_000: return 2
    if views < 50_000: return 3
    if views < 100_000: return 4
    if views < 500_000: return 5
    return 6
# ------------------------------------------------------------------------- #

@hydra.main(config_path="configs", config_name="trainEfficientNetClassifier")
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
    print(f"Model loaded from : {ckpt_path}")

    regression_sub = pd.read_csv('RegressionPredictions.csv')  # Load regression predictions - Path must be adapted
    reg_pred_map = dict(zip(regression_sub["ID"], regression_sub["views"]))
    print(f"✔ {len(reg_pred_map):,} regression predictions loaded from CSV")

    softmax = torch.nn.Softmax(dim=1)
    rows = []

    with torch.no_grad():
        for batch in test_loader:
            batch["image"]    = batch["image"].to(device)
            batch["features"] = batch["features"].to(device)

            logits = model(batch)                   
            probs  = softmax(logits).cpu()          

            for idx, vid in enumerate(batch["id"]):
                p        = probs[idx]               
                i        = int(torch.argmax(p))     
                reg_pred = float(reg_pred_map[vid]) 
                j        = class_from_views(reg_pred)

                if i == j:
                    final_views = reg_pred
                elif j < i:
                    left = CLASS_INTERVALS[i][0]
                    final_views = p[i] * left + p[j] * reg_pred
                else:  # j > i
                    right = CLASS_INTERVALS[i][1]
                    if right is None:               
                        right = reg_pred
                    final_views = p[i] * right + p[j] * reg_pred
                final_views = float(final_views)
                rows.append((vid, max(0.0, final_views)))

    submission = pd.DataFrame(rows, columns=["ID", "views"])
    submission.to_csv(f"{cfg.root_dir}/submission.csv", index=False)
    print(f"Submission saved ")

if __name__ == "__main__":
    create_submission()
