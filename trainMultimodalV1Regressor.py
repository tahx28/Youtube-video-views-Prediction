import torch
import wandb
import hydra
from tqdm import tqdm

from data.datamodule import DataModule
from utils.sanity import show_images

import torchvision.transforms as T

@hydra.main(config_path="configs", config_name="trainMultimodalV1Regressor")
def train(cfg):
    logger = (
        wandb.init(project="challenge_CSC_43M04_EP", name=cfg.experiment_name)
        if cfg.log
        else None
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    augmentation = T.Compose([
        T.RandomApply([
            T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.95, 1.05)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.RandomPerspective(distortion_scale=0.2, p=0.2),
        ], p=0.7),

        T.RandomApply([
            T.RandomRotation(degrees=10),
        ], p=0.2),
    ])

    # -- loop over epochs
    for epoch in tqdm(range(cfg.epochs), desc="Epochs"):
        # -- loop over training batches
        model.train()
        epoch_train_loss = 0
        num_samples_train = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for i, batch in enumerate(pbar):
            batch["image"] = batch["image"].to(device)
            batch["target"] = batch["target"].to(device).squeeze()
            if isinstance(batch["text"], torch.Tensor):
                batch["text"] = batch["text"].to(device)
            batch["features"] = batch["features"].to(device)

            batch["image"] = torch.stack([
                augmentation(img.cpu()).to(device) for img in batch["image"]
            ])
            
            preds = model(batch).squeeze()
            loss = loss_fn(preds, batch["target"])
            (
                logger.log({"loss": loss.detach().cpu().numpy()})
                if logger is not None
                else None
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.detach().cpu().numpy() * len(batch["image"])
            num_samples_train += len(batch["image"])
            pbar.set_postfix({"train/loss_step": loss.detach().cpu().numpy()})
        epoch_train_loss /= num_samples_train
        (
            logger.log(
                {
                    "epoch": epoch,
                    "train/loss_epoch": epoch_train_loss
                }
            )
            if logger is not None
            else None
        )

        # -- validation loop
        val_metrics = {}
        epoch_val_loss = 0
        num_samples_val = 0
        model.eval()
        if val_loader is not None: 
            for _, batch in enumerate(val_loader):
                batch["image"] = batch["image"].to(device)
                batch["target"] = batch["target"].to(device).squeeze()
                if isinstance(batch["text"], torch.Tensor):
                    batch["text"] = batch["text"].to(device)
                batch["features"] = batch["features"].to(device)

                with torch.no_grad():
                    preds = model(batch).squeeze()
                loss = loss_fn(preds, batch["target"])
                epoch_val_loss += loss.detach().cpu().numpy() * len(batch["image"])
                num_samples_val += len(batch["image"])
            epoch_val_loss /= num_samples_val
            val_metrics["val/loss_epoch"] = epoch_val_loss
            (
                logger.log(
                    {
                        "epoch": epoch,
                        **val_metrics,
                    }
                )
                if logger is not None
                else None
            )

    print(
        f"""Epoch {epoch}: 
        Training metrics:
        - Train Loss (MSLE): {epoch_train_loss:.4f},
        Validation metrics: 
        - Val Loss (MSLE): {epoch_val_loss:.4f},"""
    )


    if cfg.log:
        logger.finish()

    torch.save(model.state_dict(), cfg.checkpoint_path)


if __name__ == "__main__":
    train()
