import torch
import wandb
import hydra
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import numpy as np

from data.datamodule import DataModule
from utils.sanity import show_images

import torchvision.transforms as T

@hydra.main(config_path="configs", config_name="trainMultimodalV1Classifier")
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
        num_correct_train = 0
        num_samples_train = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for i, batch in enumerate(pbar):
            batch["image"] = batch["image"].to(device)
            batch["target"] = batch["target"].to(device)
            if isinstance(batch["text"], torch.Tensor):
                batch["text"] = batch["text"].to(device)
            batch["features"] = batch["features"].to(device)

            batch["image"] = torch.stack([
                augmentation(img.cpu()).to(device) for img in batch["image"]
            ])
            
            preds = model(batch)
            loss = loss_fn(preds, batch["target"])
            pred_classes = torch.argmax(preds, dim=1)
            correct_train = (pred_classes == batch["target"]).sum().item()

            (
                logger.log({"loss": loss.detach().cpu().numpy()})
                if logger is not None
                else None
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.detach().cpu().numpy() * len(batch["image"])
            num_correct_train += correct_train
            num_samples_train += len(batch["image"])
            pbar.set_postfix({"train/loss_step": loss.detach().cpu().numpy()})
        epoch_train_loss /= num_samples_train
        train_acc = num_correct_train / num_samples_train
        (
            logger.log(
                {
                    "epoch": epoch,
                    "train/loss_epoch": epoch_train_loss,
                    "train/accuracy": train_acc
                }
            )
            if logger is not None
            else None
        )

        # -- validation loop
        val_metrics = {}
        epoch_val_loss = 0
        num_correct_val = 0
        num_samples_val = 0
        all_preds = []
        all_targets = []
        model.eval()
        if val_loader is not None: 
            for _, batch in enumerate(val_loader):
                batch["image"] = batch["image"].to(device)
                batch["target"] = batch["target"].to(device)
                if isinstance(batch["text"], torch.Tensor):
                    batch["text"] = batch["text"].to(device)
                batch["features"] = batch["features"].to(device)

                with torch.no_grad():
                    preds = model(batch)

                loss = loss_fn(preds, batch["target"])
                pred_classes = torch.argmax(preds, dim=1)
                correct_val = (pred_classes == batch["target"]).sum().item()
                all_preds.extend(pred_classes.cpu().numpy())
                all_targets.extend(batch["target"].cpu().numpy())
                epoch_val_loss += loss.detach().cpu().numpy() * len(batch["image"])
                num_correct_val += correct_val
                num_samples_val += len(batch["image"])
            epoch_val_loss /= num_samples_val
            val_acc = num_correct_val / num_samples_val
            val_metrics["val/loss_epoch"] = epoch_val_loss
            val_metrics["val/accuracy"] = val_acc
            all_preds = np.array(all_preds)
            all_targets = np.array(all_targets)

            # Matrice de confusion
            conf_mat = confusion_matrix(all_targets, all_preds)
            f1_macro = f1_score(all_targets, all_preds, average='macro')
            f1_weighted = f1_score(all_targets, all_preds, average='weighted')

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

            if False:
                logger.log({
                    "val/f1_macro": f1_macro,
                    "val/f1_weighted": f1_weighted,
                    "val/confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=all_targets,
                        preds=all_preds,
                        class_names=[str(i) for i in sorted(set(all_targets))],
                    )
                })


    print(
    f"""Epoch {epoch}: 
    Training metrics:
    - Train Loss: {epoch_train_loss:.4f},
    - Train Acc:  {train_acc:.4f}
    """
    )

    if num_samples_val > 0:
        print("Confusion Matrix:\n", conf_mat)
        print("F1 Score (macro):", f1_macro)
        print("F1 Score (weighted):", f1_weighted)
        print("Classification Report:\n", classification_report(all_targets, all_preds))

    if cfg.log:
        logger.finish()

    torch.save(model.state_dict(), cfg.checkpoint_path)


if __name__ == "__main__":
    train()
