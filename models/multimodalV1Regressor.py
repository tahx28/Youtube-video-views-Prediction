import torch
import torch.nn as nn
from torchvision import transforms
from sentence_transformers import SentenceTransformer


class MultiModalV1Regressor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        text_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        trainable_text: bool = False,
        frozen_image: bool = True,
    ):
        super().__init__()

        # 1. Thumbnail encoder
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")
        self.backbone.head = nn.Identity()
        self.image_dim = int(self.backbone.norm.normalized_shape[0])  

        if frozen_image:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 2. Text encoder 
        self.text_encoder = SentenceTransformer(text_model_name)
        self.text_encoder.eval()

        if not trainable_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.text_dim = self.text_encoder.get_sentence_embedding_dimension()

        self.feat_dim = feature_dim

        # 3. Regression head 
        fusion_dim = self.image_dim + self.text_dim + self.feat_dim
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.ReLU(),

            nn.Linear(128, 1)
        )

    def forward(self, batch: dict):

        x_img = self.backbone(batch["image"])  

        titles: list[str] = batch["text"]
        device = next(self.text_encoder.parameters()).device
        x_txt = self.text_encoder.encode(titles, convert_to_tensor=True, device=device)

        x_tab = batch["features"]  

        # Fusion
        x = torch.cat([x_img, x_txt, x_tab], dim=1)
        out = self.regression_head(x)
        return out.squeeze(1)  



