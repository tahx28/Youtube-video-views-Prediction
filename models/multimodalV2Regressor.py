import torch
import torch.nn as nn
import torchvision.models as models
from transformers import DistilBertModel, DistilBertTokenizer


class MultiModalV2Regressor(nn.Module):

    def __init__(
        self,
        feature_dim: int,
        freeze_image: bool = True,
        freeze_text: bool = True
    ):
        super().__init__()

        # 1. Thumbnail encoder
        self.image_encoder = models.resnet50(pretrained=True)
        self.image_encoder.fc = nn.Identity()
        self.image_dim = 2048  

        if freeze_image:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        # 2. Title encoder
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.text_dim = self.text_encoder.config.hidden_size  

        if freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.feat_dim = feature_dim

        # 4. Regression head
        fusion_dim = self.image_dim + self.text_dim + self.feat_dim
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.ReLU(),

            nn.Linear(128, 1)
        )

    def forward(self, batch: dict) -> torch.Tensor:

        x_img = self.image_encoder(batch['image']) 

        texts = batch['text']  
        device = x_img.device
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(device)
        txt_outputs = self.text_encoder(
            input_ids=tokens['input_ids'],
            attention_mask=tokens['attention_mask']
        )
        x_txt = txt_outputs.last_hidden_state[:, 0, :]

        x_feat = batch['features']  

        #  Fusion 
        x = torch.cat([x_img, x_txt, x_feat], dim=1)  

        out = self.regressor(x)  
        return out  # [B, 1]
