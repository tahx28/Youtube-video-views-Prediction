# Code for Multimodal V4 Regressor inspired from Entube paper


import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------#
class ResidualAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.act  = nn.LeakyReLU(0.1)

    def forward(self, x):      
        res = x
        x = self.norm(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        return self.act(res + x)

# ---------------------------------------------------------------------------#
class MultiModalV3Regressor(nn.Module):
    def __init__(
        self,
        num_tab_features: int = 42,
        text_model_name: str = "sentence-transformers/distilbert-base-nli-mean-tokens",
        frozen_image: bool = True,
        trainable_text: bool = False,
        include_tags: bool = False,
    ):
        super().__init__()
        self.include_tags = include_tags

        # 1. Thumbnail encoder 
        res = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        res.fc = nn.Identity()
        self.image_backbone = res
        if frozen_image:
            for p in self.image_backbone.parameters():
                p.requires_grad = False
        self.image_rab = ResidualAttentionBlock(dim=2048)
        self.image_lin = nn.Linear(2048, 512)

        # 2. Title encoder
        self.text_encoder = SentenceTransformer(text_model_name)
        if not trainable_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        txt_dim = self.text_encoder.get_sentence_embedding_dimension()
        self.text_rab = ResidualAttentionBlock(dim=txt_dim)
        self.text_lin = nn.Linear(txt_dim, 512)

        # 3. Tabular features 
        #self.tab_norm = nn.BatchNorm1d(num_tab_features)
        self.tab_lin  = nn.Linear(num_tab_features, 512) 

        # 4. Tags encoder 
        if self.include_tags:
            self.tag_rab = ResidualAttentionBlock(dim=txt_dim)
            self.tag_lin = nn.Linear(txt_dim, 512)

        # 5. Fusion Conv1d 
        if self.include_tags:
            self.fusion_conv = nn.Sequential(
                nn.Conv1d(4, 64,  kernel_size=3, padding=1), 
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(8)  
            )

        else:
            self.fusion_conv = nn.Sequential(
                nn.Conv1d(3, 64,  kernel_size=3, padding=1),
                nn.BatchNorm1d(64), nn.ReLU(),

                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128), nn.ReLU(),

                nn.AdaptiveMaxPool1d(8)  
            )


        # 6. Head MLP
        self.reg_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def _encode_thumbnail(self, imgs: torch.Tensor) -> torch.Tensor:
        x = self.image_backbone(imgs)
        if x.dim() == 4:
            x = x.flatten(1)
        x = self.image_rab(x.unsqueeze(1)).squeeze(1)
        return self.image_lin(x)

    def _encode_title(self, titles: list[str], device) -> torch.Tensor:
        x = self.text_encoder.encode(titles, convert_to_tensor=True, device=device)
        x = self.text_rab(x.unsqueeze(1)).squeeze(1)
        return self.text_lin(x)

    def _encode_features(self, feats: torch.Tensor) -> torch.Tensor:
        #feats = self.tab_norm(feats) 
        return self.tab_lin(feats)

    def _encode_tags(self, tags: list[str], device) -> torch.Tensor:
        x = self.text_encoder.encode(tags, convert_to_tensor=True, device=device)
        x = self.tag_rab(x.unsqueeze(1)).squeeze(1)
        return self.tag_lin(x)

    def forward(self, batch: dict):
        device = batch["image"].device

        img_vec = self._encode_thumbnail(batch["image"])       
        txt_vec = self._encode_title(batch["text"], device)    
        tab_vec = self._encode_features(batch["features"])  

        # Fusion
        if self.include_tags:   
            tag_vec = self._encode_tags(batch["tags"], device)     
            fused = torch.stack([img_vec, txt_vec, tab_vec, tag_vec], dim=1)

        else:
            fused = torch.stack([img_vec, txt_vec, tab_vec], dim=1)


        conv_out = self.fusion_conv(fused).flatten(1)          
        return self.reg_head(conv_out)                         # (B,1)
