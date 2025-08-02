# Code for Multimodal V4 Classifier inspired from Entube paper


from __future__ import annotations
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------#
class ResidualAttentionBlock(nn.Module):
    """LayerNorm → Multi-Head-Attention → LeakyReLU + skip-connection."""
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.act  = nn.LeakyReLU(0.1)

    def forward(self, x):                       
        residual = x
        x = self.norm(x)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        return self.act(residual + attn_out)

# ---------------------------------------------------------------------------#
class MultiModalV3Classifier(nn.Module):
    def __init__(
        self,
        num_tab_features: int = 42,
        num_classes: int = 7,
        text_model_name: str = "sentence-transformers/distilbert-base-nli-mean-tokens",
        frozen_image: bool = True,
        trainable_text: bool = False,
    ):
        super().__init__()

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
        txt_dim = self.text_encoder.get_sentence_embedding_dimension()  # 768
        self.text_rab = ResidualAttentionBlock(dim=txt_dim)
        self.text_lin = nn.Linear(txt_dim, 512)     # projeter à 512

        # 3. Tabular features 
        #self.tab_norm = nn.BatchNorm1d(num_tab_features)
        self.tab_lin  = nn.Linear(num_tab_features, 512)  

        # 4. Fusion par Conv1d  
        #   Entrée : (B, 3, 512) → Conv1d → (B, 128, 8) → flatten 1024
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(3, 64,  kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),

            nn.AdaptiveMaxPool1d(8)  
        )

        # 5. Classification head  
        self.classification_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, num_classes),
        )

    # ------------------------  encoders ------------------------- #
    def _encode_thumbnail(self, imgs: torch.Tensor) -> torch.Tensor:
        x = self.image_backbone(imgs)              
        if x.dim() == 4:                           
            x = torch.flatten(x, 1)
        x = self.image_rab(x.unsqueeze(1)).squeeze(1)  
        return self.image_lin(x)                   

    def _encode_title(self, titles: list[str], device) -> torch.Tensor:
        x = self.text_encoder.encode(titles, convert_to_tensor=True,
                                     device=device)           
        x = self.text_rab(x.unsqueeze(1)).squeeze(1)           
        return self.text_lin(x)                                

    def _encode_features(self, feats: torch.Tensor) -> torch.Tensor:
        #feats = self.tab_norm(feats)               
        return self.tab_lin(feats)                 

    # ---------------------------- forward ------------------------------ #
    def forward(self, batch: dict):
        """
        batch = {
            "image": Tensor [B,3,224,224],
            "text":  list[str] taille B,
            "features": Tensor [B,42]
        }
        """
        device = batch["image"].device

        img_vec = self._encode_thumbnail(batch["image"])        
        txt_vec = self._encode_title(batch["text"], device)     
        tab_vec = self._encode_features(batch["features"])      

        # Fusion
        fused = torch.stack([img_vec, txt_vec, tab_vec], dim=1)

        conv_out = self.fusion_conv(fused).flatten(1)

        return self.classification_head(conv_out)               # (B, 7 logits)
