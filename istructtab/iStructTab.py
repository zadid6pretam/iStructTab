# ======================================================
# iStructTab.py - Multimodal (Image + Tabular) model
# with GEDS feature sequencing + OEMT (Linformer)
# ======================================================
# Requirements:
#   pip install torch torchvision linformer
# ======================================================
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from linformer import Linformer


# ======================================================
# Utility
# ======================================================
def set_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


# ======================================================
# Image Encoder (ResNet-50 → vector)
# ======================================================
class ImageFeatureEncoder(nn.Module):
    """
    ResNet-50 backbone that returns a single feature vector per image.
    Supports RGB + grayscale; optional ImageNet normalization when pretrained=True.
    """

    def __init__(self, d_model: int = 256, pretrained: bool = False, in_channels: int = 3):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
        base = torchvision.models.resnet50(weights=weights)

        # --- handle grayscale / arbitrary channels ---
        if in_channels != 3:
            old_conv = base.conv1
            base.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            # if pretrained and 1-channel, average original weights
            if pretrained and in_channels == 1:
                with torch.no_grad():
                    base.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

        # backbone up to layer4
        self.stem = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4
        )

        # global pooling and projection
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # -> (B, 2048, 1, 1)
        self.proj = nn.Linear(2048, d_model)

        # normalization (if pretrained) – reuse torchvision metadata if available
        if pretrained and weights is not None:
            meta = getattr(weights, "meta", None)
            if meta and "mean" in meta and "std" in meta:
                mean, std = meta["mean"], meta["std"]
            else:
                mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        else:
            self.mean, self.std = None, None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) or (B, H, W) or (H, W)
        returns: image feature vector (B, d_model)
        """
        if x.dim() == 3:
            # (B, H, W) -> (B, 1, H, W)
            x = x.unsqueeze(1)
        if x.dim() == 2:
            # (H, W) -> (1, 1, H, W)
            x = x.unsqueeze(0).unsqueeze(0)

        if x.shape[1] == 1:
            # repeat grayscale to 3 channels for ResNet weights
            x = x.repeat(1, 3, 1, 1)

        if (self.mean is not None) and (self.std is not None):
            x = (x - self.mean) / self.std

        f = self.stem(x)                 # (B, 2048, H', W')
        f = self.gap(f)                  # (B, 2048, 1, 1)
        f = f.view(f.size(0), -1)        # (B, 2048)
        return self.proj(f)              # (B, d_model)


# ======================================================
# Tabular Token Encoder (numeric + categorical + text)
# ======================================================
class TabularTokenEncoder(nn.Module):
    """
    Encodes tabular data into a sequence of tokens with:
      - Numeric features: scalar→token via Linear(1→d_model)
      - Categorical features: learned embedding
      - Text features: EmbeddingBag (mean pooling)
    Handles NaN imputation for numeric inputs (per-batch median).
    """

    def __init__(
        self,
        num_features: Optional[int] = None,
        d_model: int = 256,
        depth: int = 2,
        heads: int = 4,
        vocab_size_text: int = 5000,
        max_cat_card: int = 50,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size_text = vocab_size_text
        self.max_cat_card = max_cat_card

        # numeric → token
        self.scalar_to_token = nn.Linear(1, d_model)

        # categorical
        self.cat_embed = nn.Embedding(max_cat_card + 2, d_model)

        # text (bag-of-words / token ids)
        self.text_embed = nn.EmbeddingBag(vocab_size_text, d_model, mode="mean")

        # small Transformer encoder over tokens
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        # positional encoding (learnable, created lazily)
        self.pos: Optional[nn.Parameter] = None

    def forward(self, x_tab) -> torch.Tensor:
        """
        x_tab can be:
          - Tensor (B, num_numeric)  -> only numeric
          - dict:
              {
                "num":  FloatTensor (B, N_num) with NaNs allowed,
                "cat":  LongTensor  (B, N_cat) with -1 for missing,
                "text": LongTensor  (B, O_text, seq_len)
              }
        returns:
          tokens: (B, O_total, d_model)
        """
        if isinstance(x_tab, dict):
            x_num = x_tab.get("num", None)
            x_cat = x_tab.get("cat", None)
            x_text = x_tab.get("text", None)
        else:
            x_num, x_cat, x_text = x_tab, None, None

        B = (
            x_num.shape[0]
            if x_num is not None
            else x_cat.shape[0]
            if x_cat is not None
            else x_text.shape[0]
        )

        tokens = []

        # --- numeric branch ---
        if x_num is not None:
            # impute NaNs with per-column batch median
            col_median = torch.nanmedian(x_num, dim=0).values
            col_median = torch.where(
                torch.isfinite(col_median),
                col_median,
                torch.zeros_like(col_median),
            )
            x_num = x_num.clone()
            mask = torch.isnan(x_num)
            if mask.any():
                x_num[mask] = col_median.expand_as(x_num)[mask]

            # scalar → token per feature
            tok_num = self.scalar_to_token(x_num.unsqueeze(-1))  # (B, N_num, d_model)
            tokens.append(tok_num)

        # --- categorical branch ---
        if x_cat is not None:
            x_cat = x_cat.clone().to(torch.long)
            # clamp to [-1, max_cat_card-1]; map <0 to special oov index
            x_cat = torch.clamp(x_cat, min=-1, max=self.max_cat_card - 1)
            x_cat[x_cat < 0] = self.max_cat_card + 1
            tok_cat = self.cat_embed(x_cat)  # (B, N_cat, d_model)
            tokens.append(tok_cat)

        # --- text branch ---
        if x_text is not None:
            # x_text: (B, O_text, seq_len)
            B_text, O_text, seq_len = x_text.shape
            x_text_flat = x_text.view(B_text * O_text, seq_len).to(torch.long)
            tok_text = self.text_embed(x_text_flat)              # (B*O_text, d_model)
            tok_text = tok_text.view(B_text, O_text, self.d_model)
            tokens.append(tok_text)

        if not tokens:
            raise ValueError("No valid input provided to TabularTokenEncoder.")

        x = torch.cat(tokens, dim=1)  # (B, O_total, d_model)
        O = x.size(1)

        # lazily (re)create positional embeddings on correct device and length
        if (self.pos is None) or (self.pos.size(1) < O) or (self.pos.device != x.device):
            self.pos = nn.Parameter(
                torch.randn(1, O, self.d_model, device=x.device) * 0.02
            )

        x = x + self.pos[:, :O, :]
        return self.encoder(x)  # (B, O_total, d_model)


# ======================================================
# Tabular encoder wrapper: tokens → vector
# ======================================================
class TabularEncoder(nn.Module):
    """
    Wraps TabularTokenEncoder and pools tokens into a single vector per sample.
    """

    def __init__(
        self,
        num_features: Optional[int],
        d_model: int = 256,
        depth: int = 2,
        heads: int = 4,
        vocab_size_text: int = 5000,
        max_cat_card: int = 50,
        out_dim: Optional[int] = None,
    ):
        super().__init__()
        self.token_encoder = TabularTokenEncoder(
            num_features=num_features,
            d_model=d_model,
            depth=depth,
            heads=heads,
            vocab_size_text=vocab_size_text,
            max_cat_card=max_cat_card,
        )
        self.proj = nn.Linear(d_model, out_dim or d_model)

    def forward(self, x_tab) -> torch.Tensor:
        """
        x_tab: same formats supported as TabularTokenEncoder
        returns: tabular feature vector (B, out_dim)
        """
        tokens = self.token_encoder(x_tab)     # (B, O_total, d_model)
        pooled = tokens.mean(dim=1)            # (B, d_model)
        return self.proj(pooled)               # (B, out_dim)


# ======================================================
# GEDS: Graph-Enhanced Descriptor Sequencing
# ======================================================
class GEDS_GPU(nn.Module):
    """
    GEDS feature sequencing, implemented fully in torch.

    Given fused features F_combined ∈ R^{N×m}, it:
      1) Computes per-feature descriptors d_j = [μ_j, σ_j^2]
      2) Normalizes descriptors and builds cosine similarity matrix A
      3) Applies a 1-layer GCN (Â D W, with ReLU)
      4) Scores each feature s_j = ||d'_j||_2
      5) Returns π_GEDS = argsort(s)
    """

    def __init__(self, in_dim: int = 2, out_dim: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, F_combined: torch.Tensor):
        """
        F_combined: (N, m) fused feature matrix (tab + img)
        returns:
          sequence: LongTensor (m,) - feature sequence π_GEDS
          scores:   Tensor    (m,)  - GEDS scores s_j
        """
        device = F_combined.device
        N, m = F_combined.shape

        # 1) descriptors: mean and variance across samples
        mu = F_combined.mean(dim=0)                          # (m,)
        var = F_combined.var(dim=0, unbiased=False)          # (m,)
        D_desc = torch.stack([mu, var], dim=1)               # (m, 2)

        # 2) normalize descriptors
        D_norm = F.normalize(D_desc, p=2, dim=1)             # (m, 2)

        # 3) cosine similarity matrix + self loops
        A = D_norm @ D_norm.T                                # (m, m)
        A = A + torch.eye(m, device=device)

        # 4) symmetric normalization: A_hat = D^{-1/2} A D^{-1/2}
        deg = A.sum(dim=1).clamp(min=1e-6)                   # (m,)
        deg_inv_sqrt = deg.pow(-0.5)
        A_hat = deg_inv_sqrt.unsqueeze(1) * A * deg_inv_sqrt.unsqueeze(0)

        # 5) GCN refinement
        D_trans = self.fc(D_norm)                            # (m, out_dim)
        D_ref = F.relu(A_hat @ D_trans)                      # (m, out_dim)

        # 6) scores and feature sequence (ascending)
        scores = D_ref.norm(p=2, dim=1)                      # (m,)
        sequence = torch.argsort(scores, descending=False)   # (m,)

        return sequence, scores


# ======================================================
# OEMT with Linformer backbone
# ======================================================
class OEMT(nn.Module):
    """
    Order-Aware Efficient Transformer with Memory Augmentation (OEMT),
    using Linformer as the encoder over (k + M) tokens.

    Input:
      x: (B, m) - **already sequenced** feature vectors
    Output:
      logits: (B, num_classes)
      fs_out: (B, m) - feature-sequencing scores s'
    """

    def __init__(
        self,
        input_dim: int,
        d_t: int = 256,
        k: int = 128,
        M: int = 10,
        num_heads: int = 4,
        num_layers: int = 2,
        num_classes: int = 2,
        linformer_k: int = 32,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_t = d_t

        # number of pooled tokens cannot exceed input_dim
        self.k = min(k, input_dim)
        self.M = M

        # per-feature projection to d_t
        # T_in[b, j, :] = x[b, j] * proj_w[j] + proj_b[j]
        self.proj_w = nn.Parameter(torch.randn(input_dim, d_t) * 0.02)
        self.proj_b = nn.Parameter(torch.zeros(input_dim, d_t))

        # order-aware pooling: m → k tokens
        self.pool = nn.Linear(d_t, self.k)

        # M learnable memory tokens
        self.memory_tokens = nn.Parameter(torch.randn(M, d_t) * 0.02)

        # Linformer encoder on (k + M) tokens
        self.linformer = Linformer(
            dim=d_t,
            seq_len=self.k + self.M,
            depth=num_layers,
            heads=num_heads,
            k=linformer_k,
            one_kv_head=True,
            share_kv=True,
        )

        # classification head from pooled memory representation
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_t),
            nn.Linear(d_t, num_classes),
        )

        # feature-sequencing head: from M·d_t back to m scores
        self.order_head = nn.Linear(self.M * d_t, input_dim)

    def forward(self, x: torch.Tensor):
        """
        x: (B, m) - sequenced feature vectors
        returns:
          logits: (B, num_classes)
          fs_out: (B, m)
        """
        B, m = x.shape
        if m != self.input_dim:
            raise ValueError(
                f"OEMT expected input_dim={self.input_dim}, got m={m}."
            )

        # 1) per-feature projection to token space
        #    T_in[b, j, :] = x[b, j] * proj_w[j] + proj_b[j]
        T_in = (
            x.unsqueeze(-1) * self.proj_w.unsqueeze(0)
            + self.proj_b.unsqueeze(0)
        )  # (B, m, d_t)
        T_in = F.relu(T_in, inplace=True)

        # 2) order-aware pooling: m → k tokens
        scores = self.pool(T_in)                            # (B, m, k)
        P = F.softmax(scores.transpose(1, 2), dim=2)        # (B, k, m)
        Tpool = torch.bmm(P, T_in)                          # (B, k, d_t)

        # 3) prepend M memory tokens
        mem = self.memory_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, M, d_t)
        Tcat = torch.cat([Tpool, mem], dim=1)                     # (B, k+M, d_t)

        # 4) Linformer encoder
        Tout = self.linformer(Tcat)                         # (B, k+M, d_t)

        # 5a) classification from memory tokens
        mem_out = Tout[:, self.k :, :]                      # (B, M, d_t)
        h = mem_out.mean(dim=1)                             # (B, d_t)
        logits = self.classifier(h)                         # (B, num_classes)

        # 5b) feature-sequencing head
        fs_out = self.order_head(mem_out.reshape(B, -1))    # (B, m)

        return logits, fs_out


# ======================================================
# iStructTab Model: Image + Tabular → GEDS → OEMT
# ======================================================
class iStructTab(nn.Module):
    """
    End-to-end multimodal model:
      Tabular (Transformer) + Image (ResNet-50)
      → fused feature matrix F
      → GEDS feature sequencing (π_GEDS)
      → OEMT (Linformer) over sequenced features
    """

    def __init__(
        self,
        num_tab_features: Optional[int],
        num_classes: int,
        d_model: int = 128,
        # tabular encoder params
        tab_depth: int = 2,
        tab_heads: int = 4,
        vocab_size_text: int = 5000,
        max_cat_card: int = 50,
        # GEDS params
        geds_in_dim: int = 2,
        geds_out_dim: int = 2,
        # OEMT + Linformer params
        oemt_k: int = 128,
        oemt_M: int = 10,
        oemt_heads: int = 4,
        oemt_layers: int = 2,
        linformer_k: int = 32,
        # losses / training
        lambda_fs: float = 0.1,
        # image encoder
        pretrained_resnet: bool = False,
        img_in_channels: int = 3,
    ):
        super().__init__()
        self.lambda_fs = float(lambda_fs)

        # --- encoders ---
        # Tabular: tokens → pooled vector (dimension d_model)
        self.tab_enc = TabularEncoder(
            num_features=num_tab_features,
            d_model=d_model,
            depth=tab_depth,
            heads=tab_heads,
            vocab_size_text=vocab_size_text,
            max_cat_card=max_cat_card,
            out_dim=d_model,
        )

        # Image: ResNet-50 → feature vector (dimension d_model)
        self.img_enc = ImageFeatureEncoder(
            d_model=d_model,
            pretrained=pretrained_resnet,
            in_channels=img_in_channels,
        )

        # total fused feature dimension m = o + d = d_model + d_model
        self.tab_dim = d_model
        self.img_dim = d_model
        self.m = self.tab_dim + self.img_dim

        # GEDS feature sequencing
        self.geds = GEDS_GPU(in_dim=geds_in_dim, out_dim=geds_out_dim)

        # OEMT backbone with Linformer
        self.oemt = OEMT(
            input_dim=self.m,
            d_t=d_model,
            k=oemt_k,
            M=oemt_M,
            num_heads=oemt_heads,
            num_layers=oemt_layers,
            num_classes=num_classes,
            linformer_k=linformer_k,
        )

    def forward(self, x_tab, x_img, y: Optional[torch.Tensor] = None):
        """
        x_tab:  same formats as TabularTokenEncoder (Tensor or dict with num/cat/text)
        x_img:  (B, C, H, W) or (B, H, W) / (H, W)
        y:      optional labels (B,) for classification
        returns:
          out = {
            "logits":      (B, num_classes),
            "sequence":    (m,),
            "geds_scores": (m,),
            "fs_scores":   (B, m),
            "beta":        (B, m),
            "loss":        scalar (if y is not None),
            "loss_ce":     scalar,
            "loss_fs":     scalar,
          }
        """
        # 1) modality-specific encoders
        tab_vec = self.tab_enc(x_tab)        # (B, d_model)
        img_vec = self.img_enc(x_img)        # (B, d_model)

        # 2) fuse into unified feature matrix F ∈ R^{B×m}
        F_combined = torch.cat([tab_vec, img_vec], dim=1)  # (B, m)

        # 3) GEDS feature sequencing (single sequence over m dims)
        sequence, geds_scores = self.geds(F_combined)       # sequence: (m,)

        # 4) reorder features according to π_GEDS
        F_reordered = F_combined[:, sequence]               # (B, m)

        # 5) OEMT backbone with Linformer
        logits, fs_scores = self.oemt(F_reordered)          # (B, num_classes), (B, m)

        # 6) feature-sequencing target β (descending linear ramp)
        B, m = F_reordered.shape
        device = logits.device
        ranks = torch.arange(m, device=device, dtype=torch.float32)
        if m > 1:
            beta_vec = 1.0 - ranks / (m - 1.0)
        else:
            beta_vec = torch.ones(1, device=device)
        beta = beta_vec.unsqueeze(0).expand(B, m)           # (B, m)

        out = {
            "logits": logits,
            "sequence": sequence,
            "geds_scores": geds_scores,
            "fs_scores": fs_scores,
            "beta": beta,
        }

        # 7) losses (classification + feature sequencing)
        if y is not None:
            ce = F.cross_entropy(logits, y)
            fs_loss = F.mse_loss(fs_scores, beta)
            loss = ce + self.lambda_fs * fs_loss

            out["loss"] = loss
            out["loss_ce"] = ce
            out["loss_fs"] = fs_loss

        return out