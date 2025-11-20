# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
import numpy as np


# ------------------------------------------------------
# GELU
# ------------------------------------------------------
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + F.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


# ------------------------------------------------------
# TorViNet
# ------------------------------------------------------
class TorViNet(nn.Module):

    def __init__(self):
        super(TorViNet, self).__init__()

        self.frame_selector = DynamicFrameSelectionModule()
        self.enhancement_module = FeatureEnhancementModule()

    def forward(self, x):
        # x: [B, 3, 64, 224, 224]
        x = self.frame_selector(x)
        x = self.enhancement_module(x)
        return x


# ------------------------------------------------------
# DFSM
# ------------------------------------------------------
class DynamicFrameSelectionModule(nn.Module):

    def __init__(self, out_channels=4, num_frames_select=4):
        super(DynamicFrameSelectionModule, self).__init__()

        self.num_select = num_frames_select
        self.out_channels = out_channels

        self.embedding = nn.Sequential(
            nn.Conv3d(3, out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # SE Block
        self.se_block = SEBlock(in_channels=64, reduction=32)

    def forward(self, x):


        x = self.embedding(x)  # [B, C, T, H, W]
        B, C, T, H, W = x.size()

        # SE block 计算每帧 importance score
        frame_scores = self.se_block(x.permute(0, 2, 3, 4, 1))  # [B, T]

        # Top-N 与 Bottom-N index
        _, top_idx = torch.topk(frame_scores, self.num_select, dim=1)
        _, bottom_idx = torch.topk(frame_scores, self.num_select, dim=1, largest=False)

        # index reshape
        top_idx = top_idx[:, None, :, None, None].expand(B, C, self.num_select, H, W)
        bottom_idx = bottom_idx[:, None, :, None, None].expand(B, C, self.num_select, H, W)

        x_top = x.gather(2, top_idx)
        x_bottom = x.gather(2, bottom_idx)

        return torch.cat([x_top, x_bottom], dim=2)


# ------------------------------------------------------
# SE Block for temporal saliency
# ------------------------------------------------------
class SEBlock(nn.Module):

    def __init__(self, in_channels, reduction):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduction),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, T, H, W, C]
        B, T, _, _, C = x.shape
        y = self.avg_pool(x).view(B, T)  # [B, T]
        return self.fc(y)                # [B, T]


# ------------------------------------------------------
# Feature Enhancement Module
# ------------------------------------------------------
class FeatureEnhancementModule(nn.Module):

    def __init__(self, num_features=196, num_heads=8):
        super().__init__()

        self.dwt = DWTForward(J=1, wave='haar', mode='periodization')

        self.spatial_embed = PatchEmbedding([224, 224], patch_size=14,
                                            in_chans=4, num_features=num_features)

        self.frequency_embed = PatchEmbedding([112, 112], patch_size=7,
                                              in_chans=4, num_features=num_features)

        self.transformer = SpatiotemporalTransformerBlock(
            dim=num_features*2,
            num_heads=num_heads,
            mlp_ratio=4.0,
            act_layer=GELU
        )

        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(200704, 1)
        )

    def forward(self, x):
        B, C, T, H, W = x.size()

        # -----------------------
        # DWT
        # -----------------------
        x_mean = x.mean(dim=1).view(-1, 1, T, H)  # [B*T, 1, 224, 224]
        xl, (xh,) = self.dwt(x_mean)

        xhh, xhv, xhd = xh[:, :, 0], xh[:, :, 1], xh[:, :, 2]
        freq = torch.cat([xl, xhh, xhv, xhd], dim=1)
        freq = torch.sigmoid(freq)
        freq = freq.view(B, T, -1, H // 2, W // 2).permute(0, 2, 1, 3, 4)

        # -----------------------
        # Patch Embedding
        # -----------------------
        spatial_tokens = self.spatial_embed(x)
        freq_tokens = self.frequency_embed(freq)

        tokens = torch.cat([spatial_tokens, freq_tokens], dim=2)

        # Transformer Block
        tokens = self.transformer(tokens)

        # classification
        tokens = tokens.reshape(B, -1)
        out = self.classifier(tokens)
        return out


# ------------------------------------------------------
# Patch Embedding
# ------------------------------------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, input_shape, patch_size, in_chans, num_features=196):
        super().__init__()

        H, W = input_shape
        self.num_patches = (H // patch_size) * (W // patch_size) * 2

        self.proj = nn.Conv3d(in_chans, num_features,
                              kernel_size=[4, patch_size, patch_size],
                              stride=[4, patch_size, patch_size])

        self.norm = nn.LayerNorm(num_features)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)


# ------------------------------------------------------
# Transformer Block: SFMHA + LC-MLP
# ------------------------------------------------------
class SpatiotemporalTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, act_layer):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpatialFrequencyAttention(dim, num_heads=num_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = LocalizedContrastMLP(dim, int(dim * mlp_ratio), act_layer)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ------------------------------------------------------
# S-F Attention
# ------------------------------------------------------
class SpatialFrequencyAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qk = nn.Linear(dim, dim * 2)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads)
        q, k = qk[:, :, 0], qk[:, :, 1]
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


# ------------------------------------------------------
# Localized Contrast MLP
# ------------------------------------------------------
class LocalizedContrastMLP(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# ------------------------------------------------------
# DropPath
# ------------------------------------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob is None or self.drop_prob == 0.:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep + torch.rand(shape, device=x.device)
        random_tensor.floor_()
        return x / keep * random_tensor






