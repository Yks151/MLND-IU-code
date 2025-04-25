import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------- Stage 1: Enhanced RetinaNet -----------------
class EnhancedFPN(nn.Module):
    def __init__(self, in_channels=[64, 128, 256], out_channels=256):
        super().__init__()
        self.inner_blocks = nn.ModuleList([
            nn.Conv2d(ch, out_channels, 1) for ch in in_channels
        ])
        self.layer_blocks = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in range(len(in_channels))
        ])
        self.cross_attentions = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3*out_channels, 3, 1),
                nn.Softmax(dim=1)
            ) for _ in range(len(in_channels))
        ])

    def forward(self, feats):
        laterals = [block(f) for f, block in zip(feats, self.inner_blocks)]

        # 自顶向下融合
        for i in range(len(laterals)-1, 0, -1):
            laterals[i-1] += F.interpolate(laterals[i], scale_factor=2, mode='nearest')

        # 跨层注意力融合
        enhanced = []
        for i in range(len(laterals)):
            neighbors = []
            if i > 0:
                neighbors.append(F.avg_pool2d(laterals[i-1], 2))
            neighbors.append(laterals[i])
            if i < len(laterals)-1:
                neighbors.append(F.interpolate(laterals[i+1], scale_factor=2))

            fused = torch.cat(neighbors, dim=1)
            attn = self.cross_attentions[i](fused)  # (B,3,H,W)
            weighted = sum([attn[:,k] * n for k, n in enumerate(neighbors)])
            enhanced.append(self.layer_blocks[i](weighted))

        return enhanced

# ----------------- Stage 2: AG-UNet++ -----------------
class DABM(nn.Module):
    """ Dense Attention Bridging Module """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//reduction, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels//reduction, in_channels, 1),
            nn.Sigmoid()
        )
        # Spatial Attention (Deformable Conv)
        self.sa = nn.Conv2d(in_channels, 2, 3, padding=1)

    def forward(self, x):
        # Channel attention
        ca_weight = self.ca(x)  # (B,C,1,1)
        x_ca = x * ca_weight

        # Deformable spatial attention
        offset = self.sa(x)  # (B,2,H,W)
        grid = self._get_grid(offset)
        x_sa = F.grid_sample(x, grid, align_corners=False)

        return x_ca + x_sa

    def _get_grid(self, offset):
        B, _, H, W = offset.size()
        xx = torch.linspace(-1, 1, W).view(1,1,W).expand(B, H, W)
        yy = torch.linspace(-1, 1, H).view(1,H,1).expand(B, H, W)
        grid = torch.stack([xx, yy], dim=-1).to(offset.device)
        return grid + offset.permute(0,2,3,1)

class AG_UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=5, out_channels=1):
        super().__init__()
        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            DABM(64),
            nn.MaxPool2d(2)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            DABM(128),
            nn.MaxPool2d(2)
        )

        # Decoder with dense connections
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder1 = nn.Sequential(
            DABM(128),
            nn.Conv2d(128, 64, 3, padding=1)
        )

        # Deep supervision heads
        self.aux_heads = nn.ModuleList([
            nn.Conv2d(64, 1, 1),
            nn.Conv2d(64, 1, 1)
        ])

        self.final_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)  # (B,64,H/2,W/2)
        e2 = self.encoder2(e1)  # (B,128,H/4,W/4)

        # Decoder
        d1 = self.up1(e2)  # (B,64,H/2,W/2)
        d1 = torch.cat([d1, e1], dim=1)  # (B,128,H/2,W/2)
        d1 = self.decoder1(d1)  # (B,64,H/2,W/2)

        # Deep supervision
        aux_outputs = [F.interpolate(head(d1), scale_factor=2**i)
                      for i, head in enumerate(self.aux_heads)]

        final = self.final_conv(d1)
        return final, aux_outputs

# ----------------- Stage 3: 3D-CPM -----------------
class CPM3D(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        self.local_conv = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
        self.region_conv = nn.Conv3d(64, 64, kernel_size=5, padding=2)
        self.global_conv = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(64, 64, 1)
        )
        self.fusion = nn.Sequential(
            nn.Conv3d(64*3, 64, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: (B, C, D, H, W)
        local = self.local_conv(x)
        region = self.region_conv(x)
        global_feat = self.global_conv(x).expand_as(x)

        fused = torch.cat([local, region, global_feat], dim=1)
        weights = self.fusion(fused)  # (B,64, D,H,W)

        return (weights * torch.stack([local, region, global_feat])).sum(1)

# ----------------- Full Model -----------------
class MLND_IU(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = EnhancedFPN()
        self.stage2 = AG_UNetPlusPlus()
        self.stage3 = CPM3D()

    def forward(self, x):
        # x: (B,5,H,W)
        # Stage1: 候选区域生成
        proposals = self.stage1(x)  # 多尺度特征

        # Stage2: 精细分割
        seg_pred, aux_preds = self.stage2(proposals[-1])

        # Stage3: 3D上下文验证
        volume = self._stack_slices(seg_pred)  # 构造3D输入
        final_pred = self.stage3(volume)
        return final_pred, aux_preds

    def _stack_slices(self, x):
        # 将2D预测堆叠为伪3D体积
        return x.unsqueeze(2).repeat(1,1,5,1,1)  # (B,C,5,H,W)