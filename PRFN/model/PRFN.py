import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from thop import profile

class DenseAdaptiveUpsampleModule(nn.Module):
    def __init__(self, scale_factor=4, growth_rate=32, num_layers=6, dropout_rate=0.2):
        super(DenseAdaptiveUpsampleModule, self).__init__()
        self.scale_factor = scale_factor
        self.weight_generator = nn.Sequential(
            DenseBlock(4, growth_rate, num_layers, dropout_rate=dropout_rate),
            nn.Conv2d(4 + num_layers * growth_rate, 4 * scale_factor ** 2, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, lr):
        weights = self.weight_generator(lr)
        N, C, H, W = lr.shape
        upscale_factor = self.scale_factor
        weights = weights.view(N, 4, upscale_factor, upscale_factor, H, W)
        weights = weights.permute(0, 1, 4, 2, 5, 3).reshape(N, 4, H * upscale_factor, W * upscale_factor)
        return weights * F.interpolate(lr, scale_factor=upscale_factor, mode='bilinear', align_corners=True)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, dropout_rate=0.2):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout_rate)
                )
            )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, dim=1))
            features.append(new_feature)
        return torch.cat(features, dim=1)



def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class QKVProjection(nn.Module):
    def __init__(self, dim, num_head, qkv_bias=True):
        super(QKVProjection, self).__init__()
        self.dim = dim
        self.num_head = num_head

        self.qkv = nn.Conv2d(dim, 3 * dim, 1, bias=qkv_bias)

    def forward(self, x):
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b (l c) h w -> b l c h w', l=self.num_head)
        return qkv

    def flops(self, resolutions):
        return resolutions[0] * resolutions[1] * 1 * 1 * self.dim * 3 * self.dim


def get_relative_position_index(win_h, win_w):
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)], indexing='ij')) # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1) #
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None,
                                                   :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += win_h - 1
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)


class SpatialSelfAttention(nn.Module):
    def __init__(self, dim, num_head, total_head, window_size=8, shift=0, attn_drop=0.0, proj_drop=0.0, helper=True):
        super(SpatialSelfAttention, self).__init__()
        self.dim = dim
        self.num_head = num_head
        self.total_head = total_head
        self.window_size = window_size
        self.window_area = window_size ** 2
        self.shift = shift
        self.helper = helper

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_head, 1, 1))), requires_grad=True)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_head))

        self.register_buffer("relative_position_index", get_relative_position_index(window_size, window_size))

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Conv2d(dim * num_head, dim * num_head, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def _get_rel_pos_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1) # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.unsqueeze(0)

    def forward(self, qkv, ch=None):
        B, L, C, H, W = qkv.size()
        # window shift
        if self.shift > 0:
            qkv = torch.roll(qkv, shifts=(-self.shift, -self.shift), dims=(-2, -1))

        # window partition
        q, k, v = rearrange(qkv, 'b l c (h wh) (w ww) -> (b h w) l (wh ww) c',
                            wh=self.window_size, ww=self.window_size).chunk(3, dim=-1) # [B_, L1, hw, C/L] respectively
        if ch is not None and self.helper: # [B, C, H, W]
            if self.shift > 0:
                ch = torch.roll(ch, shifts=(-self.shift, -self.shift), dims=(-2, -1))
            ch = rearrange(ch, 'b (l c) (h wh) (w ww) -> (b h w) l (wh ww) c',
                           l=self.total_head - self.num_head, wh=self.window_size,
                           ww=self.window_size) # [B_, L1, hw, C/L]
            ch = torch.mean(ch, dim=1, keepdim=True) # head squeeze [B_, 1, hw, C/L]
            v = v * ch # [B_, L1, hw, C/L]

        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(2, -1) # [B_, L1, hw, hw]
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()
        attn = attn * logit_scale

        attn = attn + self._get_rel_pos_bias()
        attn = self.attn_drop(F.softmax(attn, dim=-1))

        x = attn @ v # [B_, L1, hw, C/L]

        # window unpartition + head merge
        x = window_unpartition(x, (H, W), self.window_size) # [B, L1*C/L, H, W]
        x = self.proj_drop(self.proj(x))

        # window reverse shift
        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(-2, -1))

        return x

    def flops(self, resolutions):
        H, W = resolutions
        num_wins = H // self.window_size * W // self.window_size
        flops = self.num_head * H * W * self.dim if self.helper else 0  # v = v*ch
        flops += num_wins * self.num_head * self.window_area * self.dim * self.window_area # attn = Q@K^T
        flops += num_wins * self.num_head * self.window_area * self.window_area * self.dim # attn@V
        flops += H * W * 1 * 1 * self.num_head * self.dim * self.num_head * self.dim # self.proj
        return flops


def window_unpartition(x, resolutions, window_size):
    return rearrange(x, '(b h w) l (wh ww) c -> b (l c) (h wh) (w ww)',
                     h=resolutions[0] // window_size, w=resolutions[1] // window_size, wh=window_size)


class ChannelSelfAttention(nn.Module):
    def __init__(self, dim, num_head, total_head, attn_drop=0.0, proj_drop=0.0, helper=True):
        super(ChannelSelfAttention, self).__init__()
        self.dim = dim
        self.num_head = num_head
        self.total_head = total_head
        self.helper = helper

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_head, 1, 1))), requires_grad=True)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Conv2d(dim * num_head, dim * num_head, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, qkv, sp=None):
        B, L, C, H, W = qkv.size()

        q, k, v = rearrange(qkv, 'b l c h w -> b l c (h w)').chunk(3, dim=-2) # [B, L2, C/L, HW]
        if sp is not None and self.helper:
            sp = torch.mean(sp, dim=1, keepdim=True) # channel squeeze # [B, 1, H, W]
            sp = rearrange(sp, 'b (l c) h w -> b l c (h w)', l=1) # [B, 1, 1, HW]
            v = v * sp # [B, L2, C/L, HW]

        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(2, -1) # [B, L2, C/L, C/L]
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()
        attn = attn * logit_scale

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v # [B, L2, C/L, HW]

        # head merge
        x = rearrange(x, 'b l c (h w) -> b (l c) h w', h=H) # [B, L2*C/L, H, W]
        x = self.proj_drop(self.proj(x)) # [B, L2*C/L, H, W]

        return x

    def flops(self, resolutions):
        H, W = resolutions
        flops = self.num_head * self.dim * H * W if self.helper else 0  # v = v*sp
        flops += self.num_head * self.dim * H * W * self.dim # attn = Q@K^T
        flops += self.num_head * self.dim * self.dim * H * W # attn@V
        flops += H * W * 1 * 1 * self.num_head * self.dim * self.num_head * self.dim # self.proj
        return flops


class ReshapeLayerNorm(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super(ReshapeLayerNorm, self).__init__()

        self.dim = dim
        self.norm = norm_layer(dim)

    def forward(self, x):
        B, C, H, W = x.size()
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H)
        return x

    def flops(self, resolutions):
        H, W = resolutions
        flops = 0
        flops += H * W * self.dim
        return flops


class MobiVari1(nn.Module):  # MobileNet v1 Variants
    def __init__(self, dim, kernel_size, stride, act=nn.LeakyReLU, out_dim=None):
        super(MobiVari1, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.out_dim = out_dim or dim

        self.dw_conv = nn.Conv2d(dim, dim, kernel_size, stride, kernel_size // 2, groups=dim)
        self.pw_conv = nn.Conv2d(dim, self.out_dim, 1, 1, 0)
        self.act = act()

    def forward(self, x):
        out = self.act(self.pw_conv(self.act(self.dw_conv(x)) + x))
        return out + x if self.dim == self.out_dim else out

    def flops(self, resolutions):
        H, W = resolutions
        flops = H * W * self.kernel_size * self.kernel_size * self.dim + H * W * 1 * 1 * self.dim * self.out_dim # self.dw_conv + self.pw_conv
        return flops


class MobiVari2(MobiVari1):  # MobileNet v2 Variants
    def __init__(self, dim, kernel_size, stride, act=nn.LeakyReLU, out_dim=None, exp_factor=1.2, expand_groups=4):
        super(MobiVari2, self).__init__(dim, kernel_size, stride, act, out_dim)
        self.expand_groups = expand_groups
        expand_dim = int(dim * exp_factor)
        expand_dim = expand_dim + (expand_groups - expand_dim % expand_groups)
        self.expand_dim = expand_dim

        self.exp_conv = nn.Conv2d(dim, self.expand_dim, 1, 1, 0, groups=expand_groups)
        self.dw_conv = nn.Conv2d(expand_dim, expand_dim, kernel_size, stride, kernel_size // 2, groups=expand_dim)
        self.pw_conv = nn.Conv2d(expand_dim, self.out_dim, 1, 1, 0)

    def forward(self, x):
        x1 = self.act(self.exp_conv(x))
        out = self.pw_conv(self.act(self.dw_conv(x1) + x1))
        return out + x if self.dim == self.out_dim else out

    def flops(self, resolutions):
        H, W = resolutions
        flops = H * W * 1 * 1 * (self.dim // self.expand_groups) * self.expand_dim # self.exp_conv
        flops += H * W * self.kernel_size * self.kernel_size * self.expand_dim # self.dw_conv
        flops += H * W * 1 * 1 * self.expand_dim * self.out_dim # self.pw_conv
        return flops


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_ratio, act_layer=nn.GELU, bias=True, drop=0.0):
        super(FeedForward, self).__init__()

        self.dim = dim
        self.hidden_ratio = hidden_ratio

        self.hidden = nn.Conv2d(dim, int(dim * hidden_ratio), 1, bias=bias)
        self.drop1 = nn.Dropout(drop)
        self.out = nn.Conv2d(int(dim * hidden_ratio), dim, 1, bias=bias)
        self.drop2 = nn.Dropout(drop)
        self.act = act_layer()

    def forward(self, x):
        return self.drop2(self.out(self.drop1(self.act(self.hidden(x)))))

    def flops(self, resolutions):
        H, W = resolutions
        flops = 2 * H * W * 1 * 1 * self.dim * self.dim * self.hidden_ratio # self.hidden + self.out
        return flops


class NoLayer(nn.Identity):
    def __init__(self):
        super(NoLayer, self).__init__()

    def flops(self, resolutions):
        return 0

    def forward(self, x, **kwargs):
        return x.flatten(1, 2)

class DRAMiTransformer(nn.Module):
    def __init__(self, dim, num_head=4, chsa_head_ratio=0.25, window_size=8, shift=0, head_dim=None, qkv_bias=True, mv_ver=1,
                 hidden_ratio=2.0, act_layer=nn.GELU, norm_layer=ReshapeLayerNorm, attn_drop=0.0, proj_drop=0.0,
                 drop_path=0.0, helper=True,
                 mv_act=nn.LeakyReLU, exp_factor=1.2, expand_groups=4):
        super(DRAMiTransformer, self).__init__()

        self.dim = dim
        self.num_head = num_head
        self.window_size = window_size
        self.chsa_head = int(num_head * chsa_head_ratio)
        self.shift = shift
        self.helper = helper

        self.qkv_proj = QKVProjection(dim, num_head, qkv_bias=qkv_bias)
        self.sp_attn = SpatialSelfAttention(dim // num_head, num_head - self.chsa_head, num_head,
                                            window_size, shift, attn_drop, proj_drop,
                                            helper) if num_head - self.chsa_head != 0 else NoLayer()
        self.ch_attn = ChannelSelfAttention(dim // num_head, self.chsa_head, num_head, attn_drop, proj_drop,
                                            helper) if self.chsa_head != 0 else NoLayer()
        if mv_ver == 1:
            self.mobivari = MobiVari1(dim, 3, 1, act=mv_act)
        elif mv_ver == 2:
            self.mobivari = MobiVari2(dim, 3, 1, act=mv_act, out_dim=None, exp_factor=exp_factor,
                                      expand_groups=expand_groups)

        self.norm1 = norm_layer(dim)

        self.ffn = FeedForward(dim, hidden_ratio, act_layer=act_layer)
        self.norm2 = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, sp_=None, ch_=None):
        B, C, H, W = x.size()

        # QKV projection + head split
        qkv = self.qkv_proj(x) # [B, L, C, H, W]

        # SP-SA / CH-SA
        sp = self.sp_attn(qkv[:, :self.num_head - self.chsa_head], ch=ch_) # [B, L1*C/L, H, W]
        ch = self.ch_attn(qkv[:, self.num_head - self.chsa_head:], sp=sp_) # [B, L2*C/L, H, W]
        attn0 = self.mobivari(torch.cat([sp, ch], dim=1)) # merge [B, C, H, W]
        attn = self.drop_path(self.norm1(attn0)) + x # LN, skip connection [B, C, H, W]

        # FFN
        out = self.drop_path(self.norm2(self.ffn(attn))) + attn # FFN, LN, skip connection [B, C, H, W]

        return out

    def flops(self, resolutions):
        flops = self.qkv_proj.flops(resolutions)
        flops += self.sp_attn.flops(resolutions)
        flops += self.ch_attn.flops(resolutions)
        flops += self.mobivari.flops(resolutions)
        flops += self.norm1.flops(resolutions)
        flops += self.ffn.flops(resolutions)
        flops += self.norm2.flops(resolutions)
        params = sum([p.numel() for n, p in self.named_parameters()])
        return flops


class GradualFusionModule(nn.Module):
    def __init__(self, channels=4):
        super(GradualFusionModule, self).__init__()

        self.pan_resblock = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )
        self.ms_resblock = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )

        self.fusion_transform = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(channels * 2, channels * 2, 3, 1, 1),
        )
        self.fusion_reduce1x1 = nn.Conv2d(channels * 4, channels, kernel_size=1)

        self.dram = DRAMiTransformer(dim=64)

    def forward(self, pan, ms, fusion):

        pan_delta = self.pan_resblock(pan)
        ms_delta  = self.ms_resblock(ms)

        pan_enhanced = pan + pan_delta
        ms_enhanced  = ms  + ms_delta

        fusion_cat = torch.cat([ms_enhanced, pan_enhanced, fusion], dim=1)
        fusion_cat = self.fusion_reduce1x1(fusion_cat)

        fusion_feat = self.fusion_transform(fusion_cat)
        fusion_feat = self.dram(fusion_feat)

        fusion_out = fusion + fusion_feat

        return pan_enhanced, ms_enhanced, fusion_out

class RecursiveFusionModule(nn.Module):
    def __init__(self, channels, num_recursions):
        super(RecursiveFusionModule, self).__init__()

        self.deep_feature = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 4, 3, padding=1),
            nn.BatchNorm2d(channels * 4),
            nn.ReLU(True),
            nn.Conv2d(channels * 4, channels * 2, 3, padding=1)
        )

        self.multi_scale = nn.ModuleList([
            nn.Conv2d(channels * 2, channels * 2, 3, padding=i, dilation=i)
            for i in [1, 2, 3]
        ])

        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 6, channels * 2, 1),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(True)
        )

        self.num_recursions = num_recursions

    def forward(self, fusion_pre):
        x = fusion_pre
        for i in range(self.num_recursions):
            deep_feat = self.deep_feature(x)
            multi_scale_feats = []
            for conv in self.multi_scale:
                multi_scale_feats.append(conv(deep_feat))

            combined = torch.cat(multi_scale_feats, dim=1)
            refined = self.fusion(combined)

            x = x + refined

        return x

class ProRecFusionNet(nn.Module):
    def __init__(self, channels=32, num_gradualfusion=8, num_recursions=8):
        super(ProRecFusionNet, self).__init__()

        self.channels = channels
        self.num_gradualfusion = num_gradualfusion
        self.num_recursions = num_recursions

        self.recursive_module = RecursiveFusionModule(
            channels=self.channels, num_recursions=self.num_recursions
        )
        self.pan_encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.channels, kernel_size=3, stride=1, padding=1)
        )
        self.ms_encoder = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=self.channels, kernel_size=3, stride=1, padding=1)
        )
        self.fusion_module = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=self.channels * 2, kernel_size=3, stride=1, padding=1)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.channels * 2, out_channels=self.channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(self.channels, 4, 3, 1, padding=1),
        )
        self.blocklists = nn.ModuleList(
            [GradualFusionModule(channels=self.channels) for _ in range(self.num_gradualfusion)]
        )
        self.fusion_conv = nn.Conv2d(self.channels * 2, self.channels, kernel_size=1)
        self.DenseAdaptiveUpsampleModule = DenseAdaptiveUpsampleModule(scale_factor=4)

    def forward(self, pan, lr):
        _, N, H, W = lr.shape
        lr_up = self.DenseAdaptiveUpsampleModule(lr)

        pan_features = self.pan_encoder(pan)
        ms_features  = self.ms_encoder(lr_up)
        fusion_features = self.fusion_module(torch.cat([lr_up, pan], dim=1))

        for block in self.blocklists:
            pan_features, ms_features, fusion_features = block(pan_features, ms_features, fusion_features)

        fusion_features = self.recursive_module(fusion_features)

        out = self.decoder(fusion_features)
        out = torch.sigmoid(out)
        return out

if __name__ == "__main__":
    model = ProRecFusionNet().cuda()

    pan = torch.randn(1, 1, 256, 256).cuda()
    lrms = torch.randn(1, 4, 64, 64).cuda()

    flops, params = profile(model, inputs=(pan, lrms))

    print(f"FLOPS: {flops / 1e9:.3f} G")
    print(f"Params: {params / 1e6:.3f} M")




