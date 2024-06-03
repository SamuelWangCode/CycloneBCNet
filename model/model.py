import torch
from torch import nn
import torch.nn.functional as F

from typhoon_intensity_bc.model.module import (ConvSC, ConvNeXtSubBlock, ConvMixerSubBlock, GASubBlock, gInception_ST,
                                               HorNetSubBlock, MLPMixerSubBlock, MogaSubBlock, PoolFormerSubBlock,
                                               SwinSubBlock, UniformerSubBlock, VANSubBlock, ViTSubBlock, TAUSubBlock)


class ValueEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(ValueEncoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) if i == 0 else nn.Linear(hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.activation = nn.ReLU()

    def forward(self, x):
        # x shape: (B, T, C2)
        for layer in self.layers:
            x = self.activation(layer(x))
        return x



class SpatialAttentionWeight(nn.Module):
    def __init__(self, height, width, N, center_weight):
        super(SpatialAttentionWeight, self).__init__()
        self.base_height = height
        self.base_width = width
        self.N = N
        self.center_weight = center_weight
        self.weights = None
        self.initialize_weights()

    def initialize_weights(self):
        self.weights = torch.ones(1, 1, self.base_height, self.base_width)
        center = (self.base_height // 2, self.base_width // 2)
        self.weights[:, :, center[0] - self.N // 2:center[0] + self.N // 2,
        center[1] - self.N // 2:center[1] + self.N // 2] = self.center_weight

    def forward(self, x):
        # 检查当前特征图的维度与权重维度是否匹配
        if x.size(2) != self.base_height or x.size(3) != self.base_width:
            # 调整权重以匹配输入特征图的尺寸
            current_weights = F.interpolate(self.weights, size=(x.size(2), x.size(3)), mode='bilinear',
                                            align_corners=False)
        else:
            current_weights = self.weights
        return x * current_weights.to(x.device)


class GAP(nn.Module):
    def __init__(self):
        super(GAP, self).__init__()

    def forward(self, x):
        # x shape: (B, T, C_hid, H, W)
        x = x.mean(dim=[3, 4])  # Global Average Pooling over spatial dimensions # 1 5 16
        return x


class CustomDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomDecoder, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x shape: (B, T, C_hid, H, W)
        x = self.fc(x)
        return x


class BCModel(nn.Module):

    def __init__(self, in_shape1, in_shape2, hid_S=16, hid_T=256, N_S=4, N_T=4, output_dim=3,
                 model_type='incepu', mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 act_inplace=True, center_weight=1.0, N=10, **kwargs):
        super(BCModel, self).__init__()
        T, C1, H, W = in_shape1
        _, C2 = in_shape2
        self.encoder1 = Encoder(C1, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.spatial_attention_weight = SpatialAttentionWeight(height=H, width=W, N=N, center_weight=center_weight)
        self.mid_layer = MidIncepNet(T * hid_S, hid_T, N_T)  # or use MidMetaNet
        self.gap = GAP()
        self.value_encoder = TimeDistributedValueEncoder(C2, C2)  # Same output dimension as input
        self.decoder = CustomDecoder(hid_S + C2, output_dim)
        if model_type == 'incepu':
            self.hid = MidIncepNet(T * hid_S, hid_T, N_T)
        else:
            self.hid = MidMetaNet(T * hid_S, hid_T, N_T,
                                  input_resolution=(H, W), model_type=model_type,
                                  mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)

    def forward(self, x1, x2, **kwargs):
        B, T, C1, H, W = x1.shape
        _, _, C2 = x2.shape

        embed1, skip1 = self.encoder1(x1.view(B * T, C1, H, W))
        embed2 = self.value_encoder(x2)
        features = self.spatial_attention_weight(embed1)
        _, C_, H_, W_ = features.shape
        features = features.view(B, T, C_, H_, W_)
        hid = self.mid_layer(features)
        sp_feature = self.gap(hid)
        combined = torch.cat([sp_feature, embed2], dim=-1)  # Concatenate along the feature dimension
        output = self.decoder(combined)
        corrected_output = output + x2[:, :, -2:]

        return corrected_output


def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse:
        return list(reversed(samplings[:N]))
    else:
        return samplings[:N]


class TimeDistributedValueEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TimeDistributedValueEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        # x shape: (B, T, C2)
        output, (hn, cn) = self.lstm(x)
        return output  # (B, T, hidden_dim)


class Encoder(nn.Module):

    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0],
                   act_inplace=act_inplace),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s,
                     act_inplace=act_inplace) for s in samplings[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s,
                     act_inplace=act_inplace) for s in samplings[:-1]],
            ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1],
                   act_inplace=act_inplace)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid + enc1)
        Y = self.readout(Y)
        return Y


class MidIncepNet(nn.Module):

    def __init__(self, channel_in, channel_hid, N2, incep_ker=[3, 5, 7, 11], groups=8, **kwargs):
        super(MidIncepNet, self).__init__()
        assert N2 >= 2 and len(incep_ker) > 1
        self.N2 = N2
        enc_layers = [gInception_ST(
            channel_in, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for i in range(1, N2 - 1):
            enc_layers.append(
                gInception_ST(channel_hid, channel_hid // 2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        enc_layers.append(
            gInception_ST(channel_hid, channel_hid // 2, channel_hid,
                          incep_ker=incep_ker, groups=groups))
        dec_layers = [
            gInception_ST(channel_hid, channel_hid // 2, channel_hid,
                          incep_ker=incep_ker, groups=groups)]
        for i in range(1, N2 - 1):
            dec_layers.append(
                gInception_ST(2 * channel_hid, channel_hid // 2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        dec_layers.append(
            gInception_ST(2 * channel_hid, channel_hid // 2, channel_in,
                          incep_ker=incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W) # 1 5*(16 + 8) 181 360

        # encoder
        skips = []
        z = x # 1 120 181 360
        for i in range(self.N2): # N2=4
            z = self.enc[i](z) # 1 256 181 360
            if i < self.N2 - 1:
                skips.append(z)
        # decoder
        z = self.dec[0](z) # 1 256 181 360
        for i in range(1, self.N2):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = z.reshape(B, T, C, H, W) # 1 5 24 181 360
        return y


class MetaBlock(nn.Module):
    def __init__(self, in_channels, out_channels, input_resolution=None, model_type=None,
                 mlp_ratio=8., drop=0.0, drop_path=0.0, layer_i=0):
        super(MetaBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        model_type = model_type.lower() if model_type is not None else 'gsta'

        if model_type == 'gsta':
            self.block = GASubBlock(
                in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        elif model_type == 'convmixer':
            self.block = ConvMixerSubBlock(in_channels, kernel_size=11, activation=nn.GELU)
        elif model_type == 'convnext':
            self.block = ConvNeXtSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'hornet':
            self.block = HorNetSubBlock(in_channels, mlp_ratio=mlp_ratio, drop_path=drop_path)
        elif model_type in ['mlp', 'mlpmixer']:
            self.block = MLPMixerSubBlock(
                in_channels, input_resolution, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type in ['moga', 'moganet']:
            self.block = MogaSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop_rate=drop, drop_path_rate=drop_path)
        elif model_type == 'poolformer':
            self.block = PoolFormerSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'swin':
            self.block = SwinSubBlock(
                in_channels, input_resolution, layer_i=layer_i, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path)
        elif model_type == 'uniformer':
            block_type = 'MHSA' if in_channels == out_channels and layer_i > 0 else 'Conv'
            self.block = UniformerSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop,
                drop_path=drop_path, block_type=block_type)
        elif model_type == 'van':
            self.block = VANSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        elif model_type == 'vit':
            self.block = ViTSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'tau':
            self.block = TAUSubBlock(
                in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        else:
            assert False and "Invalid model_type"

        if in_channels != out_channels:
            self.reduction = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        z = self.block(x)
        return z if self.in_channels == self.out_channels else self.reduction(z)


class MidMetaNet(nn.Module):
    def __init__(self, channel_in, channel_hid, N2,
                 input_resolution=None, model_type=None,
                 mlp_ratio=4., drop=0.0, drop_path=0.1):
        super(MidMetaNet, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2
        dpr = [  # stochastic depth decay rule
            x.item() for x in torch.linspace(1e-2, drop_path, self.N2)]

        # downsample
        enc_layers = [MetaBlock(
            channel_in, channel_hid, input_resolution, model_type,
            mlp_ratio, drop, drop_path=dpr[0], layer_i=0)]
        # middle layers
        for i in range(1, N2 - 1):
            enc_layers.append(MetaBlock(
                channel_hid, channel_hid, input_resolution, model_type,
                mlp_ratio, drop, drop_path=dpr[i], layer_i=i))
        # upsample
        enc_layers.append(MetaBlock(
            channel_hid, channel_in, input_resolution, model_type,
            mlp_ratio, drop, drop_path=drop_path, layer_i=N2 - 1))
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)

        z = x
        for i in range(self.N2):
            z = self.enc[i](z)

        y = z.reshape(B, T, C, H, W)
        return y
