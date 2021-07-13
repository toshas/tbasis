from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models.resnet as resnet
from packaging import version

assert version.parse(torch.__version__) >= version.parse('1.2')
assert version.parse(torchvision.__version__) >= version.parse('0.4.0')


class BasicBlockWithDilation(nn.Module):
    """Workaround for prohibited dilation in BasicBlock in 0.4.0"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlockWithDilation, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        self.conv1 = resnet.conv3x3(inplanes, planes, stride=stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = resnet.conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


_basic_block_layers = {
    'resnet18': (2, 2, 2, 2),
    'resnet34': (3, 4, 6, 3),
}


class ASPPpart_1(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
        )


class ASPPpart_2(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super().__init__(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class ASPPpart(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, output_stride=16, rates=(6, 12, 18)):
        super().__init__()
        assert output_stride in (8, 16), 'Invalid output stride'
        if output_stride == 8:
            rates = (2 * a for a in rates)
        self.conv0 = ASPPpart(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv1 = ASPPpart(in_channels, out_channels, kernel_size=3, stride=1, padding=rates[0], dilation=rates[0])
        self.conv2 = ASPPpart(in_channels, out_channels, kernel_size=3, stride=1, padding=rates[1], dilation=rates[1])
        self.conv3 = ASPPpart(in_channels, out_channels, kernel_size=3, stride=1, padding=rates[2], dilation=rates[2])
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            ASPPpart(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)
        )
        self.conv_out_1 = ASPPpart_1(
            out_channels * (len(rates) + 2), out_channels, kernel_size=1, stride=1, padding=0, dilation=1
        )
        self.conv_out_2 = ASPPpart_2(
            out_channels * (len(rates) + 2), out_channels, kernel_size=1, stride=1, padding=0, dilation=1
        )

    def forward(self, x):
        out = (
            self.conv0(x),
            self.conv1(x),
            self.conv2(x),
            self.conv3(x),
            F.interpolate(self.pool(x), x.shape[2:], mode='bilinear', align_corners=False)
        )
        out = torch.cat(out, dim=1)
        out_1 = self.conv_out_1(out)
        out_2 = self.conv_out_2(out_1)
        return out_1, out_2


class Encoder(nn.Module):
    def __init__(self, name, **encoder_kwargs):
        super().__init__()
        encoder = self._create(name, **encoder_kwargs)
        del encoder.avgpool
        del encoder.fc
        self.encoder = encoder

    def _create(self, name, **encoder_kwargs):
        if name not in _basic_block_layers.keys():
            fn_name = getattr(resnet, name)
            model = fn_name(**encoder_kwargs)
        else:
            # special case due to prohibited dilation in the original BasicBlock
            pretrained = encoder_kwargs.pop('pretrained', False)
            progress = encoder_kwargs.pop('progress', True)
            model = resnet._resnet(
                name, BasicBlockWithDilation, _basic_block_layers[name], pretrained, progress, **encoder_kwargs
            )
        replace_stride_with_dilation = encoder_kwargs.get('replace_stride_with_dilation', (False, False, False))
        assert len(replace_stride_with_dilation) == 3
        if replace_stride_with_dilation[0]:
            model.layer2[0].conv2.padding = (2, 2)
            model.layer2[0].conv2.dilation = (2, 2)
        if replace_stride_with_dilation[1]:
            model.layer3[0].conv2.padding = (2, 2)
            model.layer3[0].conv2.dilation = (2, 2)
        if replace_stride_with_dilation[2]:
            model.layer4[0].conv2.padding = (2, 2)
            model.layer4[0].conv2.dilation = (2, 2)
        return model

    def forward(self, x):
        features = OrderedDict()
        features['in'] = x

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        features['enc_preamble'] = x

        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        features['enc_l1'] = x

        x = self.encoder.layer2(x)
        features['enc_l2'] = x

        x = self.encoder.layer3(x)
        features['enc_l3'] = x

        x = self.encoder.layer4(x)
        features['enc_l4'] = x

        return features


class DecoderDeeplabV3p(nn.Module):
    def __init__(self, num_classes, bottleneck_ch, skip_4x_ch):
        super(DecoderDeeplabV3p, self).__init__()

        self.reduce_skip_4x = nn.Sequential(
            nn.Conv2d(skip_4x_ch, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.last_conv = nn.Sequential(
            nn.Conv2d(bottleneck_ch + 48, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        )

    def forward(self, features):
        reduced_skip_4x = self.reduce_skip_4x(features['enc_l1'])
        aspp_4x = F.interpolate(features['aspp'], size=reduced_skip_4x.shape[2:], mode='bilinear', align_corners=False)
        x_low = torch.cat((aspp_4x, reduced_skip_4x), dim=1)
        y_hat_low = self.last_conv(x_low)
        y_hat_high = F.interpolate(y_hat_low, size=features['in'].shape[2:], mode='bilinear', align_corners=False)
        features['y_hat_low'] = y_hat_low
        features['y_hat_high'] = y_hat_high
        return features


class ModelNetDeepLabV3Plus(torch.nn.Module):
    def __init__(self, cfg, c_out):
        super().__init__()

        self.stride = 16
        dilation = (False, False, True)

        self.encoder = Encoder(
            cfg.model_encoder_name,
            pretrained=cfg.model_encoder_pretrained,
            zero_init_residual=True,
            replace_stride_with_dilation=dilation,
        )

        is_basic_block = cfg.model_encoder_name in _basic_block_layers
        model_encoder_features_out = 512 if is_basic_block else 2048
        model_encoder_features_4x = 64 if is_basic_block else 256

        self.aspp = ASPP(model_encoder_features_out, 256, output_stride=self.stride)
        self.decoder = DecoderDeeplabV3p(c_out, 256, model_encoder_features_4x)

    def forward(self, x):
        features = self.encoder(x)
        low_features = features['enc_l4']
        low_features_1, low_features_2 = self.aspp(low_features)
        features['aspp_pre'] = low_features_1
        features['aspp'] = low_features_2
        features = self.decoder(features)
        return features

    @property
    def bottleneck_stride(self):
        return self.stride
