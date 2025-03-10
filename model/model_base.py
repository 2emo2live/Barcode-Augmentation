import torch
import torch.nn.functional as F


class Conv_bn_block(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._conv = torch.nn.Conv2d(*args, **kwargs)
        self._bn = torch.nn.BatchNorm2d(kwargs['out_channels'])

    def forward(self, input):
        return F.relu(self._bn(self._conv(input)))


class Res_block(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self._conv1 = torch.nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1)
        self._conv2 = torch.nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=1, padding=1)
        self._conv3 = torch.nn.Conv2d(in_channels // 4, in_channels, kernel_size=1, stride=1)
        self._bn = torch.nn.BatchNorm2d(in_channels)

    def forward(self, x):
        xin = x
        x = F.relu(self._conv1(x))
        x = F.relu(self._conv2(x))
        x = self._conv3(x)
        x = torch.add(xin, x)
        x = F.relu(self._bn(x))

        return x


class encoder_net(torch.nn.Module):
    def __init__(self, in_channels, get_feature_map=False):
        super().__init__()
        self.cnum = 32
        self.get_feature_map = get_feature_map
        self._conv1_1 = Conv_bn_block(
            in_channels=in_channels,
            out_channels=self.cnum,
            kernel_size=3,
            stride=1,
            padding=1)
        self._conv1_2 = Conv_bn_block(
            in_channels=self.cnum,
            out_channels=self.cnum,
            kernel_size=3,
            stride=1,
            padding=1)

        # --------------------------
        self._pool1 = torch.nn.Conv2d(
            in_channels=self.cnum,
            out_channels=2 * self.cnum,
            kernel_size=3,
            stride=2,
            padding=1)
        self._conv2_1 = Conv_bn_block(
            in_channels=2 * self.cnum,
            out_channels=2 * self.cnum,
            kernel_size=3,
            stride=1,
            padding=1)
        self._conv2_2 = Conv_bn_block(
            in_channels=2 * self.cnum,
            out_channels=2 * self.cnum,
            kernel_size=3,
            stride=1,
            padding=1)

        # ---------------------------
        self._pool2 = torch.nn.Conv2d(
            in_channels=2 * self.cnum,
            out_channels=4 * self.cnum,
            kernel_size=3,
            stride=2,
            padding=1)
        self._conv3_1 = Conv_bn_block(
            in_channels=4 * self.cnum,
            out_channels=4 * self.cnum,
            kernel_size=3,
            stride=1,
            padding=1)

        self._conv3_2 = Conv_bn_block(
            in_channels=4 * self.cnum,
            out_channels=4 * self.cnum,
            kernel_size=3,
            stride=1,
            padding=1)

        # ---------------------------
        self._pool3 = torch.nn.Conv2d(
            in_channels=4 * self.cnum,
            out_channels=8 * self.cnum,
            kernel_size=3,
            stride=2,
            padding=1)
        self._conv4_1 = Conv_bn_block(
            in_channels=8 * self.cnum,
            out_channels=8 * self.cnum,
            kernel_size=3,
            stride=1,
            padding=1)
        self._conv4_2 = Conv_bn_block(
            in_channels=8 * self.cnum,
            out_channels=8 * self.cnum,
            kernel_size=3,
            stride=1,
            padding=1)

    def forward(self, x):
        x = self._conv1_1(x)
        x = self._conv1_2(x)
        x = F.relu(self._pool1(x))
        x = self._conv2_1(x)
        x = self._conv2_2(x)
        f1 = x
        x = F.relu(self._pool2(x))
        x = self._conv3_1(x)
        x = self._conv3_2(x)
        f2 = x
        x = F.relu(self._pool3(x))
        x = self._conv4_1(x)
        x = self._conv4_2(x)
        if self.get_feature_map:
            return x, [f2, f1]
        else:
            return x


class build_res_block(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self._block1 = Res_block(in_channels)
        self._block2 = Res_block(in_channels)
        self._block3 = Res_block(in_channels)
        self._block4 = Res_block(in_channels)

    def forward(self, x):
        x = self._block1(x)
        x = self._block2(x)
        x = self._block3(x)
        x = self._block4(x)
        return x


class decoder_net(torch.nn.Module):
    def __init__(self, in_channels, get_feature_map=False, mt=1, fn_mt=[1, 1, 1]):
        super().__init__()
        if isinstance(fn_mt, int):
            fn_mt = [fn_mt for _ in range(3)]
        assert isinstance(fn_mt, list) and len(fn_mt) == 3

        self.cnum = 32
        self.get_feature_map = get_feature_map
        self._conv1_1 = Conv_bn_block(in_channels=int(fn_mt[0] * in_channels), out_channels=8 * self.cnum,
                                      kernel_size=3, stride=1, padding=1)
        self._conv1_2 = Conv_bn_block(in_channels=8 * self.cnum, out_channels=8 * self.cnum, kernel_size=3, stride=1,
                                      padding=1)

        # -----------------
        self._deconv1 = torch.nn.ConvTranspose2d(8 * self.cnum, 4 * self.cnum, kernel_size=3, stride=2, padding=1,
                                                 output_padding=1)
        self._conv2_1 = Conv_bn_block(in_channels=int(fn_mt[1] * mt * 4 * self.cnum), out_channels=4 * self.cnum,
                                      kernel_size=3, stride=1, padding=1)
        self._conv2_2 = Conv_bn_block(in_channels=4 * self.cnum, out_channels=4 * self.cnum, kernel_size=3, stride=1,
                                      padding=1)

        # -----------------
        self._deconv2 = torch.nn.ConvTranspose2d(4 * self.cnum, 2 * self.cnum, kernel_size=3, stride=2, padding=1,
                                                 output_padding=1)
        self._conv3_1 = Conv_bn_block(in_channels=int(fn_mt[2] * mt * 2 * self.cnum), out_channels=2 * self.cnum,
                                      kernel_size=3, stride=1, padding=1)
        self._conv3_2 = Conv_bn_block(in_channels=2 * self.cnum, out_channels=2 * self.cnum, kernel_size=3, stride=1,
                                      padding=1)

        # ----------------
        self._deconv3 = torch.nn.ConvTranspose2d(2 * self.cnum, self.cnum, kernel_size=3, stride=2, padding=1,
                                                 output_padding=1)
        self._conv4_1 = Conv_bn_block(in_channels=self.cnum, out_channels=self.cnum, kernel_size=3, stride=1, padding=1)
        self._conv4_2 = Conv_bn_block(in_channels=self.cnum, out_channels=self.cnum, kernel_size=3, stride=1, padding=1)

    def forward(self, x, fuse=None, detach_flag=False):
        if fuse and fuse[0] is not None:
            if detach_flag:
                x = torch.cat((x, fuse[0].detach()), dim=1)
            else:
                x = torch.cat((x, fuse[0]), dim=1)
        x = self._conv1_1(x)
        x = self._conv1_2(x)
        f1 = x
        x = F.relu(self._deconv1(x))
        if fuse and fuse[1] is not None:
            if detach_flag:
                x = torch.cat((x, fuse[1].detach()), dim=1)
            else:
                x = torch.cat((x, fuse[1]), dim=1)
        x = self._conv2_1(x)
        x = self._conv2_2(x)
        f2 = x
        x = F.relu(self._deconv2(x))
        if fuse and fuse[2] is not None:
            if detach_flag:
                x = torch.cat((x, fuse[2].detach()), dim=1)
            else:
                x = torch.cat((x, fuse[2]), dim=1)
        x = self._conv3_1(x)
        x = self._conv3_2(x)
        f3 = x
        x = F.relu(self._deconv3(x))
        x = self._conv4_1(x)
        x = self._conv4_2(x)
        if self.get_feature_map:
            return x, [f1, f2, f3]
        else:
            return x


class PSPModule(torch.nn.Module):
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6), norm_layer=torch.nn.BatchNorm2d):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = torch.nn.ModuleList(
            [self._make_stage(features, out_features, size, norm_layer) for size in sizes])
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=1, padding=0, dilation=1,
                            bias=False),
            norm_layer(out_features),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(0.1)
        )

    def _make_stage(self, features, out_features, size, norm_layer):
        prior = torch.nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = torch.nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = norm_layer(out_features)
        return torch.nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in
                  self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle
