import math
import random
import torch
import torchvision.transforms.functional as tf
import torch.nn.functional as F
from torchvision.models import vgg19
from PIL import Image

from model_bmm import background_reconstruction_module
from model_tmm import text_modification_module


def random_transform(cfg, i_s):
    i_s_aug = i_s
    vflip_rate = cfg.vflip_rate
    hflip_rate = cfg.hflip_rate
    angle_range = cfg.angle_range
    if random.random() < hflip_rate:
        i_s_aug = tf.hflip(i_s_aug)
    if random.random() < vflip_rate:
        i_s_aug = tf.vflip(i_s_aug)
    if len(angle_range) > 0:
        angle = random.randint(*random.choice(angle_range))
        i_s_aug = tf.rotate(i_s_aug, angle=angle, interpolation=Image.BILINEAR, expand=False)
    i_s_aug[:cfg.batch_size - cfg.real_bs] = i_s[:cfg.batch_size - cfg.real_bs]

    return i_s_aug


class Generator(torch.nn.Module):
    def __init__(self, cfg, in_channels, mode='full'):
        super().__init__()
        self.cfg = cfg
        self.cnum = 32
        self.brm = background_reconstruction_module(in_channels)
        self.mode = mode
        if mode == 'full':
            self.tmm = text_modification_module(cfg, in_channels, cfg.num_control_points, cfg.tps_margins,
                                                cfg.stn_activation)

    def forward(self, i_s, i_t=None):
        o_b, fuse, o_mask_s = self.brm(i_s)
        o_b_ori = o_b
        o_b = o_mask_s * o_b + (1 - o_mask_s) * i_s
        if self.mode == 'full':
            i_s_new = i_s * o_mask_s.detach()
            if self.training:
                i_s_new = random_transform(self.cfg, i_s_new)
            o_mask_t, o_f, x_t_tps = self.tmm(i_t, i_s_new, fuse=fuse)

            return o_b_ori, o_b, o_f, x_t_tps, o_mask_s, o_mask_t

        return o_b_ori, o_b, o_mask_s


class Discriminator(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        self.cfg = cfg
        self.cnum = 32
        self._conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1)
        self._conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self._conv3 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self._conv4 = torch.nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self._conv5 = torch.nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)
        self._conv2_bn = torch.nn.BatchNorm2d(128)
        self._conv3_bn = torch.nn.BatchNorm2d(256)
        self._conv4_bn = torch.nn.BatchNorm2d(512)
        self._conv5_bn = torch.nn.BatchNorm2d(1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()

    def forward(self, x):
        x = F.leaky_relu(self._conv1(x), 0.2)
        x = self._conv2(x)
        x = F.leaky_relu(self._conv2_bn(x), 0.2)
        x = self._conv3(x)
        x = F.leaky_relu(self._conv3_bn(x), 0.2)
        x = self._conv4(x)
        x = F.leaky_relu(self._conv4_bn(x), 0.2)
        x = self._conv5(x)
        x = self._conv5_bn(x)
        x = torch.sigmoid(x)

        return x


class Vgg19(torch.nn.Module):
    def __init__(self, vgg19_weights):
        super(Vgg19, self).__init__()
        # features = list(vgg19(pretrained = True).features)
        vgg = vgg19(pretrained=False)
        params = torch.load(vgg19_weights)
        vgg.load_state_dict(params)
        features = list(vgg.features)
        self.features = torch.nn.ModuleList(features).eval()

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)

            if ii in {1, 6, 11, 20, 29}:
                results.append(x)
        return results
