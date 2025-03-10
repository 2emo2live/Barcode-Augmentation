import math
import torch
import numpy as np
import torch.nn.functional as F

from tps_spatial_transformer import TPSSpatialTransformer
from model_base import encoder_net, build_res_block, decoder_net, Conv_bn_block, PSPModule


class text_modification_module(torch.nn.Module):
    def __init__(self, cfg, in_channels, num_ctrlpoints, margins, stn_activation=None):
        super().__init__()
        self.cfg = cfg
        self.num_ctrlpoints = num_ctrlpoints
        self.margins = margins
        self.stn_activation = stn_activation
        self.cnum = 32
        self._t_encoder = encoder_net(in_channels)
        self._t_res = build_res_block(8*self.cnum)
        self._s_encoder = encoder_net(in_channels)
        self._s_res = build_res_block(8*self.cnum)
        self._mask_decoder = decoder_net(16*self.cnum, fn_mt=[1.5, 2, 2])
        self._mask_out = torch.nn.Conv2d(self.cnum, 1, kernel_size=3, stride=1, padding=1)
        self._t_decoder = decoder_net(16*self.cnum, fn_mt=[1.5, 2, 2])
        self._t_cbr = Conv_bn_block(in_channels=2*self.cnum, out_channels=2*self.cnum, kernel_size=3, stride=1, padding=1)
        self._t_out = torch.nn.Conv2d(2*self.cnum, 3, kernel_size=3, stride=1, padding=1)
        self.ppm = PSPModule(16*self.cnum, out_features=16*self.cnum)

        if cfg.TPS_ON:
            self.stn_fc1 = torch.nn.Sequential(
                torch.nn.Linear(16*16*256, 512),    # TODO: check 16 and 256
                torch.nn.BatchNorm1d(512),
                torch.nn.ReLU(inplace=True))
            self.stn_fc2 = torch.nn.Linear(512, num_ctrlpoints * 2)
            self.tps = TPSSpatialTransformer(output_image_size=cfg.tps_outputsize, num_control_points=num_ctrlpoints, margins=cfg.tps_margins)
            self.init_weights(self.stn_fc1)
            self.init_stn(self.stn_fc2, margins)

    def init_weights(self, module):
        for m in module.modules():
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

    def init_stn(self, stn_fc2, margins=(0.01, 0.01)):
        margin = margins[0]
        sampling_num_per_side = int(self.num_ctrlpoints / 2)
        ctrl_pts_x = np.linspace(margin, 1. - margin, sampling_num_per_side)
        ctrl_pts_y_top = np.ones(sampling_num_per_side) * margin
        ctrl_pts_y_bottom = np.ones(sampling_num_per_side) * (1 - margin)
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        ctrl_points = np.concatenate(
            [ctrl_pts_top, ctrl_pts_bottom], axis=0).astype(np.float32)
        if self.stn_activation is None:
            pass
        elif self.stn_activation == 'sigmoid':
            ctrl_points = -np.log(1. / ctrl_points - 1.)
        elif self.stn_activation == 'tanh':
            ctrl_points = ctrl_points * 2 - 1
            ctrl_points = np.log((1 + ctrl_points) / (1 - ctrl_points)) / 2
        stn_fc2.weight.data.zero_()
        stn_fc2.bias.data = torch.Tensor(ctrl_points).view(-1)

    def forward(self, x_t, x_s, fuse):
        x_s = self._s_encoder(x_s)
        x_s = self._s_res(x_s)
        x_t_tps = x_t
        if self.cfg.TPS_ON:
            batch_size, _, h, w = x_s.size()
            ctrl_points = x_s.reshape(batch_size, -1)
            ctrl_points = self.stn_fc1(ctrl_points)
            ctrl_points = self.stn_fc2(0.1 * ctrl_points)
            if self.stn_activation == 'sigmoid':
                ctrl_points = F.sigmoid(ctrl_points)
            elif self.stn_activation == 'tanh':
                ctrl_points = torch.tanh(ctrl_points)

            ctrl_points = ctrl_points.view(-1, self.num_ctrlpoints, 2)
            x_t, _ = self.tps(x_t, ctrl_points)
            x_t_tps = x_t
        x_t = self._t_encoder(x_t)
        x_t = self._t_res(x_t)
        x = torch.cat((x_t, x_s), dim=1)
        x = self.ppm(x)

        mask_t = self._mask_decoder(x, fuse=fuse, detach_flag=True)
        mask_t_out = torch.sigmoid(self._mask_out(mask_t))

        o_f = self._t_decoder(x, fuse=fuse, detach_flag=True)
        o_f = torch.cat((o_f, mask_t), dim=1)
        o_f = self._t_cbr(o_f)
        o_f_out = torch.sigmoid(self._t_out(o_f))

        return mask_t_out, o_f_out, x_t_tps
