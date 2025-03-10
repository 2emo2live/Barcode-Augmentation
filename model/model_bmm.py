import torch

from model_base import encoder_net, build_res_block, decoder_net, PSPModule


class background_reconstruction_module(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cnum = 32
        self._encoder = encoder_net(in_channels, get_feature_map=True)
        self._res = build_res_block(8 * self.cnum)
        self._decoder = decoder_net(8 * self.cnum, get_feature_map=True, mt=2)
        self._out = torch.nn.Conv2d(self.cnum, 3, kernel_size=3, stride=1, padding=1)
        self._mask_s_decoder = decoder_net(8 * self.cnum)
        self._mask_s_out = torch.nn.Conv2d(self.cnum, 1, kernel_size=3, stride=1, padding=1)
        self.ppm = PSPModule(8 * self.cnum, out_features=8 * self.cnum)

    def forward(self, x):
        x, f_encoder = self._encoder(x)
        x = self._res(x)
        x = self.ppm(x)
        mask_s = self._mask_s_decoder(x, fuse=None)
        mask_s_out = torch.sigmoid(self._mask_s_out(mask_s))

        x, fs = self._decoder(x, fuse=[None] + f_encoder)
        x = torch.sigmoid(self._out(x))

        return x, fs, mask_s_out
