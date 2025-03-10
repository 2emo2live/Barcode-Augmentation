import os
import re
import itertools
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data.sampler import Sampler
from PIL import Image


# import standard_text

def linearize(img):
    image_float = np.array(img).astype(np.float32) / 255.0
    image_float_small = image_float <= 0.04045
    image_float_small = image_float * image_float_small.astype(int)
    image_float_small /= 12.92
    image_float_big = image_float > 0.04045
    image_float_big = image_float * image_float_big.astype(int)
    image_float_big += 0.055
    image_float_big /= 1.055
    image_float_big = np.power(image_float_big, 2.4)
    linear_image = image_float_big + image_float_small

    linear_image_8bit = np.uint8(linear_image * 255)
    return Image.fromarray(linear_image_8bit)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size), grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    # Collect data into fixed-length chunks or blocks
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


class custom_dataset(Dataset):
    def __init__(self, cfg, data_dir=None, mode='train', with_real_data=False):
        self.cfg = cfg
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.Resize(cfg.data_shape),
            transforms.ToTensor(),
        ])

        if self.mode == 'train':
            self.data_dir = cfg.data_dir
            if isinstance(self.data_dir, str):
                self.data_dir = [self.data_dir]
            assert isinstance(self.data_dir, list)

            self.name_list = []
            for tmp_data_dir in self.data_dir:
                self.name_list += [os.path.join(tmp_data_dir, '{}', filename) for filename in
                                   os.listdir(os.path.join(tmp_data_dir, cfg.i_s_dir))]
            self.len_synth = len(self.name_list)
            self.len_real = 0

            if with_real_data:
                self.real_data_dir = cfg.real_data_dir
                if isinstance(self.real_data_dir, str):
                    self.real_data_dir = [self.real_data_dir]
                assert isinstance(self.real_data_dir, list)

                self.real_name_list = []
                for tmp_data_dir in self.real_data_dir:
                    self.real_name_list += [os.path.join(tmp_data_dir, filename) for filename in
                                            os.listdir(os.path.join(tmp_data_dir, cfg.i_s_dir))]

                self.name_list += self.real_name_list
                self.len_real = len(self.real_name_list)
        else:
            assert data_dir is not None
            self.data_dir = data_dir
            self.name_list = []
            # TODO: check
            self.name_list += [os.path.join(self.data_dir, '{}', filename) for filename in
                               os.listdir(os.path.join(self.data_dir, cfg.i_s_dir))]

    def custom_len(self):
        return self.len_synth, self.len_real

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        img_name = self.name_list[idx]
        if self.mode == 'train':
            if idx < self.len_synth:
                i_s = Image.open(img_name.format(self.cfg.i_s_dir))
                i_t = Image.open(img_name.format(self.cfg.i_t_dir))
                if i_s.mode != 'RGB':
                    i_s = i_s.convert('RGB')
                t_b = Image.open(img_name.format(self.cfg.t_b_dir))
                t_f = Image.open(img_name.format(self.cfg.t_f_dir))
                mask_t = Image.open(img_name.format(self.cfg.mask_t_dir))
                mask_s = Image.open(img_name.format(self.cfg.mask_s_dir))
                i_t = self.transform(i_t)
                i_s = self.transform(i_s)
                t_b = self.transform(t_b)
                t_f = self.transform(t_f)
                mask_t = self.transform(mask_t)
                mask_s = self.transform(mask_s)
            else:
                i_t = Image.open(img_name.format(self.cfg.i_t_dir))
                i_s = Image.open(img_name.format(self.cfg.i_s_dir))
                if i_s.mode != 'RGB':
                    i_s = i_s.convert('RGB')
                i_t = self.transform(i_t)
                i_s = self.transform(i_s)
                t_f = i_s
                t_b = -1 * torch.ones([3] + self.cfg.data_shape)
                mask_t = -1 * torch.ones([1] + self.cfg.data_shape)
                mask_s = -1 * torch.ones([1] + self.cfg.data_shape)

            return [i_t, i_s, t_b, t_f, mask_t, mask_s]
        else:
            main_name = img_name
            i_s = Image.open(img_name.format(self.cfg.i_s_dir))
            if i_s.mode != 'RGB':
                i_s = i_s.convert('RGB')
            i_t = Image.open(img_name.format(self.cfg.i_t_dir))
            i_t = Image.fromarray(np.uint8(i_t))
            i_s = self.transform(i_s)
            i_t = self.transform(i_t)

            return [i_t, i_s, main_name]


class erase_dataset(Dataset):
    def __init__(self, cfg, data_dir=None, mode='train'):
        self.cfg = cfg
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.Resize(cfg.data_shape),
            transforms.ToTensor()
        ])
        if self.mode == 'train':
            self.data_dir = cfg.data_dir
            if isinstance(self.data_dir, str):
                self.data_dir = [self.data_dir]
            assert isinstance(self.data_dir, list)
            self.name_list = []
            for tmp_data_dir in self.data_dir:
                self.name_list += [os.path.join(tmp_data_dir, '{}', filename) for filename in
                                   os.listdir(os.path.join(tmp_data_dir, cfg.i_s_dir))]
                # self.name_list += [(os.path.join(tmp_data_dir, f"{dirname}",filename) for dirname in ['i_s', ]) for filename in
                #                   os.listdir(os.path.join(tmp_data_dir, cfg.i_s_dir))]

        else:
            assert data_dir is not None
            self.data_dir = data_dir
            self.name_list = []
            self.name_list += [os.path.join(self.data_dir, '{}', filename) for filename in
                               os.listdir(os.path.join(self.data_dir, cfg.i_s_dir))]

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        img_name = self.name_list[idx]
        if self.mode == 'train':
            i_s = Image.open(img_name.format(self.cfg.i_s_dir))
            t_b = Image.open(img_name.format(self.cfg.t_b_dir))
            mask_s = Image.open(img_name.format(self.cfg.mask_s_dir))
            #i_s = linearize(i_s)
            #t_b = linearize(t_b)
            #mask_s = linearize(mask_s)
            i_s = self.transform(i_s)
            t_b = self.transform(t_b)
            mask_s = self.transform(mask_s)

            return [i_s, t_b, mask_s]
        else:
            main_name = img_name
            i_s = Image.open(img_name.format(self.cfg.i_s_dir))
            if i_s.mode != 'RGB':
                i_s = i_s.convert('RGB')
            #i_s = linearize(i_s)
            i_s = self.transform(i_s)

            return [i_s, main_name]
