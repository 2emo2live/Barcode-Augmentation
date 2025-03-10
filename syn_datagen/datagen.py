"""
Generating data for SRNet
Copyright (c) 2019 Netease Youdao Information Technology Co.,Ltd.
Licensed under the GPL License (see LICENSE for details)
Written by Yu Qian
"""

import os
import cv2
import json
import cfg
from Synthtext.gen import datagen, multiprocess_datagen


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    i_s_dir = os.path.join(cfg.data_dir, cfg.i_s_dir)
    t_t_dir = os.path.join(cfg.data_dir, cfg.t_t_dir)
    t_b_dir = os.path.join(cfg.data_dir, cfg.t_b_dir)
    t_f_dir = os.path.join(cfg.data_dir, cfg.t_f_dir)
    mask_1_dir = os.path.join(cfg.data_dir, cfg.mask_1_dir)
    mask_2_dir = os.path.join(cfg.data_dir, cfg.mask_2_dir)
    borders_dir = os.path.join(cfg.data_dir, cfg.borders_dir)
    i_t_dir = os.path.join(cfg.data_dir, cfg.i_t_dir)

    makedirs(i_s_dir)
    makedirs(t_t_dir)
    makedirs(t_b_dir)
    makedirs(t_f_dir)
    makedirs(mask_1_dir)
    makedirs(mask_2_dir)
    makedirs(i_t_dir)

    mp_gen = multiprocess_datagen(cfg.process_num, cfg.data_capacity)
    mp_gen.multiprocess_runningqueue()
    digit_num = len(str(cfg.sample_num)) - 1
    border_dict = {}
    for idx in range(cfg.sample_num):
        print("Generating step {:>6d} / {:>6d}".format(idx + 1, cfg.sample_num))
        i_s, t_t, t_b, t_f, mask_1, mask_2, border, i_t = mp_gen.dequeue_data()
        border_dict[str(idx).zfill(digit_num)] = border.tolist()
        i_s_path = os.path.join(i_s_dir, str(idx).zfill(digit_num) + '.png')
        t_t_path = os.path.join(t_t_dir, str(idx).zfill(digit_num) + '.png')
        t_b_path = os.path.join(t_b_dir, str(idx).zfill(digit_num) + '.png')
        t_f_path = os.path.join(t_f_dir, str(idx).zfill(digit_num) + '.png')
        mask_1_path = os.path.join(cfg.data_dir, cfg.mask_1_dir, str(idx).zfill(digit_num) + '.png')
        mask_2_path = os.path.join(cfg.data_dir, cfg.mask_2_dir, str(idx).zfill(digit_num) + '.png')
        i_t_path = os.path.join(i_t_dir, str(idx).zfill(digit_num) + '.png')
        cv2.imwrite(i_s_path, i_s, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(t_t_path, t_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(t_b_path, t_b, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(t_f_path, t_f, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(mask_1_path, mask_1, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(mask_2_path, mask_2, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(i_t_path, i_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

    borders_path = os.path.join(cfg.data_dir, cfg.borders_dir + '.json')
    with open(borders_path, 'w') as f:
        json.dump(border_dict, f)

    mp_gen.terminate_pool()


if __name__ == '__main__':
    main()
