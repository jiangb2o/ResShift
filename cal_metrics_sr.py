#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Calculate PSNR / SSIM / LPIPS for SR outputs vs. GT images.
Usage example:
  python scripts/cal_metrics_sr.py --gt_dir ./ImageNet-Test/gt --sr_dir ./inference_result/imagenet_test --device cuda:0
"""
import argparse
from pathlib import Path
import sys

import torch
import lpips
import numpy as np
from loguru import logger
from utils import util_image
from PIL import Image


def load_im_tensor(im_path, device='cuda:0'):
    im = util_image.imread(str(im_path), chn='rgb', dtype='float32')
    im = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).to(device)
    im = (im - 0.5) / 0.5
    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str, required=True)
    parser.add_argument('--sr_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--ycbcr', action='store_true', help='Use Y channel PSNR/SSIM')
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    sr_dir = Path(args.sr_dir)
    assert gt_dir.exists() and sr_dir.exists(), 'gt_dir or sr_dir not found'

    logger.info(f'GT: {gt_dir}  SR: {sr_dir}  Device: {args.device}')

    lpips_vgg = lpips.LPIPS(net='vgg').to(args.device).eval()
    lpips_alex = lpips.LPIPS(net='alex').to(args.device).eval()

    files = sorted([p for p in sr_dir.glob('**/*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    assert len(files) > 0, 'No image files found in sr_dir'

    psnrs = []
    ssims = []
    lpips_v = []
    lpips_a = []

    for p in files:
        stem = p.stem
        gt_candidates = list(gt_dir.glob(f'**/{stem}.*'))
        if len(gt_candidates) == 0:
            logger.warning(f'GT not found for {p.name}, skip')
            continue
        gt_p = gt_candidates[0]

        # load for PSNR/SSIM ([0, 255] uint8)
        im_sr = util_image.imread(str(p), chn='rgb', dtype='uint8')
        im_gt = util_image.imread(str(gt_p), chn='rgb', dtype='uint8')

        # ensure same shape
        if im_sr.shape != im_gt.shape:
            logger.warning(f'Shape mismatch for {p.name}, resize SR to GT')
            import cv2
            im_sr = cv2.resize(im_sr, (im_gt.shape[1], im_gt.shape[0]), interpolation=cv2.INTER_CUBIC)

        psnr = util_image.calculate_psnr(im_sr, im_gt, border=0, ycbcr=args.ycbcr)
        ssim = util_image.calculate_ssim(im_sr, im_gt, border=0, ycbcr=args.ycbcr)
        psnrs.append(psnr)
        ssims.append(ssim)

        # LPIPS expects [-1,1] tensors
        try:
            im_sr_t = load_im_tensor(p, device=args.device)
            im_gt_t = load_im_tensor(gt_p, device=args.device)
            with torch.no_grad():
                lv = lpips_vgg(im_gt_t, im_sr_t).sum().item()
                la = lpips_alex(im_gt_t, im_sr_t).sum().item()
        except Exception as e:
            logger.warning(f'LPIPS fail for {p.name}: {e}')
            lv = np.nan
            la = np.nan
        lpips_v.append(lv)
        lpips_a.append(la)

    def mean_ignore_nan(x):
        x = np.array(x, dtype=np.float32)
        return float(np.nanmean(x))

    logger.info('RESULTS:')
    logger.info(f'  Images evaluated: {len(psnrs)}')
    logger.info(f'  PSNR: {np.mean(psnrs):.4f}  (std: {np.std(psnrs):.4f})')
    logger.info(f'  SSIM: {np.mean(ssims):.4f}  (std: {np.std(ssims):.4f})')
    logger.info(f'  LPIPS-VGG: {mean_ignore_nan(lpips_v):.4f}')
    logger.info(f'  LPIPS-AlexNet: {mean_ignore_nan(lpips_a):.4f}')


if __name__ == '__main__':
    main()
