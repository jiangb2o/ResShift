#!/usr/bin/env python
# -*- coding:utf-8 -*-


import argparse
from pathlib import Path
from loguru import logger
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List
import torchmetrics
from torchmetrics.multimodal.clip_iqa import CLIPImageQualityAssessment
import sys
from utils import util_image


def find_images(dir_path: Path) -> List[Path]:
    imgs = sorted([p for p in dir_path.glob('**/*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    return imgs



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sr_dir', type=str, required=True, help='SR / inference results folder')
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()

    sr_dir = Path(args.sr_dir)
    assert sr_dir.exists(), 'sr_dir not found'

    sr_files = find_images(sr_dir)
    assert len(sr_files) > 0, 'No images found in sr_dir'
    logger.info(f'Found {len(sr_files)} images in {sr_dir}')

    clip = []
    clipiqa_metric = CLIPImageQualityAssessment()
    for p in sr_files:
        print(p)
        im_sr = util_image.imread(str(p), chn='rgb', dtype='float32')
        clip.append(clipiqa_metric(im_sr))



    # Summary
    print('\n==== RESULTS SUMMARY ====>')
    print(f'Images evaluated: {len(sr_files)}')
    if args.clip:
        if len(clip) > 0:
            print(f'CLIP (cosine sim) : mean={np.nanmean(clip):.6f}  std={np.nanstd(clip):.6f}')
        else:
            print('CLIP: skipped (no pairs or model)')


if __name__ == '__main__':
    main()
