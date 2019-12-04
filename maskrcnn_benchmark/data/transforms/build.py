# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T
from . import ssd_transforms as S

# modify by kevin.cao at 19.11.28
def build_transforms(cfg, is_train=True):
    if cfg.INPUT.USE_SSD_DATA_AUGMENT:
        transform = build_ssd_transforms(cfg, is_train)
    else:
        transform = build_mask_transforms(cfg, is_train)

    return transform
    
# modify by kevin.cao at 19.11.28
# maskrcnn_benchmark augmentations
def build_mask_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_horizontal_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
        flip_vertical_prob = cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN
        brightness = cfg.INPUT.BRIGHTNESS
        contrast = cfg.INPUT.CONTRAST
        saturation = cfg.INPUT.SATURATION
        hue = cfg.INPUT.HUE
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_horizontal_prob = 0.0
        flip_vertical_prob = 0.0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )
    color_jitter = T.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )

    transform = T.Compose(
        [
            color_jitter,
            T.Resize(min_size, max_size),
            T.RandomHorizontalFlip(flip_horizontal_prob),
            T.RandomVerticalFlip(flip_vertical_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform
  
# add by kevin.cao at 19.11.28
# ssd_augmentations
def build_ssd_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_horizontal_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
        flip_vertical_prob = cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN
        brightness = cfg.INPUT.BRIGHTNESS
        contrast = cfg.INPUT.CONTRAST
        saturation = cfg.INPUT.SATURATION
        hue = cfg.INPUT.HUE
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_horizontal_prob = 0.0
        flip_vertical_prob = 0.0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )
    color_jitter = T.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )

    if is_train:
        transform = T.Compose(
            [
                color_jitter,
                T.RandomHorizontalFlip(flip_horizontal_prob),
                T.RandomVerticalFlip(flip_vertical_prob),
                S.ConvertFromPIL(),
                S.PhotometricDistort(),
                S.Expand(cfg.INPUT.PIXEL_MEAN),
                S.RandomSampleCrop(cfg.INPUT.USE_SSD_RANDOM_SAMPLE_CROP),
                S.ConvertFromNP(),
                T.Resize(min_size, max_size),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    else:
        transform = T.Compose(
            [
                color_jitter,
                T.RandomHorizontalFlip(flip_horizontal_prob),
                T.RandomVerticalFlip(flip_vertical_prob),
                T.Resize(min_size, max_size),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    return transform

