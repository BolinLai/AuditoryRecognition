#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""
from fvcore.common.config import CfgNode


def add_custom_config(_C):
    # Add your own customized configs.
    _C.MVIT = CfgNode()
    _C.MVIT.DEPTH = 4
    _C.MVIT.NUM_HEADS = 1
    _C.MVIT.EMBED_DIM = 96
    _C.MVIT.PATCH_KERNEL = (3, 7, 7)
    _C.MVIT.PATCH_STRIDE = (2, 4, 4)
    _C.MVIT.PATCH_PADDING = (1, 3, 3)
    _C.MVIT.MLP_RATIO = 4.0
    _C.MVIT.QKV_BIAS = True
    _C.MVIT.DROPPATH_RATE = 0.2
    _C.MVIT.CLS_EMBED_ON = True

    _C.AUDIO_DATA.INTERVAL = 4800  # the interval of two adjacent audio frames (4800 = 1/30 * 16000 * 9)
    _C.AUDIO_DATA.NUM_FRAME_CROPS = 8  # for 3d model

    _C.SOLVER.COSINE_AFTER_WARMUP = False

    pass
