# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from ._utils import _C
from maskrcnn_benchmark import _C
import logging
import math
import torch
from qd.qd_common import calculate_iou_xywh
import numpy as np
import torch.nn as nn

from apex import amp

# Only valid with fp32 inputs - give AMP the hint
nms = amp.float_function(_C.nms)
hnms = _C.hnms

# nms.__doc__ = """
# This function performs Non-maximum suppresion"""

class MultiHashNMSAnyKPt(nn.ModuleList):
    def __init__(self, num, alpha):
        all_hash_rect = []
        for i in range(num):
            curr_w0 = math.exp(1. * i / num * (-math.log(alpha)))
            curr_h0 = math.exp(1. * i / num * (-math.log(alpha)))
            bx = 1. * i / num
            by = 1. * i / num

            hr = SingleHashNMSKPtC(curr_w0, curr_h0, alpha,
                    bx, by)
            all_hash_rect.append(hr)
        super(MultiHashNMSAnyKPt, self).__init__(all_hash_rect)

    def forward(self, rects, conf):
        for i, hr in enumerate(self):
            if i == 0:
                curr_keep = hr(rects, conf)
                keep = curr_keep
            else:
                curr_keep = hr(rects[keep], conf[keep])
                keep = keep[curr_keep]
        return keep

class SingleHashNMSKPtC(nn.Module):
    def __init__(self, w0, h0, alpha, bx=0.5, by=0.5):
        super(SingleHashNMSKPtC, self).__init__()
        self.w0 = float(w0)
        self.h0 = float(h0)
        self.alpha = alpha
        self.bx = bx
        self.by = by

    def __call__(self, rects, conf):
        result = hnms(rects, conf,
                self.w0, self.h0,
                self.alpha,
                self.bx, self.by)
        return result
