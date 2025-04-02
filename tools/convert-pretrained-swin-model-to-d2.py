#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pickle as pkl
import sys

import torch

"""
Usage:
  # download pretrained swin model:
  wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
  # run the conversion
  ./convert-pretrained-model-to-d2.py swin_tiny_patch4_window7_224.pth swin_tiny_patch4_window7_224.pkl
  # Then, use swin_tiny_patch4_window7_224.pkl with the following changes in config:
MODEL:
  WEIGHTS: "/path/to/swin_tiny_patch4_window7_224.pkl"
INPUT:
  FORMAT: "RGB"
"""

if __name__ == "__main__":
    input = '/data/lwz/MABP/exp420/model_0144704.pth'

    obj = torch.load(input, map_location="cpu")["model"]
    rename_obj ={}
    print(obj.keys())
    for k,v in obj.items():
        if k.startswith('sem_seg_head.predictor.RIA_cross_attention'):
            rename_obj[k.replace('sem_seg_head.predictor.RIA_cross_attention','sem_seg_head.predictor.MMD_cross_attention')]=v
        elif k.startswith('sem_seg_head.predictor.self_attention_layers'):
            rename_obj[k.replace('sem_seg_head.predictor.self_attention_layers','sem_seg_head.predictor.MMD_self_attention')]=v 
        else:
            rename_obj[k]=v
    res = {"model": rename_obj}

    with open('/data/lwz/MABP-git/MABP_best.pth', "wb") as f:
        torch.save(res, f)
