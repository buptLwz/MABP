import time
import copy
import logging

import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from transformers import BertTokenizer
from pycocotools import mask as coco_mask

__all__ = ["RefCOCOMapperlmdb"]

'''We have replaced the cumbersome data processing flow in ReLA with an lmdb strategy similar to that in CGFormer.
ReLA https://github.com/henghuiding/ReLA
CGFormer https://github.com/SooLab/CGFormer'''

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def build_transform_train(cfg):
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE

    augmentation = []

    augmentation.extend([
        T.Resize((image_size, image_size))
    ])

    return augmentation


def build_transform_test(cfg):
    image_size = cfg.INPUT.IMAGE_SIZE

    augmentation = []

    augmentation.extend([
        T.Resize((image_size, image_size))
    ])

    return augmentation

import lmdb
import pyarrow as pa
import os
import cv2
def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)

class RefCOCOMapperlmdb:
    '''Same as CGFormer except for key names, image dimensions, dataset etc.
    Note: Key names are related to the lmdb creation process, refer to Dataset.md.
        '''
    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        image_format,
        bert_type,
        max_tokens,
        merge=True,
        lmdb_dir
    ):
        self.is_train = is_train
        self.merge = merge
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "Full TransformGens used: {}".format(str(self.tfm_gens))
        )

        self.bert_type = bert_type
        self.max_tokens = max_tokens
        logging.getLogger(__name__).info(
            "Loading BERT tokenizer: {}...".format(self.bert_type)
        )
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_type)

        self.img_format = image_format
        self.lmdb_dir = lmdb_dir
        self.env = None
        self.index=0




    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        if is_train:
            tfm_gens = build_transform_train(cfg)
            lmdb_dir = os.path.join(cfg.DATASETS.LMDB_ROOT,cfg.DATASETS.TRAIN[0].split('_')[0],cfg.DATASETS.TRAIN[0].split('_')[-1])
        else:
            tfm_gens = build_transform_test(cfg)
            lmdb_dir = os.path.join(cfg.DATASETS.LMDB_ROOT,cfg.DATASETS.TEST[0].split('_')[0],cfg.DATASETS.TEST[0].split('_')[-1])

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
            "bert_type": cfg.REFERRING.BERT_TYPE,
            "max_tokens": cfg.REFERRING.MAX_TOKENS,
            'lmdb_dir': lmdb_dir
        }
        return ret

    @staticmethod
    def _merge_masks(x):
        return x.sum(dim=0, keepdim=True).clamp(max=1)
    def _init_db(self):

        self.env = lmdb.open(self.lmdb_dir,
                             subdir=os.path.isdir(self.lmdb_dir),
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)
        self.txn = self.env.begin(write=False)

    def __call__(self, key):

        '''As per the standard Detectron2 workflow, the data being processed here 
        is the registered grefcoco dictionary. Please refer to grefcoco.py and 
        register_refcoco.py for more details.'''

        dataset_dict={}
        if self.env is None:
            self._init_db()
        byteflow = self.txn.get(key)
        ref = loads_pyarrow(byteflow)

        image = cv2.imdecode(np.frombuffer(ref['image'], np.uint8),
                               cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (480,480),interpolation=cv2.INTER_LINEAR)
        h,w = image.shape[:2] 
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0)
        

        mask = cv2.imdecode(np.frombuffer(ref['mask'], np.uint8),
                            cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.
        mask = cv2.resize(mask, (480,480),interpolation=cv2.INTER_NEAREST)

        dataset_dict["mask"] = torch.as_tensor(np.ascontiguousarray(mask),dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        dataset_dict["empty"] = torch.as_tensor(np.ascontiguousarray(ref['empty']))#.unsqueeze(0)

        if self.is_train:
            # Random sample sentences during training to alleviate instance-level long-tail issues.
            idx = np.random.choice(ref['num_sents']) 
            
            dataset_dict['lang_tokens'] = torch.as_tensor(np.ascontiguousarray(np.frombuffer(ref['lang_tokens'][idx],dtype=np.uint16).copy()),dtype=torch.int).unsqueeze(0)
            dataset_dict['lang_masks'] = torch.as_tensor(np.ascontiguousarray(np.frombuffer(ref['lang_masks'][idx],dtype=bool).copy()),dtype=torch.bool).unsqueeze(0)#.unsqueeze(-1)
        
        else:

            lang_tokens=[]
            lang_masks=[]
            
            for i in range(ref['num_sents']):
                lang_tokens.append(torch.as_tensor(np.ascontiguousarray(np.frombuffer(ref['lang_tokens'][i],dtype=np.uint16).copy()),dtype=torch.int).unsqueeze(0))
                lang_masks.append(torch.as_tensor(np.ascontiguousarray(np.frombuffer(ref['lang_masks'][i],dtype=bool).copy()),dtype=torch.bool).unsqueeze(0))
                
            dataset_dict['lang_tokens']=lang_tokens
            dataset_dict['lang_masks']=lang_masks
            dataset_dict['sents'] = ref['sents']
            dataset_dict['img_name'] = ref['img_name']
            dataset_dict['segment_id'] = ref['segment_id']
            

        dataset_dict['width'] = w
        dataset_dict['height'] = h

        return dataset_dict
