from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from transformers import BertModel

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone

from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import ReferringCriterion

'''main detectron2 model for single object'''
@META_ARCH_REGISTRY.register()
class MABP(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        lang_backbone: nn.Module,
    ):

        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        # language backbone
        self.text_encoder = lang_backbone
       # self.text_encoder = torch.compile(self.text_encoder)
        #self.backbone=torch.compile(self.backbone)
        #self.sem_seg_head=torch.compile(self.sem_seg_head)
        #self.criterion=torch.compile(self.criterion)


    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        text_encoder = BertModel.from_pretrained(cfg.REFERRING.BERT_TYPE)
        text_encoder.pooler = None

        # loss weights

        CE_final_weight = cfg.CE_final_WEIGHT
        CE_intermedia_weight = cfg.CE_intermedia_WEIGHT
        CE_nt_weight = cfg.CE_nt_WEIGHT
        Dice_final_weight = cfg.Dice_final_WEIGHT
        Dice_intermedia_weight = cfg.Dice_intermedia_WEIGHT


        weight_dict = {
            "CE_final" : CE_final_weight,
            "CE_intermedia" : CE_intermedia_weight,
            "CE_nt" : CE_nt_weight,
            "Dice_final" : Dice_final_weight,
            "Dice_intermedia" : Dice_intermedia_weight,

        }

        losses = ["masks"]

        criterion = ReferringCriterion(
            weight_dict=weight_dict,
            losses=losses,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "lang_backbone": text_encoder,
        }

    @property
    def device(self):
        return self.pixel_mean.device
    
    #@torch.compile()
    def forward(self, data):

        # The normalization is completed in gres_model/data_prepare.py (train_net.py line 104)
        images = data['images']
        gtmasks = data['masks']
        emptys = data['emptys']
        

        processed_results = []
        for lang_tokens,lang_masks,sents in zip(
                data['lang_tokens'],data['lang_masks'],data['sents']):
            
            # get linguistic features
            lang_feats = self.text_encoder(lang_tokens,lang_masks)[0] # B, Nl, 768

            lang_feats = lang_feats.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
            lang_masks = lang_masks.unsqueeze(dim=-1)  # (batch, N_l, 1)

            res2,res3,res4,res5 = self.backbone(images, lang_feats, lang_masks)    
            outputs = self.sem_seg_head(res2,res3,res4,res5, lang_feats, lang_masks)

            if self.training:

                losses = self.criterion(outputs, gtmasks, emptys)

                for k in list(losses.keys()):
                    if k in self.criterion.weight_dict:
                        losses[k] *= self.criterion.weight_dict[k]
                    else:
                        losses.pop(k)
                return losses

            else:
                # we use test batch size = 1
                pred_masks = outputs["pred_masks"][-1].unsqueeze(1)


                pred_masks = F.interpolate(
                    pred_masks,
                    size=(images.shape[-2], images.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )

                #pred_nts = outputs["nt_label"]# 4*B*2

                del outputs

                # outputs["nt_label"] is list with 4 elements, [nt_0_level, nt_1_level, nt_2_level, nt_3_level]
                # nt_i_level means no-target indicator in i-th level
                # nt_i_level shape: B*2

                #cur_bs = pred_nts[0].shape[0] # nt_0_level.shape[0]

                # make nt_label batch first
                #batched_pred_nts = [ [ j[i] for j in pred_nts] for i in range(cur_bs)] # B*4*2
                
                
                for pred_mask in zip(
                    pred_masks
                ):
                    processed_results.append({})
                    pred_mask = retry_if_cuda_oom(self.refer_inference)(pred_mask)

                    processed_results[-1]["pred_mask"] = pred_mask
                    #processed_results[-1]["pred_nt"] = pred_nt
                    processed_results[-1]['sent']=sents



            return processed_results

    def refer_inference(self, mask_pred):
        mask_pred = mask_pred.sigmoid()
        # for i in range(len(nt_pred)):
        #     nt_pred[i] = nt_pred[i].sigmoid()
        return mask_pred#, nt_pred