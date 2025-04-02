# Bring Adaptive Binding Prototypes to Generalized Referring Expression Segmentation

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bring-adaptive-binding-prototypes-to/generalized-referring-expression-segmentation)](https://paperswithcode.com/sota/generalized-referring-expression-segmentation?p=bring-adaptive-binding-prototypes-to)

 **[📄[arXiv]](https://arxiv.org/abs/2405.15169)**

This repository contains code for **TMM** paper:
> [Bring Adaptive Binding Prototypes to Generalized Referring Expression Segmentation](https://arxiv.org/abs/2405.15169)  
> Weize Li, Haochen Bai, Zhicheng Zhao, Fei Su


<div align="center">
  <img src="/data/lwz/MABP-git/fig/archi-rsh-1.png" width="100%" height="100%"/>
</div><br/>

## Update
- **(2025/04/01)** We have updated full codes.
- **(2024/09/27)** A simple version is released.

## Installation:

The code is tested under same environment as ReLA

1. Follow https://github.com/henghuiding/ReLA to prepare environment

2. Prepare the dataset following ```datasets/DATASET.md```

3. Download the best weights we provide in <a href="https://drive.google.com/file/d/1qxjwyFVtrscKNB7WF3xwi7etJMf0qm1_/view?usp=sharing" title="model">MABP_best</a>
## Inference

```
python train_net.py --config-file configs/gres-MABP.yaml --num-gpus 2 --dist-url auto MODEL.WEIGHTS [path to MABP_best.pth]   OUTPUT_DIR [output dir]
```
Then you can obtain the current best result of MABP:


| cIoU | gIoU | T_acc | N_acc | Pr@0.7 | Pr@0.8 | Pr@0.9 |
| :---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 65.72 | 68.86 | 96.35 | 64.73 | 70.45 | 58.92 | 26.17 |

Note: This result is slightly higher than what we reported in our paper, primarily because we enhance the recall for no-target samples by taking the logical OR of the intermediate results of no-target indicators. 

Additionally, the results reported in the paper were based solely on the first no-target indicator, which was a bug caused by our oversight (using the last indicator or applying a logical OR would have been more reasonable). Unfortunately, it is regrettable that the paper on TMM cannot be corrected currently.

## Training

Firstly, download the backbone weights (`swin_base_patch4_window12_384_22k.pkl`) and convert it into detectron2 format using the script:

```
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth
python tools/convert-pretrained-swin-model-to-d2.py swin_base_patch4_window12_384_22k.pth swin_base_patch4_window12_384_22k.pkl
```

Then start training:
```
python train_net.py \
    --config-file configs/gres-MABP.yaml \
    --num-gpus 2 --dist-url auto \
    MODEL.WEIGHTS [path to weights] \
    OUTPUT_DIR [path to weights]
```

Note: You can add your own configurations subsequently to the training command for customized options. For example:

```
SOLVER.IMS_PER_BATCH 48 
SOLVER.BASE_LR 0.00001 
```

For the full list of base configs, see `configs/referring_R50.yaml` and `configs/Base-COCO-InstanceSegmentation.yaml`



## Acknowledgement

This project is based on [ReLA](https://github.com/henghuiding/ReLA), [refer](https://github.com/lichengunc/refer), [Mask2Former](https://github.com/facebookresearch/Mask2Former), [Detectron2](https://github.com/facebookresearch/detectron2), [VLT](https://github.com/henghuiding/Vision-Language-Transformer). Many thanks to the authors for their great works!


