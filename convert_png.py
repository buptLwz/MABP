import torch
from pycocotools import mask as coco_mask
from PIL import Image
import numpy as np
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
from tqdm import tqdm
samples = torch.load("/data/lwz/MABP/exp419-334/inference/ref_seg_predictions.pth")
for i in tqdm(samples):
    pred_mask = np.uint8(i['pred_mask'][0])
    mask = np.uint8(i['gt_mask'])
    #print(mask)
    name = i['sent']
    if True:
        pred_mask=Image.fromarray(pred_mask*255)
        mask  = Image.fromarray( np.uint8(mask.squeeze()*255))
    
        pred_mask.save('/data/lwz/MABP/exp419-334/pred/'+str(i['img_name'][-10:])+'+'+name.replace(' ','-').replace('/','-or-')+'-pred.png')
        mask.save('/data/lwz/MABP/exp419-334/pred/'+str(i['img_name'][-10:])+'+'+name.replace(' ','-').replace('/','-or-')+'-mask.png')

    #print(i)
    #break