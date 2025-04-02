import argparse
import json
import os

import cv2
import numpy as np
from tqdm import tqdm

from refer import REFER
from grefer import G_REFER

parser = argparse.ArgumentParser(description='Data preparation')
parser.add_argument('--data_root', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--dataset',
                    type=str,
                    choices=['grefcoco','refcoco', 'refcoco+', 'refcocog', 'refclef'],
                    default='refcoco')
parser.add_argument('--split', type=str, default='umd')
parser.add_argument('--generate_mask', action='store_true')

args = parser.parse_args()

img_path = os.path.join(args.data_root, 'images', 'train2014')


h, w = (416, 416)


refer = G_REFER(args.data_root, args.dataset, args.split)

print('dataset [%s_%s] contains: ' % (args.dataset, args.split))
ref_ids = refer.getRefIds()
image_ids = refer.getImgIds()
print('%s expressions for %s refs in %s images.' %
      (len(refer.Sents), len(ref_ids), len(image_ids)))


assert args.dataset == 'grefcoco'
splits = ['train', 'val', 'testA', 'testB']


for split in splits:
    ref_ids = refer.getRefIds(split=split)
    print('%s refs are in split [%s].' % (len(ref_ids), split))


def cat_process(cat):
    if cat >= 1 and cat <= 11:
        cat = cat - 1
    elif cat >= 13 and cat <= 25:
        cat = cat - 2
    elif cat >= 27 and cat <= 28:
        cat = cat - 3
    elif cat >= 31 and cat <= 44:
        cat = cat - 5
    elif cat >= 46 and cat <= 65:
        cat = cat - 6
    elif cat == 67:
        cat = cat - 7
    elif cat == 70:
        cat = cat - 9
    elif cat >= 72 and cat <= 82:
        cat = cat - 10
    elif cat >= 84 and cat <= 90:
        cat = cat - 11
    return cat


def bbox_process(bbox):
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    x_max = x_min + int(bbox[2])
    y_max = y_min + int(bbox[3])
    return list(map(int, [x_min, y_min, x_max, y_max]))
def prepare_dataset(dataset, splits, output_dir, generate_mask=False):
    ann_path = os.path.join(output_dir, 'anns', dataset)
    mask_path = os.path.join(output_dir, 'masks', dataset)
    if not os.path.exists(ann_path):
        os.makedirs(ann_path)
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)

    for split in splits:

        dataset_array = []
        ref_ids = refer.getRefIds(split=split)
        print(ref_ids)
        print('Processing split:{} - Len: {}'.format(split, len(ref_ids)))
        for i in tqdm(ref_ids):
            ref_dict = {}

            refs = refer.Refs[i]
            #bboxs = refer.getRefBox(i)
            sentences = refs['sentences']
            sent_dict = []
            for n, sent in enumerate(sentences):
                sent_dict.append({
                    'idx': n,
                    'sent_id': sent['sent_id'],
                    'sent': sent['sent'].strip()
                })

            ref_dict['sentences'] = sent_dict
            ref_dict['sentences_num'] = len(sent_dict)
            if ref_dict['sentences_num']==0:
                continue
            image_urls = refer.loadImgs(image_ids=refs['image_id'])[0]
            cat = [ cat_process(j) for j in refs['category_id']] 
            image_urls = image_urls['file_name']
            if dataset == 'refclef' and image_urls in [
                    '19579.jpg', '17975.jpg', '19575.jpg'
            ]:
                continue
            #box_info = bbox_process(bboxs)

            #ref_dict['bbox'] = box_info
            ref_dict['cat'] = cat
            ref_dict['segment_id'] = i
            ref_dict['img_name'] = image_urls
            anns = refer.loadAnns(refs['ann_id']) #for ann in refs['ann_id']]
            if None in anns:
                ref_dict['empty'] = True
            else:
                ref_dict['empty'] = False


            if generate_mask:
                from PIL import Image
                img = Image.open(os.path.join(img_path,image_urls))
                outmask = np.zeros((img.size[1],img.size[0]),dtype=np.uint8)
                #for i in anns:
                if ref_dict['empty']:
                    #pass
                    cv2.imwrite(os.path.join(mask_path,
                                         str(i) + '.png'),
                            outmask * 255)
                else:
                    inss = []
                    for j in anns:
                        if j['iscrowd']:
                            continue
                        else:
                            inss.append(refer.getMask(j)['mask'])
                    #inss =[refer.getMask(j)['mask'] for j in anns]
                    for ins in inss:
                        outmask = outmask + ins
                        #continue
                    outmask = np.clip(outmask,0,1)
                    cv2.imwrite(os.path.join(mask_path,
                                             str(i) + '.png'),
                                outmask * 255)



            dataset_array.append(ref_dict)
        print('Dumping json file...')
        with open(os.path.join(output_dir, 'anns', dataset, split + '.json'),
                  'w') as f:
            json.dump(dataset_array, f)


prepare_dataset(args.dataset, splits, args.output_dir, args.generate_mask)
