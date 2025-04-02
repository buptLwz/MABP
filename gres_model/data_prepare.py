import torch

def prepare_data(data,test=False):
    pixel_mean=torch.as_tensor([123.675,116.28,103.53]).view(-1, 1, 1)
    pixel_std=torch.as_tensor([58.395,57.12,57.375]).view(-1, 1, 1)
    size_divisibility= 32

    images = [x["image"].cuda(non_blocking=True) for x in data]
    images = [(x - pixel_mean.cuda(non_blocking=True)) / pixel_std.cuda(non_blocking=True) for x in images]

    images = torch.cat(images,dim=0)

    emptys = [x['empty'].cuda(non_blocking=True) for x in data]
    emptys = torch.cat(emptys, dim=0)

    masks = [x["mask"].cuda(non_blocking=True) for x in data]
    masks = torch.cat(masks,dim=0)


    if test:
        assert len(data)==1
        lang_tokens = [i.cuda(non_blocking=True) for i in data[0]['lang_tokens']]
        lang_masks = [i.cuda(non_blocking=True) for i in data[0]['lang_masks']]
        sents = data[0]['sents']

    else:
                
        lang_tokens = [x['lang_tokens'].cuda(non_blocking=True) for x in data]
        lang_tokens = [torch.cat(lang_tokens, dim=0)]

        lang_masks = [x['lang_masks'].cuda(non_blocking=True) for x in data]
        lang_masks = [torch.cat(lang_masks, dim=0)]

        sents=[' ']

   
    
    return {'images':images, 
            'lang_tokens':lang_tokens, 
            'lang_masks':lang_masks, 
            'masks':masks, 
            'emptys':emptys,
            'sents':sents}
