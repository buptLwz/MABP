import logging

import torch
import torch.nn.functional as F
from torch import nn

from ..utils.misc import nested_tensor_from_tensor_list
def _get_src_permutation_idx(self, indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B,-1, window_size, window_size, C)
    return windows

def refer_bce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weight: None):

    #loss = F.cross_entropy(inputs, targets, weight=weight)
    loss = F.binary_cross_entropy_with_logits(inputs,targets,reduction='none',pos_weight=torch.tensor([1.1],device=inputs.device))

    return loss

def refer_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weight: torch.Tensor):

    loss = F.cross_entropy(inputs, targets, weight=weight,reduction='none')

    return loss

def dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """

    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()

refer_bce_loss_jit = torch.jit.script(
    refer_bce_loss
)  # type: torch.jit.ScriptModule

refer_ce_loss_jit = torch.jit.script(
    refer_ce_loss
)  # type: torch.jit.ScriptModule

refer_dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule

class ReferringCriterion(nn.Module):
    def __init__(self, weight_dict, losses, so):
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses
        self.so = so

    def get_loss(self, loss, outputs, targets):
        loss_map = {
            'masks': self.loss_masks_refer,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets)

    def loss_masks_refer(self, outputs, targets):
        src_masks = outputs["pred_masks"]
        out_mask = src_masks[-1].unsqueeze(1)
        src_masks = src_masks[:-1]
        #src_minimap = outputs["pred_logits"].permute(0,2,1)
        #src_nt_label = outputs["nt_label"]
        #mask_sw = outputs['mask_sw'] #bqhwc

        masks = [t["gt_mask_merged"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        # print(target_masks.shape)
        # print(src_masks[0].shape)
        target_masks = target_masks.to(src_masks[0])

        #target_nts = torch.stack([t["empty"] for t in targets])
        if not self.so:
            src_nt_label = outputs["nt_label"]
            target_nts = torch.stack([t["empty"] for t in targets])
        src_shape = [i.shape[-2:] for i in src_masks]
        h, w = target_masks.shape[-2:]
        #print(out_mask.shape) 
################
        # target_sw = F.interpolate(target_masks, (out_mask.shape[-2],out_mask.shape[-1]), mode='nearest')
        # pad_mask = torch.ones_like(target_sw)

        # target_sw = F.pad(target_sw,(4,4,4,4))
        # pad_mask = F.pad(pad_mask,(4,4,4,4))
        
        
        # target_sw = window_partition(target_sw.permute(0,2,3,1),8)
        # pad_mask = window_partition(pad_mask.permute(0,2,3,1),8)

##################

        out_mask = F.interpolate(out_mask, (h, w), mode='bilinear', align_corners=False)

        target_minimap = [F.interpolate(target_masks, i,  mode='bilinear', align_corners=False) for i in src_shape]

        # target_sw = F.pad(target_masks,(4,4,4,4))
        # target_sw = F.interpolate(target_sw, (out_mask.shape[-2],out_mask.shape[-1]), mode='nearest')
        # target_sw = window_partition(target_sw.permute(0,2,3,1),8)
       

        weight = torch.tensor([0.9, 1.1],device=src_masks[0].device)#.to(src_masks[0])
        # print(src_nt_label[0].shape)
        # print(target_nts.shape)s
        A = refer_bce_loss_jit(out_mask, target_masks.float(), None)
        #print(A.shape)
        
        #A_sw = F.avg_pool2d(A,4,4)#.flatten(2)
        #A_sw = F.pad(A_sw,(4,4,4,4),mode=)#.flatten(2)

        #A_sw = F.avg_pool2d(A_sw,8,8,4,count_include_pad=False).flatten(2)

        #_,ind = torch.topk(A_sw,30)
        #print(ind.shape)
        with torch.no_grad():
            out_mask_p = torch.sigmoid(out_mask.detach())
            entropy_A = torch.clamp(-out_mask_p*torch.log2(out_mask_p+1e-6)-(1-out_mask_p)*torch.log2((1-out_mask_p)+1e-6),min=0.5)*2-0.5
            src_masks_p = [ torch.sigmoid(src_masks[i].detach().unsqueeze(1)) for i in range(len(src_masks))]
            entropy_B = [ torch.clamp(-src_masks_p[i]*torch.log2(src_masks_p[i]+1e-6)-(1-src_masks_p[i])*torch.log2((1-src_masks_p[i])+1e-6),min=0.5)*2-0.5 for i in range(len(src_masks))]
        A = A*(entropy_A)
        A = A.mean() #+ refer_dice_loss_jit(out_mask.squeeze(1),target_masks.float())
        B = sum([(refer_bce_loss_jit(src_masks[i].unsqueeze(1), target_minimap[i].float(), None)*(entropy_B[i])).mean()  for i in range(len(src_masks))])/len(src_masks)
        
        #B = sum([refer_bce_loss_jit(src_masks[i].unsqueeze(1), target_minimap[i].float(), None).mean()  for i in range(len(src_masks))])/len(src_masks)
        #B = sum([(refer_bce_loss_jit(src_masks[i].unsqueeze(1), target_minimap[i].float(), None).mean() + refer_dice_loss_jit(src_masks[i],target_minimap[i].float()) )for i in range(len(src_masks))])/len(src_masks)
        #C_weight = [ 0.5 if i==0 else 1.0 for i in target_nts ]
        #C_weight = torch.tensor(C_weight,device=src_masks[0].device)#.unsqueeze(1)
        #print(refer_ce_loss_jit(src_nt_label[0], target_nts, weight).shape)
        #print(C_weight.shape)
        #C = sum([refer_ce_loss_jit(src_nt_label[i], target_nts.unsqueeze(1).float(), weight).mean()  for i in range(len(src_nt_label))])/len(src_nt_label)
        #C = sum([(refer_ce_loss_jit(src_nt_label[i], target_nts, weight)).mean()  for i in range(len(src_nt_label))])/len(src_nt_label)
        if not self.so:
            C = [(refer_ce_loss_jit(src_nt_label[i], target_nts, weight))  for i in range(len(src_nt_label))]
            C = sum([C[i].mean() for i in range(len(src_nt_label))])/len(src_nt_label)

        D = refer_dice_loss_jit(out_mask.squeeze(1),target_masks.float())
        E = sum([refer_dice_loss_jit(src_masks[i],target_minimap[i].float()) for i in range(len(src_masks))])/len(src_masks)
        #print(src_nt_label.shape)
        #C = (refer_ce_loss_jit(src_nt_label, target_nts, weight)*C_weight).sum()#.mean()
        #C = refer_ce_loss_jit(src_nt_label, target_nts, weight).mean()
        #D = torch.tensor(0).to(src_masks[0])
        #D_num = 0
        #T = 0.5

        # batch_idx = torch.cat([torch.full_like(src, i) for i, (src) in enumerate(ind)])
        # select_mask = mask_sw[batch_idx,ind.squeeze(1)]
        # select_mask = select_mask.permute(0,1,4,2,3).flatten(3) #3,10,8,8,256 -> 3,10,256,64

        # select_pad_mask = pad_mask[batch_idx,ind.squeeze(1)]
        # select_pad_mask = select_pad_mask.permute(0,1,4,2,3).flatten(3) #3,10,8,8,1 -> 3,10,1,64

        # select_tgt = target_sw[batch_idx,ind.squeeze(1)]
        # #print(select_tgt[1,0,:,:,0])
        # select_tgt = select_tgt.permute(0,1,4,2,3).flatten(3) #3,10,8,8,1 -> 3,10,1,64
        # #print(select_tgt[1,0,0,])
        # contgt = ((select_tgt==select_tgt.permute(0,1,3,2))).long()#.to(src_masks[0])
        # #print(contgt.shape)
        # #print(contgt[1,0,:,:])
        # pos = contgt - torch.eye(64,device=contgt.device)#.to(contgt)#.to(src_masks[0])
        # neg = (contgt == 0).float()
        # #print(pos[1,0,:,:])
        # #print(neg[1,0,:,:])

        # sim = F.cosine_similarity(select_mask.unsqueeze(3),select_mask.unsqueeze(4),dim=2)
        # #print(sim.shape)
        # pos_c = torch.exp((sim*pos)/T).sum(dim=-1)
        # neg_c = torch.exp((sim*neg)/T).sum(dim=-1)

        # con = -torch.log(pos_c/(pos_c+neg_c))
        # #print(con.shape)
        # #print(select_pad_mask.squeeze(-2).long().shape)
        # #con = con*select_pad_mask.squeeze(-2)
        # #print(con.shape)
        # #print(con[torch.nonzero(con,as_tuple=True)].shape)
        # #print(con[torch.nonzero(con*select_pad_mask.squeeze(-2),as_tuple=True)].shape)

        # unpad_con = con[torch.nonzero(select_pad_mask.squeeze(-2),as_tuple=True)]
        # #print(con.shape)
        # D =unpad_con.mean()

        #D = D + con
        # for b in range(mask_sw.shape[0]):
        #     for i in ind[b,0]:
        #         curmask = mask_sw[b,i].permute(2,0,1).flatten(1)
        #         curtgt = target_sw[b,i].permute(2,0,1).flatten(1)
        #         curpm = pad_mask[b,i].permute(2,0,1).flatten(1)

        #         contgt = ((curtgt==curtgt.t())).long()#.to(src_masks[0])
        #         pos = contgt - torch.eye(64).to(contgt)#.to(src_masks[0])
        #         neg = contgt == 0
        #         sim = F.cosine_similarity(curmask.unsqueeze(1),curmask.unsqueeze(2),dim=0)
        #         #print(sim.shape)
        #         pos_c = torch.exp((sim*pos)/T).sum(dim=1)
        #         neg_c = torch.exp((sim*neg)/T).sum(dim=1)
        #         con = -torch.log(pos_c/(pos_c+neg_c))[curpm].mean()
        #         D = D + con

        #         #D = D + F.smooth_l1_loss(sim,contgt)
        #         D_num = D_num +1
        #print(D_num)
        #D = D / D_num


        # loss_mask = \
        #     refer_ce_loss_jit(out_mask, target_masks.float(), weight) + \
        #     sum([refer_ce_loss_jit(src_masks[i].unsqueeze(1), target_minimap[i].float(), weight) * 0.1 for i in range(len(src_masks))]) + \
        #      sum([refer_ce_loss_jit(src_nt_label[i], target_nts.unsqueeze(1).float(), weight) * 0.1 for i in range(len(src_nt_label))])
        #loss_mask = A + B + C
        # print(A)
        # print(B)
        # print(C)
        if self.so:

            losses = {
            #"loss_mask": loss_mask,
                "A":A,
                "B":B,
                #"C":C,
                "D":D,
                "E":E,

            }
        else:
            losses = {
            #"loss_mask": loss_mask,
            "A":A,
            "B":B,
            "C":C,
            "D":D,
            "E":E,

        }

        del src_masks
        del target_masks
        return losses

    def forward(self, outputs, targets):
        # Compute all the requested losses
        losses = {}
        losses.update(self.loss_masks_refer(outputs, targets))

        return losses

