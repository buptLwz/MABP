# loss compute
import torch
import torch.nn.functional as F
from torch import nn



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
    def __init__(self,  losses,weight_dict=None):
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses

    def get_loss(self, loss, outputs, targets_m,empty):
        loss_map = {
            'masks': self.loss_masks_refer,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets_m,empty)

    def loss_masks_refer(self, outputs, target_masks,empty):
        src_masks = outputs["pred_masks"]
        out_mask = src_masks[-1].unsqueeze(1)
        src_masks = src_masks[:-1]
       

        #src_nt_label = outputs["nt_label"]


        #target_nts = empty
        
        src_shape = [i.shape[-2:] for i in src_masks]
        h, w = target_masks.shape[-2:]


        out_mask = F.interpolate(out_mask, (h, w), mode='bilinear', align_corners=False)

        target_minimap = [F.interpolate(target_masks, i,  mode='bilinear', align_corners=False) for i in src_shape]


        weight = torch.tensor([0.9, 1.1],device=src_masks[0].device)

        CE_final = refer_bce_loss_jit(out_mask, target_masks.float(), None)

        with torch.no_grad():
            out_mask_p = torch.sigmoid(out_mask.detach())
            entropy_CE_final = torch.clamp(-out_mask_p*torch.log2(out_mask_p+1e-6)-(1-out_mask_p)*torch.log2((1-out_mask_p)+1e-6),min=0.5)*2-0.5
            
            src_masks_p = [ torch.sigmoid(src_masks[i].detach().unsqueeze(1)) for i in range(len(src_masks))]
            entropy_CE_intermedia = [ torch.clamp(-src_masks_p[i]*torch.log2(src_masks_p[i]+1e-6)-(1-src_masks_p[i])*torch.log2((1-src_masks_p[i])+1e-6),min=0.5)*2-0.5 for i in range(len(src_masks))]
        
        CE_final = CE_final*(entropy_CE_final)#*weight_nt.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        CE_final = CE_final.mean() #+ refer_dice_loss_jit(out_mask.squeeze(1),target_masks.float())
        CE_intermedia = sum([(refer_bce_loss_jit(src_masks[i].unsqueeze(1), target_minimap[i].float(), None)*(entropy_CE_intermedia[i])).mean()  for i in range(len(src_masks))])/len(src_masks)
        
      
        
        #CE_nt = [(refer_ce_loss_jit(src_nt_label[i], target_nts.to(torch.int64), weight))  for i in range(len(src_nt_label))]
       
        #CE_nt = sum([CE_nt[i].mean() for i in range(len(src_nt_label))])/len(src_nt_label)
        Dice_final = refer_dice_loss_jit(out_mask.squeeze(1),target_masks.float())
        Dice_intermedia = sum([refer_dice_loss_jit(src_masks[i],target_minimap[i].float()) for i in range(len(src_masks))])/len(src_masks)

        losses = {
            #"loss_mask": loss_mask,
            "CE_final" : CE_final,
            "CE_intermedia" : CE_intermedia,
            #"CE_nt" : CE_nt,
            "Dice_final" : Dice_final,
            "Dice_intermedia" : Dice_intermedia,

        }

        del src_masks
        del target_masks
        return losses

    def forward(self, outputs, targets_m,empty):
        # Compute all the requested losses
        losses = {}
        losses.update(self.loss_masks_refer(outputs, targets_m,empty))

        return losses

