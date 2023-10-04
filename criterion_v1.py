import torch 
from tqdm import tqdm 
import numpy as np
import os 
import torch.nn.functional as F 

def focal_loss(preds, targets, weight):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
        preds (B x c x h x w x d)
        gt_regr (B x c x h x w x d)
    '''
    preds = preds.permute(0, 2, 3, 4, 1)
    targets = targets.permute(0, 2, 3, 4, 1)

    pos_inds = targets.eq(1).float()
    neg_inds = targets.lt(1).float()

    neg_weights = torch.pow(1 - targets, weight)

    loss = 0
    # for pred in preds:
        # pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
    preds = torch.clamp(preds, min=1e-4, max=1 - 1e-4)
    # print(f'in the loss shape is {preds.shape}')
    # print(f'the target shape is {pos_inds.shape}')
    pos_loss = torch.log(preds) * torch.pow(1 - preds, 2) * pos_inds * 100.
    neg_loss = torch.log(1 - preds) * torch.pow(preds, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss / len(preds)


def reg_l1_loss(pred, target, mask):
    #--------------------------------#
    #   计算l1_loss
    #--------------------------------#
    pred = pred.permute(0,2,3,4,1)
    target = target.permute(0,2,3,4,1)
    expand_mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 1, 3)
    
    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss


class Criterion:
    def __init__(self, config) -> None:
        self._config = config
    def __call__(self, hmap_pred, whd_pred, offset_pred, hmap_target, whd_target, offset_target, mask_target):

        hmap_loss = focal_loss(hmap_pred, hmap_target, self._config['point_weight'])
        r_loss =  reg_l1_loss(offset_pred, offset_target, mask_target)
        whd_loss = reg_l1_loss(whd_pred, whd_target, mask_target)

        loss_dict = {}
        loss_dict['hmap'] = hmap_loss
        loss_dict['whd'] = whd_loss
        loss_dict['offset'] = r_loss
        return loss_dict