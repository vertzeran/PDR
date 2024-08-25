import torch
from torch.nn import functional


def smooth_l1_loss(net_outputs, dx_dy_gt):
    return functional.smooth_l1_loss(net_outputs, dx_dy_gt)


def mse_loss(net_outputs, dx_dy_gt, weight=None):
    reduction = 'none' if weight is not None else 'mean'
    loss = functional.mse_loss(net_outputs, dx_dy_gt, reduction=reduction)
    if weight is not None:
        loss = torch.sum(loss * weight) / torch.sum(weight)
    return loss


def norm_loss(net_outputs, dx_dy_gt, weight=None):
    loss = torch.linalg.norm(net_outputs - dx_dy_gt, dim=1, keepdim=True)
    if weight is not None:
        loss = torch.sum(loss * weight) / torch.sum(weight)
    else:
        loss = loss.mean()
    return loss


def sqr_norm_loss(net_outputs, dx_dy_gt, weight=None):
    loss = torch.sum((net_outputs - dx_dy_gt) ** 2, dim=1, keepdim=True)
    if weight is not None:
        loss = torch.sum(loss * weight) / torch.sum(weight)
    else:
        loss = loss.mean()
    return loss


def norm_loss_with_length_mse(net_outputs, dx_dy_gt):
    out_length = torch.linalg.norm(net_outputs, dim=1, keepdim=True)
    gt_length = torch.linalg.norm(dx_dy_gt, dim=1, keepdim=True)

    loss1 = norm_loss(net_outputs, dx_dy_gt)
    loss2 = mse_loss(gt_length, out_length)

    return loss1 + loss2


def separate_length_and_direction_loss(net_outputs, dx_dy_gt, length_loss=mse_loss, direction_loss=norm_loss):
    out_length = torch.linalg.norm(net_outputs, dim=1, keepdim=True)
    gt_length = torch.linalg.norm(dx_dy_gt, dim=1, keepdim=True)

    eps = 1e-8
    out_direction = net_outputs / (out_length + eps)
    gt_direction = dx_dy_gt / (gt_length + eps)

    length_loss = length_loss(out_length, gt_length)
    direction_loss = direction_loss(out_direction, gt_direction)

    return length_loss + direction_loss


LOSSES = {
    'smooth_l1_loss': smooth_l1_loss,
    'mse_loss': mse_loss,
    'norm_loss': norm_loss,
    'sqr_norm_loss': sqr_norm_loss,
    'separate_length_and_direction_loss': separate_length_and_direction_loss,
    'norm_loss_with_length_mse': norm_loss_with_length_mse,
}
