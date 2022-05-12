# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import math
from typing import List, Tuple
import torch

from detectron2.layers import batched_nms, cat
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage

logger = logging.getLogger(__name__)


def find_top_rpn_proposals(
    proposals: List[torch.Tensor],
    pred_objectness_logits: List[torch.Tensor],
    image_sizes: List[Tuple[int, int]],
    nms_thresh: float,
    pre_nms_topk: int,
    post_nms_topk: int,
    min_box_size: float,
    training: bool,
):
    """
    For each feature map, select the `pre_nms_topk` highest scoring proposals,
    apply NMS, clip proposals, and remove small boxes. Return the `post_nms_topk`
    highest scoring proposals among all the feature maps for each image.

    Args:
        proposals (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A, 4).
            All proposal predictions on the feature maps.
        pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A).
        image_sizes (list[tuple]): sizes (h, w) for each image
        nms_thresh (float): IoU threshold to use for NMS
        pre_nms_topk (int): number of top k scoring proposals to keep before applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is per
            feature map.
        post_nms_topk (int): number of top k scoring proposals to keep after applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is total,
            over all feature maps.
        min_box_size (float): minimum proposal box side length in pixels (absolute units
            wrt input images).
        training (bool): True if proposals are to be used in training, otherwise False.
            This arg exists only to support a legacy bug; look for the "NB: Legacy bug ..."
            comment.

    Returns:
        list[Instances]: list of N Instances. The i-th Instances
            stores post_nms_topk object proposals for image i, sorted by their
            objectness score in descending order.
    
    返回值中每项记录的是第 i 个 image 的 proposals
    """
    num_images = len(image_sizes)
    device = proposals[0].device

    # 选取 每个 feature level 和 每个 image 的 topk 个 anchor
    # 1. Select top-k anchor for every level and every image
    topk_scores = []  # #lvl Tensor, each of shape N x topk
    topk_proposals = []
    level_ids = []  # #lvl Tensor, each of shape (topk,)
    batch_idx = torch.arange(num_images, device=device)
    for level_id, (proposals_i, logits_i) in enumerate(zip(proposals, pred_objectness_logits)):
        Hi_Wi_A = logits_i.shape[1]
        num_proposals_i = min(pre_nms_topk, Hi_Wi_A)

        # sort is faster than topk (https://github.com/pytorch/pytorch/issues/22812)
        # topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)
        logits_i, idx = logits_i.sort(descending=True, dim=1)
        topk_scores_i = logits_i[batch_idx, :num_proposals_i]
        topk_idx = idx[batch_idx, :num_proposals_i]

        # each is N x topk
        topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]  # N x topk x 4

        topk_proposals.append(topk_proposals_i)
        topk_scores.append(topk_scores_i)
        level_ids.append(torch.full((num_proposals_i,), level_id, dtype=torch.int64, device=device))

    # 将所有的结果都拼接起来
    # 2. Concat all levels together
    topk_scores = cat(topk_scores, dim=1)
    topk_proposals = cat(topk_proposals, dim=1)
    level_ids = cat(level_ids, dim=0)

    # 3. For each image, run a per-level NMS, and choose topk results.
    results: List[Instances] = []
    for n, image_size in enumerate(image_sizes):
        boxes = Boxes(topk_proposals[n])
        scores_per_img = topk_scores[n]
        lvl = level_ids

        valid_mask = torch.isfinite(boxes.tensor).all(dim=1) & torch.isfinite(scores_per_img)
        if not valid_mask.all():
            if training:
                raise FloatingPointError(
                    "Predicted boxes or scores contain Inf/NaN. Training has diverged."
                )
            boxes = boxes[valid_mask]
            scores_per_img = scores_per_img[valid_mask]
            lvl = lvl[valid_mask]
        boxes.clip(image_size)

        # filter empty boxes
        keep = boxes.nonempty(threshold=min_box_size)
        if keep.sum().item() != len(boxes):
            boxes, scores_per_img, lvl = boxes[keep], scores_per_img[keep], lvl[keep]

        keep = batched_nms(boxes.tensor, scores_per_img, lvl, nms_thresh)
        # In Detectron1, there was different behavior during training vs. testing.
        # (https://github.com/facebookresearch/Detectron/issues/459)
        # During training, topk is over the proposals from *all* images in the training batch.
        # During testing, it is over the proposals for each image separately.
        # As a result, the training behavior becomes batch-dependent,
        # and the configuration "POST_NMS_TOPK_TRAIN" end up relying on the batch size.
        # This bug is addressed in Detectron2 to make the behavior independent of batch size.
        keep = keep[:post_nms_topk]  # keep is already sorted

        res = Instances(image_size)
        res.proposal_boxes = boxes[keep]
        res.objectness_logits = scores_per_img[keep]
        res.level_ids = lvl[keep]
        results.append(res)
    return results


def find_top_rpn_proposals_group(
    proposals: List[torch.Tensor],
    pred_objectness_logits: List[torch.Tensor],
    image_sizes: List[Tuple[int, int]],
    num_anchors: List[int],
    nms_thresh: float,
    pre_nms_topk: int,
    post_nms_topk: int,
    min_box_size: float,
    training: bool,
):
    """
    对每个 feature map, 选取 pre_nms_topk 个最高分的 proposals, 之后会进行 nms, proposals clip, 并且移除
    掉所有的小的 boxes. 最后返回 post_nms_topk 个最高分的 proposals(image 所有 feature map 上的 proposals).
    For each feature map, select the `pre_nms_topk` highest scoring proposals,
    apply NMS, clip proposals, and remove small boxes. Return the `post_nms_topk`
    highest scoring proposals among all the feature maps for each image.

    Args:
        proposals (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A, 4).
            All proposal predictions on the feature maps.
        pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A).
        image_sizes (list[tuple]): sizes (h, w) for each image
        num_anchors: 对应于 mrrp 每个 feature map 上每个位置的 anchor 数量, 要求是所有 feature map 的该值保持一致
        nms_thresh (float): IoU threshold to use for NMS (默认 0.7)
        pre_nms_topk (int): number of top k scoring proposals to keep before applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is per
            feature map.
        post_nms_topk (int): number of top k scoring proposals to keep after applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is total,
            over all feature maps.
        min_box_size (float): minimum proposal box side length in pixels (absolute units
            wrt input images).
        training (bool): True if proposals are to be used in training, otherwise False.
            This arg exists only to support a legacy bug; look for the "NB: Legacy bug ..."
            comment.

    Returns:
        list[Instances]: list of N Instances. The i-th Instances
            stores post_nms_topk object proposals for image i, sorted by their
            objectness score in descending order.
    """
    num_images = len(image_sizes)
    device = proposals[0].device
    num_preprepre_nms = 0
    num_prepre_nms = 0
    num_pre_nms = 0
    num_post_nms = 0

    # 1. Select top-k anchor for every level and every image
    topk_scores = []  # #lvl Tensor, each of shape N x topk
    topk_proposals = []
    level_ids = []  # #lvl Tensor, each of shape (topk,)
    batch_idx = torch.arange(num_images, device=device)
    # print(num_anchors)
    for level_id, (proposals_i, logits_i) in enumerate(zip(proposals, pred_objectness_logits)):
        # 当前 level 的总的anchors数量
        Hi_Wi_A = logits_i.shape[1]
        # 总的采样点数量
        Hi_Wi = int(Hi_Wi_A / num_anchors[level_id])
        # 将 anchors 的相关输出 按照位置进行展开
        logits_i = logits_i.view(-1, Hi_Wi, num_anchors[level_id])
        proposals_i = proposals_i.view(-1, Hi_Wi, num_anchors[level_id], 4)
        # print(logits_i.shape)
        # print(proposals_i.shape)
        num_preprepre_nms += Hi_Wi_A
        # 这里好像是每个位置, 每种 shape 选 topk ? 而不是所有位置所有 shape 选择 topk 了... em...
        # 遍历所有的 anchor id
        for anchor_id in range(num_anchors[level_id]):
            # 每层选取最多的数量
            num_proposals_i_a = min(pre_nms_topk, Hi_Wi)

            # 获取所有空间位置上 anchor_id 位置上 logits 和 proposals
            logits_i_a = logits_i[:, :, anchor_id]
            proposals_i_a = proposals_i[:, :, anchor_id, :]

            if True and False:
                width_i_a = proposals_i_a[..., 2] - proposals_i_a[..., 0]
                height_i_a = proposals_i_a[..., 3] - proposals_i_a[..., 1]
                size_i_a = width_i_a * height_i_a
                aspect_i_a = width_i_a / height_i_a
                print(device, size_i_a.sqrt().mean(), size_i_a.mean().sqrt(), aspect_i_a.mean())

            # sort is faster than topk (https://github.com/pytorch/pytorch/issues/22812)
            # topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)
            # 对每张图的 anchor_id 位置的 proposals 进行排序, 选取 topk 的 proposals
            logits_i_a, idx = logits_i_a.sort(descending=True, dim=1)
            topk_scores_i_a = logits_i_a[batch_idx, :num_proposals_i_a]
            topk_idx = idx[batch_idx, :num_proposals_i_a]

            # each is N x topk
            topk_proposals_i_a = proposals_i_a[batch_idx[:, None], topk_idx]  # N x topk x 4

            # 将每层中的 topk 进行填充
            topk_proposals.append(topk_proposals_i_a)
            topk_scores.append(topk_scores_i_a)
            # level_id 是用来区分层次的, 后面 batched_nms 进行处理的时候会按照 level_ids 进行区分
            level_ids.append(
                torch.full(
                    (num_proposals_i_a,),
                    level_id * 1000 + anchor_id,
                    dtype=torch.int64,
                    device=device,
                )
            )
    # exit()
    # 2. Concat all levels together
    # 将所有层次的 anchors 相关信息 cat 到一起
    topk_scores = cat(topk_scores, dim=1)
    topk_proposals = cat(topk_proposals, dim=1)
    level_ids = cat(level_ids, dim=0)

    # 3. For each image, run a per-level NMS, and choose topk results.
    # 对每个 image 都会进行一次 按照 level 的 nms, 然后选择其中 topk 的结果
    results: List[Instances] = []
    for n, image_size in enumerate(image_sizes):
        # 当前层次的 topk boxes
        boxes = Boxes(topk_proposals[n])
        scores_per_img = topk_scores[n]
        lvl = level_ids

        num_prepre_nms += len(boxes)

        # 进行 bbox 是否有效的 mask 的生成
        valid_mask = torch.isfinite(boxes.tensor).all(dim=1) & torch.isfinite(scores_per_img)
        if not valid_mask.all():
            # 训练的时候并不允许这种情况 的出现
            if training:
                raise FloatingPointError(
                    "Predicted boxes or scores contain Inf/NaN. Training has diverged."
                )
            boxes = boxes[valid_mask]
            scores_per_img = scores_per_img[valid_mask]
            lvl = lvl[valid_mask]
        # boxes 会根据 image_size 进行 clip, 防止超出边界
        boxes.clip(image_size)

        # filter empty boxes
        # 筛选掉所有的空的 boxes
        keep = boxes.nonempty(threshold=min_box_size)
        if keep.sum().item() != len(boxes):
            boxes, scores_per_img, lvl = boxes[keep], scores_per_img[keep], lvl[keep]

        # nms_pre_nms 会添加 剩余的 boxes 的数量
        num_pre_nms += keep.sum().item()
        # 进行 nms 处理, batched_nms 会按照类别进行 nms 操作, 这里 lvl 对应了每个 anchor 的类别, 因此实际上是按照 feature map 以及 anchor_id 进行 nms 的
        keep = batched_nms(boxes.tensor, scores_per_img, lvl, nms_thresh)
        # nms_post_nms 会添加 nms 处理之后的 boxes 的数量
        num_post_nms += keep.numel()
        # In Detectron1, there was different behavior during training vs. testing.
        # (https://github.com/facebookresearch/Detectron/issues/459)
        # During training, topk is over the proposals from *all* images in the training batch.
        # During testing, it is over the proposals for each image separately.
        # As a result, the training behavior becomes batch-dependent,
        # and the configuration "POST_NMS_TOPK_TRAIN" end up relying on the batch size.
        # This bug is addressed in Detectron2 to make the behavior independent of batch size.
        # 会保留最终的 post_nms_topk 个 anchors
        keep = keep[:post_nms_topk]  # keep is already sorted
        # if keep.numel() > post_nms_topk:
        # keep = torch.multinomial(scores_per_img[keep] - scores_per_img[keep].min(), post_nms_topk)
        # 进行 result 的记录
        res = Instances(image_size)
        res.proposal_boxes = boxes[keep]
        res.objectness_logits = scores_per_img[keep]
        res.level_ids = lvl[keep]
        results.append(res)

    if training:
        # 下面是统计数据相关
        storage = get_event_storage()
        storage.put_scalar("rpn/num_proposals_preprepre_nms", num_preprepre_nms / num_images)
        storage.put_scalar("rpn/num_proposals_prepre_nms", num_prepre_nms / num_images)
        storage.put_scalar("rpn/num_proposals_pre_nms", num_pre_nms / num_images)
        storage.put_scalar("rpn/num_proposals_post_nms", num_post_nms / num_images)

    return results


def add_ground_truth_to_proposals(gt_boxes, proposals):
    """
    Call `add_ground_truth_to_proposals_single_image` for all images.

    Args:
        gt_boxes(list[Boxes]): list of N elements. Element i is a Boxes
            representing the gound-truth for image i.
        proposals (list[Instances]): list of N elements. Element i is a Instances
            representing the proposals for image i.

    Returns:
        list[Instances]: list of N Instances. Each is the proposals for the image,
            with field "proposal_boxes" and "objectness_logits".
    """
    assert gt_boxes is not None

    assert len(proposals) == len(gt_boxes)
    if len(proposals) == 0:
        return proposals

    return [
        add_ground_truth_to_proposals_single_image(gt_boxes_i, proposals_i)
        for gt_boxes_i, proposals_i in zip(gt_boxes, proposals)
    ]


def add_ground_truth_to_proposals_single_image(gt_boxes, proposals):
    """
    Augment `proposals` with ground-truth boxes from `gt_boxes`.

    Args:
        Same as `add_ground_truth_to_proposals`, but with gt_boxes and proposals
        per image.

    Returns:
        Same as `add_ground_truth_to_proposals`, but for only one image.
    """
    device = proposals.objectness_logits.device
    # Assign all ground-truth boxes an objectness logit corresponding to
    # P(object) = sigmoid(logit) =~ 1.
    gt_logit_value = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))
    gt_logits = gt_logit_value * torch.ones(len(gt_boxes), device=device)

    # Concatenating gt_boxes with proposals requires them to have the same fields
    gt_proposal = Instances(proposals.image_size)
    gt_proposal.proposal_boxes = gt_boxes
    gt_proposal.objectness_logits = gt_logits
    new_proposals = Instances.cat([proposals, gt_proposal])

    return new_proposals
