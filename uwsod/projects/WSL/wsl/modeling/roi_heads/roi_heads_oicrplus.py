from time import time
import inspect
import logging
import numpy as np
import copy
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, batched_nms
# from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.keypoint_head import build_keypoint_head
from detectron2.modeling.roi_heads.mask_head import build_mask_head
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

from wsl.modeling.poolers import ROIPooler
from wsl.modeling.roi_heads.fast_rcnn_oicr import OICROutputLayers
from wsl.modeling.roi_heads.fast_rcnn_wsddn import WSDDNOutputLayers
from wsl.modeling.roi_heads.roi_heads import (
    ROIHeads,
    get_image_level_gt,
    select_foreground_proposals,
    select_proposals_with_visible_keypoints,
)

logger = logging.getLogger(__name__)
@ROI_HEADS_REGISTRY.register()
class OICRPlusHeads(ROIHeads):
    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        vis_period: int = 0,
        refine_K: int = 4,
        refine_mist: bool = False,
        mist_p: float = 0.10,
        mist_thre: float = 0.05,
        mist_type: str = "nms",
        refine_reg: List[bool] = [False, False, False, False],
        box_refinery: List[nn.Module] = [None, None, None, None],
        cls_agnostic_bbox_reg: bool = False,
        pooler_type: str = "ROIPool",
        cfg = None,
        **kwargs 
    ):
        """
            NOTE: this interface is experimental.
        """
        super().__init__(**kwargs)
        assert mist_type in ["nms", "wetectron"], f"{mist_type} is wrong"
        self.mist_type = mist_type
        self.mist_p = mist_p
        self.mist_thre = mist_thre
        self.cfg = cfg
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor
        self.pooler_type = pooler_type

        self.iter = 0
        self.iter_test = 0    

        self.vis_period = vis_period

        self.refine_K = refine_K
        self.refine_mist = refine_mist
        self.refine_reg = refine_reg
        self.box_refinery = box_refinery
        for k in range(self.refine_K):
            self.add_module("box_refinery_{}".format(k), self.box_refinery[k])
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
    
    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        ret["cfg"] = cfg
        ret["pooler_type"] = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        ret["mist_type"] = cfg.WSL.MIST_TYPE
        ret["mist_p"] = cfg.WSL.MIST_P
        ret["mist_thre"] = cfg.WSL.MIST_THRE
        ret.update(
            cls._init_box_head(cfg, input_shape)
        )
        return ret
    
    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        in_channels = [input_shape[f].channels for f in in_features]
        in_channels = in_channels[-1]

        pooler_scale = tuple(i for i in [pooler_scales[-1]])
        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scale,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        box_predictor = WSDDNOutputLayers(cfg, box_head.output_shape)

        cls_agnostic_bbox_reg = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        refine_K = cfg.WSL.REFINE_NUM
        refine_mist = cfg.WSL.REFINE_MIST
        refine_reg = cfg.WSL.REFINE_REG

        box_refinery = []
        for k in range(refine_K):
            box_refinery_k = OICROutputLayers(cfg, box_head.output_shape, k)
            box_refinery.append(box_refinery_k)
        
        vis_period = cfg.VIS_PERIOD
        
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
            "vis_period": vis_period,
            "refine_K": refine_K,
            "refine_mist": refine_mist,
            "refine_reg": refine_reg,
            "box_refinery": box_refinery,
            "cls_agnostic_bbox_reg": cls_agnostic_bbox_reg,
        }
    
    def forward(
        self,
        images_list: List[ImageList],
        features_list: List[Dict[str, torch.Tensor]],
        proposals_list: List[List[Instances]],
        targets_list: List[Optional[List[Instances]]] = [None, None, None, None],
    ):
        """
            args:
                images_list: [images1, images1_flip, images2, images2_flip]
                features_list: [features1, features2]
                proposals_list: [proposals1, proposals1_flip, proposals2, proposals2_flip]
                targets_list: [gt_instances1, gt_instances1_flip, gt_instances2, gt_instances2_flip]
        """

        if not self.training:
            pred_instances, all_scores, all_boxes = self._forward_box_test(features_list, proposals_list, targets_list)
            # self.iter_test = self.iter_test + 1
            return pred_instances, {}, all_scores, all_boxes
        images1, images1_flip, images2, images2_flip = images_list
        features1, features2 = features_list
        proposals1, proposals1_flip, proposals2, proposals2_flip = proposals_list
        targets1, targets1_flip, targets2, targets2_flip = targets_list


        self.gt_classes_img, self.gt_classes_img_int, self.gt_classes_img_oh = get_image_level_gt(
            targets1, self.num_classes
        )


        features1 = [features1[f] for f in self.box_in_features]
        features2 = [features2[f] for f in self.box_in_features]
        features1_flip = [f[1].unsqueeze(0) for f in features1]
        features1 = [f[0].unsqueeze(0) for f in features1]
        features2_flip = [f[1].unsqueeze(0) for f in features2]
        features2 = [f[0].unsqueeze(0) for f in features2]
        losses = self._forward_box(features1, features1_flip, features2, features2_flip, proposals1, proposals1_flip, proposals2, proposals2_flip)
        self.iter = self.iter + 1
        
        return None, losses
    
    def _forward_box(
        self, features1, features1_flip, features2, features2_flip, proposals1, proposals1_flip, proposals2, proposals2_flip
    ):
        assert len(proposals1) == len(proposals1_flip) == len(proposals2) == len(proposals2_flip) == 1, "the batchsize should be 1"
        
        box_features1 = self.box_pooler(features1, [x.proposal_boxes for x in proposals1])
        box_features1_flip = self.box_pooler(features1_flip, [x.proposal_boxes for x in proposals1_flip])
        box_features2 = self.box_pooler(features2, [x.proposal_boxes for x in proposals2])
        box_features2_flip = self.box_pooler(features2_flip, [x.proposal_boxes for x in proposals2_flip])

        objectness_logits1 = torch.cat([x.objectness_logits + 1 for x in proposals1], dim=0)
        objectness_logits1_flip = torch.cat([x.objectness_logits + 1 for x in proposals1_flip], dim=0)
        objectness_logits2 = torch.cat([x.objectness_logits + 1 for x in proposals2], dim=0)
        objectness_logits2_flip = torch.cat([x.objectness_logits + 1 for x in proposals2_flip], dim=0)
        if self.pooler_type == "ROILoopPool":
            objectness_logits1 = torch.cat(
                [objectness_logits1, objectness_logits1, objectness_logits1], dim = 0
            )
            objectness_logits1_flip = torch.cat(
                [objectness_logits1_flip, objectness_logits1_flip, objectness_logits1_flip], dim = 0
            )
            objectness_logits2 = torch.cat(
                [objectness_logits2, objectness_logits2, objectness_logits2], dim = 0
            )
            objectness_logits2_flip = torch.cat(
                [objectness_logits2_flip, objectness_logits2_flip, objectness_logits2_flip], dim = 0
            )

        box_features1_new = box_features1 * objectness_logits1.view(-1, 1, 1, 1)
        box_features1_flip_new = box_features1_flip * objectness_logits1_flip.view(-1, 1, 1, 1)
        box_features2_new = box_features2 * objectness_logits2.view(-1, 1, 1, 1)
        box_features2_flip_new = box_features2_flip * objectness_logits2_flip.view(-1, 1, 1, 1)

        pool5_roi1 = box_features1_new.clone().detach()
        pool5_roi1_flip = box_features1_flip_new.clone().detach()
        pool5_roi2 = box_features2_new.clone().detach()
        pool5_roi2_flip = box_features2_flip_new.clone().detach()


        box_features_new1 = self.box_head(box_features1_new)
        box_features_new1_flip = self.box_head(box_features1_flip_new)
        box_features_new2 = self.box_head(box_features2_new)
        box_features_new2_flip = self.box_head(box_features2_flip_new)


        if self.pooler_type == "ROILoopPool":
            pool5_roi1, _, _ = torch.chunk(
                pool5_roi1, 3, dim=0
            )
            pool5_roi1_flip, _, _ = torch.chunk(
                pool5_roi1_flip, 3, dim=0
            )
            pool5_roi2, _, _ = torch.chunk(
                pool5_roi2, 3, dim=0
            )
            pool5_roi2_flip, _, _ = torch.chunk(
                pool5_roi2_flip, 3, dim=0
            )


            box_features_new1, box_features_frame1, box_features_context1 = torch.chunk(
                box_features_new1, 3, dim=0
            )
            box_features_new1_flip, box_features_frame1_flip, box_features_context1_flip = torch.chunk(
                box_features_new1_flip, 3, dim=0
            )
            box_features_new2, box_features_frame2, box_features_context2 = torch.chunk(
                box_features_new2, 3, dim=0
            )
            box_features_new2_flip, box_features_frame2_flip, box_features_context2_flip = torch.chunk(
                box_features_new2_flip, 3, dim=0
            )

            predictions1 = self.box_predictor(
                [box_features_new1, box_features_frame1, box_features_context1], proposals1, context=True
            )
            predictions1_flip = self.box_predictor(
                [box_features_new1_flip, box_features_frame1_flip, box_features_context1_flip], proposals1_flip, context=True
            )
            predictions2 = self.box_predictor(
                [box_features_new2, box_features_frame2, box_features_context2], proposals2, context=True
            )
            predictions2_flip = self.box_predictor(
                [box_features_new2_flip, box_features_frame2_flip, box_features_context2_flip], proposals2_flip, context=True
            )

        else:
            predictions1 = self.box_predictor(box_features_new1, proposals1)
            predictions1_flip = self.box_predictor(box_features_new1_flip, proposals1_flip)
            predictions2 = self.box_predictor(box_features_new2, proposals2)
            predictions2_flip = self.box_predictor(box_features_new2_flip, proposals2_flip)


        mil_loss1 = self.box_predictor.losses(predictions1, proposals1, self.gt_classes_img_oh)
        mil_loss1_flip = self.box_predictor.losses(predictions1_flip, proposals1_flip, self.gt_classes_img_oh)
        mil_loss2 = self.box_predictor.losses(predictions2, proposals2, self.gt_classes_img_oh)
        mil_loss2_flip = self.box_predictor.losses(predictions2_flip, proposals2_flip, self.gt_classes_img_oh)
        losses = dict()
        losses["loss_cls"] = (mil_loss1["loss_cls"] + mil_loss1_flip["loss_cls"] + mil_loss2["loss_cls"] + mil_loss2_flip["loss_cls"]) / 4.0

        prev_pred_scores1 = predictions1[0].detach()
        prev_pred_scores1_flip = predictions1_flip[0].detach()
        prev_pred_scores2 = predictions2[0].detach()
        prev_pred_scores2_flip = predictions2_flip[0].detach()
        prev_pred_scores = (prev_pred_scores1 + prev_pred_scores1_flip + prev_pred_scores2 + prev_pred_scores2_flip) / 4.0
    
        prev_pred_boxes1 = [p.proposal_boxes for p in proposals1]
        prev_pred_boxes1_flip = [p.proposal_boxes for p in proposals1_flip]
        prev_pred_boxes2 = [p.proposal_boxes for p in proposals2]
        prev_pred_boxes2_flip = [p.proposal_boxes for p in proposals2_flip]


        for k in range(self.refine_K):
            suffix = "_r" + str(k)
            term_weight = 1
            if k == 0 and (not self.refine_mist) and (not self.cfg.WSL.REFINE_REG[0]):
                term_weight = 3
            if self.refine_mist:
                if self.mist_type == "wetectron":
                    targets1 = self.get_pgt_mist_mist(
                        prev_pred_boxes1, prev_pred_scores, proposals1, top_pro=self.mist_p, thres=self.mist_thre, suffix=suffix
                    )
                elif self.mist_type == "nms":
                    targets1 = self.get_pgt_mist(
                        prev_pred_boxes1, prev_pred_scores, proposals1, top_pro=self.mist_p, thres=self.mist_thre, suffix=suffix
                    )
                else:
                    assert False, "The mist_type is not wetectron or oicrplus!"
                if k == 0:
                    term_weight = 1
            else:
                targets1 = self.get_pgt_top_k(
                    prev_pred_boxes1, prev_pred_scores, proposals1, suffix=suffix
                )

            
            proposals_k1 = self.label_and_sample_proposals(proposals1, targets1, suffix=suffix)
            proposals_k1_flip = [
                Instances(
                    proposal.image_size,
                    proposal_boxes = proposal.proposal_boxes,
                    objectness_logits = proposal.objectness_logits,
                    gt_boxes = proposal.proposal_boxes.clone()[proposal_gt.gt_index],
                    gt_classes = proposal_gt.gt_classes.clone(),
                    gt_scores = proposal_gt.gt_scores.clone(),
                    gt_weights = proposal_gt.gt_weights.clone(),
                    gt_index = proposal_gt.gt_index.clone()
                )
                for i, (proposal_gt, proposal) in enumerate(
                    zip(proposals_k1, proposals1_flip)
                )
            ]
            proposals_k2 = [
                Instances(
                    proposal.image_size,
                    proposal_boxes = proposal.proposal_boxes,
                    objectness_logits = proposal.objectness_logits,
                    gt_boxes = proposal.proposal_boxes.clone()[proposal_gt.gt_index],
                    gt_classes = proposal_gt.gt_classes.clone(),
                    gt_scores = proposal_gt.gt_scores.clone(),
                    gt_weights = proposal_gt.gt_weights.clone(),
                    gt_index = proposal_gt.gt_index.clone()
                )
                for i, (proposal_gt, proposal) in enumerate(
                    zip(proposals_k1, proposals2)
                )
            ]
            proposals_k2_flip = [
                Instances(
                    proposal.image_size,
                    proposal_boxes = proposal.proposal_boxes,
                    objectness_logits = proposal.objectness_logits,
                    gt_boxes = proposal.proposal_boxes.clone()[proposal_gt.gt_index],
                    gt_classes = proposal_gt.gt_classes.clone(),
                    gt_scores = proposal_gt.gt_scores.clone(),
                    gt_weights = proposal_gt.gt_weights.clone(),
                    gt_index = proposal_gt.gt_index.clone()
                )
                for i, (proposal_gt, proposal) in enumerate(
                    zip(proposals_k1, proposals2_flip)
                )
            ]

            predictions_k1 = self.box_refinery[k](box_features_new1)
            predictions_k1_flip = self.box_refinery[k](box_features_new1_flip)
            predictions_k2 = self.box_refinery[k](box_features_new2)
            predictions_k2_flip = self.box_refinery[k](box_features_new2_flip)

            losses_k1 = self.box_refinery[k].losses(predictions_k1, proposals_k1)
            losses_k1_flip = self.box_refinery[k].losses(predictions_k1_flip, proposals_k1_flip)
            losses_k2 = self.box_refinery[k].losses(predictions_k2, proposals_k2)
            losses_k2_flip = self.box_refinery[k].losses(predictions_k2, proposals_k2_flip)


            losses_oicr = {}
            for key in losses_k1.keys():
                losses_oicr[key] = (losses_k1[key] + losses_k1_flip[key] + losses_k2[key] + losses_k2_flip[key]) / 4.0
            for key in losses_oicr.keys():
                losses_oicr[key] = losses_oicr[key] * term_weight
            
            prev_pred_scores1 = self.box_refinery[k].predict_probs(predictions_k1, proposals_k1)
            prev_pred_scores1_flip = self.box_refinery[k].predict_probs(predictions_k1_flip, proposals_k1_flip)
            prev_pred_scores2 = self.box_refinery[k].predict_probs(predictions_k2, proposals_k2)
            prev_pred_scores2_flip = self.box_refinery[k].predict_probs(predictions_k2_flip, proposals_k2_flip)
            prev_pred_scores = [(prev_pred_scores1[i]+prev_pred_scores1_flip[i]+prev_pred_scores2[i]+prev_pred_scores2_flip[i])/4.0 for i in range(len(prev_pred_scores1))]
            prev_pred_scores = [score.detach() for score in prev_pred_scores]
            
            
            if self.cfg.OICRPLUS.BBOX_UPDATE:
                N, B = proposals_k1[0].proposal_boxes.tensor.shape
                K = predictions_k1[1].shape[1] // B
                delta1 = predictions_k1[1].view(N, K, B)
                delta1_flip = predictions_k1_flip[1].view(N, K, B)
                delta2 = predictions_k2[1].view(N, K, B)
                delta2_flip = predictions_k2_flip[1].view(N, K, B)
                # delta 是 [dx, dy, dw, dh] 的形式
                delta = torch.zeros(delta1.shape)
                delta[:, :, 0] = (delta1[:, :, 0] - delta1_flip[:, :, 0] + delta2[:, :, 0] - delta2_flip[:, :, 0])/4
                delta[:, :, 1] = (delta1[:, :, 1] + delta1_flip[:, :, 1] + delta2[:, :, 1] + delta2_flip[:, :, 1])/4
                delta[:, :, 2] = (delta1[:, :, 2] + delta1_flip[:, :, 2] + delta2[:, :, 2] + delta2_flip[:, :, 2])/4
                delta[:, :, 3] = (delta1[:, :, 3] + delta1_flip[:, :, 3] + delta2[:, :, 3] + delta2_flip[:, :, 3])/4
                
                delta2 = delta.clone()
                delta2[:, :, 0] = -delta2[:, :, 0]
                
                delta = delta.view(predictions_k1[1].shape).cuda()
                delta2 = delta2.view(predictions_k1[1].shape).cuda()

                prev_pred_boxes1 = self.box_refinery[k].predict_boxes((None, delta), proposals_k1)
                prev_pred_boxes1_flip = self.box_refinery[k].predict_boxes((None, delta2), proposals_k1_flip)
                prev_pred_boxes2 = self.box_refinery[k].predict_boxes((None, delta), proposals_k2)
                prev_pred_boxes2_flip = self.box_refinery[k].predict_boxes((None, delta2), proposals_k2_flip)
                prev_pred_boxes1 = [box.detach() for box in prev_pred_boxes1]
                prev_pred_boxes1_flip = [box.detach() for box in prev_pred_boxes1_flip]
                prev_pred_boxes2 = [box.detach() for box in prev_pred_boxes2]
                prev_pred_boxes2_flip = [box.detach() for box in prev_pred_boxes2_flip] 


            losses.update(losses_oicr)

        return losses

    def _forward_box_test(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances], targets_list=None
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        if len(self.box_in_features) > 1:
            box_f = [self.box_in_features[-1]]
        else:
            box_f = self.box_in_features
        features = [features[f] for f in box_f]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        objectness_logits = torch.cat([x.objectness_logits + 1 for x in proposals], dim=0)
        if self.pooler_type == "ROILoopPool":
            objectness_logits = torch.cat(
                [objectness_logits, objectness_logits, objectness_logits], dim=0
            )
        box_features = box_features * objectness_logits.view(-1, 1, 1, 1)
        box_features = self.box_head(box_features)
        if self.pooler_type == "ROILoopPool":
            # ROILoopPool好像对应的是 ContextLocNet 的三个 pool
            box_features, box_features_frame, box_features_context = torch.chunk(
                box_features, 3, dim=0
            )
            predictions = self.box_predictor(
                [box_features, box_features_frame, box_features_context], proposals, context=True
            )
            
            del box_features_frame
            del box_features_context
        else:
            predictions = self.box_predictor(box_features, proposals)
        if self.refine_reg[-1] and False:
            predictions_k = self.box_refinery[-1](box_features)
            pred_instances, _, all_scores, all_boxes = self.box_refinery[-1].inference(
                predictions_k, proposals
            )
        else:
            predictions_K = []
            for k in range(self.refine_K):
                predictions_k = self.box_refinery[k](box_features)
                predictions_K.append(predictions_k)

            pred_instances, _, all_scores, all_boxes = self.box_refinery[-1].inference(
                predictions_K, proposals
            )
        return pred_instances, all_scores, all_boxes

    


    @torch.no_grad()
    def get_pgt_mist_mist(self, prev_pred_boxes, prev_pred_scores, proposals, top_pro=0.15, thres=0.01, suffix=""):
        iou_thre = 0.2
        score_thre = thres
        pgt_scores, pgt_boxes, pgt_classes, pgt_weights = self.get_pgt_top_k(
            prev_pred_boxes,
            prev_pred_scores,
            proposals,
            top_k=top_pro,
            thres=0.00,
            need_instance=False,
            need_weight=True,
            suffix=suffix,
        )
        def iou_cal(boxes):
            assert len(boxes.shape) == 3 and boxes.shape[2] == 4
            num_gt_cls, topk, _ = boxes.shape
            iou_matrix = torch.zeros(
                num_gt_cls, topk, topk
            )
            for i in range(num_gt_cls):
                box1 = boxes[i].clone().detach()
                box2 = boxes[i].clone().detach()
                area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
                width_height = torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:,None,:2], box2[:, :2])
                width_height.clamp_(min=0)
                inter = width_height.prod(dim=2)
                iou_matrix[i] = torch.where(
                    inter > 0,
                    inter / (area[:, None] + area - inter),
                    torch.zeros(1, dtype=inter.dtype, device=inter.device)
                )
            return iou_matrix
        index = 0
        for pgt_box, pgt_score, pgt_class, pgt_weight, gt_int in zip(pgt_boxes, pgt_scores, pgt_classes, pgt_weights, self.gt_classes_img_int):
            num_gt_cls = len(gt_int)
            num_seed = pgt_box.shape[0]
            assert num_seed % num_gt_cls == 0
            num_topk = num_seed // num_gt_cls
            keep = torch.zeros(num_gt_cls, num_topk, dtype=torch.bool, device=pgt_weight.device)

            pgt_box = pgt_box.view(num_topk, num_gt_cls, 4).permute(1,0,2).contiguous()
            pgt_score = pgt_score.view(num_topk, num_gt_cls).t().contiguous()
            pgt_class = pgt_class.view(num_topk, num_gt_cls).t().contiguous()
            pgt_weight = pgt_weight.view(num_topk, num_gt_cls).t().contiguous()

            iou_matrix = iou_cal(pgt_box)
            keep[:, 0] = 1
            for ii in range(1, num_topk):
                max_iou, _ = torch.max(iou_matrix[:, ii:ii+1, :ii], dim=2)
                keep[:, ii] = (max_iou < iou_thre).byte().squeeze(-1)
            score_mask = (pgt_score >= score_thre)
            keep = torch.logical_and(keep, score_mask)
            keep[:, 0] = 1
            pgt_boxes[index] = pgt_box[keep]
            pgt_scores[index] = pgt_score[keep]
            pgt_classes[index] = pgt_class[keep]
            pgt_weights[index] = pgt_weight[keep]
            index += 1

        pgt_boxes = [Boxes(pgt_box) for pgt_box in pgt_boxes]

        targets = [
            Instances(
                proposals[i].image_size,
                gt_boxes=pgt_box,
                gt_classes=pgt_class,
                gt_scores=pgt_score,
                gt_weights=pgt_weight,
            )
            for i, (pgt_box, pgt_class, pgt_score, pgt_weight) in enumerate(
                zip(pgt_boxes, pgt_classes, pgt_scores, pgt_scores)
            )
        ]

        self._vis_pgt(targets, "pgt_mist", suffix)

        return targets


    @torch.no_grad()
    def get_pgt_mist(self, prev_pred_boxes, prev_pred_scores, proposals, top_pro=0.10, thres=0.05, suffix=""):
        pgt_scores, pgt_boxes, pgt_classes, pgt_weights, pgt_idxs_save = self.get_pgt_top_k(
            prev_pred_boxes,
            prev_pred_scores,
            proposals,
            top_k=top_pro,
            thres=thres,
            # thres=0.05,
            # thres=0.0,
            need_instance=False,
            need_weight=True,
            suffix=suffix,
        )

        # NMS
        pgt_idxs = [torch.zeros_like(pgt_class) for pgt_class in pgt_classes]
        keeps = [
            # batched_nms(pgt_box, pgt_score, pgt_class, 0.2)
            batched_nms(pgt_box, pgt_score, pgt_class, 0.01)
            for pgt_box, pgt_score, pgt_class in zip(pgt_boxes, pgt_scores, pgt_idxs)
        ]
        pgt_scores = [pgt_score[keep] for pgt_score, keep in zip(pgt_scores, keeps)]
        pgt_boxes = [pgt_box[keep] for pgt_box, keep in zip(pgt_boxes, keeps)]
        pgt_classes = [pgt_class[keep] for pgt_class, keep in zip(pgt_classes, keeps)]
        pgt_weights = [pgt_weight[keep] for pgt_weight, keep in zip(pgt_weights, keeps)]
        pgt_idxs = [pgt_idx[keep] for pgt_idx, keep in zip(pgt_idxs_save, keeps)]
    
        pgt_boxes = [Boxes(pgt_box) for pgt_box in pgt_boxes]

        targets = [
            Instances(
                proposals[i].image_size,
                gt_boxes=pgt_box,
                gt_classes=pgt_class,
                gt_scores=pgt_score,
                gt_weights=pgt_weight,
                gt_index=pgt_idx
            )
            for i, (pgt_box, pgt_class, pgt_score, pgt_weight, pgt_idx) in enumerate(
                zip(pgt_boxes, pgt_classes, pgt_scores, pgt_scores, pgt_idxs)
            )
        ]
        

        return targets

    @torch.no_grad()
    def get_pgt_top_k(
        self,
        prev_pred_boxes,
        prev_pred_scores,
        proposals,
        top_k=1,
        thres=0,
        need_instance=True,
        need_weight=True,
        suffix="",
        oicr_w=True
    ):
        assert isinstance(prev_pred_boxes, tuple) or isinstance(prev_pred_boxes, list)
        if isinstance(prev_pred_boxes[0], Boxes):
            num_preds = [len(prev_pred_box) for prev_pred_box in prev_pred_boxes]
            prev_pred_boxes = [
                prev_pred_box.tensor.unsqueeze(1).expand(num_pred, self.num_classes, 4)
                for num_pred, prev_pred_box in zip(num_preds, prev_pred_boxes)
            ]
        else:
            assert isinstance(prev_pred_boxes[0], torch.Tensor)
            if self.cls_agnostic_bbox_reg:
                num_preds = [prev_pred_box.size(0) for prev_pred_box in prev_pred_boxes]
                prev_pred_boxes = [
                    prev_pred_box.unsqueeze(1).expand(num_pred, self.num_classes, 4)
                    for num_pred, prev_pred_box in zip(num_preds, prev_pred_boxes)
                ]
        prev_pred_boxes = [
            prev_pred_box.view(-1, self.num_classes, 4) for prev_pred_box in prev_pred_boxes
        ]

        if isinstance(prev_pred_scores, torch.Tensor):
            num_preds_per_image = [len(p) for p in proposals]
            prev_pred_scores = prev_pred_scores.split(num_preds_per_image, dim=0)
        else:
            assert isinstance(prev_pred_scores, list)
            assert isinstance(prev_pred_scores[0], torch.Tensor)

        prev_pred_scores = [
            torch.index_select(prev_pred_score, 1, gt_int)
            for prev_pred_score, gt_int in zip(prev_pred_scores, self.gt_classes_img_int)
        ]
        prev_pred_boxes = [
            torch.index_select(prev_pred_box, 1, gt_int)
            for prev_pred_box, gt_int in zip(prev_pred_boxes, self.gt_classes_img_int)
        ]

        # get top k
        num_preds = [prev_pred_score.size(0) for prev_pred_score in prev_pred_scores]
        if top_k >= 1:
            top_ks = [min(num_pred, int(top_k)) for num_pred in num_preds]
        elif top_k < 1 and top_k > 0:
            top_ks = [max(int(num_pred * top_k), 1) for num_pred in num_preds]
        else:
            top_ks = [min(num_pred, 1) for num_pred in num_preds]
        pgt_scores_idxs = [
            torch.topk(prev_pred_score, top_k, dim=0)
            for prev_pred_score, top_k in zip(prev_pred_scores, top_ks)
        ]
        pgt_scores = [item[0] for item in pgt_scores_idxs]
        pgt_idxs = [item[1] for item in pgt_scores_idxs]
        pgt_idxs_save = [item[1].clone() for item in pgt_scores_idxs]

        pgt_idxs = [
            torch.unsqueeze(pgt_idx, 2).expand(top_k, gt_int.numel(), 4)
            for pgt_idx, top_k, gt_int in zip(pgt_idxs, top_ks, self.gt_classes_img_int)
        ]
        pgt_boxes = [
            torch.gather(prev_pred_box, 0, pgt_idx)
            for prev_pred_box, pgt_idx in zip(prev_pred_boxes, pgt_idxs)
        ]
        pgt_classes = [
            torch.unsqueeze(gt_int, 0).expand(top_k, gt_int.numel())
            for gt_int, top_k in zip(self.gt_classes_img_int, top_ks)
        ]
        if need_weight:
            if oicr_w:
                pgt_weights = [
                    pgt_score.clone().detach()
                    for pgt_score in pgt_scores
                ]
            else:
                assert False
                pgt_weights = [
                    torch.index_select(pred_logits, 1, gt_int).expand(top_k, gt_int.numel())
                    for pred_logits, gt_int, top_k in zip(
                        self.pred_class_img_logits.split(1, dim=0), self.gt_classes_img_int, top_ks
                    )
                ]

        if thres > 0:
            # get large scores
            masks = [pgt_score.ge(thres) for pgt_score in pgt_scores]
            masks = [
                torch.cat([torch.full_like(mask[0:1, :], True), mask[1:, :]], dim=0)
                for mask in masks
            ]
            pgt_idxs_save = [
                torch.masked_select(pgt_idx, mask) for pgt_idx, mask in zip(pgt_idxs_save, masks)
            ]
            pgt_scores = [
                torch.masked_select(pgt_score, mask) for pgt_score, mask in zip(pgt_scores, masks)
            ]
            pgt_boxes = [
                torch.masked_select(
                    pgt_box, torch.unsqueeze(mask, 2).expand(top_k, gt_int.numel(), 4)
                )
                for pgt_box, mask, top_k, gt_int in zip(
                    pgt_boxes, masks, top_ks, self.gt_classes_img_int
                )
            ]
            pgt_classes = [
                torch.masked_select(pgt_class, mask) for pgt_class, mask in zip(pgt_classes, masks)
            ]
            if need_weight:
                pgt_weights = [
                    torch.masked_select(pgt_weight, mask)
                    for pgt_weight, mask in zip(pgt_weights, masks)
                ]

        pgt_scores = [pgt_score.reshape(-1) for pgt_score in pgt_scores]
        pgt_idxs_save = [pgt_idx.reshape(-1) for pgt_idx in pgt_idxs_save]
        pgt_boxes = [pgt_box.reshape(-1, 4) for pgt_box in pgt_boxes]
        pgt_classes = [pgt_class.reshape(-1) for pgt_class in pgt_classes]
        if need_weight:
            pgt_weights = [pgt_weight.reshape(-1) for pgt_weight in pgt_weights]

        if not need_instance and need_weight:
            return pgt_scores, pgt_boxes, pgt_classes, pgt_weights, pgt_idxs_save
        elif not need_instance and not need_weight:
            assert False, "could not reach this branch"
            return pgt_scores, pgt_boxes, pgt_classes

        pgt_boxes = [Boxes(pgt_box) for pgt_box in pgt_boxes]

        targets = [
            Instances(
                proposals[i].image_size,
                gt_boxes=pgt_box,
                gt_classes=pgt_class,
                gt_scores=pgt_score,
                gt_weights=pgt_weight,
                gt_index=pgt_idx
            )
            for i, (pgt_box, pgt_class, pgt_score, pgt_weight, pgt_idx) in enumerate(
                zip(pgt_boxes, pgt_classes, pgt_scores, pgt_weights, pgt_idxs_save)
            )
        ]

        return targets
