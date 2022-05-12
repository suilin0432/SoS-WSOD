import logging
import numpy as np
from typing import Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY

__all__ = ["MultiInputRCNN"]


@META_ARCH_REGISTRY.register()
class MultiInputRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element,
                representing the per-channel mean and std to be used to normalize
                the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    # TODO: As MultiInputRCNN adopt multiple input images (with different transformation), visualization should be modified if you need.
    # def visualize_training(self, batched_inputs, proposals):
    #     """
    #     A function used to visualize images and proposals. It shows ground truth
    #     bounding boxes on the original image and up to 20 predicted object
    #     proposals on the original image. Users can implement different
    #     visualization functions for different models.

    #     Args:
    #         batched_inputs (list): a list that contains input to the model.
    #         proposals (list): a list that contains predicted proposals. Both
    #             batched_inputs and proposals should have the same length.
    #     """
    #     from detectron2.utils.visualizer import Visualizer

    #     storage = get_event_storage()
    #     max_vis_prop = 20

    #     for input, prop in zip(batched_inputs, proposals):
    #         img = input["image"]
    #         img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
    #         v_gt = Visualizer(img, None)
    #         v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
    #         anno_img = v_gt.get_image()
    #         box_size = min(len(prop.proposal_boxes), max_vis_prop)
    #         v_pred = Visualizer(img, None)
    #         v_pred = v_pred.overlay_instances(
    #             boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
    #         )
    #         prop_img = v_pred.get_image()
    #         vis_img = np.concatenate((anno_img, prop_img), axis=1)
    #         vis_img = vis_img.transpose(2, 0, 1)
    #         vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
    #         storage.put_image(vis_name, vis_img)
    #         break  # only visualize one image in a batch

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        assert len(batched_inputs) == 1, "now, MultiInputRCNN only support the setting -> imgs_per_gpu=1"
        if not self.training:
            return self.inference(batched_inputs)

        images1, images2, images1_flip, images2_flip = self.preprocess_image(batched_inputs)
        images1_input = torch.cat([images1.tensor, images1_flip.tensor], 0)
        images2_input = torch.cat([images2.tensor, images2_flip.tensor], 0)

        if "instances1" in batched_inputs[0]:
            gt_instances1 = [x["instances1"].to(self.device) for x in batched_inputs]
        else:
            gt_instances1 = None
        if "instances2" in batched_inputs[0]:
            gt_instances2 = [x["instances2"].to(self.device) for x in batched_inputs]
        else:
            gt_instances2 = None
        if "instances1_flip" in batched_inputs[0]:
            gt_instances1_flip = [x["instances1_flip"].to(self.device) for x in batched_inputs]
        else:
            gt_instances1_flip = None
        if "instances2_flip" in batched_inputs[0]:
            gt_instances2_flip = [x["instances2_flip"].to(self.device) for x in batched_inputs]
        else:
            gt_instances2_flip = None

        # 每个 features 对应于一个 scale 的原始输入和 flip 之后的输入 的 feature (两者是相同大小的)
        features1 = self.backbone(images1_input)
        features2 = self.backbone(images2_input)

        # 不考虑使用 rpn
        # if self.proposal_generator:
        #     proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # else:
        assert ("proposals1" in batched_inputs[0]) and ("proposals1_flip" in batched_inputs[0]) and ("proposals2" in batched_inputs[0]) and ("proposals2_flip" in batched_inputs[0])
        proposals1 = [x["proposals1"].to(self.device) for x in batched_inputs]
        proposals2 = [x["proposals2"].to(self.device) for x in batched_inputs]
        proposals1_flip = [x["proposals1_flip"].to(self.device) for x in batched_inputs]
        proposals2_flip = [x["proposals2_flip"].to(self.device) for x in batched_inputs]

        images_list = [
            images1, images1_flip, images2, images2_flip
        ]
        features_list = [
            features1, features2
        ]
        proposals_list = [
            proposals1, proposals1_flip, proposals2, proposals2_flip
        ]
        gt_instances_list = [
            gt_instances1, gt_instances1_flip, gt_instances2, gt_instances2_flip
        ]
        _, detector_losses = self.roi_heads(images_list, features_list, proposals_list, gt_instances_list)
        # if self.vis_period > 0:
        #     storage = get_event_storage()
        #     if storage.iter % self.vis_period == 0:
        #         self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        # losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image_inference(batched_inputs)
        features = self.backbone(images.tensor)
        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            targets = None
            if "instances" in batched_inputs[0]:
                targets = [x["instances"].to(self.device) for x in batched_inputs]
            # image_id = None
            # if "image_id" in batched_inputs[0]:
            #     image_ids = [x["image_id"] for x in batched_inputs]
            results, _, all_scores, all_boxes = self.roi_heads(images, features, proposals, targets)
            # print(all_scores)
            # print(all_scores.shape)
            # exit()
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results, all_scores, all_boxes = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            return MultiInputRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results, all_scores, all_boxes

    def preprocess_image(self, batched_inputs):
        images1 = [x["image1"].to(self.device) for x in batched_inputs]
        images1 = [(x - self.pixel_mean) / self.pixel_std for x in images1]
        images1 = ImageList.from_tensors(images1, self.backbone.size_divisibility)
        images2 = [x["image2"].to(self.device) for x in batched_inputs]
        images2 = [(x - self.pixel_mean) / self.pixel_std for x in images2]
        images2 = ImageList.from_tensors(images2, self.backbone.size_divisibility)
        images1_flip = [x["image1_flip"].to(self.device) for x in batched_inputs]
        images1_flip = [(x - self.pixel_mean) / self.pixel_std for x in images1_flip]
        images1_flip = ImageList.from_tensors(images1_flip, self.backbone.size_divisibility)
        images2_flip = [x["image2_flip"].to(self.device) for x in batched_inputs]
        images2_flip = [(x - self.pixel_mean) / self.pixel_std for x in images2_flip]
        images2_flip = ImageList.from_tensors(images2_flip, self.backbone.size_divisibility)
        return images1, images2, images1_flip, images2_flip

    def preprocess_image_inference(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results
