# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import json
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager

__all__ = ["load_voc_instances", "register_pascal_voc", "register_pascal_voc_wsl"]


# fmt: off
CLASS_NAMES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
)
# fmt: on


def load_voc_instances(dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]]):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    # print(1)
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    # Needs to read many small annotation files. Makes sense at local
    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        if not os.path.isfile(anno_file):
            with Image.open(jpeg_file) as img:
                width, height = img.size
            r = {"file_name": jpeg_file, "image_id": fileid, "height": height, "width": width}
            instances = []
            r["annotations"] = instances
            dicts.append(r)
            continue

        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            difficult = int(obj.find("difficult").text)
            if difficult == 1:
                continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instances.append(
                {"category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts

def load_voc_instances_wsl(dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]]):
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)
    if "07" in dirname:
        annotation_wsl = json.load(open(
            "{}/pseudo_labels/oicr_plus_voc_2007_{}.json".format(dirname, split), "r"
        ))
    elif "12" in dirname:
        annotation_wsl = json.load(open(
            "{}/pseudo_labels/oicr_plus_voc_2012_{}.json".format(dirname, split), "r"
        ))
    else:
        assert False, "Wrong dirname: {}".format(dirname)
    multi_class_labels = None
    if "multi_label" in annotation_wsl:
        multi_class_labels = annotation_wsl.pop("multi_label")
    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    dicts = []
    for fileid in fileids:
        anno = annotation_wsl[str(int(fileid))]
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")

        if not os.path.isfile(anno_file):
            with Image.open(jpeg_file) as img:
                width, height = img.size
            r = {"file_name": jpeg_file, "image_id": fileid, "height": height, "width": width}
            instances = []
            for obj in anno:
                bbox = obj["bbox"]
                bbox = [int(i) for i in bbox] # 因为 predict 出来的 bbox 是float, 要转化为 int list
                category_id = obj["category_id"] # release 版本并没有 + 1
                instances.append(
                    {
                        "category_id": category_id, "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS
                    }
                )
            r["annotations"] = instances
            if multi_class_labels is not None:
                r["multi_label"] = multi_class_labels[str(int(fileid))]
            dicts.append(r)
            continue

        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []
        # 这里是从 annotation_wsl 中进行 gt 信息的提取, 而不是 从 anno file 中提取真正的 gt 信息出来
        for obj in anno:
            bbox = obj["bbox"]
            bbox = [int(i) for i in bbox]
            category_id = obj["category_id"] 
            instances.append(
                {
                    "category_id": category_id, "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS
                }
            )
        r["annotations"] = instances
        if multi_class_labels is not None:
            r["multi_label"] = multi_class_labels[str(int(fileid))]
        dicts.append(r)
    return dicts

def register_pascal_voc(name, dirname, split, year, class_names=CLASS_NAMES):
    DatasetCatalog.register(name, lambda: load_voc_instances(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )

def register_pascal_voc_wsl(name, dirname, split, year, class_names=CLASS_NAMES):
    DatasetCatalog.register(name, lambda: load_voc_instances_wsl(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )

