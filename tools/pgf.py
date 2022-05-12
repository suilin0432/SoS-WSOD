import json
from detectron2.data import build_detection_test_loader, get_detection_dataset_dicts
from detectron2.config import get_cfg
import torch
import numpy as np
import argparse
import copy
import os
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser("Perform PGF.")
    parser.add_argument("--det-path", default='uwsod/datasets/VOC2007/detection_results/')
    parser.add_argument("--save-path", default='uwsod/datasets/VOC2007/pseudo_labels/')
    parser.add_argument("--prefix", default='oicr_plus_')
    parser.add_argument("--dataset", default='voc2007', choices=('voc2007', 'voc2012', 'coco'))
    parser.add_argument("--coco-path", default="uwsod/datasets/coco/")
    parser.add_argument("--t-con", default=0.85)
    parser.add_argument("--t-keep", default=0.2)
    parser.add_argument("--use-diff", action="store_true")
    args = parser.parse_args()
    return args

def pgf_voc(det_path, save_path, prefix, t_con, t_keep, use_diff, year):
    print("loading voc datasets...")
    os.chdir("uwsod/")
    trainset = get_detection_dataset_dicts((f'voc_{year}_train',))
    valset = get_detection_dataset_dicts((f'voc_{year}_val',))
    os.chdir("../")

    print("loading voc detection results...")
    train_detection_result = json.load(open(f"{det_path}/{prefix}voc_{year}_train.json"))
    val_detection_result = json.load(open(f"{det_path}/{prefix}voc_{year}_val.json"))

    # image_id 2 anns
    train_gt_anns = {}
    val_gt_anns = {}
    for i in range(len(trainset)):
        message = trainset[i]
        image_id = int(message["image_id"])
        train_gt_anns[image_id] = message["annotations"]
    for i in range(len(valset)):
        message = valset[i]
        image_id = int(message["image_id"])
        val_gt_anns[image_id] = message["annotations"]

    train_result = {}
    val_result = {}
    for i in range(len(train_detection_result)):
        message = train_detection_result[i]
        image_id = message["image_id"]
        message["category_id"] = message["category_id"] - 1
        if image_id not in train_gt_anns:
            continue
        if not train_result.get(image_id, False):
            train_result[image_id] = [message]
        else:
            train_result[image_id].append(message)
    for i in range(len(val_detection_result)):
        message = val_detection_result[i]
        image_id = message["image_id"]
        message["category_id"] = message["category_id"] - 1
        if image_id not in val_gt_anns:
            continue
        if not val_result.get(image_id, False):
            val_result[image_id] = [message]
        else:
            val_result[image_id].append(message)

    # multi-label messages of images
    train_class_dict = {}
    val_class_dict = {}
    for img_id in tqdm(train_gt_anns):
        anns = train_gt_anns[img_id]
        classes = []
        for ann in anns:
            c = ann["category_id"]
            if c not in classes:
                classes.append(c)
        train_class_dict[img_id] = classes

    for img_id in tqdm(val_gt_anns):
        anns = val_gt_anns[img_id]
        classes = []
        for ann in anns:
            c = ann["category_id"]
            if c not in classes:
                classes.append(c)
        val_class_dict[img_id] = classes

    # perform pgf
    print("performing pgf...")
    # 1. filter by class label
    class_filter(train_result, train_class_dict, "train")
    class_filter(val_result, val_class_dict, "val")
    # 2. pgf
    diff_classes = [4, 5, 6, 8, 9, 15, 16]
    print(use_diff,  type(use_diff))
    pgf(train_result, "train", t_con, t_keep, use_diff, diff_classes)
    pgf(val_result, "val", t_con, t_keep, use_diff, diff_classes)

    print("saving results...")
    json.dump(train_result, open(f"{save_path}/{prefix}voc_{year}_train.json", "w"))
    json.dump(val_result, open(f"{save_path}/{prefix}voc_{year}_val.json", "w"))


def pgf_coco(det_path, save_path, prefix, t_con, t_keep, use_diff, coco_path):
    print("loading coco datasets...")
    os.chdir("uwsod/")
    trainset = get_detection_dataset_dicts(('coco_2014_train',))
    valset = get_detection_dataset_dicts(('coco_2014_valminusminival',))
    os.chdir("../")
    print("loading coco detection results...")
    train_detection_result = json.load(open(f"{det_path}/{prefix}coco_2014_train.json"))
    val_detection_result = json.load(open(f"{det_path}/{prefix}coco_2014_valminusminival.json"))

    # image_id 2 anns
    train_gt_anns = {}
    val_gt_anns = {}
    for i in range(len(trainset)):
        message = trainset[i]
        image_id = message["image_id"]
        train_gt_anns[image_id] = message["annotations"]
    for i in range(len(valset)):
        message = valset[i]
        image_id = message["image_id"]
        val_gt_anns[image_id] = message["annotations"]

    # filter images which do not contain objects
    train_result = {}
    val_result = {}
    for i in range(len(train_detection_result)):
        message = train_detection_result[i]
        image_id = message["image_id"]
        if image_id not in train_gt_anns:
            continue
        train_result[image_id] = message["instances"]
    for i in range(len(val_detection_result)):
        message = val_detection_result[i]
        image_id = message["image_id"]
        if image_id not in val_gt_anns:
            continue
        val_result[image_id] = message["instances"]

    # multi-label messages of images
    train_class_dict = {}
    val_class_dict = {}
    for img_id in tqdm(train_gt_anns):
        anns = train_gt_anns[img_id]
        classes = []
        for ann in anns:
            c = ann["category_id"]
            if c not in classes:
                classes.append(c)
        train_class_dict[img_id] = classes

    for img_id in tqdm(val_gt_anns):
        anns = val_gt_anns[img_id]
        classes = []
        for ann in anns:
            c = ann["category_id"]
            if c not in classes:
                classes.append(c)
        val_class_dict[img_id] = classes

    # perform pgf
    print("performing pgf...")
    # 1. filter by class label
    class_filter(train_result, train_class_dict, "train")
    class_filter(val_result, val_class_dict, "val")
    # 2. pgf
    pgf(train_result, "train", t_con, t_keep, use_diff, None)
    pgf(val_result, "val", t_con, t_keep, use_diff, None)

    # load gt annotations and replace gt annotations by pseudo labels
    print("saving results...")
    coco_train_gt_path = f"{coco_path}/annotations/instances_train2014.json"
    coco_train = json.load(open(coco_train_gt_path))
    coco_val_gt_path = f"{coco_path}/annotations/instances_valminusminival2014.json"
    coco_val = json.load(open(coco_val_gt_path))

    new_train_annotations = gen_annotations(train_result)
    new_val_annotations = gen_annotations(val_result)
    coco_train["annotations"] = new_train_annotations
    coco_val["annotations"] = new_val_annotations
    
    # save pseudo labels
    json.dump(coco_train, open(f"{save_path}/{prefix}coco_2014_train.json", "w"))
    json.dump(coco_val, open(f"{save_path}/{prefix}coco_2014_valminusminival2014.json", "w"))

def gen_annotations(result):
    new_annotations = []
    INDEX = 0
    id2cat = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}
    for img_id in tqdm(result):
        predictions = result[img_id]
        for prediction in predictions:
            new_annotations.append(
                {
                    "image_id": img_id,
                    "bbox": prediction["bbox"],
                    "category_id": id2cat[prediction["category_id"]],
                    "id": INDEX
                }
            )
            INDEX += 1
    return new_annotations

def contain_cal(a_, b_):
    a = copy.deepcopy(a_)
    b = copy.deepcopy(b_)
    a[2]+=a[0]
    a[3]+=a[1]
    b[2]+=b[0]
    b[3]+=b[1]
    c = [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]
    area_c = max(0, c[2]-c[0]) * max(0, c[3]-c[1])
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    return area_c/(area_a+1e-6)

def pgf(result, split, t_con, t_keep, use_diff, diff_classes):
    length = 0
    for i in result:
        length += len(result[i])
    print(f"{split} split length before pgf: {length}")
    see_list = {}
    for img_id in result:
        see_list[img_id] = []
    
    for img_id in tqdm(result):
        predictions = result[img_id]
        drop_list = []
        for i in range(len(predictions)):
            c = predictions[i]["category_id"]
            if (c not in see_list[img_id]):
                see_list[img_id].append(c)
                continue
            if predictions[i]["score"] < t_keep:
                drop_list.append(i)
        for i in drop_list[::-1]:
            result[img_id].pop(i)
    
    length = 0
    for i in result:
        length += len(result[i])
    print(f"{split} split length in middle of pgf: {length}")

    for i in tqdm(result):
        anns = result[i]
        save = [True] * len(anns)
        bboxes = [b["bbox"] for b in anns]
        cats = [b["category_id"] for b in anns]
        new_anns = []
        for b_i in range(len(save)):
            for b_j in range(len(save)):
                if b_i == b_j or (cats[b_i] != cats[b_j]): continue
                if not use_diff and cats[b_i] in diff_classes: continue
                val = contain_cal(bboxes[b_i], bboxes[b_j])
                if val >= t_con:
                    save[b_i] = False
        for j in range(len(save)):
            if save[j]:
                new_anns.append(copy.deepcopy(anns[j]))
        result[i] = new_anns

    length = 0
    for i in result:
        length += len(result[i])

    print(f"{split} split length after pgf: {length}")


def class_filter(result, class_dict, split):
    length = 0
    for i in result:
        length += len(result[i])
    print(f"{split} split length before multi-class filter: {length}")
    for img_id in tqdm(result):
        predictions = result[img_id]
        gt_classes = class_dict[img_id]
        drop_list = []
        for i in range(len(predictions)):
            if predictions[i]["category_id"] not in gt_classes:
                drop_list.append(i)
        for i in drop_list[::-1]:
            result[img_id].pop(i)
    length = 0
    for i in result:
        length += len(result[i])
    print(f"{split} split length after multi-class filter: {length}")
    



def main():
    args = parse_args()
    det_path = args.det_path
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.system(f"mkdir -p {save_path}")
    prefix = args.prefix
    dataset = args.dataset
    t_con = args.t_con
    t_keep = args.t_keep
    use_diff = args.use_diff
    if dataset == "coco":
        coco_path = args.coco_path
        pgf_coco(det_path, save_path, prefix, t_con, t_keep, use_diff, coco_path)
    elif "voc" in dataset:
        year = dataset[3:]
        pgf_voc(det_path, save_path, prefix, t_con, t_keep, use_diff, year)
    else:
        raise ValueError(f"{dataset} is not supported.")

if __name__ == "__main__":
    main()
