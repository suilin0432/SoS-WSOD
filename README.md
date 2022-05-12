# Salvage of Supervision in Weakly Supervised Object Detection

This is the official repository of our paper:

[Salvage of Supervision in Weakly Supervised Object Detection](https://cs.nju.edu.cn/wujx/paper/CVPR2022_SoS.pdf)

[Lin Sui](http://www.lamda.nju.edu.cn/suil/), [Chen-Lin Zhang](https://tzzcl.github.io/) and [Jianxin Wu](https://cs.nju.edu.cn/wujx/index.htm)

IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022

<p align="center">
<img src="pipeline.pdf" width="85%">
</p>



##### Install

Notes:

- As we build our stage 1 model based on UWSOD repo which adopts an earlier detectron2 version, please prepare two different conda environments as follows.

###### Install UWSOD for Stage 1

```
# create conda environment for Stage 1
conda create -n wsod python=3.7 -y
conda activate wsod


conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch

# You may need to install appropriate version of pytorch according to you device and driver
# For example 30XX GPU w/ pytorch 1.9.0 cudatoolkit 11.1
# pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# download SoS-WSOD
git clone https://github.com/suilin0432/SoS-WSOD.git

# install detectron2 and wsl
cd SoS-WSOD
cd uwsod
# install detectron2
python3 -m pip install -v -e .

# install wsl
cd projects/WSL

pip install git+https://github.com/lucasb-eyer/pydensecrf.git
pip install opencv-python sklearn shapely
pip install git+https://github.com/cocodataset/panopticapi.git

git submodule update --init --recursive
python3 -m pip install -v -e .
```



###### Install Detectron2 for Stage 2 & 3

```
# create conda environment for Stage 2 and 3
conda create -n fsod python=3.7 -y
conda activate fsod

conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch

# You may need to install appropriate version of pytorch according to you device and driver
# For example 30XX GPU w/ pytorch 1.9.0 cudatoolkit 11.1
# pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# install detectron2
cd ${Detectron2_Path}
python3 -m pip install -v -e .

```

 

##### Prepare Datasets

###### PASCAL VOC

1. Download PASCAL VOC Dataset from official website

2. Link the dataset into uwsod, detectron2 and unbias

   ```
   # uwsod
   cd SoS-WSOD/uwsod/
   mkdir datasets
   ln -s $VOC_PATH datasets/VOC2007
   
   # detectron2
   cd ../detectron2
   mkdir datasets
   ln -s $VOC_PATH datasets/VOC2007
   
   # unbias
   cd ../unbias
   mkdir datasets
   ln -s $VOC_PATH datasets/VOC2007
   ```

3. Prepare Proposal file following [here](https://github.com/shenyunhang/DRN-WSOD-pytorch/tree/DRN-WSOD/projects/WSL)

###### MS-COCO

1. Download MS-COCO Dataset from official website

2. Link the dataset into uwsod, detectron2 and unbias

   ```
   # uwsod
   cd SoS-WSOD/uwsod/
   ln -s $COCO_PATH datasets/coco
   
   # detectron2
   cd ../detectron2
   ln -s $COCO_PATH datasets/coco
   
   # unbias
   cd ../unbias
   ln -s $COCO_PATH datasets/coco
   ```

3. Prepare Proposal file following [here](https://github.com/shenyunhang/DRN-WSOD-pytorch/tree/DRN-WSOD/projects/WSL)



##### Get Started

We will use VOC2007 as the example.

###### Backbone Preparation

1. Download VGG Backbone from [here](https://1drv.ms/f/s!Am1oWgo9554dgRQ8RE1SRGvK7HW2) following [UWSOD repo](https://github.com/shenyunhang/UWSOD/tree/UWSOD/projects/WSL).

2. Create models category and put the VGG backbone into it.

   ```
   cd SoS-WSOD/uwsod
   mkdir -p models/VGG
   ln -s $VGG_PATH modles/VGG/
   ```

   

###### Stage 1: WSOD stage

1. Train a basic WSOD model first:

   ```
   bash run/code_release/oicr_plus_voc07.sh
   ```

2. Generate prediction result:

   ```
   bash run/code_release/oicr_plus_voc07_detection_result.sh
   ```

   

###### Stage 2: Pseudo-FSOD stage

1. Generate pseudo labels with PGF

   ```
   # VOC2007
   python tools/pgf.py --det-path uwsod/datasets/VOC2007/detection_results/ --save-path uwsod/datasets/VOC2007/pseudo_labels --prefix oicr_plus_ --dataset voc2007
   # COCO
   python tools/pgf.py --det-path uwsod/datasets/coco/detection_results/ --save-path uwsod/datasets/coco/pseudo_labels --prefix oicr_plus_ --dataset coco --use-diff
   ```

2. Register a new dataset in Detectron2

3. Generate the base split (keep len(dataset)-1 images for simplicity)

   ```
   cd unbias/
   python generate_base_split.py --config configs/code_release/voc_baseline.yaml --save-path ./dataseed/VOC_all.txt
   ```

4. Perform pseudo-FSOD

   ```
   # using VOC2007 for example
   bash run/code_release/voc_baseline.sh
   ```

   

###### Stage 3: SSOD stage

1. add multi-label messages into pseudo-label annotation files:

   ```
   # VOC2007
   python tools/add_multi_label.py --pgt-temp unbias/datasets/VOC2007/pseudo_labels/oicr_plus_voc_2007_{}.json --dataset voc2007
   # COCO
   python tools/add_multi_label.py --pgt-temp unbias/datasets/coco/pseudo_labels/oicr_plus_coco_2014_{}.json --dataset coco
   ```

2. dataset split & get the split percent:

   ```
   # Note: After splitting process, the percentage is printed.
   
   # Use split_single.py (single process & single gpu)
   # VOC2007
   python split_single.py --config ./configs/code_release/voc_split.yaml --ckpt ./output/voc_baseline/model_0007999.pth --save-path ./dataseed/VOC07_oicr_plus_split.txt --k 2000
   
   # COCO
   python split_single.py --config ./configs/code_release/coco_split.yaml --ckpt ./output/coco_baseline/model_0029999.pth --save-path ./dataseed/COCO_oicr_plus_split.txt --k 2000
   
   # Use split_multi.py (multiple process & multiple gpu)
   # VOC2007
   python split_multi.py --config ./configs/code_release/voc_split.yaml --ckpt ./output/voc_baseline/model_0007999.pth --save-path ./dataseed/VOC07_oicr_plus_split.txt --k 2000 --gpu 8
   
   # COCO
   python split_multi.py --config ./configs/code_release/coco_split.yaml --ckpt unbias/output/coco_baseline/model_0029999.pth --save-path unbias/dataseed/COCO_oicr_plus_split.txt --k 2000 --gpu 8
   ```

3. perform ssod training

   ````
   # using VOC2007 as example
   # 1. change the DATALOADER.SUP_PERCENT in bash file
   # 2. run the bash file
   bash run/code_release/voc_ssod.sh
   ````

   

##### TTA Test:

1. extract single branch of the model

   ```
   python tools/convert2detectron2.py ${MODEL_PATH} ${OUTPUT_PATH} -m [teacher(default) | student]
   ```

2. Perform TTA test

   ```
   python train_net_test_tta.py \
   --num-gpus 8 \
   --config configs/code_release/voc07_tta_test.yaml \
   --dist-url tcp://0.0.0.0:21197 --eval-only \
   MODEL.WEIGHTS ${MODEL_PATH} \
   OUTPUT_DIR ${OUTPUT_DIR}
   ```

   

##### Models

###### VOC2007

| stage                                     | $mAP_{50:95}$ | $mAP_{50}$ | $mAP_{75}$ |                          model link                          |
| ----------------------------------------- | :-----------: | :--------: | :--------: | :----------------------------------------------------------: |
| SoS-WSOD stage 1                          |     26.2      |    54.1    |    22.8    | [link](https://drive.google.com/file/d/1gYuxnzjoyKyjiU3jK5MKGKKb3nMbVq-i/view?usp=sharing) |
| SoS-WSOD stage 1+2                        |     27.3      |    57.6    |    22.5    | [link](https://drive.google.com/file/d/1_obOkQ41j9jWHqGeA0vzpKyTn9vKV3Dx/view?usp=sharing) |
| SoS-WSOD stage 1+2+3                      |     31.6      |    62.7    |    28.1    | [link](https://drive.google.com/file/d/1_KlpJy0BTRA8HGp2oY4tXj-Bq8bRY7m_/view?usp=sharing) |
| SoS-WSOD stage 1+2+3 (low threshold test) |     31.7      |    63.1    |    28.1    |                        same as above                         |

SoS-WSOD+ on VOC2007 (WSOD part in the Journal Version)

| stage                            | $mAP_{50:95}$ | $mAP_{50}$ | $mAP_{75}$ |                          model link                          |
| -------------------------------- | :-----------: | :--------: | :--------: | :----------------------------------------------------------: |
| SoS-WSOD (adopt CASD as stage 1) |     32.8      |    64.1    |    29.8    | [link](https://drive.google.com/file/d/1Aiemr7zsBsTUEhNrtZB2JoScca6UveFs/view?usp=sharing) |
| SoS-WSOD+ (VOC2007 Only)         |     16.0      |    37.8    |    11.5    | [link](https://drive.google.com/file/d/1Dyu6q_N-RX1QJXV9ulweSkUX4U8sEyzL/view?usp=sharing) |
| SoS-WSOD+ (COCO Pretrain)        |     30.4      |    59.8    |    27.2    | [link](https://drive.google.com/file/d/1cm3wOs2BQaA114rI3uCRdrnKL1CUgGgY/view?usp=sharing) |
| SoS-WSOD+                        |     35.5      |    65.3    |    33.1    | [link](https://drive.google.com/file/d/1nCNyvDYDboqspNgbfXj7rptpEXoCYoWK/view?usp=sharing) |

Note: 

1. All results are obtained w/o TTA.
2. low threshold test denotes we adopt lower prediction threshold which is widely used in WSOD instead of the default value (0.05) in FSOD. In our paper, results of experiments, which shrink the technique gap (stage 2 & 3), use the default 0.05.
3. In stage 1 of SoS-WSOD w/ CASD, we directly use the model released by CASD official repo.
4. SoS-WSOD+ means we enable the vanilla ResNet in stage 1 with the help of contrastive learning.
5. VOC2007 Only indicates that in all stages we do not rely on any other datasets besides VOC2007.
6. COCO Pretrain means we rid of ImageNet dependency and pretrain the model w/ unlabeled MS-COCO in all stages.

###### MS-COCO

| stage                            | $mAP_{50:95}$ | $mAP_{50}$ | $mAP_{75}$ |                          model link                          |
| -------------------------------- | :-----------: | :--------: | :--------: | :----------------------------------------------------------: |
| stage 1                          |     11.6      |    23.6    |    10.4    | [link](https://drive.google.com/file/d/1W1GU9AlNG9k2rxq6-6EdcmthhTt-JH1x/view?usp=sharing) |
| stage 1+2                        |     13.7      |    27.5    |    12.2    | [link](https://drive.google.com/file/d/1w-atGR3aYSzbfjrz0kpxALeqqMN3swqc/view?usp=sharing) |
| stage 1+2+3                      |     15.5      |    30.6    |    14.4    | [link](https://drive.google.com/file/d/1rFhwTnhs8e-fn9lyRqFZuzgYK755WsVk/view?usp=sharing) |
| stage 1+2+3 (low threshold test) |     15.9      |    31.6    |    14.6    |                        same as above                         |



##### Inference with Provided Models

###### Stage 1

```
python3 projects/WSL/tools/train_net.py \
--num-gpus 4 \
--config-file projects/WSL/configs/Detection/code_release/voc07_oicr_plus.yaml \
--dist-url tcp://0.0.0.0:17346 --eval-only \
MODEL.WEIGHTS ${MODEL_PATH} \
OUTPUT_DIR ${OUTPUT_DIR} TEST.AUG.ENABLED False
```

###### Stage 2 & 3

```
python train_net_test_tta.py \
--num-gpus 8 \
--config configs/code_release/voc07_tta_test.yaml \
--dist-url tcp://0.0.0.0:21197 --eval-only \
MODEL.WEIGHTS ${MODEL_PATH} \
OUTPUT_DIR ${OUTPUT_DIR} TEST.AUG.ENABLED False
```

###### SoS-WSOD+

1. SoS-WSOD+ w/ unlabeled ImageNet Pretrain

   ```
   python train_net_test_tta.py \
   --num-gpus 8 \
   --config configs/code_release/sos_plus_test.yaml \
   --dist-url tcp://0.0.0.0:21197 --eval-only \
   MODEL.WEIGHTS ${MODEL_PATH} \
   OUTPUT_DIR ${OUTPUT_DIR} TEST.AUG.ENABLED False
   ```

2. SoS-WSOD+ w/o ImageNet (unlabeled COCO Pretrain & VOC2007 only)

   ```
   python train_net_test_tta.py \
   --num-gpus 8 \
   --config configs/code_release/sos_plus_wo_imagenet_test.yaml \
   --dist-url tcp://0.0.0.0:21197 --eval-only \
   MODEL.WEIGHTS ${MODEL_PATH} \
   OUTPUT_DIR ${OUTPUT_DIR} TEST.AUG.ENABLED False
   ```

##### Note:

1. For readability and usability, we clean and rewrite our codes. We do evaluate the codebase on VOC2007 and could get comparable or even better performance than results which are reported in our CVPR 2022 paper. However, we do not evaluate on COCO yet.
2. As we tried, inference result of the provided detector model may have some deviation according to the experiment environment. For example, compared with using the experiment environment with gcc 5, we find a little bit lower performance with gcc 7. Such a phenomenon is founded under both pytorch 1.9.0 and pytorch 1.6.0. But, results obtained by training following the SoS framework from scratch will not be affected.



##### Acknowledgment

This code is built upon UWSOD, unbiased-teacher and detectron2, thanks all the contributors of these codebases.

---



##### TODO List:

1. Evaluate results on MS-COCO.
2. Salvage of Supervision in Weakly Supervised Instance Segmentation (SoS-WSIS) and Salvage of Supervision in Weakly Supervised Semantic Segmentation  (SoS-WSSS) are coming.
3. Enable vanilla ResNet in stage 1 and get rid of ImageNet Pretraining.


