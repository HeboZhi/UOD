# Class-Agnostic Foreground Response Calibration for Robust Unknown Object Detection
[Hebo Zhi],[Junhao Li],[Shiyan Fan],[Jun Zhang]

This work is currently under review at The Visual Computer.
# Create Environment

```bash
conda create -n CAFR python=3.8
conda activate CAFR
pip install -r requirements.txt
```


# Install Detectron2

Please install detectron2 following [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

# Dataset Preparation

The datasets can be downloaded using this [link](https://drive.google.com/drive/folders/1Mh4xseUq8jJP129uqCvG9cSLdjqdl0Jo?usp=drive_link).

#### PASCAL VOC

Please put the corresponding json files in Google Cloud Disk into ./anntoations

Please download the JPEGImages data from the [Link](
https://drive.google.com/file/d/1n9C4CiBURMSCZy2LStBQTzR17rD_a67e/view?usp=sharing) provided by [VOS](https://github.com/deeplearning-wisc/vos#).

The VOC dataset folder should have the following structure:

```text
 └── VOC_DATASET_ROOT
     |
     ├── JPEGImages
     ├── voc0712_train_all.json
     ├── voc0712_train_completely_annotation200.json
     └── val_coco_format.json
```

#### COCO

Please put the corresponding json files in Google Cloud Disk into ./anntoations

The COCO dataset folder should have the following structure:

```text
 └── COCO_DATASET_ROOT
     |
     ├── annotations
        ├── xxx (the original json files)
        ├── instances_val2017_coco_ood.json
        ├── instances_val2017_mixed_ID.json
        └── instances_val2017_mixed_OOD.json
     ├── train2017
     └── val2017
```

# Training
```bash
python train_net.py --dataset-dir VOC_DATASET_ROOT --num-gpus 2 --config-file VOC-Detection/faster-rcnn/Iou_FFN.yaml --random-seed 0 --resume
```
The pretrained models for Pascal-VOC can be downloaded from [Here](https://drive.google.com/file/d/1-rSYdKAsvhJDNT7SRuq9K55rFPzo_lfy/view?usp=drive_link). Please put the model in ./detection/.

# Pretesting
The function of this process is to obtain the threshold, which only uses part of the training data.
```bash
sh pretest.sh
```


# Evaluation on the VOC
```bash
python apply_net.py --dataset-dir VOC_DATASET_ROOT --test-dataset voc_custom_val  --config-file VOC-Detection/faster-rcnn/Iou_FFN.yaml --inference-config Inference/standard_nms.yaml --random-seed 0 --image-corruption-level 0 --visualize 0
```
# Evaluation on the COCO-OOD
```bash
sh test_ood.sh
```
# Evaluation on the COCO-Mix
```bash
sh test_mixed.sh
```
# Visualize prediction results
```bash
sh vis.sh
```



