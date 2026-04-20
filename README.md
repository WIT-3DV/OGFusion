# OGFusion

Official implementation of:

**“Occupancy Guided Radar-Camera Fusion for Robust 3D Object Detection in Autonomous Driving”**

submitted to **The Visual Computer**. This repository contains the source code, configuration files, and usage instructions for OGFusion.

## Abstract

Three-dimensional object detection is a critical task in autonomous driving perception, where robust sensing under varying environmental conditions remains challenging. Radar-camera fusion has emerged as a promising solution by combining radar's weather robustness with camera's dense semantic information. However, existing methods suffer from radar sparsity, angular uncertainty, and inefficient cross-modal attention. To address these issues, we present OGFusion, an occupancy guided radar-camera fusion framework for 3D object detection. The method first applies Geometric Radar Densification (GRD) to obtain denser radar cues and improve geometric modeling abilities, then uses a Spatial-Context Feature Pyramid (SCFP) to alleviate the lack of positional information and insufficient contextual modeling across pyramid levels. Finally, we design the Occupancy Guided Attention module (OGA) to enable radar-camera feature fusion through occupancy guided local cross-attention and occupancy feature integration. Extensive experiments on TJ4DRadSet and VoD show that OGFusion improves 3D mAP on TJ4DRadSet by 4.70% and EAA mAP on VoD by 1.99% over the baseline while maintaining lightweight efficiency.

## Method Overview

OGFusion is an occupancy guided radar-camera fusion framework for robust 3D object detection in autonomous driving scenarios. The framework mainly consists of the following components:

1. **Geometric Radar Densification (GRD)**  
   Enhances radar spatial support through angular perturbation and image-guided filtering to alleviate radar sparsity and angular uncertainty.

2. **Spatial-Context Feature Pyramid (SCFP)**  
   Incorporates explicit spatial prior modeling and context-aware selective response refinement to improve positional and contextual representation.

3. **Occupancy Guided Attention (OGA)**  
   Enables radar-camera feature fusion through occupancy guided local cross-attention and occupancy feature integration.

The implementation is organized around the OpenPCDet-style detection pipeline, with additional modules for radar-camera fusion, geometric densification, and occupancy-guided attention.

## Citation

This repository is associated with the manuscript:

**“Occupancy Guided Radar-Camera Fusion for Robust 3D Object Detection in Autonomous Driving”**

The manuscript has been submitted to **The Visual Computer**. If you find this code useful, please consider citing our work.

## Requirements

Recommended environment:

- Python 3.9
- Conda
- CUDA 11.7
- PyTorch 1.13.0
- torchvision 0.14.0
- torchaudio 0.13.0

## Installation

```bash
git clone https://github.com/WIT-3DV/OGFusion.git
cd OGFusion
```

```bash
conda create -n OGFusion python=3.9 -y
conda activate OGFusion
```

```bash
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117
```

```bash
pip install openmim
pip install mmcv==2.1.0
pip install mmdet==3.3.0
```

```bash
python setup.py develop
```

```bash
pushd pcdet/ops/pillar_ops
python setup.py develop
popd
```

## Dataset Preparation

Prepare the VoD and TJ4DRadSet datasets under the `data/` directory.

The raw datasets can be obtained from their official repositories:

- **View-of-Delft (VoD)**:  
  https://github.com/tudelft-iv/view-of-delft-dataset

- **TJ4DRadSet**:  
  https://github.com/TJRadarLab/TJ4DRadSet

Please follow the instructions and license terms provided by the official dataset repositories to download and prepare the raw data.

Expected structure:

```text
data
├── dataset_name
│   ├── ImageSets
│   ├── kitti_infos_test.pkl
│   ├── kitti_infos_train.pkl
│   ├── kitti_infos_trainval.pkl
│   ├── kitti_infos_val.pkl
│   ├── testing
│   └── training
|   |   ├── calib                               
|   |   ├── pose
|   |   ├── velodyne
|   |   ├── image_2
|   |   ├── mask2former_geometric_densification      
|   |   └── label_2                                        
```
Generate dataset information files before training:
```text
python pcdet.datasets.kitti.dataset_name
```
```text
kitti_infos_train.pkl
kitti_infos_val.pkl
kitti_infos_trainval.pkl
kitti_infos_test.pkl
```

Dataset utilities are provided under:

```text
pcdet/datasets/
```

## Training

Training configuration files are placed under:

```text
tools/cfgs/
```

Example command:

```bash
bash ./tools/scripts/dist_train.sh <num_gpus> --cfg_file <path_to_config_file>
```


## Evaluation

Example command:

```bash
python ./tools/test.py \
  --cfg_file <path_to_config_file> \
  --ckpt <path_to_checkpoint>
```

## Acknowledgements

This project is built upon or inspired by the following open-source repositories:

- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
- [Mask2Former](https://github.com/facebookresearch/Mask2Former)
- [HGSFusion](https://github.com/garfield-cpp/HGSFusion)

We sincerely thank the authors for their excellent work.