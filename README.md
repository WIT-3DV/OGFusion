# OGFusion

Official implementation of the manuscript:

**“Occupancy Guided Radar-Camera Fusion for Robust 3D Object Detection in Autonomous Driving”**

This repository contains the source code, configuration files, and usage instructions for the manuscript currently submitted to **The Visual Computer**. The code is released to improve transparency, reproducibility, and long-term accessibility of the proposed method.

This repository is directly related to the manuscript above. If you find this code useful, please consider citing the corresponding manuscript once it becomes available.

## Abstract

Three-dimensional object detection is a critical task in autonomous driving perception, where robust sensing under varying environmental conditions remains challenging. Radar-camera fusion has emerged as a promising solution by combining radar's weather robustness with camera's dense semantic information. However, existing methods suffer from radar sparsity, angular uncertainty, and inefficient cross-modal attention. To address these issues, we present OGFusion, an occupancy guided radar-camera fusion framework for 3D object detection. The method first applies Geometric Radar Densification (GRD) to obtain denser radar cues and improve geometric modeling abilities, then uses a Spatial-Context Feature Pyramid (SCFP) to alleviate the lack of positional information and insufficient contextual modeling across pyramid levels. Finally, we design the Occupancy Guided Attention module (OGA) to enable radar-camera feature fusion through occupancy guided local cross-attention and occupancy feature integration. Extensive experiments on TJ4DRadSet and VoD show that OGFusion improves 3D mAP on TJ4DRadSet by 4.70% and EAA mAP on VoD by 1.99% over the baseline while maintaining lightweight efficiency. This framework contributes to reliable and efficient multi-modal perception for autonomous driving.

## Table of Contents

- [Abstract](#abstract)
- [Method Overview](#method-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)


## Method Overview

OGFusion is an occupancy guided radar-camera fusion framework for robust 3D object detection in autonomous driving scenarios. The framework mainly consists of the following components:

1. **Geometric Radar Densification (GRD)**  
   GRD is used to enhance sparse radar observations with geometry-aware cues derived from image-based semantic information.

2. **Spatial-Context Feature Pyramid (SCFP)**  
   SCFP improves multi-level feature representation by strengthening spatial and contextual modeling across feature pyramid levels.

3. **Occupancy Guided Attention (OGA)**  
   OGA performs radar-camera feature interaction using occupancy-guided local cross-attention and occupancy feature integration.

The implementation is organized around the OpenPCDet-style detection pipeline, with additional modules for radar-camera fusion, geometric densification, and occupancy-guided attention.

## Requirements

The following environment is recommended:

- Python 3.9
- Conda
- CUDA 11.7
- PyTorch 1.13.0
- torchvision 0.14.0
- torchaudio 0.13.0
- GCC/NVCC compatible with the CUDA and PyTorch versions above

## Installation

Clone this repository and enter the project root directory:

```bash
git clone https://github.com/WIT-3DV/OGFusion
cd OGFusion
```

Create and activate the conda environment:

```bash
conda create -n OGFusion python=3.9 -y
conda activate OGFusion
```

Install PyTorch with CUDA 11.7 support:

```bash
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117
```

Install MMDetection-related dependencies:

```bash
pip install openmim
pip install mmcv==2.1.0
pip install mmdet==3.3.0
```

Install the main project in editable mode:

```bash
python setup.py develop
```

Compile and install the pillar operation module:

```bash
pushd pcdet/ops/pillar_ops
python setup.py develop
popd
```

Set up the Mask2Former environment according to the official Mask2Former installation instructions. The geometric densification pipeline relies on a correctly configured Mask2Former environment, including its required dependencies and custom operators.

## Dataset Preparation

Please prepare the VoD and TJ4D datasets and place them under the `data/` directory.

The expected dataset format follows the KITTI-style structure:

```text
data
└── <dataset_name>
    ├── ImageSets
    ├── training
    │   ├── calib
    │   ├── pose
    │   ├── velodyne
    │   ├── image_2
    │   ├── mask2former_geometric_densification
    │   └── label_2
    └── testing
```

The dataset information files should be generated before training:

```text
kitti_infos_train.pkl
kitti_infos_val.pkl
kitti_infos_trainval.pkl
kitti_infos_test.pkl
```

Dataset conversion and information-file generation utilities are provided under:

```text
pcdet/datasets/
```

Please follow the data licenses and official download instructions of the original datasets.

The Geometric Radar Densification module is used to generate denser radar cues from image-based semantic and geometric information.

The related scripts are provided under:

```text
Mask2Former/demo/
```

Please make sure that the Mask2Former environment is properly configured before running the densification pipeline.

The generated densified radar features should be saved under the following directory:

```text
data/<dataset_name>/training/mask2former_geometric_densification/
```

Detailed dataset-specific execution commands will be updated together with the release of the corresponding reproducibility package.

## Training

Training configurations and scripts are provided under:

```text
tools/cfgs/
tools/scripts/
```

To train the model, select the corresponding configuration file for the target dataset and run the distributed training script according to your hardware setup.

```bash
bash ./tools/scripts/dist_train.sh <num_gpus>
```

Please replace `<num_gpus>` with the number of GPUs used for training.

Dataset-specific configuration names, pretrained checkpoints, random seeds, and full training logs will be released in the corresponding reproducibility package after the review process.

## Evaluation

Evaluation can be performed using the testing script:

```bash
python ./tools/test.py \
  --cfg_file <path_to_config_file> \
  --ckpt <path_to_checkpoint>
```

Please replace `<path_to_config_file>` and `<path_to_checkpoint>` with the corresponding configuration and trained checkpoint.

The detailed evaluation settings corresponding to the manuscript tables will be updated after the review and publication process.

## Citation

This repository is directly related to the manuscript currently submitted to **The Visual Computer**.

If you use this code, dataset preparation pipeline, configuration files, or experimental results in your research, please cite the related manuscript once it becomes available.

The final citation information, including DOI, volume, issue, and page numbers, will be updated after the manuscript is accepted or published.

## Acknowledgements

Many thanks to the following open-source repositories for their valuable contributions to the community:

- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) for the 3D object detection framework and related utilities.
- [Mask2Former](https://github.com/facebookresearch/Mask2Former) for the image segmentation framework and related components.
- [HGSFusion](https://github.com/garfield-cpp/HGSFusion) for its inspiring multimodal fusion design and implementation.

Our implementation is built upon or inspired by these excellent projects. We sincerely appreciate the authors for making their code publicly available. Please also consider citing the corresponding papers and repositories if you use this code.

