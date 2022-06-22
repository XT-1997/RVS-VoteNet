## RVS-VoteNet: Revisiting VoteNet with Inner-group Relation and Weighted Relation-Aware Proposal
This repository contains an implementation of RVS-VoteNet, a 3D object detection method introduced in our paper:
> **RVS-VoteNet: Revisiting VoteNet with Inner-group Relation and Weighted Relation-Aware Proposal**<br>

### Installation
- Python 3.6+
- PyTorch 1.10.1
- CUDA 10.1+ 
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

**Step 0.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.** Create a conda environment and activate it.

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

On GPU platforms:

```shell
conda install pytorch torchvision -c pytorch
```

On CPU platforms:

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

# Installation

We recommend that users follow our best practices to install MMDetection3D. However, the whole process is highly customizable. See [Customize Installation](#customize-installation) section for more information.

## Best Practices
Assuming that you already have CUDA 11.0 installed, here is a full script for quick installation of MMDetection3D with conda.
Otherwise, you should refer to the step-by-step installation instructions in the next section.

```shell
pip install openmim
mim install mmcv-full
mim install mmdet
mim install mmsegmentation
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
pip install -e .
```

**Step 0.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

**Step 1.** Install [MMDetection](https://github.com/open-mmlab/mmdetection).


```shell
pip install mmdet
```

Optionally, you could also build MMDetection from source in case you want to modify the code:

```shell
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout v2.24.0  # switch to v2.24.0 branch
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

**Step 2.** Install [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).

```shell
pip install mmsegmentation
```

Optionally, you could also build MMSegmentation from source in case you want to modify the code:

```shell
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
git checkout v0.20.0  # switch to v0.20.0 branch
pip install -e .  # or "python setup.py develop"
```

**Step 3.** Clone the MMDetection3D repository.

```shell
git clone https://github.com/XT-1997/RVS-VoteNet.git
cd RVS-VoteNet/mmdetection3d
```

**Step 4.** Install build requirements and then install MMDetection3D.

```shell
pip install -v -e .  # or "python setup.py develop"
```
