# RootPose


## Environment

The code is conducted under the following environment:

* Ubuntu 18.04
* Python 3.6.10
* PyTorch 1.8.1
* CUDA 10.2


## Dataset

The Human3.6M dataset and HumanEva dataset setting follow the [VideoPose3D](https://github.com/facebookresearch/VideoPose3D).
Please refer to it to set up the Human3.6M dataset (under ./data directory).

The MPI-INF-3DHP dataset setting follows the [MMPose](https://github.com/open-mmlab/mmpose).
Please refer it to set up the MPI-INF-3DHP dataset (also under ./data directory).

# Training from scratch

Training on the 243 frames with two GPUs:

>  torchrun --nproc_per_node=2 run.py -c checkpoint