# PoseAnchor: Robust Root Position Estimation for 3D Human Pose Estimation
Official implementation of ICCV 2025 paper


## Environment

The code is conducted under the following environment:

* Ubuntu 18.04
* Python 3.6.10
* PyTorch 1.8.1
* CUDA 10.2


## Dataset

The Human3.6M dataset dataset setting follow the [VideoPose3D](https://github.com/facebookresearch/VideoPose3D).
Please refer to it to set up the Human3.6M dataset (under ./data directory).

The MPI-INF-3DHP dataset setting follows the [MMPose](https://github.com/open-mmlab/mmpose).
Please refer it to set up the MPI-INF-3DHP dataset (also under ./data directory).

```bash
${POSE_ROOT}/
|-- data
|   |-- data_3d_h36m.npz
|   |-- data_2d_h36m_gt.npz
|   |-- data_2d_h36m_cpn_ft_h36m_dbb.npz
```

## Training from scratch

Training on the 243 frames with two GPUs:

```bash
torchrun --nproc_per_node=2 run.py -c checkpoint
```

## Acknowledgement

Thanks for the baselines, we construct the code based on them:

* MixSTE
* VideoPose3D
* SimpleBaseline
