import numpy as np
from common.skeleton import Skeleton
from torch.utils.data import Dataset
from itertools import zip_longest

class ChunkedDataset_Seq(Dataset):
    """
    Batched data generator (re-written as a PyTorch Dataset).
    The sequences are split into equal-length chunks and padded as necessary.
    
    Arguments:
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    chunk_length -- number of output frames to predict for each training example (usually 1)
    stride -- distance between the start frames of consecutive chunks (usually chunk)
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    random_seed -- initial seed to use for the random generator
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    shuffle -- if True, shuffle the order of chunked pairs on initialization
    """

    def __init__(self, 
                 cameras,
                 poses_3d,
                 poses_2d,
                 chunk_length,
                 stride,
                 pad=0,
                 causal_shift=0,
                 random_seed=1234,
                 augment=False,
                 kps_left=None,
                 kps_right=None,
                 joints_left=None,
                 joints_right=None,
                 shuffle=True):
        
        # (1) 기본 검증
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (
            "poses_3d와 poses_2d의 길이가 일치하지 않습니다.",
            len(poses_3d), len(poses_2d)
        )
        assert cameras is None or len(cameras) == len(poses_2d), (
            "cameras와 poses_2d의 길이가 일치하지 않습니다.",
            len(cameras), len(poses_2d)
        )
        
        self.cameras = cameras
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d
        
        self.chunk_length = chunk_length
        self.pad = pad
        self.causal_shift = causal_shift
        
        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right

        pairs = [] # (seq_idx, start_frame, end_frame, flip) tuples
        for i in range(len(poses_2d)):
            assert poses_3d is None or poses_2d[i].shape[0] == poses_3d[i].shape[0]
            n_chunks = (poses_2d[i].shape[0] + stride - 1) // stride

            # bounds = np.arange(n_chunks+1)*stride - offset
            starts = np.arange(n_chunks) * stride
            # offset = ((poses_2d[i].shape[0] - starts[-1]) + chunk_length) /2 - (poses_2d[i].shape[0] - starts[-1])
            # starts -= int(offset)
            ends = starts + chunk_length
            ends[-1] = min(ends[-1], poses_2d[i].shape[0])
            starts[-1] = min(starts[-1], poses_2d[i].shape[0] - chunk_length)
            augment_vector = np.full(len(starts), False, dtype=bool)
            pairs += zip(np.repeat(i, len(starts)), starts, ends, augment_vector)
            if augment:
                pairs += zip(np.repeat(i, len(starts)), starts, ends, ~augment_vector)
        self.pairs = pairs

    def __len__(self):
        """
        전체 chunk(혹은 flip 포함 chunk)의 개수 반환
        """
        return len(self.pairs)
    
    def __getitem__(self, index):
        """
        index번째 chunk를 가져와서 2D, 3D, camera 데이터를 추출하고
        flip이 True면 좌우 반전 수행 후,
        최종적으로 (cam, pose_3d, pose_2d)의 형태로 반환.
        """
        seq_i, start_3d, end_3d, flip = self.pairs[index]
        seq_i = int(seq_i)  # numpy에서 튜플을 가져오면 dtype=object라 int 캐스팅
        
        # ----------------------------
        # 2D poses 추출
        # ----------------------------
        seq_2d = self.poses_2d[seq_i]
        random_shift = 0
        start_2d = start_3d + random_shift  # pad, causal_shift 반영하려면 추가할 수도 있음
        end_2d   = end_3d + random_shift

        low_2d  = max(start_2d, 0)
        high_2d = min(end_2d, seq_2d.shape[0])
        pad_left_2d  = low_2d - start_2d
        pad_right_2d = end_2d - high_2d

        if pad_left_2d != 0 or pad_right_2d != 0:
            chunk_2d = np.pad(
                seq_2d[low_2d:high_2d],
                ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 
                'edge'
            )
        else:
            chunk_2d = seq_2d[low_2d:high_2d]

        if flip:
            # x 좌표 반전
            chunk_2d[..., 0] *= -1
            # 좌/우 keypoint swap
            if (self.kps_left is not None) and (self.kps_right is not None):
                left = self.kps_left
                right = self.kps_right
                # chunk_2d[:, left + right] 이런 식으로 스왑
                chunk_2d[..., left + right, :] = chunk_2d[..., right + left, :]
        # ----------------------------
        # 3D poses 추출 (optional)
        # ----------------------------
        chunk_3d = None
        if self.poses_3d is not None:
            seq_3d = self.poses_3d[seq_i]
            low_3d  = max(start_3d, 0)
            high_3d = min(end_3d, seq_3d.shape[0])
            pad_left_3d  = low_3d - start_3d
            pad_right_3d = end_3d - high_3d

            if pad_left_3d != 0 or pad_right_3d != 0:
                chunk_3d = np.pad(
                    seq_3d[low_3d:high_3d],
                    ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)),
                    'edge'
                )
            else:
                chunk_3d = seq_3d[low_3d:high_3d]

            if flip:
                chunk_3d[..., 0] *= -1
                if (self.joints_left is not None) and (self.joints_right is not None):
                    left = self.joints_left
                    right = self.joints_right
                    chunk_3d[..., left + right, :] = chunk_3d[..., right + left, :]

        # ----------------------------
        # Camera 추출 (optional)
        # ----------------------------
        cam_param = None
        if self.cameras is not None:
            cam_param = self.cameras[seq_i].copy()
            if flip:
                # distortion 등 x 관련 파라미터 반전
                # 원래는 cam[7] *= -1 하던 로직
                if len(cam_param) > 7:
                    cam_param[7] *= -1


        return cam_param, chunk_3d, chunk_2d