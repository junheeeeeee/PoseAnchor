# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# command:
# CUDA_VISIBLE_DEVICES=1,2 OMP_NUM_THREADS=8 torchrun --nproc_per_node=2 run.py -k cpn_ft_h36m_dbb -f 243 -s 243 -l log/root -c checkpoint/root -m CSTE

import numpy as np

from common.arguments import parse_args
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno
import math
from einops import rearrange, repeat
from copy import deepcopy

from common.camera import *
import collections
from model.MixSTEs import *
from common.skeleton import *

import random
from common.loss import *
from common.generators import ChunkedGenerator_Seq, UnchunkedGenerator_Seq
from common.chunk_dataset import *
from time import time
from common.utils import *
from common.logging import Logger
from model.load_model import load_model
# from model.PoseMamba import PoseMamba
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from progress.bar import Bar
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
import socket
#cudnn.benchmark = True       
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# import ptvsd
# ptvsd.enable_attach(address = ('192.168.210.130', 5678))
# print("ptvsd start")
# ptvsd.wait_for_attach()
# print("start debuging")
# joints_errs = []
args = parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


# initial setting
TIMESTAMP = "{0:%Y%m%dT%H-%M-%S/}".format(datetime.now())
# tensorboard



print('Loading dataset...')
dataset_path = 'data/data_3d_' + args.dataset + '.npz'
if args.dataset == 'h36m':
    from common.h36m_dataset import Human36mDataset
    dataset = Human36mDataset(dataset_path)
elif args.dataset.startswith('humaneva'):
    from common.humaneva_dataset import HumanEvaDataset
    dataset = HumanEvaDataset(dataset_path)
elif args.dataset.startswith('custom'):
    from common.custom_dataset import CustomDataset
    dataset = CustomDataset('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz')
else:
    raise KeyError('Invalid dataset')

print('Preparing data...')
for subject in dataset.subjects():
    for action in dataset[subject].keys():
        anim = dataset[subject][action]

        if 'positions' in anim:
            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_3d[:, 1:] -= pos_3d[:, :1] # Remove global offset, but keep trajectory in first position
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d

print('Loading 2D detections...')
keypoints = np.load('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz', allow_pickle=True)
keypoints_metadata = keypoints['metadata'].item()
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
keypoints = keypoints['positions_2d'].item()

###################
for subject in dataset.subjects():
    assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
    for action in dataset[subject].keys():
        assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
        if 'positions_3d' not in dataset[subject][action]:
            continue

        for cam_idx in range(len(keypoints[subject][action])):

            # We check for >= instead of == because some videos in H3.6M contain extra frames
            mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
            assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

            if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                # Shorten sequence
                keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

        assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

for subject in keypoints.keys():
    for action in keypoints[subject]:
        for cam_idx, kps in enumerate(keypoints[subject][action]):
            # Normalize camera frame
            cam = dataset.cameras()[subject][cam_idx]
            kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
            keypoints[subject][action][cam_idx] = kps

subjects_train = args.subjects_train.split(',')
subjects_semi = [] if not args.subjects_unlabeled else args.subjects_unlabeled.split(',')
if not args.render:
    subjects_test = args.subjects_test.split(',')
else:
    subjects_test = [args.viz_subject]


def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)): # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)): # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    stride = args.downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i])//stride * subset)*stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start+n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start+n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]


    return out_camera_params, out_poses_3d, out_poses_2d

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

action_filter = None if args.actions == '*' else args.actions.split(',')
if action_filter is not None:
    print('Selected actions:', action_filter)

cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, action_filter)

# set receptive_field as number assigned
receptive_field = args.number_of_frames
pad = (receptive_field -1) // 2 # Padding on each side
min_loss = args.min_loss
width = cam['res_w']
height = cam['res_h']
num_joints = keypoints_metadata['num_joints']

#########################################PoseTransformer
if args.model == 'MotionAGFormer':
        model_pos_train = load_model(args.model, args)
        model_pos = load_model(args.model, args)
else:
    try:
        model_pos_train =  eval(args.model)(num_frame=receptive_field, num_joints=num_joints, in_chans=2, embed_dim_ratio=args.cs, depth=args.dep,
            num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0.1)

        model_pos =  eval(args.model)(num_frame=receptive_field, num_joints=num_joints, in_chans=2, embed_dim_ratio=args.cs, depth=args.dep,
                num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0)
    except:
        raise Exception("Undefined model name")


################ load weight ########################
# posetrans_checkpoint = torch.load('./checkpoint/pretrained_posetrans.bin', map_location=lambda storage, loc: storage)
# posetrans_checkpoint = posetrans_checkpoint["model_pos"]
# model_pos_train = load_pretrained_weights(model_pos_train, posetrans_checkpoint)

#################
causal_shift = 0
model_params = 0
for parameter in model_pos.parameters():
    model_params += parameter.numel()


wandb_id = args.wandb_id if args.wandb_id != '' else wandb.util.generate_id()
# make model parallel
if torch.cuda.is_available():
    model_pos = nn.DataParallel(model_pos)
    model_pos = model_pos.cuda()

if args.resume or args.evaluate:
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    chk_filename = "checkpoint/" + chk_filename
    # chk_filename = args.resume or args.evaluate

    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))
    model_pos_train.load_state_dict(checkpoint['model_pos'], strict=False)
    model_pos.load_state_dict(checkpoint['model_pos'], strict=False)
    wandb_id = checkpoint['wandb_id'] if 'wandb_id' in checkpoint else wandb_id
    min_loss = checkpoint['min_loss'] if 'min_loss' in checkpoint else min_loss
    print('Best validation loss so far:', min_loss)
    print('wandb_id:', wandb_id)



test_generator = UnchunkedGenerator_Seq(cameras_valid, poses_valid, poses_valid_2d,
                                    pad=pad, causal_shift=causal_shift, augment=False,
                                    kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)

print('INFO: Testing on {} frames'.format(test_generator.num_frames()))


def eval_data_prepare(receptive_field, inputs_2d, inputs_3d):
    # inputs_2d_p = torch.squeeze(inputs_2d)
    # inputs_3d_p = inputs_3d.permute(1,0,2,3)
    # out_num = inputs_2d_p.shape[0] - receptive_field + 1
    # eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
    # for i in range(out_num):
    #     eval_input_2d[i,:,:,:] = inputs_2d_p[i:i+receptive_field, :, :]
    # return eval_input_2d, inputs_3d_p
    ### split into (f/f1, f1, n, 2)
    assert inputs_2d.shape[:-1] == inputs_3d.shape[:-1], "2d and 3d inputs shape must be same! "+str(inputs_2d.shape)+str(inputs_3d.shape)
    inputs_2d_p = torch.squeeze(inputs_2d)
    inputs_3d_p = torch.squeeze(inputs_3d)

    if inputs_2d_p.shape[0] / receptive_field > inputs_2d_p.shape[0] // receptive_field: 
        out_num = inputs_2d_p.shape[0] // receptive_field+1
    elif inputs_2d_p.shape[0] / receptive_field == inputs_2d_p.shape[0] // receptive_field:
        out_num = inputs_2d_p.shape[0] // receptive_field

    eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
    eval_input_3d = torch.empty(out_num, receptive_field, inputs_3d_p.shape[1], inputs_3d_p.shape[2])

    for i in range(out_num-1):
        eval_input_2d[i,:,:,:] = inputs_2d_p[i*receptive_field:i*receptive_field+receptive_field,:,:]
        eval_input_3d[i,:,:,:] = inputs_3d_p[i*receptive_field:i*receptive_field+receptive_field,:,:]
    if inputs_2d_p.shape[0] < receptive_field:
        from torch.nn import functional as F
        pad_right = receptive_field-inputs_2d_p.shape[0]
        inputs_2d_p = rearrange(inputs_2d_p, 'b f c -> f c b')
        inputs_2d_p = F.pad(inputs_2d_p, (0,pad_right), mode='replicate')
        # inputs_2d_p = np.pad(inputs_2d_p, ((0, receptive_field-inputs_2d_p.shape[0]), (0, 0), (0, 0)), 'edge')
        inputs_2d_p = rearrange(inputs_2d_p, 'f c b -> b f c')
    if inputs_3d_p.shape[0] < receptive_field:
        pad_right = receptive_field-inputs_3d_p.shape[0]
        inputs_3d_p = rearrange(inputs_3d_p, 'b f c -> f c b')
        inputs_3d_p = F.pad(inputs_3d_p, (0,pad_right), mode='replicate')
        inputs_3d_p = rearrange(inputs_3d_p, 'f c b -> b f c')
    eval_input_2d[-1,:,:,:] = inputs_2d_p[-receptive_field:,:,:]
    eval_input_3d[-1,:,:,:] = inputs_3d_p[-receptive_field:,:,:]

    return eval_input_2d, eval_input_3d


###################

# Evaluate
def evaluate(test_generator, action=None, return_predictions=False, use_trajectory_model=False, newmodel=None):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_mrpe = 0
    epoch_loss_3d_vel = 0
    with torch.no_grad():
        model_eval = model_pos
        model_eval.eval()
        N = 0
        for cam, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            inputs_3d = torch.from_numpy(batch.astype('float32'))
            cam = torch.from_numpy(cam.astype('float32'))


            ##### apply test-time-augmentation (following Videopose3d)
            inputs_2d_flip = inputs_2d.clone()
            inputs_2d_flip [:, :, :, 0] *= -1
            inputs_2d_flip[:, :, kps_left + kps_right,:] = inputs_2d_flip[:, :, kps_right + kps_left,:]

            ##### convert size
            inputs_3d_p = inputs_3d
            if newmodel is not None:
                def eval_data_prepare_pf(receptive_field, inputs_2d, inputs_3d):
                    inputs_2d_p = torch.squeeze(inputs_2d)
                    inputs_3d_p = inputs_3d.permute(1,0,2,3)
                    padding = int(receptive_field//2)
                    inputs_2d_p = rearrange(inputs_2d_p, 'b f c -> f c b')
                    inputs_2d_p = F.pad(inputs_2d_p, (padding,padding), mode='replicate')
                    inputs_2d_p = rearrange(inputs_2d_p, 'f c b -> b f c')
                    out_num = inputs_2d_p.shape[0] - receptive_field + 1
                    eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
                    for i in range(out_num):
                        eval_input_2d[i,:,:,:] = inputs_2d_p[i:i+receptive_field, :, :]
                    return eval_input_2d, inputs_3d_p
                
                inputs_2d, inputs_3d = eval_data_prepare_pf(81, inputs_2d, inputs_3d_p)
                inputs_2d_flip, _ = eval_data_prepare_pf(81, inputs_2d_flip, inputs_3d_p)
            else:
                inputs_2d, inputs_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d_p)
                inputs_2d_flip, _ = eval_data_prepare(receptive_field, inputs_2d_flip, inputs_3d_p)

            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()
                inputs_2d_flip = inputs_2d_flip.cuda()
                inputs_3d = inputs_3d.cuda()
                b = inputs_3d.shape[0]
                cam = cam.cuda()
                cam = cam.repeat(b,1)
            
            inputs_traj = inputs_3d[:, :, :1].clone()
            inputs_3d[:, :, 0] = 0
            
            predicted_3d_pos = model_eval(inputs_2d)
            predicted_3d_pos_flip = model_eval(inputs_2d_flip)
            predicted_3d_pos_flip[:, :, :, 0] *= -1
            predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :,
                                                                    joints_right + joints_left]
            for i in range(predicted_3d_pos.shape[0]):
                predicted_3d_pos[i,:,:,:] = (predicted_3d_pos[i,:,:,:] + predicted_3d_pos_flip[i,:,:,:])/2
            predicted_3d_pos[:, :, 0] = 0
            pred_root, residuals = get_root(predicted_3d_pos, inputs_2d, cam)
            
            # gt_2d = project_to_2d_linear(inputs_3d + inputs_traj, cam)
            # pred_2d = project_to_2d_linear(predicted_3d_pos + pred_root, cam)
            # best_2d = project_to_2d_linear(predicted_3d_pos + inputs_traj, cam)
            # predicted_3d_pos, pred_root, pred_2d = refine_pose(predicted_3d_pos, pred_2d, cam)
            # torch.set_printoptions(sci_mode=False, precision=6)
            # print((gt_2d[2,0] - pred_2d[2,0]) * 1000)
            # print(residuals[2,0] * 1000)
            # print((inputs_traj[2,0] - pred_root[2,0]) * 1000)

            # print(f"{'MPJPE:':<10} {mpjpe(predicted_3d_pos, inputs_3d).item() * 1000:.2f}")
            # print(f"{'MRPE:':<10} {mpjpe(pred_root, inputs_traj).item() * 1000:.2f}")
            # print(f"{'Original:':<10} {mpjpe(inputs_2d, gt_2d).item() * 1000:.2f}")
            # print(f"{'Refined:':<10} {mpjpe(pred_2d, gt_2d).item() * 1000 - mpjpe(inputs_2d, gt_2d).item() * 1000:.2f}")
            # print(f"{'Best:':<10} {mpjpe(best_2d, gt_2d).item() * 1000 - mpjpe(inputs_2d, gt_2d).item() * 1000:.2f}")
            # print(f"{'Changed:':<10} {mpjpe(inputs_2d, pred_2d).item() * 1000:.2f}")
            # print('----------')
            # exit()

            if return_predictions:
                return predicted_3d_pos.squeeze().cpu().numpy()
            

            error = mpjpe(predicted_3d_pos, inputs_3d)

            epoch_loss_3d_pos_scale += inputs_3d.shape[0]*inputs_3d.shape[1] * mpjpe(predicted_3d_pos + pred_root, inputs_3d + inputs_traj).item()
            epoch_mrpe += inputs_3d.shape[0]*inputs_3d.shape[1] * mpjpe(pred_root, inputs_traj).item()

            epoch_loss_3d_pos += inputs_3d.shape[0]*inputs_3d.shape[1] * error.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

            epoch_loss_3d_pos_procrustes += inputs_3d.shape[0]*inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

            # Compute velocity error
            epoch_loss_3d_vel += inputs_3d.shape[0]*inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)
    if action is None:
        print('----------')
    else:
        print('----'+action+'----')
    e1 = (epoch_loss_3d_pos / N)*1000
    e2 = (epoch_loss_3d_pos_procrustes / N)*1000
    e3 = (epoch_loss_3d_pos_scale / N)*1000
    e4 = (epoch_mrpe / N)*1000
    ev = (epoch_loss_3d_vel / N)*1000
    print('Test time augmentation:', test_generator.augment_enabled())
    print(f'Protocol #1 Error (MPJPE)    :', f'{e1:.1f}', 'mm')
    print(f'Protocol #2 Error (Abs-MPJPE):', f'{e3:.1f}', 'mm')
    print(f'Protocol #3 Error (MRPE)     :', f'{e4:.1f}', 'mm')
    print(f'Protocol #4 Error (P-MPJPE)  :', f'{e2:.1f}', 'mm')
    print(f'Velocity    Error (MPJVE)    :', f'{ev:.2f}', 'mm')
    print('----------')

    return e1, e2, e3, e4, ev




print('Evaluating...')
all_actions = {}
all_actions_by_subject = {}
for subject in subjects_test:
    if subject not in all_actions_by_subject:
        all_actions_by_subject[subject] = {}

    for action in dataset[subject].keys():
        action_name = action.split(' ')[0]
        if action_name not in all_actions:
            all_actions[action_name] = []
        if action_name not in all_actions_by_subject[subject]:
            all_actions_by_subject[subject][action_name] = []
        all_actions[action_name].append((subject, action))
        all_actions_by_subject[subject][action_name].append((subject, action))

def fetch_actions(actions):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []

    for subject, action in actions:
        poses_2d = keypoints[subject][action]
        for i in range(len(poses_2d)): # Iterate across cameras
            out_poses_2d.append(poses_2d[i])

        poses_3d = dataset[subject][action]['positions_3d']
        assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
        for i in range(len(poses_3d)): # Iterate across cameras
            out_poses_3d.append(poses_3d[i])
        
        if subject in dataset.cameras():
            cams = dataset.cameras()[subject]
            assert len(cams) == len(poses_2d), 'Camera count mismatch'
            for cam in cams:
                if 'intrinsic' in cam:
                    out_camera_params.append(cam['intrinsic'])


    stride = args.downsample
    if stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_camera_params ,out_poses_3d, out_poses_2d

def run_evaluation(actions, action_filter=None):
    errors_p1 = []
    errors_p2 = []
    errors_p3 = []
    errors_p4 = []
    errors_vel = []
    # joints_errs_list=[]

    for action_key in actions.keys():
        # action_key = 'SittingDown' # 제일 루트 안 좋은 예
        # action_key = 'Greeting' # p2d가 제일 안 좋은 예
        if action_filter is not None:
            found = False
            for a in action_filter:
                if action_key.startswith(a):
                    found = True
                    break
            if not found:
                continue

        cams_act ,poses_act, poses_2d_act = fetch_actions(actions[action_key])
        gen = UnchunkedGenerator_Seq(cams_act, poses_act, poses_2d_act,
                                pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                                kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                joints_right=joints_right)
        e1, e2, e3, e4, ev = evaluate(gen, action_key)
        
        # joints_errs_list.append(joints_errs)

        errors_p1.append(e1)
        errors_p2.append(e2)
        errors_p3.append(e3)
        errors_p4.append(e4)
        errors_vel.append(ev)
    
    print('Protocol #1     (MPJPE) action-wise average:', round(np.mean(errors_p1), 1), 'mm')
    print('Protocol #2 (Abs-MPJPE) action-wise average:', round(np.mean(errors_p3), 1), 'mm')
    print('Protocol #3      (MRPE) action-wise average:', round(np.mean(errors_p4), 1), 'mm')
    print('Protocol #4   (P-MPJPE) action-wise average:', round(np.mean(errors_p2), 1), 'mm')
    print('Velocity        (MPJVE) action-wise average:', round(np.mean(errors_vel), 2), 'mm')


    # joints_errs_np = np.array(joints_errs_list).reshape(-1, 17)
    # joints_errs_np = np.mean(joints_errs_np, axis=0).reshape(-1)
    # with open('output/mpjpe_joints.csv', 'a+') as f:
    #     for i in joints_errs_np:
    #         f.write(str(i)+'\n')

if not args.by_subject:
    run_evaluation(all_actions, action_filter)
else:
    for subject in all_actions_by_subject.keys():
        print('Evaluating on subject', subject)
        run_evaluation(all_actions_by_subject[subject], action_filter)
        print('')


    


