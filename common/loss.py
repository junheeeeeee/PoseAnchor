from matplotlib.pyplot import bone
import torch
import numpy as np

def mpjpe(predicted, target, return_joints_err=False):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    if not return_joints_err:
        return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))
    else:
        errors = torch.norm(predicted - target, dim=len(target.shape)-1)
        # errors: [B, T, N]
        from einops import rearrange
        errors = rearrange(errors, 'B T N -> N (B T)')
        errors = torch.mean(errors, dim=-1).cpu().numpy().reshape(-1) * 1000
        return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1)), errors
    
def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    # assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))

def get_limb_lens(x):
    '''
        Input: (N, T, 17, 3)
        Output: (N, T, 16)
    '''
    limbs_id = [[0,1], [1,2], [2,3],
         [0,4], [4,5], [5,6],
         [0,7], [7,8], [8,9], [9,10],
         [8,11], [11,12], [12,13],
         [8,14], [14,15], [15,16]
        ]
    limbs = x[:,:,limbs_id,:]
    limbs = limbs[:,:,:,0,:]-limbs[:,:,:,1,:]
    limb_lens = torch.norm(limbs, dim=-1)
    return limb_lens

def change_bone_length(p3d, bone_length):
    joints = p3d + 0
    index = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12,13],[8,14], [14,15],[15,16]]
    bone = []
    conf = []
    for i in index:
        conf.append(torch.max(torch.linalg.norm(joints[:, :, i[1]] - joints[:, :, i[0]], axis=-1, keepdims=True), torch.tensor([1e-6]).cuda()))
        bone.append((joints[:, :, i[1]] - joints[:, :, i[0]]) / conf[-1])
    
    for i in range(len(index)):
        joints[:, :, index[i][1]] = joints[:, :, index[i][0]] + bone[i] * bone_length[:, :, i, None]
    conf = torch.nn.Sigmoid()(torch.stack(conf, axis=-1).permute(0, 1, 3, 2))
    conf = torch.cat([torch.ones_like(conf[:, :, :1]), conf], dim=-2)
    return joints, conf

def get_bone_length(p3d):
    index = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]]
    bone = []
    for i in index:
        bone.append(torch.linalg.norm(p3d[:, :, i[1]] - p3d[:, :, i[0]], axis=-1))
    bone = torch.stack(bone, axis=-1)
    return bone

def tmpjpe(predicted, target):
    p_bone = get_bone_length(predicted)
    t_bone = get_bone_length(target)    
    tpose = torch.tensor([[[0.,0.,0],[1,0,0],[1,-1,0],[1,-2,0],[-1,0,0],[-1,-1,0],[-1,-2,0],[0,1,0],[0,2,0],[0,3,0],[0,4,0],[-1,2,0],[-2,2,0],[-3,2,0],[1,2,0],[2,2,0],[3,2,0]]]).squeeze(0).to(predicted.device)
    tpose = tpose.repeat(predicted.shape[0], predicted.shape[1],1, 1)


    pred, conf = change_bone_length(tpose, p_bone)

    targ, conf = change_bone_length(tpose, t_bone)
    return mpjpe(pred, targ)



def smpjpe(predicted, target):
    """
    Weighted subject independent mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    # assert w.shape[0] == predicted.shape[0]
    index = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12,13],[8,14], [14,15],[15,16]]

    bone_length = []
    for i in index:
        bone_length.append(torch.linalg.norm(target[:,0,i[1]] - target[:,0,i[0]], axis=-1))
    bone_length = torch.stack(bone_length, dim=-1)

    bone = []
    for i in index:
        bone.append((predicted[:,:,i[1]] - predicted[:,:,i[0]]) / torch.linalg.norm(predicted[:,:,i[1]] - predicted[:,:,i[0]], axis=-1, keepdims=True))
    
    for i in range(len(index)):
        predicted[:, :, index[i][1]] = predicted[:, :, index[i][0]] + bone[i] * bone_length[:, None, i, None]

    return mpjpe(predicted, target)

def integrate_normal_pdf(a, b, mean=0.0, std=1.0):
    """
    정규분포의 주어진 구간 [a, b]에서의 적분 값을 계산합니다.
    
    매개변수:
    a (float): 적분 구간의 시작점
    b (float): 적분 구간의 끝점
    mean (float, optional): 정규분포의 평균 (기본값: 0.0)
    std (float, optional): 정규분포의 표준편차 (기본값: 1.0)
    
    반환값:
    float: 정규분포의 [a, b] 구간에서의 적분 값
    """
    # 정규분포의 CDF (누적 분포 함수)
    def normal_cdf(x, mean, std):
        return 0.5 * (1 + torch.erf((x - mean) / (std * torch.sqrt(torch.tensor(2.0)))))

    return normal_cdf(b, mean, std) - normal_cdf(a, mean, std)

def cross_entropy_loss(predicted, target, sigma=1.0):
    """
    Cross entropy loss
    """

    bins_num = predicted.shape[-1]
    bins = torch.linspace(-1, 1, bins_num).to(predicted.device)
    term = (bins[1] - bins[0]) / 2
    sigma = term * sigma
    target_dist = torch.zeros_like(predicted)
    for i, j in enumerate(bins):
        target_dist[..., i] = integrate_normal_pdf(j - term, j + term, mean=target, std=sigma)
    

    # import matplotlib.pyplot as plt
    # distributions = target_dist[:,:,:,2].detach().cpu()

    # plt.figure()
    # distribution = distributions[0, 0, 13].numpy()
    # plt.plot(np.linspace(-2, 2, bins_num), distribution)
    # plt.axvline(x=target[0,0,13,2].cpu(), color='r', linestyle='-')
    # mean = torch.sum(bins.cpu() * distribution)
    # plt.axvline(x=mean, color='g', linestyle='--')
    # plt.xlabel('Bins')
    # plt.ylabel('Probability')

    # plt.title(f'Distribution for sample {0}, frame {0}, joint {13}')
    # plt.show()

    return -torch.mean(target_dist * torch.log(predicted))

def weighted_cross(predicted, p3d,target, sigma=0.1, w=None):
    """
    Cross entropy loss
    """

    bins_num = predicted.shape[-1]
    bins = torch.linspace(-1.2, 1.2, bins_num).to(predicted.device)
    term = (bins[1] - bins[0]) / 2
    sigma = term * sigma
    target_dist = torch.zeros_like(predicted)
    for i, j in enumerate(bins):
        target_dist[..., i] = integrate_normal_pdf(j - term, j + term, mean= target, std=sigma)
    
    target_dist = target_dist / torch.sum(target_dist, dim=-1, keepdim=True)

    # import matplotlib.pyplot as plt
    # distributions = target_dist[:,:,:,2].detach().cpu()

    # plt.figure()
    # distribution = distributions[0, 0, 13].numpy()
    # plt.plot(np.linspace(-1.2, 1.2, bins_num), distribution)
    # plt.axvline(x=target[0,0,13,2].cpu(), color='r', linestyle='-')
    # mean = torch.sum(bins.cpu() * distribution)
    # plt.axvline(x=mean, color='g', linestyle='--')
    # plt.xlabel('Bins')
    # plt.ylabel('Probability')
    # plt.title(f'Distribution for sample {0}, frame {0}, joint {13}, gt: {target[0,0,13,2] * 1000}, mean: {mean * 1000}')
    # plt.show()

    cross_entropy_loss = -torch.mean(target_dist * torch.log(predicted) * w.reshape(1,1,-1,1,1)) 
    mse_loss = torch.mean(torch.norm(p3d - target, dim= -1) * w.reshape(1,1,-1))

    loss = cross_entropy_loss + mse_loss * 0
    return loss

def show_distribution(predicted, target, sigma=0.01):
    """
    Cross entropy loss
    """

    bins_num = predicted.shape[-1]
    bins = torch.linspace(-1.2, 1.2, bins_num).to(predicted.device)
    term = (bins[1] - bins[0]) / 2
    # sigma = term * sigma
    target_dist = torch.zeros_like(predicted)
    for i, j in enumerate(bins):
        target_dist[..., i] = integrate_normal_pdf(j - term, j + term, mean= target, std=sigma)
    
    target_dist = target_dist / torch.sum(target_dist, dim=-1, keepdim=True)

    import matplotlib.pyplot as plt
    distributions = target_dist[:,:,:,2].detach().cpu()
    dis = predicted[:,:,:,2].detach().cpu()

    plt.figure()
    distribution = distributions[0, 0, 13].numpy()
    # plt.plot(np.linspace(-1.2, 1.2, bins_num), distribution, color='b', label='gt')
    plt.plot(np.linspace(-1.2, 1.2, bins_num), dis[0,0,13].numpy(), color='g', label='pred')
    plt.axvline(x=target[0,0,13,2].cpu(), color='b', linestyle='--')
    mean = torch.sum(bins.cpu() * distribution)
    predicted_mean = torch.sum(bins.cpu() * dis[0,0,13])
    plt.axvline(x=predicted_mean, color='g', linestyle='--')
    plt.xlabel('Bins')
    plt.ylabel('Probability')
    plt.title(f'Distribution for sample {0}, frame {0}, joint {13}, gt: {target[0,0,13,2] * 1000}, pred: {predicted_mean * 1000}')
    plt.show()


def p_mpjpe(predicted, target):
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1))
    
def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape
    
    norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe(scale * predicted, target)


def mean_velocity_error_train(predicted, target, axis=0):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    
    velocity_predicted = torch.diff(predicted, dim=axis)
    velocity_target = torch.diff(target, dim=axis)

    return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=len(target.shape)-1))

def mean_velocity_error(predicted, target, axis=0):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape

    velocity_predicted = np.diff(predicted, axis=axis)
    velocity_target = np.diff(target, axis=axis)
    
    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape)-1))

def sym_penalty(dataset,keypoints,pred_out):
    """
    get penalty for the symmetry of human body
    :return:
    """
    loss_sym = 0
    if dataset == 'h36m':
        if keypoints.startswith('hr'):
            left_bone = [(0,4),(4,5),(5,6),(8,10),(10,11),(11,12)]
            right_bone = [(0,1),(1,2),(2,3),(8,13),(13,14),(14,15)]
        else:
            left_bone = [(0,4),(4,5),(5,6),(8,11),(11,12),(12,13)]
            right_bone = [(0,1),(1,2),(2,3),(8,14),(14,15),(15,16)]
        for (i_left,j_left),(i_right,j_right) in zip(left_bone,right_bone):
            left_part = pred_out[:,:,i_left]-pred_out[:,:,j_left]
            right_part = pred_out[:, :, i_right] - pred_out[:, :, j_right]
            loss_sym += torch.mean(torch.abs(torch.norm(left_part, dim=-1) - torch.norm(right_part, dim=-1)))
    elif dataset.startswith('STB'):
        loss_sym = 0
    return 0.01*loss_sym

def bonelen_consistency_loss(dataset,keypoints,pred_out):
    loss_length = 0
    if dataset == 'h36m':
        if keypoints.startswith('hr'):
            assert "hrnet has not completed"
        else:
            bones = [(0,1), (0,4), (1,2), (2,3), (4,5), (5,6), (0,7), (7,8), (8,9), (9,10), 
                    (8,11), (11,12), (12,13), (8,14), (14,15), (15,16)]
        for (i,j) in bones:
            bonelen = pred_out[:,:,i]-pred_out[:,:,j]
            bone_diff = bonelen[:,1:,:]-bonelen[:,:-1,:]
            loss_length += torch.mean(torch.norm(bone_diff, dim=-1))
    elif dataset.startswith('heva'):
        loss_length = 0

    return 0.01 * loss_length