import torch
import numpy as np
import hashlib

def wrap(func, *args, unsqueeze=False):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """
    
    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)
        
    result = func(*args)
    
    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result
    
def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2**32 - 1) * (max_value - min_value)) + min_value

def load_pretrained_weights(model, checkpoint):
    """Load pretrianed weights to model
    Incompatible layers (unmatched in name or size) will be ignored
    Args:
    - model (nn.Module): network model, which must not be nn.DataParallel
    - weight_path (str): path to pretrained weights
    """
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    # new_state_dict.requires_grad = False
    model_dict.update(new_state_dict)

    model.load_state_dict(model_dict)
    print('load_weight', len(matched_layers))
    # model.state_dict(model_dict).requires_grad = False
    return model

def differentiable_pinv(A):
    # A shape: (..., M, N)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    # S: (..., min(M,N))
    S_inv = torch.where(S > 1e-6, 1/S, torch.zeros_like(S))  # 안정성 처리
    A_pinv = Vh.transpose(-2, -1) @ torch.diag_embed(S_inv) @ U.transpose(-2, -1)
    return A_pinv

def get_root(p3d, p2d, camera):
    """
    Calculate the root position (X_r, Y_r, Z_r) from 3D joint coordinates, 2D joint projections, and camera parameters.

    Args:
        p3d (np.ndarray): 3D joint coordinates of shape (..., N, 3), where N is the number of joints.
        p2d (np.ndarray): 2D joint projections of shape (..., N, 2).
        camera (np.ndarray): Camera intrinsic parameters of shape (3, 4).

    Returns:
        root_position (torch.Tensor): Root position of shape (..., 3) containing (X_r, Y_r, Z_r).
        residuals (torch.Tensor): Residuals of the least squares fit.
    """
    # Extract camera parameters
    fx = camera[..., 0][:, None]
    fy = camera[..., 1][:, None]
    cx = camera[..., 2][:, None]
    cy = camera[..., 3][:, None]

    # Prepare A and b matrices
    b, t, N = p3d.shape[0], p3d.shape[1], p3d.shape[2]
    d = torch.zeros_like(p2d).reshape(b, t, -1)
    A = torch.stack([d.clone(), d.clone(), d.clone()], dim=-1)
 

    for i in range(N):
        x = p2d[..., i, 0] - cx
        y = p2d[..., i, 1] - cy

        # Populate A matrix
        A[..., 2 * i, 0] = -fx
        A[..., 2 * i, 2] = x
        A[..., 2 * i + 1, 1] = -fy
        A[..., 2 * i + 1, 2] = y

        # Populate b vector
        d[..., 2 * i] = fx * p3d[..., i, 0] - x * p3d[..., i, 2]
        d[..., 2 * i + 1] = fy * p3d[..., i, 1] - y * p3d[..., i, 2]

    A = A.view(-1, N * 2, 3)  # Flatten batch and time
    d = d.view(-1, N * 2, 1)     # Flatten batch and time

    A_pinv = torch.linalg.pinv(A)
    x = torch.matmul(A_pinv, d).squeeze(-1)
    residuals = (A @ x.unsqueeze(-1)).squeeze(-1) - d.squeeze()

    thresholds = torch.tensor(170 / 1000, device=residuals.device)
    mask = residuals.abs() > thresholds
    d[mask] = 0
    A[mask.unsqueeze(-1).expand_as(A)] = 0

    A_pinv = differentiable_pinv(A)
    x = torch.matmul(A_pinv, d).squeeze(-1)


    # alpha = 10  # 가중치 조정 파라미터
    # weights = 1.0 / (1.0 + alpha * residuals.pow(2))

    # # 3. A와 d에 가중치 적용
    # W = weights.unsqueeze(-1)  # 가중치 형태 변환
    # A_weighted = A * W.expand_as(A)
    # d_weighted = d * W

    # # 4. Weighted Least Squares 계산
    # A_t = A_weighted.transpose(-2, -1)
    # A_pinv = torch.linalg.pinv(A_t @ A_weighted)
    # x = torch.matmul(A_pinv @ A_t, d_weighted).squeeze(-1)

    x = x.view(b, t, 3)
    residuals = residuals.view(b, t, N , 2)

    return x.unsqueeze(2), residuals


def refine_pose(p3d, p2d, camera, iterations=3, alpha=10):
    """
    Refine 3D joint coordinates using accurate gradient calculation based on residuals
    from get_root.

    Args:
        p3d (torch.Tensor): Initial 3D joint coordinates of shape (B, t, N, 3).
        p2d (torch.Tensor): 2D joint projections of shape (B, t, N, 2).
        camera (torch.Tensor): Camera intrinsic parameters of shape (B, 4).
        iterations (int): Number of refinement iterations.
        alpha (float): Learning rate for updating 3D and 2D coordinates.

    Returns:
        refined_p3d (torch.Tensor): Refined 3D joint coordinates of shape (B, t, N, 3).
        pred_root (torch.Tensor): Predicted root position of shape (B, t, 1, 3).
        refined_p2d (torch.Tensor): Refined 2D joint projections of shape (B, t, N, 2).
    """

    for _ in range(iterations):
        # Forward pass: Calculate residuals and root position
        pred_root, residuals = get_root(p3d, p2d, camera)

        # Compute A^T * residuals (gradient w.r.t. x)
        B_t, _, N, _ = residuals.shape  # Extract batch and temporal dimensions
        residuals_flat = residuals.view(B_t * N, -1)  # Flatten for batched matrix operations

        # Backpropagation through A and d
        grad_p3d = compute_grad_p3d(residuals_flat, p3d, camera)
        grad_p2d = compute_grad_p2d(residuals_flat, p2d, camera)

        # Update p3d and p2d using computed gradients
        p3d = p3d - alpha * grad_p3d
        # p2d = p2d - alpha * grad_p2d

    return p3d, pred_root, p2d

def compute_grad_p3d(residuals_flat, p3d, camera):
    """
    Compute the gradient of the loss w.r.t. p3d using residuals.

    Args:
        residuals_flat (torch.Tensor): Flattened residuals of shape (B * t * N, 2).
        p3d (torch.Tensor): 3D joint coordinates of shape (B, t, N, 3).
        camera (torch.Tensor): Camera intrinsic parameters of shape (B, 4).

    Returns:
        grad_p3d (torch.Tensor): Gradient of the loss w.r.t. p3d of shape (B, t, N, 3).
    """
    B, t, N, _ = p3d.shape

    # Reshape residuals back to (B, t, N, 2)
    residuals = residuals_flat.view(B, t, N, 2)

    # Extract camera parameters and expand dimensions
    fx = camera[..., 0].unsqueeze(-1).unsqueeze(-1).repeat(1, t, N)
    fy = camera[..., 1].unsqueeze(-1).unsqueeze(-1).repeat(1, t, N)

    # Compute gradient for p3d
    grad_p3d = torch.zeros_like(p3d)
    grad_p3d[..., 0] = fx * residuals[..., 0]
    grad_p3d[..., 1] = fy * residuals[..., 1]
    grad_p3d[..., 2] = -(fx * residuals[..., 0] + fy * residuals[..., 1])

    return grad_p3d


def compute_grad_p2d(residuals_flat, p2d, camera):
    """
    Compute the gradient of the loss w.r.t. p2d using residuals.

    Args:
        residuals_flat (torch.Tensor): Flattened residuals of shape (B * t * N, 2).
        p2d (torch.Tensor): 2D joint projections of shape (B, t, N, 2).
        camera (torch.Tensor): Camera intrinsic parameters of shape (B, 4).

    Returns:
        grad_p2d (torch.Tensor): Gradient of the loss w.r.t. p2d of shape (B, t, N, 2).
    """
    B, t, N, _ = p2d.shape

    # Reshape residuals back to (B, t, N, 2)
    residuals = residuals_flat.view(B, t, N, 2)

    # Extract camera parameters and expand dimensions
    fx = camera[..., 0].unsqueeze(-1).unsqueeze(-1).repeat(1, t, N)
    fy = camera[..., 1].unsqueeze(-1).unsqueeze(-1).repeat(1, t, N)

    # Compute gradient for p2d
    grad_p2d = torch.zeros_like(p2d)
    grad_p2d[..., 0] = -fx * residuals[..., 0]
    grad_p2d[..., 1] = -fy * residuals[..., 1]

    return grad_p2d


def project_to_2d_linear(X, camera_params):
    """
    Project 3D points to 2D using only linear parameters (focal length and principal point).
    
    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    # assert X.shape[0] == camera_params.shape[0]
    
    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)
        
    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    
    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    
    return f*XX + c

def get_residuals(p3d, p2d, root,camera):
    # Extract camera parameters
    fx = camera[..., 0][:, None]
    fy = camera[..., 1][:, None]
    cx = camera[..., 2][:, None]
    cy = camera[..., 3][:, None]

    # Prepare A and b matrices
    b, t, N = p3d.shape[0], p3d.shape[1], p3d.shape[2]
    d = torch.zeros_like(p2d).reshape(b, t, -1)
    A = torch.stack([d.clone(), d.clone(), d.clone()], dim=-1)
 

    for i in range(N):
        x = p2d[..., i, 0] - cx
        y = p2d[..., i, 1] - cy

        # Populate A matrix
        A[..., 2 * i, 0] = -fx
        A[..., 2 * i, 2] = x
        A[..., 2 * i + 1, 1] = -fy
        A[..., 2 * i + 1, 2] = y

        # Populate b vector
        d[..., 2 * i] = fx * p3d[..., i, 0] - x * p3d[..., i, 2]
        d[..., 2 * i + 1] = fy * p3d[..., i, 1] - y * p3d[..., i, 2]

    A = A.view(-1, N * 2, 3)  # Flatten batch and time
    d = d.view(-1, N * 2, 1)     # Flatten batch and time
    root = root.reshape(-1, 3, 1)
    
    residuals = (A @ root).squeeze(-1) - d.squeeze()

    residuals = residuals.view(b, t, N , 2)

    return residuals