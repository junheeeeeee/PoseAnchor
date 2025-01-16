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

    x = x.view(b, t, 3)
    residuals = residuals.view(b, t, N , 2)

    return x.unsqueeze(2), residuals




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