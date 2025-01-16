from model.MotionAGFormer import MotionAGFormer
from torch import nn
import torch


def load_model(model_name, args):
    act_mapper = {
        "gelu": nn.GELU,
        'relu': nn.ReLU
    }

    if model_name == "MotionAGFormer":
        model = MotionAGFormer(n_layers=16,
                               dim_in=2,
                               dim_feat=128,
                               dim_rep=512,
                               dim_out=3,
                               mlp_ratio=4,
                               act_layer=act_mapper["gelu"],
                               attn_drop=0.0,
                               drop=0.0,
                               drop_path=0.0,
                               use_layer_scale=True,
                               layer_scale_init_value=0.00001,
                               use_adaptive_fusion=True,
                               num_heads=8,
                               qkv_bias=False,
                               qkv_scale=None,
                               hierarchical=False,
                               num_joints=17,
                               use_temporal_similarity=True,
                               temporal_connection_len=1,
                               use_tcn=False,
                               graph_only=False,
                               neighbour_num=2,
                               n_frames=243)
    else:
        raise Exception("Undefined model name")

    return model