"""
Data loading utilities for AgeStyle model.
Simplified for inference only.
"""
import os
import sys
import torch
import yaml
import numpy as np

BASEPATH = os.path.dirname(__file__)
from os.path import join as pjoin
sys.path.insert(0, BASEPATH)

from utils.animation_data import AnimationData


def normalize_motion(motion, mean_pose, std_pose):
    """
    Normalize motion data.
    
    Parameters
    ----------
    motion : tensor
        Motion data, shape (V, C, T) or (C, T)
    mean_pose : tensor
        Mean pose, shape (C, 1)
    std_pose : tensor
        Standard deviation, shape (C, 1)
        
    Returns
    -------
    tensor
        Normalized motion
    """
    return (motion - mean_pose) / std_pose


def single_to_batch(data):
    """
    Convert single sample to batch format.
    
    Parameters
    ----------
    data : dict
        Single sample data
        
    Returns
    -------
    dict
        Batched data with batch dimension added
    """
    for key, value in data.items():
        if key == "meta":
            data[key] = {sub_key: [sub_value] for sub_key, sub_value in value.items()}
        elif key == "content_label":
            pass
        elif key == "label":
            pass
        else:
            data[key] = value.unsqueeze(0)
    return data


def process_single_bvh(filename, config, norm_data_dir=None, downsample=4, skel=None, to_batch=False):
    """
    Process a single BVH file for inference.
    
    Parameters
    ----------
    filename : str
        Path to the BVH file
    config : Config
        Configuration object
    norm_data_dir : str, optional
        Directory containing normalization data
    downsample : int
        Downsample factor for BVH loading
    skel : Skel, optional
        Skeleton object
    to_batch : bool
        Whether to convert to batch format
        
    Returns
    -------
    dict
        Processed data dictionary containing:
        - meta: metadata
        - foot_contact: foot contact labels
        - contentraw: raw content data
        - style3draw: raw style 3d data
        - content_label: content type label
        - label: style label
        - content: normalized content
        - style3d: normalized style 3d
    """
    def to_tensor(x):
        return torch.tensor(x).float().to(config.device)

    # Load and process BVH
    anim = AnimationData.from_BVH(filename, downsample=downsample, skel=skel, trim_scale=4)
    foot_contact = anim.get_foot_contact(transpose=True)  # [4, T]
    content = to_tensor(anim.get_content_input())
    style3d = to_tensor(anim.get_style3d_input())

    # Load dataset configuration for label mapping
    dataset_config = pjoin(BASEPATH, "global_info/xia_dataset.yml")
    with open(dataset_config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    
    content_namedict = [full_name.split('_')[0] for full_name in cfg["content_full_names"]]
    content_names = cfg["content_names"]
    style_names = cfg["style_names"]
    
    style_name_to_idx = {name: i for i, name in enumerate(style_names)}
    content_name_to_idx = {name: i for i, name in enumerate(content_names)}
    
    # Parse filename to get labels
    # Expected format: {style}_{content_idx}_{clip_idx}.bvh
    filenameforBVH = filename.split('/')[-1].split('\\')[-1]  # Handle both / and \ separators
    parts = filenameforBVH.replace('.bvh', '').split('_')
    
    if len(parts) >= 3:
        style_name = parts[0]
        content_idx_str = parts[1]
        
        # Get content label from index
        content_name = content_namedict[int(content_idx_str) - 1]
        content_idx = content_name_to_idx[content_name]
        content_idx = torch.tensor(int(content_idx))
        content_idx = content_idx[None]
        
        # Get style label
        style_idx = style_name_to_idx.get(style_name, 0)
        style_idx = torch.tensor(int(style_idx))
        style_idx = style_idx[None]
    else:
        # Default labels if parsing fails
        content_idx = torch.tensor([0])
        style_idx = torch.tensor([0])

    data = {
        "meta": {"style": "test", "content": filenameforBVH},
        "foot_contact": to_tensor(foot_contact),
        "contentraw": content,
        "style3draw": style3d,
        "content_label": content_idx,
        "label": style_idx
    }

    # Load normalization parameters and normalize data
    if norm_data_dir is None:
        norm_data_dir = config.extra_data_dir
        
    for key, raw in zip(["content", "style3d"], [content, style3d]):
        norm_path = os.path.join(norm_data_dir, f'train_{key}.npz')
        if os.path.exists(norm_path):
            norm = np.load(norm_path, allow_pickle=True)
            data[key] = normalize_motion(
                raw,
                to_tensor(norm['mean']).unsqueeze(-1),
                to_tensor(norm['std']).unsqueeze(-1)
            )
        else:
            print(f"Warning: Normalization file not found: {norm_path}")
            data[key] = raw

    if to_batch:
        data = single_to_batch(data)

    return data
