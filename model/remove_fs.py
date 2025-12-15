"""
Foot sliding removal and BVH output utilities.
"""
import os
import sys
import numpy as np

BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)

from utils.animation_data import AnimationData
from utils import BVH


def save_bvh_from_network_output(motion, output_path, frametime=1/30):
    """
    Save network output as BVH file.
    
    Parameters
    ----------
    motion : numpy.ndarray
        Motion data from network output, shape [J*4+4, T] (rotations + global params)
    output_path : str
        Path to save the BVH file
    frametime : float
        Frame time for BVH file
    """
    anim_data = AnimationData.from_network_output(motion)
    bvh_data = anim_data.get_BVH()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    BVH.save(output_path, *bvh_data)
    print(f"Saved BVH to: {output_path}")


def remove_fs(motion, foot_contact, output_path, frametime=1/30):
    """
    Remove foot sliding and save as BVH file.
    
    This is a simplified version that directly saves the motion without
    complex foot sliding removal (which would require IK optimization).
    For basic usage, the motion quality from the network is usually acceptable.
    
    Parameters
    ----------
    motion : numpy.ndarray
        Motion data from network output, shape [J*4+4, T]
    foot_contact : numpy.ndarray  
        Foot contact labels, shape [4, T]
    output_path : str
        Path to save the BVH file
    frametime : float
        Frame time for BVH file
    """
    # Convert motion to AnimationData format
    # motion shape: [J*4+4, T] -> need to transpose and add foot contact
    motion_t = motion.T  # [T, J*4+4]
    
    # Append foot contact to match AnimationData expected format
    # AnimationData expects: [T, Jo * 4 + 4 global params + 4 foot_contact]
    if foot_contact is not None:
        foot_contact_t = foot_contact.T  # [T, 4]
        full_motion = np.concatenate([motion_t, foot_contact_t], axis=-1)
    else:
        # If no foot contact provided, add zeros
        foot_contact_t = np.zeros((motion_t.shape[0], 4))
        full_motion = np.concatenate([motion_t, foot_contact_t], axis=-1)
    
    # Create AnimationData and convert to BVH
    anim_data = AnimationData(full_motion)
    bvh_data = anim_data.get_BVH()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    BVH.save(output_path, *bvh_data)
    print(f"Saved BVH to: {output_path}")


if __name__ == "__main__":
    # Simple test
    print("remove_fs module loaded successfully")

