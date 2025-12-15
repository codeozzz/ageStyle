"""
Configuration for AgeStyle Model.
Simplified for inference only.
"""
import os
import sys
import torch
from os.path import join as pjoin

BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


class Config:
    """
    Configuration class for AgeStyle model.
    Contains model architecture parameters and paths.
    """
    
    # Experiment name (uses pretrained model)
    name = 'pretrained'
    
    # CUDA device
    cuda_id = 0

    # Data paths
    data_dir = pjoin(BASEPATH, 'data')
    expr_dir = BASEPATH
    data_filename = "xia.npz"
    data_path = pjoin(data_dir, data_filename)
    extra_data_dir = pjoin(data_dir, data_filename.split('.')[-2].split('/')[-1] + "_norms")

    # Model paths (will be set in initialize)
    main_dir = None
    model_dir = None
    output_dir = None

    # Input channel configurations
    rot_channels = 128  # rotation channels (with y-axis rotation)
    pos3d_channels = 64  # 3D position channels
    proj_channels = 42   # projection channels

    num_channel = rot_channels
    num_style_joints = 21

    style_channel_2d = proj_channels
    style_channel_3d = pos3d_channels

    # Style encoder configuration
    enc_cl_down_n = 2
    enc_cl_channels = [0, 96, 144]
    enc_cl_kernel_size = 8
    enc_cl_stride = 2

    # Content encoder configuration
    enc_co_down_n = 1
    enc_co_channels = [num_channel, 144]
    enc_co_kernel_size = 8
    enc_co_stride = 2
    enc_co_resblks = 1

    # MLP configuration
    mlp_dims = [enc_cl_channels[-1], 192, 256]
    content_mlp_dim = 6
    style_mlp_dim = 8
    out_dim = 256

    # Embedding parameters
    emb_num = 6
    emb_dim = 6

    # Decoder configuration
    dec_bt_channel = 144
    dec_resblks = enc_co_resblks
    dec_channels = enc_co_channels.copy()
    dec_channels.reverse()
    dec_channels[-1] = 31 * 4  # Output rotations only
    dec_up_n = enc_co_down_n
    dec_kernel_size = 8
    dec_stride = 1

    # Discriminator configuration
    disc_channels = [pos3d_channels, 96, 144]
    disc_down_n = 2
    disc_kernel_size = 6
    disc_stride = 1
    disc_pool_size = 3
    disc_pool_stride = 2

    # Number of style classes
    num_classes = 8

    # Device
    device = None
    gpus = 1

    # Dataset normalization config (for loading test data)
    dataset_norm_config = {
        "train": {"content": None, "style3d": None, "style2d": None},
        "test": {"content": "train", "style3d": "train", "style2d": "train"},
        "trainfull": {"content": "train", "style3d": "train", "style2d": "train"}
    }

    # Batch size for inference
    batch_size = 1

    def initialize(self, args=None, save=False):
        """
        Initialize configuration with command line arguments.
        
        Parameters
        ----------
        args : argparse.Namespace, optional
            Command line arguments
        save : bool
            Whether to save config (disabled for inference)
        """
        if hasattr(args, 'name') and args.name is not None:
            self.name = args.name

        if hasattr(args, 'batch_size') and args.batch_size is not None:
            self.batch_size = args.batch_size

        self.main_dir = os.path.join(self.expr_dir, self.name)
        self.model_dir = os.path.join(self.main_dir, "pth")
        self.output_dir = os.path.join(self.main_dir, "output")

        # Ensure directories exist
        ensure_dir(self.main_dir)
        ensure_dir(self.model_dir)
        ensure_dir(self.output_dir)
        ensure_dir(self.extra_data_dir)

        # Set device
        self.device = torch.device(
            "cuda:%d" % self.cuda_id if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")
