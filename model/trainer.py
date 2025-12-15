"""
Trainer module for AgeStyle model.
Simplified for inference only - training code removed.
"""
import os
import torch
import torch.nn as nn

from model import Model


def get_model_list(dirname, key):
    """Get the latest model checkpoint file."""
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and
                  key in f and ".pt" in f]
    if gen_models is None or len(gen_models) == 0:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


class Trainer(nn.Module):
    """
    Trainer class for AgeStyle model.
    Simplified for inference only.
    """
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.model = Model(cfg)
        self.model_dir = cfg.model_dir
        self.cfg = cfg

    def test(self, co_data, cl_data, status):
        """Run inference on content and style data."""
        return self.model.test(co_data, cl_data, status)

    def resume(self):
        """Load pretrained model weights."""
        model_dir = self.model_dir

        last_model_name = get_model_list(model_dir, "gen")
        if last_model_name is None:
            print('No pretrained model found, using random initialization')
            return 0

        state_dict = torch.load(last_model_name, map_location=self.cfg.device)
        self.model.gen.load_state_dict(state_dict['gen'])
        
        last_model_name = get_model_list(model_dir, "dis")
        state_dict = torch.load(last_model_name, map_location=self.cfg.device)
        self.model.dis.load_state_dict(state_dict['dis'])

        iterations = int(last_model_name[-11:-3])
        print('Resume from iteration %d' % iterations)
        return iterations
