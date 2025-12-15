"""
AgeStyle Model - Motion Style Transfer Network.
Simplified for inference only.
"""
import torch
import torch.nn as nn
import numpy as np

from networks import PatchDis, JointGen

# Optional CLIP import for text-guided features (not required for basic inference)
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP not available. Text-guided features disabled.")


class Model(nn.Module):
    """
    AgeStyle Model for motion style transfer.
    
    This model takes content motion and style motion as inputs,
    and generates motion with the content's movement but the style's characteristics.
    """
    def __init__(self, config):
        super(Model, self).__init__()

        self.gen = JointGen(config)
        self.dis = PatchDis(config)
        self.device = config.device

    @staticmethod
    def split_pos_glb(raw):
        """Split position and global info from raw data."""
        # raw: [B, (J - 1) * 3 + 4, T]
        return raw[:, :-4, :], raw[:, -4:, :]

    @staticmethod
    def merge_pos_glb(pos, glb):
        """Merge position and global info."""
        # [B, (J - 1) * 3, T], [B, 4, T]
        return torch.cat([pos, glb], dim=-2)

    def test(self, co_data, cl_data, status):
        """
        Perform motion style transfer.
        
        Parameters
        ----------
        co_data : dict
            Content motion data (the motion to transform)
        cl_data : dict
            Style motion data (the target style)
        status : str
            Output status, typically "3d"
            
        Returns
        -------
        dict
            Output dictionary containing:
            - content_meta: metadata of content
            - style_meta: metadata of style
            - foot_contact: foot contact labels
            - content: original content motion
            - recon: reconstructed motion (same style as input)
            - trans: transferred motion (content + target style)
            - style: style motion
        """
        self.eval()
        self.gen.eval()

        xtgt = co_data["style3draw"]
        xa = co_data["content"]
        stylestr = "style" + status
        
        if stylestr in co_data:
            content_stylestr = stylestr
        else:
            content_stylestr = "style3d"
        ya = co_data[content_stylestr]
        yb = cl_data[stylestr]

        la = co_data["content_label"]

        xo, xglb = self.split_pos_glb(xtgt)
        
        # Encode content
        c_xa = self.gen.enc_content(xa)
        
        # Encode styles using transformer
        s_xao = self.gen.transformer(ya, la.to(self.device))
        s_xbo = self.gen.transformer(yb, la.to(self.device))

        # Add content label embedding to style code
        la_emb = self.gen.content_embedding(la.to(self.device))
        c_la = self.gen.content_mlp(la_emb)
        c_la = c_la.unsqueeze(-1)
        
        s_xa = torch.mul(torch.sigmoid(c_la), torch.sigmoid(s_xao))
        s_xb = torch.mul(torch.sigmoid(c_la), torch.sigmoid(s_xbo))

        # Residual connection
        s_xa = s_xa + torch.sigmoid(s_xao)
        s_xb = s_xb + torch.sigmoid(s_xbo)

        # Decode
        _, rxt = self.gen.decode(c_xa, s_xb)  # transferred motion
        _, rxr = self.gen.decode(c_xa, s_xa)  # reconstructed motion
        
        full_r = self.merge_pos_glb(rxr, xglb)
        full_t = self.merge_pos_glb(rxt, xglb)

        out_dict = {
            "content_meta": co_data["meta"],
            "style_meta": cl_data["meta"],
            "foot_contact": co_data["foot_contact"],
            "content": co_data["contentraw"],
            "recon": full_r,
            "trans": full_t,
        }

        if status == "3d":
            out_dict["style"] = cl_data["contentraw"]
        else:
            out_dict["style"] = cl_data["style2draw"]

        return out_dict
