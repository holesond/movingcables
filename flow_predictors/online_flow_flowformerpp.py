import os
import sys
import pathlib
import inspect
import time

import torch
import torch.nn.functional as F
import numpy as np
import cv2

script_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
flowformerpp_root = os.path.join(script_dir,"../FlowFormerPlusPlus")
sys.path.append(flowformerpp_root)
sys.path.append(os.path.join(flowformerpp_root, "core"))
#sys.path.append(os.path.join(flowformerpp_root, "configs"))

try:
    from core.FlowFormer import build_flowformer
    from utils.utils import InputPadder
    from configs.submissions import get_cfg
except ImportError as e:
    print(
        f"Note: Failed to import the optional FlowFormerPlusPlus module. "
        f"(Maybe you have not downloaded it.)\n"
        f"    ({e})")



class OnlineFlowFlowFormerPP():
    """Online optical flow computation interface to the FlowFormerPP method."""
    def __init__(self, gpu=False, seed=326):
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
        torch.backends.cudnn.benchmark = True
        if gpu and not torch.cuda.is_available():
            raise RuntimeError(
                "Asked for execution on a GPU but CUDA is not available.")
        self.device = torch.device('cuda' if gpu else 'cpu')
        cfg = get_cfg()
        cfg.model = os.path.join(flowformerpp_root,"checkpoints/sintel.pth")
        self.model = torch.nn.DataParallel(build_flowformer(cfg))
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(cfg.model))
        self.model.eval()
    
    def flow(self, img1, img2):
        """Return optical flow given a pair of BGR/RGB images."""
        weights = None
        flow = None
        assert(img1.dtype == np.uint8)
        assert(img2.dtype == np.uint8)
        with torch.no_grad():
            image1 = img1
            image2 = img2
            if len(image1.shape) == 2:  # gray image
                image1 = np.tile(image1[..., None], (1, 1, 3))
                image2 = np.tile(image2[..., None], (1, 1, 3))
            else:
                image1 = image1[..., :3]
                image2 = image2[..., :3]

            image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
            image1 = image1.unsqueeze(0).to(self.device)
            #image1 = image1.to(self.device)
            image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
            image2 = image2.unsqueeze(0).to(self.device)
            #image2 = image2.to(self.device)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_pre, _ = self.model(image1, image2)
            flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()
        return flow, None, None

