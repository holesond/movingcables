import os
import sys
import pathlib
import inspect
import time

import numpy as np
import cv2

script_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
unimatch_root = os.path.join(script_dir,"../unimatch")
sys.path.append(unimatch_root)

try:
    import torch
    import torch.nn.functional as F

    from unimatch.unimatch import UniMatch
except ImportError as e:
    print(
        f"Note: Failed to import the optional unimatch module. "
        f"(Maybe you have not installed it.)\n"
        f"    ({e})")



class OnlineFlowGMFlow():
    """Online optical flow computation interface to the GMFlow method."""
    def __init__(self, gpu=False, mode="scale1", seed=326):
        self.feature_channels = 128
        self.num_head = 1
        self.ffn_dim_expansion = 4
        self.num_transformer_layers = 6
        self.reg_refine = False
        self.attn_type = 'swin'
        self.num_reg_refine = 1
        self.pred_bidir_flow = False
        self.local_rank = 0
        self.strict_resume = False
        if mode == "scale1":
            self.num_scales = 1
            self.upsample_factor = 8
            self.padding_factor = 16
            self.attn_splits_list = [2]
            self.corr_radius_list = [-1]
            self.prop_radius_list = [-1]
            self.resume = os.path.join(
                unimatch_root,
                "pretrained/gmflow-scale1-mixdata-train320x576-4c3a6e9a.pth")
        elif mode == "scale2":
            self.num_scales = 2
            self.upsample_factor = 4
            self.padding_factor = 32
            self.attn_splits_list = [2, 8]
            self.corr_radius_list = [-1, 4]
            self.prop_radius_list = [-1, 1]
            self.resume = os.path.join(
                unimatch_root,
                "pretrained/gmflow-scale2-mixdata-train320x576-9ff1c094.pth")
        else:
            ValueError(
                f"Invalid GMFlow mode {mode}. "
                f"Supported modes are scale1 or scale2.")
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
        torch.backends.cudnn.benchmark = True
        if gpu and not torch.cuda.is_available():
            raise RuntimeError(
                "Asked for execution on a GPU but CUDA is not available.")
        self.device = torch.device('cuda' if gpu else 'cpu')
        self.model = UniMatch(
            feature_channels=self.feature_channels,
            num_scales=self.num_scales,
            upsample_factor=self.upsample_factor,
            num_head=self.num_head,
            ffn_dim_expansion=self.ffn_dim_expansion,
            num_transformer_layers=self.num_transformer_layers,
            reg_refine=self.reg_refine,
            task="flow").to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=4e-4, weight_decay=1e-4)
        print('Load checkpoint: %s' % self.resume)
        loc = 'cuda:{}'.format(self.local_rank) if gpu else 'cpu'
        checkpoint = torch.load(self.resume, map_location=loc)
        self.model.load_state_dict(
            checkpoint['model'], strict=self.strict_resume)
        if ('optimizer' in checkpoint and 'step' in checkpoint
                and 'epoch' in checkpoint):
            print('Load optimizer')
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_step = checkpoint['step']
            print(f"    start_epoch: {start_epoch}, start_step: {start_step}")
        self.model.eval()
        
        
    
    def flow(self, img1, img2):
        """Return optical flow given a pair of BGR/RGB images."""
        transpose_img = False
        assert(img1.dtype == np.uint8)
        assert(img2.dtype == np.uint8)
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
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
        image2 = image2.unsqueeze(0).to(self.device)

        # the model is trained with size: width > height
        if image1.size(-2) > image1.size(-1):
            image1 = torch.transpose(image1, -2, -1)
            image2 = torch.transpose(image2, -2, -1)
            transpose_img = True

        nearest_size = [
            int(np.ceil(image1.size(-2) / self.padding_factor)) * self.padding_factor,
            int(np.ceil(image1.size(-1) / self.padding_factor)) * self.padding_factor,
            ]

        inference_size = nearest_size

        assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
        ori_size = image1.shape[-2:]

        # resize before inference
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            image1 = F.interpolate(
                image1, size=inference_size, mode='bilinear',
                align_corners=True)
            image2 = F.interpolate(
                image2, size=inference_size, mode='bilinear',
                align_corners=True)
        results_dict = self.model(
            image1, image2,
            attn_type=self.attn_type,
            attn_splits_list=self.attn_splits_list,
            corr_radius_list=self.corr_radius_list,
            prop_radius_list=self.prop_radius_list,
            num_reg_refine=self.num_reg_refine,
            task='flow',
            pred_bidir_flow=self.pred_bidir_flow,
            )
        flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]

        # resize back
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                    align_corners=True)
            flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
            flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

        if transpose_img:
            flow_pr = torch.transpose(flow_pr, -2, -1)

        flow = flow_pr[0].permute(1, 2, 0).cpu().detach().numpy()  # [H, W, 2]
        return flow, None, None

