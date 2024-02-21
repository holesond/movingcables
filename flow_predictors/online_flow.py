import os
import time
import pathlib
import sys
import inspect

import numpy as np
import cv2

script_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
mfn_repo_root = os.path.join(script_dir,"../MaskFlownet")
sys.path.append(mfn_repo_root)
import predict_new_data as predict_flow
predict_flow.repoRoot = mfn_repo_root



class OnlineFlow():
    """Online optical flow prediction interface to MaskFlownet(S)(Prob)."""
    def __init__(
            self, gpu=False, args=None, probabilistic=False, small=False):
        """Initialize the interface and load the chosen neural network models.
        
        Keyword arguments:
        gpu -- compute flow on a GPU (default False)
        args -- configuration options to pass directly to setup_pipeline,
            overrides all other arguments if not None (default None)
        probabilistic -- If True, use MaskFlownetProb(S). Otherwise
            use MaskFlownet(S). (default False)
        small -- load the small (S) version of MaskFlownet(Prob)
            (default False)
        """
        if args is not None:
            self.setup_pipeline(args)
            return
        args = lambda:0
        args.network = "MaskFlownet"
        args.gpu_device = ""
        if gpu:
            args.gpu_device = "0"
        args.clear_steps = False
        args.batch = 8
        args.threads = 8
        if small:
            args.config = "MaskFlownet_S.yaml"
            args.checkpoint = "dbbSep30"    # stage 3
        else:
            args.config = "MaskFlownet.yaml"
            args.checkpoint = "8caNov12"    # stage 6
        if probabilistic:
            args.network = "MaskFlownetProb"
            if small:
                args.config = "MaskFlownetSProb.yaml"
                #args.checkpoint = "376May03-1502"    # softplus stage 1
                #args.checkpoint = "ba1Apr13-1948"    # stage 1
                #args.checkpoint = "80cApr25-1625"    # stage 2
                #args.checkpoint = "038May09-1453"    # stage 2 softplus
                #args.checkpoint = "abaApr26-1816"    # stage 3
                #args.checkpoint = "e75May11-0845"    # stage 3 softplus q=0.4
                args.checkpoint = "b11May11-0920"    # stage 3 softplus q=None
            else:
                args.config = "MaskFlownetProb.yaml"
                #args.checkpoint = "0f6May06-2059"    # stage 6
                args.checkpoint = "99bMay18-1454"    # stage 6 softplus q=None
        self.setup_pipeline(args)
        
    def setup_pipeline(self, args):
        """Load and initialize the MaskFlownet pipeline."""
        checkpoint, steps = predict_flow.find_checkpoint(args.checkpoint, args)
        print("MaskFlownet checkpoint found.")
        config = predict_flow.load_model(args.config)
        print("MaskFlownet model loaded.")
        self.pipe = predict_flow.instantiate_model(
            args.gpu_device, config, args)
        print("MaskFlownet model instantiated.")
        self.pipe = predict_flow.load_checkpoint(self.pipe, config, checkpoint)
        print("MaskFlownet checkpoint loaded OK. Setup complete.")
        
    def flow(self, img1, img2):
        """Return optical flow given a pair of BGR/RGB images."""
        res = predict_flow.predict_image_pair_flow(
            img1, img2, self.pipe)
        flow, flow_var, occ_mask, warped = res
        if flow_var is None:
            return flow, occ_mask, warped
        else:
            return flow, flow_var, occ_mask, warped
