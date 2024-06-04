"""
Run motion segmentation on an image sequence.
Save images with overlaid segmentation masks.
"""

import os
import sys
import inspect
import pathlib
import argparse
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as imageio
import cv2

from evaluator.evaluator import ProgressKeeper
from evaluator.segmenters import FlowSegmenter
from flow_predictors.online_flow import OnlineFlow as OnlineFlowMaskFlownet
from flow_predictors.online_flow_farneback import OnlineFlow as \
    OnlineFlowFarneback
from flow_predictors.online_flow_gmflow import OnlineFlowGMFlow
from flow_predictors.online_flow_flowformerpp import OnlineFlowFlowFormerPP



def get_flow_segmenter(method_name, gpu=False, debug=False):
    if method_name == "maskflownet":
        flow_predictor = OnlineFlowMaskFlownet(
            gpu=gpu, probabilistic=False, small=False)
        flow_predictor_prob = None
    elif method_name == "maskflownet_ft":
        flow_predictor = None
        flow_predictor_prob = OnlineFlowMaskFlownet(
            gpu=gpu, probabilistic=False, small=False, finetuned=True)
    elif method_name == "mfnprob":
        flow_predictor = None
        flow_predictor_prob = OnlineFlowMaskFlownet(
            gpu=gpu, probabilistic=True, small=False)
    elif method_name == "mfnprob_ft":
        flow_predictor = None
        flow_predictor_prob = OnlineFlowMaskFlownet(
            gpu=gpu, probabilistic=True, small=False, finetuned=True)
    elif method_name == "farneback":
        flow_predictor = OnlineFlowFarneback(gpu)
        flow_predictor_prob = None
    elif method_name == "gmflow":
        flow_predictor = OnlineFlowGMFlow(gpu)
        flow_predictor_prob = None
    elif method_name == "flowformerpp":
        flow_predictor = OnlineFlowFlowFormerPP(gpu)
        flow_predictor_prob = None
    else:
        raise ValueError(f"Method {method_name} is not supported.")
    return FlowSegmenter(flow_predictor, flow_predictor_prob, debug)


def main(
        method_name, folder_rgb, folder_out,
        motion_threshold=2.5, gpu=False, debug=False, progress_queue=None,
        resize=None):
    """Run motion segmentation evaluation and save the results.
    
    Arguments:
    method_name -- segmentation method name
        ('maskflownet', 'maskflownet_ft', 'mfnprob', 'mfnprob_ft',
        'farneback', 'gmflow', 'flowformerpp')
    folder_rgb -- folder with an RGB image sequence to run inference on
    folder_out -- path of the output images with segmentation masks
    
    Keyword arguments:
    motion_threshold -- the flow magnitude segmentation threshold (default 2.5)
    gpu -- run on a GPU (default False)
    debug -- debug mode (default False)
    """
    mask_opacity = 0.7
    mask_color = np.array([0,255,0], dtype=np.float32)
    pathlib.Path(folder_out).mkdir(parents=True, exist_ok=True)
    filenames = sorted(os.listdir(folder_rgb))
    segmenter = get_flow_segmenter(method_name, gpu, debug)
    set_reference = True
    for filename in filenames:
        path_rgb = os.path.join(folder_rgb, filename)
        path_out = os.path.join(folder_out, filename)
        rgb = imageio.imread(path_rgb)
        rgb = rgb[...,0:3]
        if resize is not None:
            rgb = cv2.resize(rgb, resize, interpolation=cv2.INTER_AREA)
        actor_mask = None
        motion_mask, flow, uncertainty = segmenter.next_image(
            rgb, motion_threshold, set_reference, actor_mask)
        motion_mask = motion_mask > 0
        rgb_mask = rgb.astype(np.float32)
        rgb_mask[motion_mask,:] = (
            (1-mask_opacity)*rgb_mask[motion_mask,:]
            + mask_opacity*mask_color[None, None,:])
        rgb_mask = rgb_mask.astype(np.uint8)
        imageio.imwrite(path_out, rgb_mask)
        if set_reference:
            set_reference = False
        if progress_queue is not None:
            progress_queue.put(1)


def add_parser_options(parser):
    """Add common motion segmentation options to parser."""
    parser.add_argument(
        "-d", "--debug", action="store_true", default=False, help="debug mode")
    parser.add_argument(
        "-g", "--gpu", action="store_true", default=False,
        help="run the method on a GPU")
    parser.add_argument(
        '--mt', action='store', dest='motion_threshold', type=float,
        default=2.5, help="set the flow magnitude segmentation threshold")


def cmd_main():
    """Run the CLI stand-alone program."""
    parser = argparse.ArgumentParser(
        description=("Run motion segmentation on an image sequence. "
            "Save images with overlaid segmentation masks."))
    add_parser_options(parser)
    parser.add_argument(
        "method_name", default=None, type=str,
        help=("segmentation method name to run "
            "(maskflownet, maskflownet_ft, mfnprob, mfnprob_ft, "
            "farneback, gmflow, flowformerpp)"))
    parser.add_argument(
        "folder_rgb", default=None, type=str, help="input RGB(A) clip folder")
    parser.add_argument(
        "folder_out", default=None, type=str,
        help="output folder path")
    args = parser.parse_args()
    progress_keeper = ProgressKeeper()
    progress_keeper.setup([args.folder_rgb], cli_position=0)
    main(
        args.method_name, args.folder_rgb, args.folder_out,
        args.motion_threshold, args.gpu, args.debug,
        progress_queue=progress_keeper.progress_queue)
    progress_keeper.close()


if __name__ == "__main__":
    cmd_main()
