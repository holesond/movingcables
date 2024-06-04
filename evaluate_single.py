import os
import sys
import inspect
import pathlib
import argparse
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from evaluator import evaluator
from evaluator.evaluator import ProgressKeeper
from evaluator.segmenters import FlowSegmenter
from flow_predictors.online_flow import OnlineFlow as OnlineFlowMaskFlownet
from flow_predictors.online_flow_farneback import OnlineFlow as \
    OnlineFlowFarneback
from flow_predictors.online_flow_gmflow import OnlineFlowGMFlow
from flow_predictors.online_flow_flowformerpp import OnlineFlowFlowFormerPP



def get_flow_predictor(
        gpu=False, small=False, farneback=False, finetuned=False,
        gmflow=False,
        flowformerpp=False):
    if farneback:
        return OnlineFlowFarneback(gpu)
    if gmflow:
        return OnlineFlowGMFlow(gpu)
    if flowformerpp:
        return OnlineFlowFlowFormerPP(gpu)
    return OnlineFlowMaskFlownet(
        gpu=gpu, probabilistic=False, small=small, finetuned=finetuned)


def get_mfnprob_predictor(gpu=False, small=False, finetuned=False):
    return OnlineFlowMaskFlownet(
        gpu=gpu, probabilistic=True, small=small, finetuned=finetuned)


def main(folder_rgb, fn_stats_out, gpu=False, debug=False,
        probabilistic=True, small=False, mask_save_folder=None,
        flow_save_folder=None, uncertainty_save_folder=None,
        thr_motion=2.5, compute_mui=True, farneback=False,
        finetuned=False,
        gmflow=False,
        flowformerpp=False,
        progress_queue=None):
    """Run motion segmentation evaluation and save the results.
    
    Arguments:
    folder_rgb -- folder with RGB images of a clip to evaluate on
    fn_stats_out -- path of the output *.npz stats file
    
    Keyword arguments:
    gpu -- run on a GPU (default False)
    debug -- debug mode (default False)
    probabilistic -- use MaskFlownetProb (default True)
    small -- use the small MaskFlownetS or MaskFlownetProbS (default False)
    mask_save_folder -- If not None, a path where to store segmentation masks.
        (default None)
    thr_motion -- the flow magnitude segmentation threshold (default 2.5)
    compute_mui -- whether to compute IoU, precision and recall for
            several different motion and uncertainty thresholds (default True)
    farneback -- If True, use Farneback's optical flow from OpenCV instead
            of MaskFlownet(S) or MaskFlownetProb(S). (default False)
    gmflow -- If True, use GMFlow optical flow instead
            of MaskFlownet(S) or MaskFlownetProb(S). (default False)
    flowformerpp -- If True, use FlowFormer++ optical flow instead
            of MaskFlownet(S) or MaskFlownetProb(S). (default False)
    """
    folder_stats_out = os.path.dirname(fn_stats_out)
    pathlib.Path(folder_stats_out).mkdir(parents=True, exist_ok=True)
    flow_predictor = None
    flow_predictor_prob = None
    if probabilistic:
        flow_predictor_prob = get_mfnprob_predictor(gpu, small, finetuned)
    else:
        flow_predictor = get_flow_predictor(
            gpu, small, farneback, finetuned, gmflow, flowformerpp)
    segmenter = FlowSegmenter(
        flow_predictor, flow_predictor_prob, debug=debug)
    ce = evaluator.ClipEvaluator(
        folder_rgb, segmenter,
        gt_type="normal_flow", load_stick_masks=True,
        mask_save_folder=mask_save_folder,
        flow_save_folder=flow_save_folder,
        uncertainty_save_folder=uncertainty_save_folder,
        thr_motion=thr_motion, compute_mui=compute_mui,
        progress_queue=progress_queue)
    ce.run()
    ce.save_stats(fn_stats_out)
    d = np.load(fn_stats_out)
    print("Mean EPE: {:.4f}".format(np.mean(d["epe"])))
    print("Mean recall: {:.4f}".format(np.nanmean(d["recall"])))
    print("Mean precision: {:.4f}".format(np.nanmean(d["precision"])))
    print("Mean IoU: {:.4f}".format(np.nanmean(d["iou"])))
    print(
        "Mean false motion rate: {:.4f}".format(
            np.nanmean(d["false_motion"])))


def add_parser_options(parser):
    """Add common motion segmentation evaluation options to parser."""
    parser.add_argument(
        "-d", "--debug", action="store_true", default=False, help="debug mode")
    parser.add_argument(
        "-g", "--gpu", action="store_true", default=False,
        help="run MaskflowNet on GPU")
    parser.add_argument(
        "-p", "--probabilistic", action="store_true", default=False,
        help="run MfnProb")
    parser.add_argument(
        "--finetuned", action="store_true", default=False,
        help="run MfnProb or MaskFlownet finetuned on MovingCables")
    parser.add_argument(
        "-s", "--small", action="store_true", default=False,
        help="run MaskFlownet_S architecture instead of MaskFlownet")
    parser.add_argument(
        "-f", "--farneback", action="store_true", default=False,
        help="run Farneback's optical flow instead of MaskFlownet")
    parser.add_argument(
        "--gmflow", action="store_true", default=False,
        help="run GMFlow optical flow instead of MaskFlownet")
    parser.add_argument(
        "--flowformerpp", action="store_true", default=False,
        help="run FlowFormer++ optical flow instead of MaskFlownet")
    parser.add_argument(
        '-o', action='store', dest='mask_save_folder', default=None,
        help="if set, save computed segmentation masks to the given folder")
    parser.add_argument(
        '-of', action='store', dest='flow_save_folder', default=None,
        help="if set, save predicted optical flow images to the given folder")
    parser.add_argument(
        '--mt', action='store', dest='motion_threshold', type=float,
        default=2.5, help="set the flow magnitude segmentation threshold")
    parser.add_argument(
        '--no-mui', action='store_false', dest='compute_mui', default=True,
        help="disable motion uncertainty IoU (performance) evaluation")


def cmd_main():
    """Run the CLI stand-alone program."""
    parser = argparse.ArgumentParser(
        description=("Evaluate MaskFlownet motion segmentation on a clip. "
            "Save performance metrics."))
    add_parser_options(parser)
    parser.add_argument(
        "folder_rgb", default=None, type=str, help="input RGB(A) clip folder")
    parser.add_argument(
        "fn_stats_out", default=None, type=str,
        help="output stats npz file path")
    args = parser.parse_args()
    progress_keeper = ProgressKeeper()
    progress_keeper.setup([args.folder_rgb], cli_position=0)
    main(
        args.folder_rgb, args.fn_stats_out, args.gpu, args.debug,
        args.probabilistic, args.small, args.mask_save_folder,
        args.flow_save_folder,
        thr_motion=args.motion_threshold, compute_mui=args.compute_mui,
        farneback=args.farneback, finetuned=args.finetuned,
        gmflow=args.gmflow,
        flowformerpp=args.flowformerpp,
        progress_queue=progress_keeper.progress_queue)
    progress_keeper.close()


if __name__ == "__main__":
    cmd_main()
