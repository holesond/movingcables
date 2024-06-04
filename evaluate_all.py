import os
import sys
import pathlib
import inspect
import argparse
import warnings
from copy import deepcopy
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import matplotlib.pyplot as plt

import evaluate_single
from evaluator.evaluator import ProgressKeeper



def evaluate_job(args):
    folder_rgb = args[0]
    print(folder_rgb)
    evaluate_single.main(*args)


def main(
        folder_clips, folder_stats_out, gpu=False, debug=False,
        probabilistic=True, small=False, mask_save_folder=None,
        flow_save_folder=None, uncertainty_save_folder=None,
        thr_motion=2.5, compute_mui=True, farneback=False,
        finetuned=False,
        gmflow=False,
        flowformerpp=False,
        max_workers=2):
    pathlib.Path(folder_stats_out).mkdir(parents=True, exist_ok=True)
    rgba_root = os.path.join(folder_clips,"rgba_clips")
    if not os.path.isdir(rgba_root):
        rgba_root = os.path.join(folder_clips,"rgb_clips")
    progress_keeper = ProgressKeeper()
    args_buffer = []
    rgb_folders = []
    for dirpath, dirnames, filenames in os.walk(rgba_root):
        for name in dirnames:
            folder_rgb = os.path.join(rgba_root, name)
            fn_stats_out = os.path.join(folder_stats_out, name + ".npz")
            if mask_save_folder is None:
                clip_mask_save_folder = None
            else:
                clip_mask_save_folder = os.path.join(
                    mask_save_folder, name)
            if flow_save_folder is None:
                clip_flow_save_folder = None
            else:
                clip_flow_save_folder = os.path.join(
                    flow_save_folder, name)
            if uncertainty_save_folder is None:
                clip_uncertainty_save_folder = None
            else:
                clip_uncertainty_save_folder = os.path.join(
                    uncertainty_save_folder, name)
            args_buffer.append(
                [
                folder_rgb, fn_stats_out, gpu, debug,
                probabilistic, small, clip_mask_save_folder,
                clip_flow_save_folder, clip_uncertainty_save_folder,
                thr_motion, compute_mui, farneback, finetuned, gmflow,
                flowformerpp,
                progress_keeper.progress_queue,
                ])
            rgb_folders.append(folder_rgb)
    progress_keeper.setup(rgb_folders, cli_position=0)
    if max_workers == 1:
        for args in args_buffer:
            evaluate_job(args)
        progress_keeper.close()
        return
    with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=multiprocessing.get_context("spawn")) as threads:
        for res in threads.map(evaluate_job, args_buffer):
            if res is not None:
                print(str(res))
    progress_keeper.close()


def cmd_main():
    """Run the CLI stand-alone program."""
    parser = argparse.ArgumentParser(
        description=("Evaluate MaskFlownet motion segmentation "
        "on multiple clips. Save performance metrics."))
    evaluate_single.add_parser_options(parser)
    parser.add_argument(
        '-j', action='store', dest='jobs', default=2, type=int,
        help="the number of parallel jobs")
    parser.add_argument(
        "folder_clips", default=None, type=str, help="input root clip folder")
    parser.add_argument(
        "folder_stats_out", default=None, type=str, help="output stats folder")
    args = parser.parse_args()
    main(
        args.folder_clips, args.folder_stats_out, args.gpu,args.debug,
        args.probabilistic, args.small, args.mask_save_folder,
        args.flow_save_folder,
        thr_motion=args.motion_threshold, compute_mui=args.compute_mui,
        farneback=args.farneback, finetuned=args.finetuned,
        gmflow=args.gmflow,
        flowformerpp=args.flowformerpp,
        max_workers=args.jobs)


if __name__ == "__main__":
    cmd_main()
