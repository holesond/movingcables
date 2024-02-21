"""Show optical flow, normal flow and instance segmentation masks. CLI."""

import os
import argparse
import math

import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import flow_vis



def test_normal_flow_magnitude(im_flow, im_normal_flow):
    if im_flow is None or im_normal_flow is None:
        return
    mag = np.linalg.norm(im_flow[...,0:2],axis=2)
    norm_mag = np.linalg.norm(im_normal_flow[...,0:2],axis=2)
    if not np.all(norm_mag<mag+1e-1):
        # Using 1e-1 tolerance because flow resolution in 16bit png is 1/64. 
        print("WARNING/ERROR: Not all normal flow magnitudes are "
            "lower than or equal to their respective optical flow "
            "magnitudes!")


def flow_plot_single(
        ax, im_flow, im_rgb=None, max_mag=50, show_magnitude=False):
    if im_flow is None:
        return None
    if im_rgb is not None:
        im_rgb = im_rgb[...,0:3]
    alpha = 1.0
    mag = np.linalg.norm(im_flow[...,0:2], axis=2)
    if im_rgb is not None:
        fl = ax.imshow((im_rgb*0.66).astype(np.uint8))
        alpha = (mag > 0).astype(np.float32)
    if show_magnitude:
        fl = ax.imshow(
            mag, cmap='plasma', vmin=0, vmax=max_mag, alpha=alpha)
    else:
        im_flow_vis = flow_vis.flow_to_color(
            im_flow[...,0:2], convert_to_bgr=False)
        im_flow_vis_rgba = np.zeros((
            im_flow_vis.shape[0],
            im_flow_vis.shape[1],
            4,
            ), dtype=np.uint8)
        im_flow_vis_rgba[...,0:3] = im_flow_vis
        im_flow_vis_rgba[...,3] = 255*alpha
        fl = ax.imshow(im_flow_vis_rgba)
    #ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    return fl


def show_images(
        im_flow, im_normal_flow, im_rgb=None, show_mask=True,
        show_magnitude=False, estimated_flows=None, est_titles=None,
        max_mag=50):
    test_normal_flow_magnitude(im_flow, im_normal_flow)
    n_extra_plots = 0
    if estimated_flows is not None:
        n_extra_plots = len(estimated_flows)
        assert(len(estimated_flows) == len(est_titles))
    n_plots = 2 + n_extra_plots
    if show_mask:
        n_plots += 1
    if n_plots <= 3:
        n_cols = n_plots
        n_rows = 1
    elif n_plots == 4:
        n_cols, n_rows = 2, 2
    else:
        n_cols = 3
        n_rows = math.ceil(n_plots/n_cols)
    fig, axs = plt.subplots(n_rows, n_cols)
    fig.set_size_inches(0.5+2.5*n_cols, 0.2+2.2*n_rows, forward=True)
    fl_1 = flow_plot_single(
        axs.flat[1], im_normal_flow, im_rgb, max_mag, show_magnitude)
    fl_0 = flow_plot_single(
        axs.flat[0], im_flow, im_rgb, max_mag, show_magnitude)
    if fl_0 is not None:
        fl = fl_0
    else:
        fl = fl_1
    axs.flat[0].set_title("Optical flow GT")
    axs.flat[1].set_title("Normal flow GT")
    if show_mask:
        mask = im_flow[...,2]
        print(np.unique(mask))
        axs.flat[-1].imshow(mask, cmap='tab20', interpolation='nearest')
        axs.flat[-1].set_title("Segmentation GT")
        axs.flat[-1].set_xticks([])
        axs.flat[-1].set_yticks([])
    if n_extra_plots > 0:
        ax_indices = [idx for idx in range(2, 2+n_extra_plots)]
        for idx, e_flow, tit in zip(ax_indices, estimated_flows, est_titles):
            fl_n = flow_plot_single(
                axs.flat[idx], e_flow, None, max_mag, show_magnitude)
            axs.flat[idx].set_title(tit)
    if show_magnitude and fl is not None:
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
        cax = plt.axes([0.93, 0.1, 0.02, 0.8])
        plt.colorbar(fl, cax=cax)
    if not show_magnitude:
        plt.tight_layout()
    #plt.axis('off')
    plt.show()


def read_png_flow(fn_flow):
    im_flow = cv.imread(fn_flow, cv.IMREAD_UNCHANGED)
    if im_flow is None:
        raise RuntimeError(f"Cannot read flow image from {fn_flow}.")
    im_flow = im_flow[...,::-1]
    im_flow = im_flow.astype(np.float32)
    im_flow[...,0:2] = (im_flow[...,0:2] - 2**15)/64.0
    return im_flow


def read_png_rgb(fn_rgb):
    im_rgb = cv.imread(fn_rgb, cv.IMREAD_UNCHANGED)
    im_rgb = im_rgb[...,::-1]
    return im_rgb


def infer_method_name(fn):
    if "farneback" in fn:
        return "Farnebäck"
    if "mfnprob" in fn:
        return "MfnProb"
    if "mfn" in fn:
        return "MaskFlownet"
    return "Estimate"


def show_files(
        fn_flow, fn_normal_flow, fn_rgb, show_mask, show_magnitude,
        fn_estimated_flows=None):
    im_flow = None
    im_normal_flow = None
    im_rgb = None
    if not fn_flow is None:
        im_flow = read_png_flow(fn_flow)
    if not fn_normal_flow is None:
        im_normal_flow = read_png_flow(fn_normal_flow)
    if fn_rgb is not None:
        im_rgb = read_png_rgb(fn_rgb)
    estimated_flows = None
    est_titles = None
    if fn_estimated_flows is not None:
        estimated_flows = []
        est_titles = []
        for fn in fn_estimated_flows:
            estimated_flows.append(read_png_flow(fn))
            est_titles.append(infer_method_name(fn))
    show_images(
        im_flow, im_normal_flow, im_rgb, show_mask, show_magnitude,
        estimated_flows, est_titles)


def main():
    parser = argparse.ArgumentParser(
        description=("Show optical flow, normal flow "
            "and instance segmentation masks."))
    parser.add_argument(
        "-nm", "--no-masks", action="store_false", dest="show_mask",
        default=True, help="do not show the instance segmentation masks")
    parser.add_argument(
        "-mag", "--magnitude", action="store_true", default=False,
        help="show flow magnitude instead of the flow vectors")
    parser.add_argument(
        "fn_rgba", default=None, type=str,
        help="input RGB(A) image file path")
    parser.add_argument(
        "-e", "--estimated", action="append", default=None, type=str,
        help="estimated flow png image file path")
    args = parser.parse_args()
    flow_type = "flow_first_back"
    dataset_root = os.path.dirname(
        os.path.dirname(os.path.dirname(args.fn_rgba)))
    sp = os.path.split(args.fn_rgba)
    assert(len(sp)==2)
    fn = sp[1]
    clip = os.path.split(sp[0])[1]
    fn_flow = os.path.join(dataset_root, flow_type, clip, fn)
    fn_normal_flow = os.path.join(dataset_root, "normal_"+flow_type, clip, fn)
    show_files(
        fn_flow, fn_normal_flow, args.fn_rgba, args.show_mask, args.magnitude,
        args.estimated)


if __name__=="__main__":
    main()
