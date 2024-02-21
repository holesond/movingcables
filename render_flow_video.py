import os
import sys
import argparse
from copy import deepcopy

import imageio.v3 as imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, rcParams
import matplotlib
import matplotlib.animation as manimation
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
import cv2 as cv
import flow_vis



def read_png_flow(fn_flow):
    im_flow = cv.imread(fn_flow, cv.IMREAD_UNCHANGED)
    if im_flow is None:
        raise RuntimeError(f"Cannot load optical flow image from: {fn_flow}")
    im_flow = im_flow[...,::-1]
    im_flow = im_flow.astype(np.float32)
    im_flow[...,0:2] = (im_flow[...,0:2] - 2**15)/64.0
    return im_flow


def flow_to_rgba(im_flow, min_mag=1.0):
    mag = np.linalg.norm(im_flow[...,0:2], axis=2)
    im_flow_vis = flow_vis.flow_to_color(
        im_flow[...,0:2], convert_to_bgr=False)
    alpha = 255*(mag > min_mag).astype(np.uint8)
    im_flow_vis_rgba = np.zeros(
        (im_flow_vis.shape[0], im_flow_vis.shape[1], 4), dtype=np.uint8)
    im_flow_vis_rgba[...,0:3] = im_flow_vis
    im_flow_vis_rgba[...,3] = alpha
    return im_flow_vis_rgba


def color_wheel_flow(width=320, height=240, radius=120, step=0.5):
    """Generate a flow image containing the flow values for the color wheel.
    
    - Caution: flow[...,0] is horizontal direction and flow[...,1] is vertical
    direction, but image[...,0] is vertical axis (rows) and image[...,1] is
    horizontal image axis (columns).
    - See the documentation of calcOpticalFlowFarneback in OpenCV:
        - https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af
        - 𝚙𝚛𝚎𝚟(y,x)∼𝚗𝚎𝚡𝚝(y+𝚏𝚕𝚘𝚠(y,x)[1],x+𝚏𝚕𝚘𝚠(y,x)[0])
    - In the moving cable segmentation dataset, the flow (flow_first_back) is:
        - current(y,x)∼first(y+𝚏𝚕𝚘𝚠(y,x)[1],x+𝚏𝚕𝚘𝚠(y,x)[0])
    - Now the color wheel is identical to the one from Baker2007,
    page 18, Fig. 7 (A Database and Evaluation Methodology for Optical Flow;
    aka the middlebury optical flow dataset).
    """
    axis_0 = np.arange(-height//2, height//2+step, step)
    axis_1 = np.arange(-width//2, width//2+step, step)
    flow_0, flow_1 = np.meshgrid(axis_0, axis_1, indexing='ij')
    flow_img = np.zeros((axis_0.size, axis_1.size, 2))
    flow_img[...,0] = flow_1
    flow_img[...,1] = flow_0
    mag = np.linalg.norm(flow_img, axis=2)
    flow_img[mag>radius,:] = 0
    return flow_img


def run_render(
        fn_video,
        fns_rgb, fns_flows, flow_titles, flow_positions,
        segmentation_pos=3, colorwheel_pos=7, plot_grid=(2,4), fps=30):
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='MaskFlownet motion segmentation', artist='', comment='')
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    
    # print("If Times New Roman cannot be found, remove the matplotlib cache dir:", matplotlib.get_cachedir())
    # plt.rcParams['font.family'] = "Times New Roman"
    # plt.rcParams['font.size'] = 8
    assert(len(fns_flows) == len(flow_titles))
    assert(len(fns_flows) >= 2)
    assert(segmentation_pos not in flow_positions)
    assert(0 not in flow_positions)
    assert(segmentation_pos != 0)
    assert(colorwheel_pos not in flow_positions)
    assert(colorwheel_pos != segmentation_pos)
    
    fig,axs = plt.subplots(*plot_grid)
    fig.set_size_inches(12, 6, forward=True)
    
    im_colorwheel = flow_to_rgba(color_wheel_flow(), min_mag=0)
    if colorwheel_pos is not None:
        colorwheel_plot = axs.flat[colorwheel_pos].imshow(im_colorwheel)
        axs.flat[colorwheel_pos].set_title("Flow color wheel")
        axs.flat[colorwheel_pos].spines[:].set_visible(False)
    
    rgb = imageio.imread(fns_rgb[0])
    im_rgb_plot = axs.flat[0].imshow(rgb)
    axs.flat[0].set_title("RGB")
    
    im_flow_plots = []
    first = True
    for pos, fns, title in zip(flow_positions, fns_flows, flow_titles):
        im_flow = read_png_flow(fns[0])
        im_flow_vis = flow_to_rgba(im_flow)
        im_plot = axs.flat[pos].imshow(im_flow_vis)
        im_flow_plots.append(im_plot)
        axs.flat[pos].set_title(title)
        if first:
            first = False
            mask = im_flow[...,2]
            im_segmentation_plot = axs.flat[segmentation_pos].imshow(
                mask, cmap='tab20', interpolation='nearest')
            axs.flat[segmentation_pos].set_title("Instance segmentation GT")
    
    for i in range(axs.size):
        axs.flat[i].get_xaxis().set_visible(False)
        axs.flat[i].get_yaxis().set_visible(False)
    plt.tight_layout()
    
    with writer.saving(fig, fn_video, dpi=100):
        for frame_idx, fn_rgb in enumerate(fns_rgb):
            rgb = imageio.imread(fn_rgb)
            im_rgb_plot.set(data=rgb)
            first = True
            for im_plot, fns in zip(im_flow_plots, fns_flows):
                im_flow = read_png_flow(fns[frame_idx])
                im_flow_vis = flow_to_rgba(im_flow)
                im_plot.set(data=im_flow_vis)
                if first:
                    first = False
                    mask = im_flow[...,2]
                    im_segmentation_plot.set(data=mask)
            writer.grab_frame()
    plt.close(fig)


def infer_method_name(fn):
    if "farneback" in fn:
        return "Farnebäck"
    if "mfnprob" in fn:
        return "MfnProb"
    if "mfn" in fn:
        return "MaskFlownet"
    return "Estimate"


def get_frame_number(name):
    number_string = os.path.splitext(name)[0]
    return int(number_string)


def main(
        fn_video, folder_rgb, folders_estimated_flow):
    fn_video = os.path.expanduser(fn_video)
    folder_rgb = os.path.expanduser(folder_rgb)
    flow_type = "flow_first_back"
    dataset_root = os.path.dirname(os.path.dirname(folder_rgb))
    sp = os.path.split(folder_rgb)
    assert(len(sp)==2)
    clip = sp[1]
    folder_flow = os.path.join(dataset_root, flow_type, clip)
    folder_normal_flow = os.path.join(dataset_root, "normal_"+flow_type, clip)
    
    fns_rgb = []
    fns_flows = [[] for i in range(len(folders_estimated_flow)+2)]
    flow_titles = ["Optical flow GT", "Normal flow GT"]
    flow_positions = [1, 2]
    for i, folder in enumerate(folders_estimated_flow):
        flow_titles.append(infer_method_name(folder))
        flow_positions.append(i+4)
    
    skip = False
    i = 0
    last_frame_number = None
    fps = 30
    for name in sorted(os.listdir(folder_flow)):
        if not name.endswith('png'):
            continue
        current_frame_number = get_frame_number(name)
        if last_frame_number is None:
            last_frame_number = get_frame_number(name)
        elif current_frame_number - last_frame_number <= 2:
            continue
        else:
            fps = int(120.0/(current_frame_number - last_frame_number))
            last_frame_number = current_frame_number
        fns_rgb.append(os.path.join(folder_rgb, name))
        fns_flows[0].append(os.path.join(folder_flow, name))
        fns_flows[1].append(os.path.join(folder_normal_flow, name))
        for i, folder in enumerate(folders_estimated_flow):
            fns_flows[i+2].append(os.path.join(folder, name))
        i += 1
    run_render(
        fn_video, fns_rgb, fns_flows, flow_titles, flow_positions,
        segmentation_pos=3, colorwheel_pos=7, plot_grid=(2,4), fps=fps)


def cmd_main():
    parser = argparse.ArgumentParser(
        description=("Render an optical flow video."))
    parser.add_argument(
        "fn_video", default=None, type=str,
        help="output video file path")
    parser.add_argument(
        "folder_rgb", default=None, type=str,
        help="input RGB image sequence folder path")
    parser.add_argument(
        "-e", "--estimated", action="append", default=None, type=str,
        help="estimated flow png image sequence folder path")
    args = parser.parse_args()
    main(args.fn_video, args.folder_rgb, args.estimated)


if __name__=="__main__":
    cmd_main()


