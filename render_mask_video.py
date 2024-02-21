import os
import sys
import pathlib
from copy import deepcopy

import imageio.v3 as imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, rcParams
import matplotlib
import matplotlib.animation as manimation
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'



def run_render_common_gt(
        fn_video,
        fns_rgb, fns_masks_orig, fns_masks_prob, fns_masks_farneback=None,
        mark_pred_static=True, fps=30):
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='MaskFlownet motion segmentation', artist='', comment='')
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    
    # print("If Times New Roman cannot be found, remove the matplotlib cache dir:", matplotlib.get_cachedir())
    # plt.rcParams['font.family'] = "Times New Roman"
    # plt.rcParams['font.size'] = 8
    cmap_gt_moving = matplotlib.colors.ListedColormap(['black', 'tab:green'])
    cmap_gt_static = matplotlib.colors.ListedColormap(['black', 'tab:red'])
    # cmap_pred_moving = matplotlib.colors.ListedColormap(['black', 'tab:blue'])
    cmap_pred_moving = matplotlib.colors.ListedColormap(['black', 'tab:green'])
    cmap_pred_static = matplotlib.colors.ListedColormap(['black', 'tab:red'])
    
    
    predicted_fns = [fns_masks_orig, fns_masks_prob]
    titles = ["Ground Truth", "MaskFlownet", "MfnProb"]
    if fns_masks_farneback is None:
        n_plots = 3
        fig,axs = plt.subplots(1,3)
        fig.set_size_inches(12, 4, forward=True)
    else:
        n_plots = 4
        fig,axs = plt.subplots(2,2)
        fig.set_size_inches(10, 8, forward=True)
        predicted_fns.append(fns_masks_farneback)
        titles.append("Farnebäck")
    rgb = imageio.imread(fns_rgb[0])
    im_rgb_plots = []
    for i in range(n_plots):
        im_rgb_plots.append(axs.flat[i].imshow(rgb))
    mask_images = [imageio.imread(fns[0]) for fns in predicted_fns]
    
    im_static0 = axs.flat[0].imshow(
        1.0*(mask_images[0][...,1]>0),
        cmap=cmap_gt_static,
        alpha=0.5*(mask_images[0][...,1]>0),
        vmin=0, vmax=1)
    im_moving0 = axs.flat[0].imshow(
        1.0*(mask_images[0][...,0]>0),
        cmap=cmap_gt_moving,
        alpha=0.7*(mask_images[0][...,0]>0),
        vmin=0, vmax=1)
    im_pred_static_plots = []
    if mark_pred_static:
        for i in range(1, n_plots):
            im_plot = axs.flat[i].imshow(
                1.0*(mask_images[i-1][...,2]==0),
                cmap=cmap_pred_static,
                alpha=0.5*(mask_images[i-1][...,2]==0),
                vmin=0, vmax=1)
            im_pred_static_plots.append(im_plot)
    im_pred_moving_plots = []
    for i in range(1, n_plots):
        im_plot = axs.flat[i].imshow(
            1.0*(mask_images[i-1][...,2]>0),
            cmap=cmap_pred_moving,
            alpha=0.7*(mask_images[i-1][...,2]>0),
            vmin=0, vmax=1)
        im_pred_moving_plots.append(im_plot)
    
    for i, title in enumerate(titles):
        axs.flat[i].set_title(title)
    for i in range(n_plots):
        axs.flat[i].get_xaxis().set_visible(False)
        axs.flat[i].get_yaxis().set_visible(False)
    plt.tight_layout()
    
    with writer.saving(fig, fn_video, dpi=100):
        for frame_idx, fn_rgb in enumerate(fns_rgb):
            rgb = imageio.imread(fn_rgb)
            mask_images = [
                imageio.imread(fns[frame_idx]) for fns in predicted_fns]
            
            for im_rgb_plot in im_rgb_plots:
                im_rgb_plot.set(data=rgb)
            im_static0.set(
                array=1.0*(mask_images[0][...,1]>0),
                alpha=0.5*(mask_images[0][...,1]>0))
            if mark_pred_static:
                for im_plot, mask in zip(im_pred_static_plots, mask_images):
                    im_plot.set(
                        data=1.0*(mask[...,2]==0),
                        alpha=0.5*(mask[...,2]==0),
                        cmap=cmap_pred_static)
            im_moving0.set(
                array=1.0*(mask_images[0][...,0]>0),
                alpha=0.7*(mask_images[0][...,0]>0))
            for im_plot, mask in zip(im_pred_moving_plots, mask_images):
                im_plot.set(
                    data=1.0*(mask[...,2]>0),
                    alpha=0.7*(mask[...,2]>0),
                    cmap=cmap_pred_moving)
            writer.grab_frame()
    plt.close(fig)


def run_render(
        fn_video, fns_rgb, fns_masks, titles, mark_pred_static=True, fps=30):
    assert(len(titles) == len(fns_masks))
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='MaskFlownet motion segmentation', artist='', comment='')
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    
    # print("If Times New Roman cannot be found, remove the matplotlib cache dir:", matplotlib.get_cachedir())
    # plt.rcParams['font.family'] = "Times New Roman"
    # plt.rcParams['font.size'] = 8
    cmap_gt_moving = matplotlib.colors.ListedColormap(['black', 'tab:green'])
    cmap_gt_static = matplotlib.colors.ListedColormap(['black', 'tab:red'])
    # cmap_pred_moving = matplotlib.colors.ListedColormap(['black', 'tab:blue'])
    cmap_pred_moving = matplotlib.colors.ListedColormap(['black', 'tab:green'])
    cmap_pred_static = matplotlib.colors.ListedColormap(['black', 'tab:red'])
    
    n_plots = 2*len(fns_masks)
    fig, axs = plt.subplots(2, len(fns_masks))
    fig.set_size_inches(3*len(fns_masks), 5, forward=True)
    
    rgb = imageio.imread(fns_rgb[0])
    im_rgb_plots = []
    for i in range(n_plots):
        im_rgb_plots.append(axs.flat[i].imshow(rgb))
    mask_images = [imageio.imread(fns[0]) for fns in fns_masks]
    
    gt_images_static = []
    gt_images_moving = []
    pred_images_static = []
    pred_images_moving = []
    for i, mask in enumerate(mask_images):
        im_static = axs[0,i].imshow(
            1.0*(mask[...,1]>0),
            cmap=cmap_gt_static,
            alpha=0.5*(mask[...,1]>0),
            vmin=0, vmax=1)
        gt_images_static.append(im_static)
        im_moving = axs[0,i].imshow(
            1.0*(mask[...,0]>0),
            cmap=cmap_gt_moving,
            alpha=0.7*(mask[...,0]>0),
            vmin=0, vmax=1)
        gt_images_moving.append(im_moving)
        if mark_pred_static:
            im_pred_static = axs[1,i].imshow(
                1.0*(mask[...,2]==0),
                cmap=cmap_pred_static,
                alpha=0.5*(mask[...,2]==0),
                vmin=0, vmax=1)
            pred_images_static.append(im_pred_static)
        im_pred_moving = axs[1,i].imshow(
            1.0*(mask[...,2]>0),
            cmap=cmap_pred_moving,
            alpha=0.7*(mask[...,2]>0),
            vmin=0, vmax=1)
        pred_images_moving.append(im_pred_moving)
    
    for i, title in enumerate(titles):
        axs[0,i].set_title(title + " GT")
        axs[1,i].set_title(title)
    for i in range(n_plots):
        axs.flat[i].get_xaxis().set_visible(False)
        axs.flat[i].get_yaxis().set_visible(False)
    plt.tight_layout()
    
    with writer.saving(fig, fn_video, dpi=100):
        for frame_idx, fn_rgb in enumerate(fns_rgb):
            rgb = imageio.imread(fn_rgb)
            mask_images = [
                imageio.imread(fns[frame_idx]) for fns in fns_masks]
            for im_rgb_plot in im_rgb_plots:
                im_rgb_plot.set(data=rgb)
            for i, mask in enumerate(mask_images):
                gt_images_static[i].set(
                    array=1.0*(mask[...,1]>0),
                    alpha=0.5*(mask[...,1]>0))
                if mark_pred_static:
                    pred_images_static[i].set(
                        data=1.0*(mask[...,2]==0),
                        alpha=0.5*(mask[...,2]==0),
                        cmap=cmap_pred_static)
                gt_images_moving[i].set(
                    array=1.0*(mask[...,0]>0),
                    alpha=0.7*(mask[...,0]>0))
                pred_images_moving[i].set(
                    data=1.0*(mask[...,2]>0),
                    alpha=0.7*(mask[...,2]>0),
                    cmap=cmap_pred_moving)
            writer.grab_frame()
    plt.close(fig)


def get_frame_number(name):
    number_string = os.path.splitext(name)[0]
    return int(number_string)


def main(
        fn_video, folder_rgb, folder_masks_orig, folder_masks_prob,
        folder_masks_farneback=None):
    folder_rgb = os.path.expanduser(folder_rgb)
    folder_masks_orig = os.path.expanduser(folder_masks_orig)
    folder_masks_prob = os.path.expanduser(folder_masks_prob)
    fn_video = os.path.expanduser(fn_video)
    video_folder = os.path.dirname(fn_video)
    pathlib.Path(video_folder).mkdir(parents=True, exist_ok=True)
    if folder_masks_farneback is None:
        fns_masks_farneback = None
    else:
        folder_masks_farneback = os.path.expanduser(folder_masks_farneback)
        fns_masks_farneback = []
    fns_rgb, fns_masks_orig, fns_masks_prob = [], [], []
    i = 0
    last_frame_number = None
    fps = 30
    for name in sorted(os.listdir(folder_masks_orig)):
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
        fns_masks_orig.append(os.path.join(folder_masks_orig, name))
        fns_masks_prob.append(os.path.join(folder_masks_prob, name))
        if folder_masks_farneback is not None:
            fns_masks_farneback.append(
                os.path.join(folder_masks_farneback, name))
        i += 1
    
    fns_masks = [fns_masks_orig, fns_masks_prob]
    titles = ["MaskFlownet", "MfnProb"]
    if fns_masks_farneback is not None:
        fns_masks.append(fns_masks_farneback)
        titles.append("Farnebäck")
    run_render(
        fn_video, fns_rgb, fns_masks, titles, fps=fps)
    
    #run_render_common_gt(
    #    fn_video, fns_rgb, fns_masks_orig, fns_masks_prob, fns_masks_farneback,
    #    fps=fps)


if __name__=="__main__":
    if len(sys.argv) not in [5,6]:
        print("Usage: render_mask_video.py fn_video_out folder_rgb"
            " folder_masks_orig folder_masks_prob [folder_masks_farneback]")
    else:
        main(*sys.argv[1:])


