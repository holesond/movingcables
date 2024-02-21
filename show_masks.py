import os
import sys
from copy import deepcopy

import numpy as np
import imageio.v3 as imageio
import matplotlib
import matplotlib.pyplot as plt



def show_masks_common_gt(
        rgb, masks_orig, masks_prob, masks_farneback=None,
        mark_pred_static=True):
    """Show a plot comparing segmentation masks with the ground truth mask.
    
    The two (or three) methods are labeled as MaskFlownet, MaskFlownetProb,
    (and Farneback).
    
    Arguments:
    rgb -- the RGB image being segmented
    masks_orig -- an np.array of three motion segmentation binary masks
        from MaskFlownet
        - masks_orig[:,:,0] -- ground truth moving mask
        - masks_orig[:,:,1] -- ground truth static mask
        - masks_orig[:,:,2] -- predicted moving mask
    masks_prob -- an np.array of three motion segmentation binary masks
        from MaskFlownetProb, the same format as masks_orig
    masks_farneback -- an np.array of three motion segmentation binary masks
        from Farneback, the same format as masks_orig, optional (default None)
    mark_pred_static -- if True, explicitly mark predicted static areas 
        (default True)
    """
    cmap_gt_moving = matplotlib.colors.ListedColormap(['black', 'tab:green'])
    cmap_gt_static = matplotlib.colors.ListedColormap(['black', 'tab:red'])
    cmap_pred_moving = matplotlib.colors.ListedColormap(['black', 'tab:blue'])
    cmap_pred_moving = matplotlib.colors.ListedColormap(['black', 'tab:green'])
    cmap_pred_static = matplotlib.colors.ListedColormap(['black', 'tab:red'])
    """
    rgb_masked = deepcopy(rgb)
    rgb_masked[masks_prob[...,0]>0,1] = rgb_masked[masks_prob[...,0]>0,1]//2 + 128
    rgb_masked[masks_prob[...,1]>0,2] = rgb_masked[masks_prob[...,1]>0,2]//2 + 128
    rgb_masked[masks_prob[...,2]>0,0] = rgb_masked[masks_prob[...,2]>0,0]//2 + 128
    plt.imshow(rgb_masked)
    plt.show()
    """
    predicted_masks = [masks_orig, masks_prob]
    titles = ["Ground Truth", "MaskFlownet", "MfnProb"]
    if masks_farneback is None:
        n_plots = 3
        fig,axs = plt.subplots(1,3)
        fig.set_size_inches(6, 1.7, forward=True)
        #fig.set_size_inches(12, 4, forward=True)
    else:
        n_plots = 4
        fig,axs = plt.subplots(2,2)
        fig.set_size_inches(5, 4, forward=True)
        predicted_masks.append(masks_farneback)
        titles.append("Farnebäck")
    for i in range(n_plots):
        axs.flat[i].imshow(rgb)
    
    axs.flat[0].imshow(
        (masks_orig[...,1]>0),
        cmap=cmap_gt_static,
        alpha=0.5*(masks_orig[...,1]>0))
    axs.flat[0].imshow(
        (masks_orig[...,0]>0),
        cmap=cmap_gt_moving,
        alpha=0.7*(masks_orig[...,0]>0))
    if mark_pred_static:
        for i in range(1, n_plots):
            axs.flat[i].imshow(
                (predicted_masks[i-1][...,2]==0),
                cmap=cmap_pred_static,
                alpha=0.5*(predicted_masks[i-1][...,2]==0))
    for i in range(1, n_plots):
        axs.flat[i].imshow(
            (predicted_masks[i-1][...,2]>0),
            cmap=cmap_pred_moving,
            alpha=0.7*(predicted_masks[i-1][...,2]>0))
    
    #axs[0].imshow((masks_orig[...,2]>0),cmap=cmap_pred_moving,
    #    alpha=0.5*(masks_orig[...,2]>0))
    #axs[1].imshow((masks_prob[...,2]>0),cmap=cmap_pred_moving,
    #    alpha=0.5*(masks_prob[...,2]>0))
    for i, title in enumerate(titles):
        axs.flat[i].set_title(title)
    for i in range(n_plots):
        axs.flat[i].get_xaxis().set_visible(False)
        axs.flat[i].get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()


def show_masks(
        rgb, masks, titles, mark_pred_static=True):
    """Show a plot comparing segmentation masks with their ground truth masks.
    
    Display the ground truth mask of each method above the segmentation
    mask predicted by the method.
    
    Arguments:
    rgb -- the RGB image being segmented
    masks -- a list of np.arrays, each array contains three motion segmentation
        binary masks
        - masks[i][:,:,0] -- ground truth moving mask
        - masks[i][:,:,1] -- ground truth static mask
        - masks[i][:,:,2] -- predicted moving mask
    titles -- a list of method titles, len(titles) == len(masks)
    mark_pred_static -- if True, explicitly mark predicted static areas 
        (default True)
    """
    assert(len(titles) == len(masks))
    cmap_gt_moving = matplotlib.colors.ListedColormap(['black', 'tab:green'])
    cmap_gt_static = matplotlib.colors.ListedColormap(['black', 'tab:red'])
    cmap_pred_moving = matplotlib.colors.ListedColormap(['black', 'tab:blue'])
    cmap_pred_moving = matplotlib.colors.ListedColormap(['black', 'tab:green'])
    cmap_pred_static = matplotlib.colors.ListedColormap(['black', 'tab:red'])
    
    n_plots = 2*len(masks)
    fig, axs = plt.subplots(2, len(masks))
    fig.set_size_inches(0.5+2*len(masks), 4, forward=True)
    
    for i in range(n_plots):
        axs.flat[i].imshow(rgb)
    
    for i, mask in enumerate(masks):
        axs[0,i].imshow(
            (mask[...,1]>0),
            cmap=cmap_gt_static,
            alpha=0.5*(mask[...,1]>0))
        axs[0,i].imshow(
            (mask[...,0]>0),
            cmap=cmap_gt_moving,
            alpha=0.7*(mask[...,0]>0))
    if mark_pred_static:
        for i, mask in enumerate(masks):
            axs[1,i].imshow(
                (mask[...,2]==0),
                cmap=cmap_pred_static,
                alpha=0.5*(mask[...,2]==0))
    for i, mask in enumerate(masks):
        axs[1,i].imshow(
            (mask[...,2]>0),
            cmap=cmap_pred_moving,
            alpha=0.7*(mask[...,2]>0))
    for i, title in enumerate(titles):
        axs[0,i].set_title(title + " GT")
        axs[1,i].set_title(title)
    for i in range(n_plots):
        axs.flat[i].get_xaxis().set_visible(False)
        axs.flat[i].get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()


def main(fn_rgb, fn_masks_orig, fn_masks_prob, fn_masks_farneback=None):
    rgb = imageio.imread(fn_rgb)
    masks_orig = imageio.imread(fn_masks_orig)
    masks_prob = imageio.imread(fn_masks_prob)
    all_masks = [masks_orig, masks_prob]
    titles = ["MaskFlownet", "MfnProb"]
    if fn_masks_farneback is not None:
        masks_farneback = imageio.imread(fn_masks_farneback)
        all_masks.append(masks_farneback)
        titles.append("Farnebäck")
    else:
        masks_farneback = None
    #show_masks_common_gt(rgb, masks_orig, masks_prob, masks_farneback)
    show_masks(rgb, all_masks, titles)


if __name__=="__main__":
    if len(sys.argv) not in [4, 5]:
        print(
            "Usage: show_masks.py fn_rgb fn_masks_orig fn_masks_prob "
            "[fn_masks_farneback]")
    else:
        main(*sys.argv[1:])
