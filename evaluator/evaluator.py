import os
import pathlib
from copy import deepcopy
from hashlib import sha256
import multiprocessing
import queue
import time

import png
import cv2 as cv
import numpy as np
import imageio.v3 as imageio
from tqdm import tqdm



def infer_flow_folder(rgba_folder, flow_type, stick_masks=False):
    """Return the folders containing flow, stick and stick mask images.
    
    Assume that the dataset path structure is '/dataset/rgba_clips/0001/'.
    
    Arguments:
    rgba_folder -- path of the folder containing the RGB(A) images
        of the clip, e.g. /dataset/rgba_clips/0001
    flow_type -- either "optical_flow" or "normal_flow"
    stick_masks -- whether to return stick folders or not (default False)
    
    Return:
    flow_first_back -- flow folder (string),
        e.g. '/dataset/flow_first_back/0001/'
    rgba_stick_folder -- RGBA stick folder (string),
        e.g. '/dataset/rgba_clips_stick/0001/'
    stick_mask_folder -- stick mask folder (string),
        e.g. '/dataset/stick_masks/0001/'
    clip_name -- clip number string (name)
    
    Raises:
    ValueError: unknown flow type
    """
    sp = ["", ""]
    head = deepcopy(rgba_folder)
    while head[-1] == "/" or  head[-1] == "\\":
        head = head[0:-1]
    for i in range(len(sp)-1,-1,-1):
        head, tail = os.path.split(head) 
        sp[i] = tail
    assert(sp[0] in ['rgba_clips', 'rgb_clips', 'clips'])
    clip_name = sp[-1]
    if flow_type == "optical_flow":
        sp[0] = 'flow_first_back'
    elif flow_type == "normal_flow":
        sp[0] = 'normal_flow_first_back'
    else:
        raise ValueError("infer_flow_folder: unknown flow type {}.".format(
            flow_type))
    flow_first_back = os.path.join(head,*sp)
    rgba_stick_folder = None
    stick_mask_folder = None
    if stick_masks:
        sp[0] = "rgba_clips_stick"
        rgba_stick_folder = os.path.join(head,*sp)
        sp[0] = "stick_masks"
        stick_mask_folder = os.path.join(head, *sp)
    return flow_first_back, rgba_stick_folder, stick_mask_folder, clip_name


def load_flow_png_float(fn):
    """Load a uint16 PNG optical flow image as a numpy float array."""
    im = cv.imread(fn, cv.IMREAD_UNCHANGED)
    if im is None:
        return None
    assert(im.dtype==np.uint16)
    im = im[...,::-1] # Convert from reversed 'BGR' order to original 'RGB'.
    im = im.astype(np.float32)
    im[...,0:2] = (im[...,0:2] - 2**15)*(1.0/64.0)
    return im


def flow2uint16(flow):
    flow_u16 = np.floor(flow*64 + 2**15)
    flow_u16[flow_u16>2**16-1] = 2**16-1
    flow_u16[flow_u16<0] = 0
    flow_u16 = flow_u16.astype(np.uint16)
    return flow_u16


def save_flow_png(fn, im):
    # https://stackoverflow.com/questions/25696615/can-i-save-a-numpy-array-as-a-16-bit-image-using-normal-enthought-python
    # Use pypng to write im as a color PNG.
    with open(fn, 'wb') as f:
        writer = png.Writer(
            width=im.shape[1], height=im.shape[0],
            bitdepth=16, greyscale=False)
        # Convert im to the Python list of lists expected by
        # the png writer.
        im_list = im.reshape(-1, im.shape[1]*im.shape[2]).tolist()
        writer.write(f, im_list)


def binned_sum_count(x, y, x_bin_edges):
    bin_sums, _ = np.histogram(x, bins=x_bin_edges, weights=y)
    bin_counts, _ = np.histogram(x, bins=x_bin_edges)
    return bin_sums, bin_counts


def multi_iou_precision_recall(gt, pred, debug=False):
    """Compute IoU, precision and recall for multiple parameter values.
    
    Assume that the first two dimensions of gt and pred are pixel coordinates.
    The other dimensions (axes >= 2) can contain different parameter values.
    
    Arguments:
    gt -- ground truth binary segmentation masks (np.array)
    pred -- predicted binary segmentation masks (np.array)
    
    Return: iou, precision, recall (all np.array)
    Invalid values are equal to np.nan.
    """
    assert(gt.shape == pred.shape)
    intersection = np.logical_and(gt, pred)
    union = np.logical_or(gt, pred)
    n_intersection = np.sum(intersection, axis=(0,1)).astype(np.float32)
    n_union = np.sum(union, axis=(0,1)).astype(np.float32)
    n_gt = np.sum(gt, axis=(0,1)).astype(np.float32)
    n_pred = np.sum(pred, axis=(0,1)).astype(np.float32)
    if debug:
        print("n_union.shape", n_union.shape)
        print("n_intersection.shape", n_intersection.shape)
        print("n_gt.shape", n_gt.shape)
        print("n_pred.shape", n_pred.shape)
    assert(n_union.shape == n_intersection.shape)
    assert(n_union.shape == n_gt.shape)
    assert(n_union.shape == n_pred.shape)
    
    iou = np.zeros(n_union.shape, dtype=np.float32)
    recall = np.zeros(n_union.shape, dtype=np.float32)
    precision = np.zeros(n_union.shape, dtype=np.float32)
    iou[...] = np.nan
    recall[...] = np.nan
    precision[...] = np.nan
    
    valid = n_union > 0
    iou[valid] = n_intersection[valid]/n_union[valid]
    valid = n_gt > 0
    recall[valid] = n_intersection[valid]/n_gt[valid]
    valid = n_pred > 0
    precision[valid] = n_intersection[valid]/n_pred[valid]
    return iou, precision, recall


def motion_uncertainty_iou_precision_recall(
        gt_flow, pred_flow, pred_uncertainty,
        uncertainty_thresholds, motion_thresholds):
    """Compute IoU, precision and recall at multiple flow thresholds. 
    
    Arguments:
    gt_flow -- ground truth optical flow (np.array)
    pred_flow -- predicted optical flow (np.array)
    pred_uncertainty -- predicted flow uncertainty (np.array)
    uncertainty_thresholds -- list of uncertainty thresholds to evaluate
    motion_thresholds -- list of flow magnitude thresholds to evaluate
    
    Return: iou, precision, recall (all np.array)
    - Invalid values are equal to np.nan.
    - If pred_uncertainty is None, the returned arrays are 1D, their size is
    equal to motion_thresholds.size. Otherwise return 2D arrays, shape
    (motion_thresholds.size, uncertainty_thresholds.size)
    """
    gt_flow_mag = np.linalg.norm(gt_flow[...,0:2], axis=2)
    pred_flow_mag = np.linalg.norm(pred_flow[...,0:2], axis=2)
    gt_motion_masks = (
        gt_flow_mag[:,:,None] > motion_thresholds[None,None,:])
    assert(gt_motion_masks.shape == (
        gt_flow_mag.shape[0],
        gt_flow_mag.shape[1],
        motion_thresholds.size,
        ))
    pred_motion_masks = (
        pred_flow_mag[:,:,None] > motion_thresholds[None,None,:])
    if pred_uncertainty is None:
        res = multi_iou_precision_recall(
            gt_motion_masks, pred_motion_masks)
        return res
    
    uncertainty_mag = np.linalg.norm(pred_uncertainty, axis=-1)
    certain_masks = (
        uncertainty_mag[:,:,None] < uncertainty_thresholds[None,None,:])
    # x_0, x_1, motion_thresholds, uncertainty_thresholds
    pred_seg_masks = np.logical_and(
        certain_masks[:,:,None,:], pred_motion_masks[:,:,:,None])
    gt_motion_masks = np.tile(
        gt_motion_masks[...,None], pred_seg_masks.shape[3])
    res = multi_iou_precision_recall(gt_motion_masks, pred_seg_masks)
    return res


class ClipEvaluator():
    def __init__(
            self,rgb_folder,segmenter,thr_motion=2.5,thr_static=1.0,
            gt_type="optical_flow",load_stick_masks=False,
            mask_save_folder=None,
            flow_save_folder=None,
            uncertainty_save_folder=None,
            compute_mui=True,
            seed_prefix="93426543",
            progress_queue=None):
        """Initialize a ClipEvaluator for a given segmenter object and clip.
        
        Arguments:
        rgb_folder -- RGB(A) clip folder to evaluate on (string),
            e.g. '/dataset/rgb_clips/0001'
        segmenter -- an initialized segmenter object having a next_image
            member function
        
        Keyword arguments:
        thr_motion -- the flow magnitude threshold for motion segmentation
            (default 2.5)
        gt_type -- "optical_flow" or "normal_flow" (default "optical_flow")
        load_stick_masks -- whether to load and use poking stick mask images
            (default False)
        mask_save_folder -- where to save the computed segmentation masks
            (default None, meaning do not save them)
        compute_mui -- whether to compute IoU, precision and recall for
            several different motion and uncertainty thresholds (default True)
        progress_queue -- The evaluator puts a 1 to this queue when it
            processes one image.
            (default None)
        """
        self.rgb_folder = rgb_folder
        self.rgba_stick_folder = None
        self.segmenter = segmenter
        (self.flow_folder, self.rgba_stick_folder,
            self.stick_mask_folder, clip_name) = infer_flow_folder(
            rgb_folder, gt_type, stick_masks=load_stick_masks)
        self.thr_motion = thr_motion
        self.thr_static = thr_static
        self.gt_type = gt_type
        self.mask_save_folder = mask_save_folder
        self.flow_save_folder = flow_save_folder
        self.uncertainty_save_folder = uncertainty_save_folder
        self.progress_queue = progress_queue
        folders = [
            mask_save_folder,
            flow_save_folder,
            uncertainty_save_folder,
            ]
        for folder in folders:
            if folder is None:
                continue
            pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
        
        self.epe = []
        self.epe_pred_lt_1_sum = []
        self.epe_pred_lt_1_count = []
        self.epe_pred_gt_1_sum = []
        self.epe_pred_gt_1_count = []
        self.magnitude_bin_edges = np.arange(0,40.1,0.1)
        n_bins_mag = self.magnitude_bin_edges.size - 1
        self.epe_vs_magnitude_sum = np.zeros(n_bins_mag)
        self.epe_vs_magnitude_count = np.zeros(n_bins_mag)
        
        self.recall = []
        self.precision = []
        self.iou = []
        self.false_motion = []
        self.N_gt_all_pixels = []
        self.N_gt_static = []
        self.N_gt_moving = []
        self.N_predicted_moving = []
        self.N_predicted_static = []
        self.N_intersection = []
        self.N_union = []
        
        self.compute_mui = compute_mui
        self.mu_iou = []
        self.mu_precision = []
        self.mu_recall = []
        self.uncertainty_thresholds = np.array(
            [
                0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4, 4.5, 5,
                6, 7, 8, 9, 10, np.inf,
            ],
            dtype=np.float32)
        self.motion_thresholds = np.concatenate((
            np.arange(1,5,0.5,dtype=np.float32),
            np.arange(5,21,1,dtype=np.float32),
            ))
        
        seed_string = seed_prefix + clip_name
        hsh = sha256(seed_string.encode()).digest()
        seed = int.from_bytes(hsh[0:4], byteorder='little')
        self.rng = np.random.RandomState(seed)
    
    
    def append_epe_lt_gt_1(self, epe, flow):
        flow_mag = np.linalg.norm(flow, axis=2)
        sel_lt_1 = flow_mag <= 1
        cnt = np.sum(sel_lt_1)
        self.epe_pred_lt_1_count.append(cnt)
        if cnt > 0:
            self.epe_pred_lt_1_sum.append(np.sum(epe[sel_lt_1]))
        else:
            self.epe_pred_lt_1_sum.append(0)
        sel_gt_1 = flow_mag > 1
        cnt = np.sum(sel_gt_1)
        self.epe_pred_gt_1_count.append(cnt)
        if cnt > 0:
            self.epe_pred_gt_1_sum.append(np.sum(epe[sel_gt_1]))
        else:
            self.epe_pred_gt_1_sum.append(0)
    
    
    def append_flow_epe(self, gt_flow, gt_flow_mag, flow):
        epe = np.linalg.norm(gt_flow[...,0:2] - flow, axis=2)
        mean_epe = np.mean(epe)
        self.epe.append(mean_epe)
        self.append_epe_lt_gt_1(epe, flow)

        bin_sums, bin_counts = binned_sum_count(
            gt_flow_mag, epe, self.magnitude_bin_edges)
        self.epe_vs_magnitude_sum += bin_sums
        self.epe_vs_magnitude_count += bin_counts
    

    def save_masks(self, gt_moving, gt_static, pred_motion_mask, idx):
        if self.mask_save_folder is None:
            return
        masks = np.zeros(
            (gt_moving.shape[0], gt_moving.shape[1], 3), dtype=np.uint8)
        masks[...,0] = 255*gt_moving
        masks[...,1] = 255*gt_static
        masks[...,2] = 255*pred_motion_mask
        fn = os.path.join(self.mask_save_folder, "{:08d}.png".format(idx))
        imageio.imwrite(fn, masks)
    
    
    def save_flow(self, flow, uncertainty, idx):
        if self.flow_save_folder is None:
            return
        fn = os.path.join(self.flow_save_folder, "{:08d}.png".format(idx))
        flow_3 = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
        flow_3[...,0:2] = flow
        if uncertainty is not None:
            uncertainty_mag = np.linalg.norm(uncertainty, axis=2)
            flow_3[...,2] = uncertainty_mag
        else:
            flow_3[...,2] = -1
        flow_3_uint16 = flow2uint16(flow_3)
        save_flow_png(fn, flow_3_uint16)
    
    
    def append_stats(self, gt_flow, motion_mask, flow, uncertainty, idx):
        assert(motion_mask.dtype == bool)
        gt_mag = np.linalg.norm(gt_flow[...,0:2],axis=2)
        if flow is not None:
            self.append_flow_epe(gt_flow, gt_mag, flow)
        else:
            self.epe.append(np.nan)
        gt_moving = gt_mag > self.thr_motion
        gt_static = gt_mag < self.thr_static
        intersection = np.logical_and(gt_moving, motion_mask)
        union = np.logical_or(gt_moving, motion_mask)
        N_moving = float(np.sum(gt_moving))
        N_intersection = float(np.sum(intersection))
        N_union = float(np.sum(union))
        N_motion_mask = float(np.sum(motion_mask))
        self.N_gt_moving.append(N_moving)
        self.N_gt_static.append(float(np.sum(gt_static)))
        self.N_intersection.append(N_intersection)
        self.N_union.append(N_union)
        self.N_predicted_moving.append(N_motion_mask)
        self.N_predicted_static.append(motion_mask.size-N_motion_mask)
        self.N_gt_all_pixels.append(gt_mag.size)
        if N_moving > 0:
            recall = N_intersection/N_moving
            self.recall.append(recall)
        else:
            self.recall.append(np.nan)
        if N_motion_mask > 0:
            precision = N_intersection/N_motion_mask
            self.precision.append(precision)
            static_intersection = np.logical_and(gt_static, motion_mask)
            false_motion = float(np.sum(static_intersection))/N_motion_mask
            self.false_motion.append(false_motion)
        else:
            self.precision.append(np.nan)
            self.false_motion.append(np.nan)
        
        if N_union > 0:
            iou = N_intersection/N_union
            self.iou.append(iou)
        else:
            self.iou.append(np.nan)
        
        if self.compute_mui:
            res = motion_uncertainty_iou_precision_recall(
                gt_flow, flow, uncertainty,
                self.uncertainty_thresholds, self.motion_thresholds)
            mu_iou, mu_precision, mu_recall = res
            self.mu_iou.append(mu_iou)
            self.mu_precision.append(mu_precision)
            self.mu_recall.append(mu_recall)
        
        if self.mask_save_folder is not None:
            self.save_masks(gt_moving, gt_static, motion_mask, idx)
        if self.flow_save_folder is not None and flow is not None:
            self.save_flow(flow, uncertainty, idx)
    
    
    def save_stats(self, fn):
        """Save the statistics computed by run() to a npz file at path fn."""
        r = np.array(self.recall)
        p = np.array(self.precision)
        epe = np.array(self.epe)
        iou = np.array(self.iou)
        fpm = np.array(self.false_motion)
        Ngs = np.array(self.N_gt_static)
        Ngm = np.array(self.N_gt_moving)
        Npm = np.array(self.N_predicted_moving)
        Nps = np.array(self.N_predicted_static)
        Ni = np.array(self.N_intersection)
        Nu = np.array(self.N_union)
        np.savez_compressed(
            fn, recall=r, precision=p,
            iou=iou, false_motion=fpm,
            N_gt_static=Ngs, N_gt_moving=Ngm,
            N_predicted_moving=Npm, N_predicted_static=Nps,
            N_intersection=Ni, N_union=Nu,
            epe=epe,
            epe_pred_lt_1_sum=self.epe_pred_lt_1_sum,
            epe_pred_lt_1_count=self.epe_pred_lt_1_count,
            epe_pred_gt_1_sum=self.epe_pred_gt_1_sum,
            epe_pred_gt_1_count=self.epe_pred_gt_1_count,
            magnitude_bin_edges=self.magnitude_bin_edges,
            epe_vs_magnitude_sum=self.epe_vs_magnitude_sum,
            epe_vs_magnitude_count=self.epe_vs_magnitude_count,
            N_gt_all_pixels=self.N_gt_all_pixels,
            mu_iou = np.array(self.mu_iou, dtype=np.float32),
            mu_precision = np.array(self.mu_precision, dtype=np.float32),
            mu_recall = np.array(self.mu_recall, dtype=np.float32),
            uncertainty_thresholds = self.uncertainty_thresholds,
            motion_thresholds = self.motion_thresholds,
            )
    
    
    def run(self):
        """Run the evaluation of the given segmenter on the given clip."""
        idx = 1
        fn = "{:08}.png".format(idx)
        fn_rgb = os.path.join(self.rgb_folder,fn)
        fn_flow = os.path.join(self.flow_folder,fn)
        try:
            rgb = imageio.imread(fn_rgb)
        except FileNotFoundError:
            rgb = None
        gt_flow = load_flow_png_float(fn_flow)
        assert(gt_flow is not None and rgb is not None)
        stick_mask = None
        if self.stick_mask_folder is not None:
            fn_stick_mask = os.path.join(self.stick_mask_folder,fn)
            try:
                stick_mask = imageio.imread(fn_stick_mask)
            except FileNotFoundError:
                raise RuntimeError("""Stick mask clip is required but
                    its first image file {} was not found!""".format(
                    fn_stick_mask))
        file_names = sorted(os.listdir(self.rgb_folder))
        fn_idx = 0
        assert(file_names[fn_idx] == fn)
        assert(int(file_names[fn_idx].split('.')[0]) == idx)
        while gt_flow is not None and rgb is not None:
            if idx == 1:
                set_reference = True
            else:
                set_reference = False
            actor_mask = None
            if stick_mask is not None:
                actor_mask = stick_mask.astype(bool)
            if rgb.shape[2] > 3:
                rgb = rgb[...,0:3]
            motion_mask, flow, uncertainty = self.segmenter.next_image(
                rgb, self.thr_motion, set_reference, actor_mask)
            motion_mask = motion_mask > 0
            self.append_stats(gt_flow, motion_mask, flow, uncertainty, idx)

            fn_idx += 1
            if self.progress_queue is not None:
                self.progress_queue.put(1)
            if fn_idx >= len(file_names):
                rgb = None
                gt_flow = None
                break
            fn = file_names[fn_idx]
            idx = int(fn.split('.')[0])
            assert(idx%2 == 1)
            assert(fn == "{:08}.png".format(idx))
            fn_rgb = os.path.join(self.rgb_folder, fn)
            fn_flow = os.path.join(self.flow_folder, fn)
            try:
                rgb = imageio.imread(fn_rgb)
            except FileNotFoundError:
                rgb = None
            if rgb is not None:
                gt_flow = load_flow_png_float(fn_flow)
            else:
                gt_flow = None
            if rgb is not None and self.stick_mask_folder is not None:
                fn_stick_mask = os.path.join(self.stick_mask_folder,fn)
                stick_mask = imageio.imread(fn_stick_mask)
            else:
                stick_mask = None


class ProgressKeeper():
    def __init__(self):
        self.manager = multiprocessing.Manager()
        self.progress_queue = self.manager.Queue()
        self.n_all = 0
        self.pbar = None
        self.running = self.manager.Value("i", 0)
        self.process = multiprocessing.Process(target=self.run)
    def setup(self, rgb_folders, cli_position=0):
        self.n_all = 0
        for rgb_folder in rgb_folders:
            self.n_all += len(os.listdir(rgb_folder))
        self.pbar = tqdm(total=self.n_all, position=cli_position, leave=False)
        self.running.value = 1
        self.process.start()
    def close(self):
        time.sleep(0.2)
        self.running.value = 0
        self.process.join()
        if self.pbar is not None:
            self.pbar.close()
    def update_progress(self):
        assert(self.pbar is not None)
        try_again = True
        while try_again:
            try:
                progress_increment = self.progress_queue.get(block=False)
                self.pbar.update(progress_increment)
            except queue.Empty as e:
                try_again = False
                pass
    def run(self):
        while self.running.value:
            self.update_progress()
            time.sleep(0.1)

