"""Compose the clips specified in sampled_compositions.json file.
Usage: python compositor.py sampled_compositions.json
    /recorded/dataset/root/folder
    /composed/dataset/output/folder
"""

import os
import sys
import inspect
import pathlib
import argparse
import json
from copy import deepcopy
from hashlib import sha256
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import queue
import time

import cv2
import numpy as np
import imageio.v3 as imageio
from skimage.color import rgb2hsv, hsv2rgb
import png
from tqdm import tqdm

script_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
sys.path.append(script_dir)
sys.path.append(os.path.join(
    script_dir, "../Noise2NoiseFlow/sRGB_noise_modeling"))
from add_noise import SRGBNoiseModel
from color_transform_torch import transform_color_numpy as transform_color_bcsh



def infer_flow_folder(rgba_folder, flow_type, stick_masks=False):
    # path structure /dataset/rgba_clips/0001/
    sp = ["", ""]
    head = deepcopy(rgba_folder)
    while head[-1] == "/" or  head[-1] == "\\":
        head = head[0:-1]
    for i in range(len(sp)-1,-1,-1):
        head, tail = os.path.split(head) 
        sp[i] = tail
    assert(sp[0] in ['rgba_clips', 'rgb_clips', 'clips'])
    if flow_type == "optical_flow":
        sp[0] = 'flow_first_back'
    elif flow_type == "normal_flow":
        sp[0] = 'normal_flow_first_back'
    else:
        raise ValueError("infer_flow_folder: unknown flow type {}.".format(
            flow_type))
    flow_first_back = os.path.join(head, *sp)
    rgba_stick_folder = None
    stick_mask_folder = None
    if stick_masks:
        sp[0] = "rgba_clips_stick"
        rgba_stick_folder = os.path.join(head, *sp)
        sp[0] = "stick_masks"
        stick_mask_folder = os.path.join(head, *sp)
    return flow_first_back, rgba_stick_folder, stick_mask_folder


def get_rgb_clip_folder(rgba_folder):
    sp = os.path.normpath(rgba_folder).split(os.path.sep)
    assert(len(sp) > 1)
    assert(sp[-2] == 'rgba_clips')
    sp[-2] = 'clips'
    if sp[0] == '':
        sp[0] = os.path.sep
    fn_rgb = os.path.join(*sp)
    return fn_rgb


def denoise_bilateral_filter(img):
    d = 9
    sigmaColor = 9.0
    sigmaSpace = 5.0    # No effect when d>0.
    if img.shape[2] > 3:
        dst = cv2.bilateralFilter(img[...,0:3], d, sigmaColor, sigmaSpace)
        img[...,0:3] = dst
    else:
        dst = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
        img[...,:] = dst


class Clip():
    def __init__(
            self, rgba_folder, single_image_index=None,
            stick_masks=False, plain_background=False):
        # If single_image_index is not None, the clip is a static one.
        self.rgba_folder = rgba_folder
        self.single_image_index = single_image_index
        self.plain_background = plain_background
        self.static_image_rgba = None
        self.bcsh_transform = None
        self.to_grayscale = False
        self.color_shuffle = None
        self.invert_color = False
        self.bg_color_transform = None
        self.fg_color_transform = None
        (self.flow_first_back_folder, self.rgba_stick_folder,
            self.stick_mask_folder) = infer_flow_folder(
                rgba_folder, "optical_flow", stick_masks)
        self.normal_flow_first_back_folder, _, _ = infer_flow_folder(
            rgba_folder, "normal_flow", stick_masks)
        if self.single_image_index is not None:
            try:
                self.static_image_rgba = self.load_rgba(
                    self.single_image_index)
            except FileNotFoundError as exc:
                raise RuntimeError("Could not load the required "
                    "static image.") from exc
            denoise_bilateral_filter(self.static_image_rgba)
            self.static_flow_first_back = self.load_flow_first_back(
                self.single_image_index)
            if self.static_flow_first_back is None:
                raise RuntimeError("Could not load the required "
                    "static image flow.")
            self.static_normal_flow_first_back = self.load_normal_flow_first_back(
                self.single_image_index)
            if self.static_normal_flow_first_back is None:
                raise RuntimeError("Could not load the required "
                    "static image normal flow.")
            # Set the flow to zero (uint16 format).
            self.static_flow_first_back[...,0] = 2**15
            self.static_flow_first_back[...,1] = 2**15
            self.static_normal_flow_first_back[...,0] = 2**15
            self.static_normal_flow_first_back[...,1] = 2**15
            if stick_masks:
                self.static_stick_mask = self.load_stick_mask(
                    self.single_image_index)
    def load_rgba(self, index):
        name = "{:08d}.png".format(index)
        fn_rgba = os.path.join(self.rgba_folder, name)
        rgba = imageio.imread(fn_rgba).copy()
        if not self.plain_background:
            return rgba
        rgb_folder = get_rgb_clip_folder(self.rgba_folder)
        fn_rgb = os.path.join(rgb_folder, name)
        rgb = imageio.imread(fn_rgb)
        rgba[...,0:3] = rgb
        return rgba
    def load_flow_first_back(self, index):
        name = "{:08d}.png".format(index)
        fn_flow_first_back = os.path.join(
            self.flow_first_back_folder, name)
        return load_flow_png(fn_flow_first_back)
    def load_normal_flow_first_back(self, index):
        name = "{:08d}.png".format(index)
        fn_normal_flow_first_back = os.path.join(
            self.normal_flow_first_back_folder, name)
        return load_flow_png(fn_normal_flow_first_back)
    def load_stick_mask(self, index):
        name = "{:08d}.png".format(index)
        fn_stick_mask = os.path.join(self.stick_mask_folder, name)
        im_stick_mask = imageio.imread(fn_stick_mask).copy()
        # Change mask values from {0, 255} to {0, 1}.
        im_stick_mask[im_stick_mask > 0] = 1
        return im_stick_mask
    def set_bcsh_transform(self, bcsh_transform):
        """
        brightness e.g. [0.5,1.5]
        contrast e.g. [0.5,1.5]
        saturation e.g. [0.5,1.5]
        hue [-0.5,0.5]
        """
        self.bcsh_transform = bcsh_transform
    def set_color_shuffle(self, indices):
        assert(len(indices)==3)
        self.color_shuffle = indices
    def set_invert_color(self):
        self.invert_color = True
    def set_grayscale(self):
        self.to_grayscale = True
    def set_bg_color_transform(self, bg_transform):
        """
        Background color transform.
        bg_transform = [type, parameters]
        type: "grayscale", "invert", "channel_shuffle", "color_jitter"
        parameters:
            - None for "grayscale" or "invert" type
            - permuted [0,1,2] for "channel_shuffle" type
            - for type "color_jitter": dict containing keys 
                "brightness_factor", "contrast_factor",
                "saturation_factor", "hue_factor" 
        """
        self.bg_color_transform = bg_transform
    def set_fg_color_transform(self, fg_transform):
        """
        Foreground color transform.
        """
        self.fg_color_transform = fg_transform


def transform_color(clip, im_rgba):
    im_rgba = deepcopy(im_rgba)
    if clip.bcsh_transform:
        brightness_factor = clip.bcsh_transform[0]
        contrast_factor = clip.bcsh_transform[1]
        saturation_factor = clip.bcsh_transform[2]
        hue_factor = clip.bcsh_transform[3]
        im_rgba[...,0:3] = transform_color_bcsh(
            im_rgba[...,0:3],
            brightness_factor,
            contrast_factor,
            saturation_factor,
            hue_factor)
    if clip.invert_color:
        im_rgba[...,0:3] = 255-im_rgba[...,0:3]
    if clip.color_shuffle:
        im_rgba[...,0:3] = im_rgba[...,clip.color_shuffle]
    if clip.to_grayscale:
        gray = 0.30*im_rgba[...,0] + 0.59*im_rgba[...,1] + 0.11*im_rgba[...,2]
        im_rgba[...,0:3] = gray[...,None].astype(np.uint8)
    return im_rgba


def transform_color_config(im_rgba, config):
    im_rgba = deepcopy(im_rgba)
    if config is None:
        return im_rgba
    transform_type = config[0]
    if transform_type == "grayscale":
        gray = 0.30*im_rgba[...,0] + 0.59*im_rgba[...,1] + 0.11*im_rgba[...,2]
        im_rgba[...,0:3] = gray[...,None].astype(np.uint8)
        return im_rgba
    if transform_type == "invert":
        im_rgba[...,0:3] = 255-im_rgba[...,0:3]
        return im_rgba
    if transform_type == "channel_shuffle":
        params = config[1]
        assert(set(params) == set([0,1,2]))
        im_rgba[...,0:3] = im_rgba[...,params]
        return im_rgba
    if transform_type == "color_jitter":
        params = config[1]
        im_rgba[...,0:3] = transform_color_bcsh(
            im_rgba[...,0:3],
            params["brightness_factor"],
            params["contrast_factor"],
            params["saturation_factor"],
            params["hue_factor"])
        return im_rgba
    raise ValueError("Invalid color transform type: {}".format(
        transform_type))


def transform_color_bg_fg(clip, im_rgba):
    if (clip.bg_color_transform is None and
            clip.fg_color_transform is None):
        return deepcopy(im_rgba)
    bg_img = transform_color_config(im_rgba, clip.bg_color_transform)
    fg_img = transform_color_config(im_rgba, clip.fg_color_transform)
    alpha = im_rgba[...,3]/255.0
    rgb_blend = (1.0-alpha[...,None])*bg_img[...,0:3] + \
        alpha[...,None]*fg_img[...,0:3]
    fg_img[...,0:3] = rgb_blend
    return fg_img


class Frameset():
    def __init__(
            self, rgba, flow_first_back, normal_flow_first_back,
            rgba_stick, stick_mask, is_static=False):
        self.rgba = rgba
        self.flow_first_back = flow_first_back
        self.normal_flow_first_back = normal_flow_first_back
        self.rgba_stick = rgba_stick
        self.stick_mask = stick_mask
        self.is_static = is_static


def load_flow_png(fn):
    im = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    im = im[...,::-1] # Convert from reversed 'BGR' order to original 'RGB'.
    im[im[...,2]>0, 2] = 1 # Change mask values from {0, 255} to {0, 1}. 
    return im


def save_flow_png(fn,im):
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


def add_noise_to_image(clean_image, noise_model):
    noisy_image = deepcopy(clean_image)
    rgb = noisy_image[...,0:3].astype(np.float32)
    noisy_rgb = noise_model.process(rgb)
    noisy_rgb = noisy_rgb.astype(np.uint8)
    noisy_image[...,0:3] = noisy_rgb
    return noisy_image


def load_framesets(clips, index, noise_model=None):
    framesets = []
    for c in clips:
        im_stick_mask = None
        im_rgba_stick = None
        is_static = False
        if c.single_image_index is not None:
            assert(noise_model is not None)
            is_static = True
            idx = c.single_image_index
            im_rgba = add_noise_to_image(c.static_image_rgba, noise_model)
            im_flow_first_back = deepcopy(
                c.static_flow_first_back)
            im_normal_flow_first_back = deepcopy(
                c.static_normal_flow_first_back)
            if c.rgba_stick_folder is not None:
                im_stick_mask = deepcopy(c.static_stick_mask)
                # Stick mask only for masking flows during composition.
                # This mask is not added to the output stick mask.
                # This mask "only" masks out the output stick mask.
                # So do not even load the stick rgba image here.
        else:
            idx = index
            try:
                im_rgba = c.load_rgba(idx)
            except FileNotFoundError:
                return None
            im_flow_first_back = c.load_flow_first_back(idx)
            if im_flow_first_back is None:
                return None
            im_normal_flow_first_back = c.load_normal_flow_first_back(idx)
            if im_normal_flow_first_back is None:
                return None
            if c.rgba_stick_folder is not None:
                name = "{:08d}.png".format(idx)
                assert(c.stick_mask_folder is not None)
                fn_rgba_stick = os.path.join(c.rgba_stick_folder, name)
                im_rgba_stick = imageio.imread(fn_rgba_stick)
                im_stick_mask = c.load_stick_mask(idx)
        framesets.append(
            Frameset(
                im_rgba, im_flow_first_back,
                im_normal_flow_first_back,
                im_rgba_stick, im_stick_mask, is_static))
    return framesets


def fuse_rgb(rgb_bottom, rgba_top):
    if rgb_bottom is None:
        return rgba_top[...,0:3]
    alpha = rgba_top[...,3]/255.0
    rgb_bottom = (1.0-alpha[...,None])*rgb_bottom + \
        alpha[...,None]*rgba_top[...,0:3]
    return rgb_bottom


def fuse_flows(flow_bottom, flow_top, stick_mask_top=None, label=1):
    assert(label != 0)
    overwrite = flow_top[...,2].astype(bool)
    flow_bottom[overwrite,0] = flow_top[overwrite,0]
    flow_bottom[overwrite,1] = flow_top[overwrite,1]
    flow_bottom[overwrite,2] = label
    if stick_mask_top is not None:
        overwrite_stick = stick_mask_top > 0
        # Set flow to zero (2**15 in uint16 encoding).
        # Set cable mask to zero (no cable but background). 
        flow_bottom[overwrite_stick,0] = 2**15
        flow_bottom[overwrite_stick,1] = 2**15
        flow_bottom[overwrite_stick,2] = 0
    return flow_bottom


def fuse_sticks(stick_bottom, stick_top, rgba_top):
    alpha = rgba_top[...,3]/255.0
    stick_bottom = (1.0-alpha[...,None])*stick_bottom + \
        alpha[...,None]*stick_top
    return stick_bottom


def fuse_stick_masks(stick_bottom, stick_top, flow_top, label=1):
    assert(stick_bottom.dtype == np.uint8)
    assert(stick_top.dtype == np.uint8)
    overwrite_cable = flow_top[...,2].astype(bool)
    overwrite_stick = stick_top.astype(bool)
    stick_bottom[overwrite_cable] = 0
    stick_bottom[overwrite_stick] = label
    return stick_bottom


def find_clip_rgb_shape(bg_images, fg_clips):
    x_shape_clip = None
    for c in fg_clips:
        if c.static_image_rgba is None:
            im_rgba = c.load_rgba(1)
            x_shape = deepcopy(im_rgba.shape)
        else:
            x_shape = deepcopy(c.static_image_rgba.shape)
        assert(len(x_shape) == 3)
        if x_shape[2] == 4:
            x_shape = (x_shape[0], x_shape[1], 3)
        if x_shape_clip is None:
            x_shape_clip = x_shape
        else:
            assert(x_shape == x_shape_clip)
    if not bg_images:
        return x_shape_clip
    try:
        x_shape_background = bg_images[0].shape
    except AttributeError:
        return x_shape_clip
    if x_shape_clip is None:
        return x_shape_background
    if x_shape_background != x_shape_clip:
        print("    Shape of foreground clip images:", x_shape_clip)
        print("    Shape of the first background image:", x_shape_background)
    assert(x_shape_background == x_shape_clip)
    return x_shape_clip
    

def compose(
        bg_images, fg_clips, folder_rgb_out, seed=None,
        require_stick_masks=False, save_rgb_stick=False,
        progress_queue=None):
    """Create a composed clip and save its files (images) to disk.
    
    Arguments:
    - bg_images - a list of background RGB images with resolution matching the
        clip resolution
    - fg_clips - a layer-ordered list of Clips
        (the first is bottom, the last is top)
    - folder_rgb_out - folder for the RGB images of the composed output clip
        - folders for the flow images are inferred automatically
        - e.g. /dataset/rgba_clips/0001
        - data path structure /dataset/rgba_clips/0001/*.png
    - progress_queue -- The compositor puts a 1 to this queue when it
            processes one frameset.
            (default None)
    
    - The length of the output clip matches the length of the shortest fg_clip.
    - For each output clip frame, randomly sample a background image from
        bg_images.
    - Apply the requested data augmentation transforms to each clip.
    """
    idx = 1
    noise_model = None
    x_shape = find_clip_rgb_shape(bg_images, fg_clips)
    if x_shape is not None:
        noise_model = SRGBNoiseModel(x_shape, seed=seed)
    framesets = load_framesets(fg_clips, idx, noise_model)
    if framesets is None:
        print("WARNING (compose): No frameset at the first index!")
        return
    save_stick_masks = False
    for clip in fg_clips:
        if clip.rgba_stick_folder is not None:
            save_stick_masks = True
            break
    (flow_first_back_out, rgba_stick_folder_out,
        stick_mask_folder_out) = infer_flow_folder(
        folder_rgb_out, "optical_flow", save_stick_masks)
    normal_flow_first_back_out, _, _ = infer_flow_folder(
        folder_rgb_out, "normal_flow", save_stick_masks)
    #_,_,flow_first_back_out = infer_flow_folders(folder_rgb_out)
    pathlib.Path(folder_rgb_out).mkdir(parents=True, exist_ok=True)
    pathlib.Path(flow_first_back_out).mkdir(parents=True, exist_ok=True)
    pathlib.Path(normal_flow_first_back_out).mkdir(parents=True, exist_ok=True)
    if save_stick_masks == True:
        pathlib.Path(stick_mask_folder_out).mkdir(parents=True, exist_ok=True)
        if save_rgb_stick:
            pathlib.Path(rgba_stick_folder_out).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng()
    while framesets is not None:
        if len(bg_images) == 1:
            # Add noise.
            assert(noise_model is not None)
            rgb_comp = add_noise_to_image(bg_images[0], noise_model)
            rgb_comp = rgb_comp.astype(np.float32)
        elif len(bg_images) > 1:
            # Do not add noise but randomly sample one of the images.
            bg_idx = rng.integers(low=0, high=len(bg_images), size=1)[0]
            rgb_comp = deepcopy(bg_images[bg_idx].astype(np.float32))
        else:
            # Plain background.
            rgb_comp = None
        flow_first_back_comp = None
        normal_flow_first_back_comp = None
        rgba_stick_comp = None
        stick_mask_comp = None
        for i_clip, (fs, clip) in enumerate(zip(framesets, fg_clips)):
            rgba_trans = transform_color_bg_fg(clip, fs.rgba)
            rgb_comp = fuse_rgb(rgb_comp, rgba_trans.astype(np.float32))
            if flow_first_back_comp is None:
                flow_first_back_comp = deepcopy(fs.flow_first_back)
                normal_flow_first_back_comp = deepcopy(fs.normal_flow_first_back)
                if not fs.is_static:
                    # Stick masks from single image clips only occlude
                    # the moving stick. (They are not added to it.)
                    rgba_stick_comp = deepcopy(fs.rgba_stick)
                    stick_mask_comp = deepcopy(fs.stick_mask)
                continue
            if require_stick_masks:
                assert(fs.stick_mask is not None)
            flow_first_back_comp = fuse_flows(
                flow_first_back_comp, fs.flow_first_back,
                fs.stick_mask, i_clip+1)
            normal_flow_first_back_comp = fuse_flows(
                normal_flow_first_back_comp, fs.normal_flow_first_back,
                fs.stick_mask, i_clip+1)
            if fs.stick_mask is not None:
                if stick_mask_comp is None:
                    if not fs.is_static:
                        rgba_stick_comp = deepcopy(fs.rgba_stick)
                        stick_mask_comp = deepcopy(fs.stick_mask)
                        stick_mask_comp[stick_mask_comp>0] = i_clip+1
                else:
                    if not fs.is_static:
                        rgba_stick_comp = fuse_sticks(
                            rgba_stick_comp, fs.rgba_stick,
                            rgba_trans.astype(np.float32))
                    if fs.is_static:
                        stick_label = 0
                    else:
                        stick_label = i_clip+1
                    stick_mask_comp = fuse_stick_masks(
                        stick_mask_comp, fs.stick_mask,
                        fs.flow_first_back, label=stick_label)
        name = "{:08d}.png".format(idx)
        fn_rgb = os.path.join(folder_rgb_out, name)
        fn_rgb_stick = os.path.join(rgba_stick_folder_out, name)
        fn_stick_mask = os.path.join(stick_mask_folder_out, name)
        fn_flow_first_back = os.path.join(flow_first_back_out, name)
        fn_normal_flow_first_back = os.path.join(normal_flow_first_back_out, name)
        imageio.imwrite(fn_rgb, rgb_comp.astype(np.uint8))
        if require_stick_masks:
            assert(stick_mask_comp is not None)
        if stick_mask_comp is not None:
            imageio.imwrite(
                fn_stick_mask, stick_mask_comp.astype(np.uint8))
            if save_rgb_stick:
                assert(rgba_stick_comp is not None)
                imageio.imwrite(
                    fn_rgb_stick, rgba_stick_comp.astype(np.uint8))
        save_flow_png(fn_flow_first_back, flow_first_back_comp)
        save_flow_png(fn_normal_flow_first_back, normal_flow_first_back_comp)
        idx += 2
        if progress_queue is not None:
            progress_queue.put(1)
        framesets = load_framesets(fg_clips, idx, noise_model)


def compose_clip_config(config, data_root, folder_out, seed, progress_queue):
    """Compose one clip.
    
    Arguments:
    config -- clip composition configuration
    data_root -- root folder containing the recorded dataset
    folder_out -- output folder for the composed dataset
    """
    bg_images = []
    if config["background_image_path"] is not None:
        fn_background = os.path.join(
            data_root,
            config["background_image_path"])
        bg_images = [imageio.imread(fn_background)]
    fg_clips = []
    first = True
    for c_rgba_dir, c_image_idx in zip(
            config["clip_rgba_dirs"], config["clip_images"]):
        plain_background = False
        if first:
            first = False
            if not bg_images:
                plain_background = True
        clip = Clip(
            os.path.join(data_root, c_rgba_dir),
            single_image_index=c_image_idx,
            stick_masks=True,
            plain_background=plain_background)
        clip.set_bg_color_transform(config["background_transform"])
        clip.set_fg_color_transform(config["cable_transform"])
        fg_clips.append(clip)
    folder_rgb_out = os.path.join(folder_out, "rgb_clips", config["name"])
    print(folder_rgb_out)
    compose(
        bg_images, fg_clips, folder_rgb_out, seed=seed,
        require_stick_masks=True, progress_queue=progress_queue)


def compose_job(args):
    compose_clip_config(*args)


class ProgressKeeper():
    def __init__(self):
        self.manager = multiprocessing.Manager()
        self.progress_queue = self.manager.Queue()
        self.n_all = 0
        self.pbar = None
        self.running = self.manager.Value("i", 0)
        self.process = multiprocessing.Process(target=self.run)
    def setup(self, n_all, cli_position=0):
        self.n_all = n_all
        self.pbar = tqdm(
            total=self.n_all, position=cli_position, leave=False,
            smoothing=0.01)
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


def compose_sampled_compositions(
        compositions, data_root, folder_out, seed_prefix="03068216"):
    """Compose the clips specified in the compositions data structure.
    
    Arguments:
    compositions -- sampled compositions data structure (e.g. loaded from json)
    data_root -- root folder containing the recorded dataset
    folder_out -- output folder for the composed dataset
    """
    progress_keeper = ProgressKeeper()
    args_buffer = []
    for split, configs in compositions.items():
        f_out = os.path.join(folder_out, split)
        for cfg in configs:
            text = "/".join(
                [seed_prefix, split, cfg["name"], cfg["motion_clip"]])
            hsh = sha256(text.encode()).digest()
            seed = int.from_bytes(hsh[0:8], byteorder='little')
            seed = hash(seed)
            args_buffer.append(
                [cfg, data_root, f_out, seed, progress_keeper.progress_queue])
    n_clips = len(args_buffer)
    n_images_per_clip = 600
    progress_keeper.setup(n_clips * n_images_per_clip, cli_position=0)
    with ProcessPoolExecutor(max_workers=8) as workers:
        for res in workers.map(compose_job, args_buffer):
            if res is not None:
                print(str(res))
    progress_keeper.close()


def main():
    """Compose the clips specified in sampled_compositions.json file.
    
    CLI interface.
    """
    if len(sys.argv) != 4:
        print("Usage: python compositor.py sampled_compositions.json "
            "/recorded/dataset/root/folder "
            "/composed/dataset/output/folder")
        print("")
        print("Compose the clips specified in sampled_compositions.json file.")
        return
    fn_json = os.path.expanduser(sys.argv[1])
    data_root = os.path.expanduser(sys.argv[2])
    folder_out = os.path.expanduser(sys.argv[3])
    with open(fn_json) as f:
        compositions = json.load(f)
    compose_sampled_compositions(
        compositions, data_root, folder_out)


if __name__ == "__main__":
    main()
