import os
import cv2
import numpy as np
from .dataset_prefix import data_prefix



# ======== PLEASE MODIFY ========
mc_root = '/home/holesond/datasets/CableTAMP/moving_hoses/sampled_compositions_small'



def load_flow_png_float32(fn):
    """Load a uint16 PNG optical flow image as a numpy float32 array."""
    image = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    assert(image is not None)
    assert(image.dtype==np.uint16)
    image = image[...,::-1] # Convert from reversed 'BGR' order to original 'RGB'.
    flow = image[...,0:2]
    flow = flow.astype(np.float32)
    flow = (flow - 2**15)*(1.0/64.0)
    return flow


def read_dataset(
        parts = 'train', resize = None, samples = -1,
        normalize = False, crop = None, shard=1):
    dataset = dict()
    dataset['image_0'] = []
    dataset['image_1'] = []
    dataset['flow'] = []
    dataset['occ'] = []
    if parts == 'train':
        split_name = 'train'
    elif parts == 'valid':
        split_name = 'validation'
    else:
        raise ValueError(
            f"Invalid MovingCables parts (split) {parts}. "
            f"Only train or valid are supported.")
    split_root = os.path.join(mc_root, split_name)
    path_images = os.path.join(split_root, "rgb_clips")
    path_flows = os.path.join(split_root, "normal_flow_first_back")
    path_occ = os.path.join(split_root, "stick_masks")
    list_clips = sorted(os.listdir(path_flows))
    list_files = []
    for clip_name in list_clips:
        clip_folder = os.path.join(path_flows, clip_name)
        for image_name in sorted(os.listdir(clip_folder)):
            if image_name == "00000001.png":
                continue
            list_files.append([clip_name, image_name])
    num_files = len(list_files)
    if samples != -1:
        num_files = min(num_files, samples)
    for k in range(0, num_files, shard):
        clip_name, image_name = list_files[k]
        fn_img0 = os.path.join(path_images, clip_name, image_name)
        fn_img1 = os.path.join(path_images, clip_name, "00000001.png")
        fn_flow = os.path.join(path_flows, clip_name, image_name)
        fn_occ = os.path.join(path_occ, clip_name, image_name)
        img0 = cv2.imread(fn_img0)
        img0 = img0[...,::-1]
        img1 = cv2.imread(fn_img1)
        img1 = img1[...,::-1]
        flow = load_flow_png_float32(fn_flow)
        stick_mask = cv2.imread(fn_occ, cv2.IMREAD_UNCHANGED)
        if crop is not None:
            img0 = img0[crop[0]: -crop[0], crop[1]: -crop[1]]
            img1 = img1[crop[0]: -crop[0], crop[1]: -crop[1]]
            flow = flow[crop[0]: -crop[0], crop[1]: -crop[1]]
            stick_mask = stick_mask[crop[0]: -crop[0], crop[1]: -crop[1]]
        occ = (stick_mask == 0).astype(np.uint8)
        # Stick mask: 0 - background or foreground, 1,2,3,4,... - poking stick
        # Occlusion mask: 0 - occluded/invalid flow, 1 - valid flow
        if normalize:
            img_min, img_max = min(img0.min(), img1.min()), max(img0.max(), img1.max())
            img0, img1 = [((img - img_min) * (255.0 / (img_max - img_min))).astype(np.uint8) for img in (img0, img1)]
        if resize is not None:
            # cv2.resize assumes resize=(width, height)
            img0 = cv2.resize(img0, resize)
            img1 = cv2.resize(img1, resize)
            flow = cv2.resize(flow, resize) * ((np.array(resize, dtype = np.float32) - 1.0) / (
                    np.array([flow.shape[d] for d in (1, 0)], dtype = np.float32) - 1.0))[np.newaxis, np.newaxis, :]
            occ = cv2.resize(occ.astype(np.float32), resize)[..., np.newaxis]
            flow = flow / (occ + (occ == 0))
            occ = (occ * 255).astype(np.uint8)
        else:
            occ = occ * 255
        flow = flow.astype(np.float16)
        assert(img0.dtype == np.uint8)
        assert(img1.dtype == np.uint8)
        assert(flow.dtype == np.float16)
        assert(occ.dtype == np.uint8)
        dataset['image_0'].append(img0)
        dataset['image_1'].append(img1)
        dataset['flow'].append(flow)
        dataset['occ'].append(occ)
    return dataset


if __name__ == '__main__':
    dataset = read_dataset()
