from PIL import Image
import numpy as np
import torchvision
import imageio.v3 as imageio
import matplotlib.pyplot as plt



def transform_color(
        img,
        brightness_factor,
        contrast_factor,
        saturation_factor,
        hue_factor):
    img = torchvision.transforms.functional.adjust_brightness(
        img, brightness_factor)
    img = torchvision.transforms.functional.adjust_contrast(
        img, contrast_factor)
    img = torchvision.transforms.functional.adjust_saturation(
        img, saturation_factor)
    img = torchvision.transforms.functional.adjust_hue(
        img, hue_factor)
    return img


def transform_color_numpy(
        img,
        brightness_factor,
        contrast_factor,
        saturation_factor,
        hue_factor):
    img_pil = Image.fromarray(img, 'RGB')
    img_transformed = transform_color(
        img_pil, brightness_factor, contrast_factor,
        saturation_factor, hue_factor)
    img_transformed = np.array(img_transformed)
    return img_transformed


def sample_color_transform(
        rng,
        brightness=[0.5,1.5],
        contrast=[0.5,1.5],
        saturation=[0.5,1.5],
        hue=[-0.5,0.5]):
    brightness_factor = rng.uniform(brightness[0], brightness[1])
    contrast_factor = rng.uniform(contrast[0], contrast[1])
    saturation_factor = rng.uniform(saturation[0], saturation[1])
    hue_factor = rng.uniform(hue[0], hue[1])
    res = [
        brightness_factor,
        contrast_factor,
        saturation_factor,
        hue_factor,
        ]
    return res
