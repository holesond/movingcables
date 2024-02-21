"""
Sample a composed dataset configuration given its compositing recipe.
"""


import os
import json
import argparse
from copy import deepcopy

import numpy as np
import pandas as pd

from dataset_samplers import SourceClipSampler, CompositionSampler
import verify_compositions



def verify_subset_recipe(subset):
    """Verify basic internal consistency of a subset (dataset split) recipe."""
    n_composed_1 = (subset["source_clips"]["poking"]
        + subset["source_clips"]["push-pull"]
        + subset["source_clips"]["lateral"])
    n_composed_1 = n_composed_1*subset["motion_clip_augmentation_factor"]
    n_composed_2 = sum(subset["background_classes"].values())
    n_composed_3 = sum(subset["static_cable_classes"].values())
    n_composed_4 = sum(subset["cable_colors"].values())
    n_composed_5 = sum([v["n_clips"] for v in
        subset["cable_densities"].values()])
    assert(n_composed_1 == n_composed_2)
    assert(n_composed_1 == n_composed_3)
    assert(n_composed_1 == n_composed_4)
    assert(n_composed_1 == n_composed_5)
    assert(subset["cable_colors"]["original_cable_and_background"]
        <= subset["background_classes"]["plain_original"])
    if subset["static_cable_classes"]["static_clips"] > 0:
        assert(subset["source_clips"]["static"] > 0)


def verify_source_class(recipe, clips, motion_class):
    """Verify that there are enough clips of a given motion class."""
    n_motion = np.sum(clips["motion"] == motion_class)
    assert(recipe["test"]["source_clips"][motion_class]
        + recipe["train"]["source_clips"][motion_class]
        + recipe["validation"]["source_clips"][motion_class]
        == n_motion)


def verify_recipe(recipe, clips, n_clutter=254, n_distractors=138):
    """Verify that the dataset compositing recipe satisfies basic assumptions.
    
    Arguments:
    recipe -- compositing recipe data structure (already loaded from json)
    clips -- available recorded clips (Pandas data frame)
    
    Keyword arguments:
    n_clutter -- number of available clutter background images
    n_distractors -- number of available distractor background images
    """
    verify_subset_recipe(recipe["test"])
    verify_subset_recipe(recipe["train"])
    verify_subset_recipe(recipe["validation"])
    assert(recipe["test"]["background_classes"]["clutter"]
        + recipe["train"]["background_classes"]["clutter"]
        + recipe["validation"]["background_classes"]["clutter"]
        <= n_clutter)
    assert(recipe["test"]["background_classes"]["distractor"]
        + recipe["train"]["background_classes"]["distractor"]
        + recipe["validation"]["background_classes"]["distractor"]
        <= n_distractors)
    verify_source_class(recipe, clips, "poking")
    verify_source_class(recipe, clips, "push-pull")
    verify_source_class(recipe, clips, "lateral")
    verify_source_class(recipe, clips, "static")


def sample_compositions(
        recipe, all_clips, bg_clutter, bg_distractors):
    """Sample dataset compositions given recipe, clips and backgrounds."""
    source_sampler = SourceClipSampler(recipe, all_clips)
    source_sampler.split_clips()
    source_splits = source_sampler.splits
    #print(source_splits)
    composition_sampler = CompositionSampler(
        recipe, all_clips, source_splits, bg_clutter, bg_distractors)
    composition_sampler.sample_all()
    #print(composition_sampler.compositions)
    return composition_sampler.compositions


def test_compositions_equal(compositions_1, compositions_2):
    s1 = json.dumps(compositions_1, sort_keys=True, indent=4)
    s2 = json.dumps(compositions_2, sort_keys=True, indent=4)
    assert(s1 == s2)


def main(
        fn_recipe, fn_clip_list, fn_clutter_list, fn_distractor_list,
        fn_out_config):
    """Sample a composed dataset configuration given a recipe.
    
    Arguments:
    fn_recipe -- path of the compositing recipe json file
    fn_clip_list -- path of a csv file listing the recorded clips
        - it contains four columns named: clip, configuration, motion, rgba_dir
        - clip - clip name, e.g. 0011
        - configuration - cable configuration code, e.g. 001
        - motion - motion type, one of: poking, static, push-pull, lateral
        - rgba_dir - path to the clip's RGBA image dir, e.g. rgba_clips/0011
    fn_clutter_list -- path of a txt file listing available clutter
        backgrounds, one image file path per line
    fn_distractor_list -- path of a txt file listing available distractor
        backgrounds, one image file path per line
    fn_out_config -- output json file path in which to store the sampled
        composed dataset configuration
    """
    with open(os.path.abspath(fn_recipe)) as f:
        recipe = json.load(f)
    #clips = np.genfromtxt(
    #    os.path.abspath(fn_clip_list), delimiter=',', dtype='S80')
    clips = pd.read_csv(
        os.path.abspath(fn_clip_list), sep=',', header=0, dtype=str)
    # clips.to_csv("recorded_clips_2.csv", sep=',', index=False)
    bg_clutter = list(np.loadtxt(fn_clutter_list, dtype=str))
    bg_distractors = list(np.loadtxt(fn_distractor_list, dtype=str))
    verify_recipe(recipe, clips)
    compositions = sample_compositions(
        recipe, clips, bg_clutter, bg_distractors)
    compositions_2 = sample_compositions(
        recipe, clips, bg_clutter, bg_distractors)
    test_compositions_equal(compositions, compositions_2)
    verify_compositions.main(
        compositions, recipe, clips, bg_clutter, bg_distractors)
    with open(fn_out_config, "w") as fp:
        json.dump(compositions, fp, sort_keys=True, indent=4)


def cmd_main():
    """Run the CLI stand-alone program."""
    parser = argparse.ArgumentParser(
        description=("Sample a composed dataset configuration given "
            "its compositing recipe."))
    parser.add_argument(
        "recipe", default=None, type=str,
        help="the input compositing recipe (json)")
    parser.add_argument(
        "recorded_clips_csv", default=None, type=str,
        help="a csv file listing the recorded dataset clips")
    parser.add_argument(
        "clutter_txt", default=None, type=str,
        help=("a txt file listing the paths of the VGA clutter "
            "background images"))
    parser.add_argument(
        "distractors_txt", default=None, type=str,
        help=("a txt file listing the paths of the VGA distractor "
            "background images"))
    parser.add_argument(
        "sampled_compositions", default=None, type=str,
        help="the output sampled compositions file (json)")
    args = parser.parse_args()
    main(
        args.recipe, args.recorded_clips_csv, args.clutter_txt,
        args.distractors_txt, args.sampled_compositions)


if __name__ == "__main__":
    cmd_main()


"""
Lists of clutter and distractor background images obtained via:

ls -d background_images/vga_cc0/clutter/* > vga_cc0_clutter.txt
ls -d background_images/vga_cc0/distractors/* > vga_cc0_distractors.txt
"""
