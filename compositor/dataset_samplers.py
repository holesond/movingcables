from copy import deepcopy

import numpy as np
import pandas as pd

from color_transform_torch import sample_color_transform



class SourceClipSampler():
    """Randomly split source recorded clips according to the given recipe.
    
    Usage:
    source_sampler = SourceClipSampler(recipe, all_clips)
    source_sampler.split_clips()
    source_splits = source_sampler.splits
    
    The generated source_sampler.splits is a dict with three keys, 
    "test", "train", "validation".
    """
    
    def __init__(self, recipe, clips, seed=28415260):
        self.rng = np.random.RandomState(seed)
        self.clips = clips
        self.configurations = np.unique(clips["configuration"])
        self.recipe = recipe
        self.constraints = deepcopy(recipe)
        self.splits = {k:[] for k in recipe}
        self.assigned_clips = np.zeros(len(clips.index), dtype=bool)
        self.motion_sel = ((clips["motion"] == "poking")
            | (clips["motion"] == "push-pull")
            | (clips["motion"] == "lateral"))
        self.motion_types = ["poking", "push-pull", "lateral", "static"]
    
    
    def sample_unassigned_clips(self, clip_pool, n):
        mask = self.assigned_clips[clip_pool.index]
        mask = np.logical_not(mask)
        admissible = clip_pool[mask]
        samples = admissible.sample(
            n=n, random_state=self.rng, axis='index')
        return samples
    
    
    def can_assign_clips_to_splits(self, clips):
        assert(len(clips.index) == len(self.splits))
        assert(len(clips.index) == len(self.constraints))
        for i, (kc, constraint) in enumerate(self.constraints.items()):
            if constraint["source_clips"][clips["motion"].iat[i]] <= 0:
                return False
        return True
    
    def assign_clips_to_splits(self, clips):
        for i, ((ks, split), (kc, constraint)) in enumerate(zip(
                self.splits.items(), self.constraints.items())):
            constraint["source_clips"][clips["motion"].iat[i]] -= 1
            split.append(clips["clip"].iat[i])
            self.assigned_clips[clips.index[i]] = True
    
    def sample_one_clip_per_config_and_split(self):
        """
        - Proceed configuration by configuration, from 1 to 22. (From the poorest to the richest.)
            - From each configuration, assign one random moving cable clip to each split (constraint (2)). If the assignment is not admissible (would violate constraint (1)), reject it and sample another clip. If this fails, need to forget the entire assignment and start again.
            - Now constraint (2) is satisfied.
        """
        max_trials = 20
        for conf in self.configurations:
            sel = (self.clips["configuration"] == conf) & self.motion_sel
            conf_clips = self.clips[sel]
            n_trials = 1
            samples = self.sample_unassigned_clips(
                conf_clips, len(self.splits))
            while not self.can_assign_clips_to_splits(samples):
                samples = self.sample_unassigned_clips(
                    conf_clips, len(self.splits))
                n_trials += 1
                if n_trials > max_trials:
                    raise RuntimeError("Source clip sampling failed.")
            # print(samples)
            self.assign_clips_to_splits(samples)
    
    
    def finish_sampling(self):
        """
        - Group the remaining moving and static clips by motion type ("poking", "push-pull", "lateral", "static").
        - Proceed by the motion groups.
        - Sample a random clip from the group, assign it to the current data split and remove it from the group. Once the split is satisfied, start assigning to the next split. This satisfies constraint (1).
        """
        for mt in self.motion_types:
            unassigned_sel = np.logical_not(self.assigned_clips)
            sel = (self.clips["motion"] == mt) & unassigned_sel
            class_clips = self.clips[sel]
            shuffled = class_clips.sample(
                frac=1, random_state=self.rng, axis='index')
            i_clip = 0
            for (ks, split), (kc, constraint) in zip(
                    self.splits.items(), self.constraints.items()):
                n_remaining = constraint["source_clips"][mt]
                assert(n_remaining >= 0)
                if n_remaining == 0:
                    continue
                for i in range(n_remaining):
                    assert(i_clip < len(shuffled.index))
                    split.append(shuffled["clip"].iat[i_clip])
                    self.assigned_clips[shuffled.index[i_clip]] = True
                    i_clip += 1
        for k in self.splits:
            self.splits[k] = sorted(self.splits[k])
    
    
    def split_clips(self):
        # Constraints to satisfy:
        #    (1) The source_clips class count ("poking", "push-pull",
        #        "lateral", "static") of the recipe for each dataset split.
        #    (2) At least one moving clip per configuration in each split.
        self.sample_one_clip_per_config_and_split()
        self.finish_sampling()
        self.validate_splits()
    
    
    def validate_splits(self):
        assert(len(self.splits) == 3)
        set_test = set(self.splits["test"])
        set_train = set(self.splits["train"])
        set_validation = set(self.splits["validation"])
        assert(not set_test.intersection(set_train))
        assert(not set_test.intersection(set_validation))
        assert(not set_validation.intersection(set_train))
        for (ks, split), (kc, req) in zip(
                self.splits.items(), self.recipe.items()):
            split_clips = self.clips.loc[self.clips['clip'].isin(split)]
            for mt in self.motion_types:
                n_class = np.sum(split_clips["motion"] == mt)
                assert(n_class == req["source_clips"][mt])
            unique_configs = np.unique(split_clips["configuration"])
            assert(np.all(unique_configs == self.configurations))



class CompositionSampler():
    """Sample clip compositions.
    
    Usage:
    composition_sampler = CompositionSampler(
        recipe, all_clips, source_splits, bg_clutter, bg_distractors)
    composition_sampler.sample_all()
    result = composition_sampler.compositions
    
    The result is a data structure serializable to json.
    """
    
    def __init__(
            self, recipe, clips, source_splits,
            bg_clutter, bg_distractors,
            seed=71853291):
        self.rng = np.random.RandomState(seed)
        self.recipe = recipe
        self.clips = clips
        self.source_splits = source_splits
        self.bg_clutter = bg_clutter
        self.bg_distractors = bg_distractors
        self.clip_image_indices = np.arange(1,1190,2,dtype=np.int32)
        
        self.constraints = deepcopy(recipe)
        self.augment_motion_clips()
        self.configurations = np.unique(clips["configuration"])
        self.compositions = {subset_name:[] for subset_name in recipe}
        self.available_clutter_idx = [i for i in range(len(bg_clutter))]
        self.available_distractors_idx = [
            i for i in range(len(bg_distractors))]
        self.all_clips_by_motion = {
            subset_name: self.sort_augment_source_clips_by_motion(subset_name)
            for subset_name in recipe
            }
        self.remaining_clips_by_motion = deepcopy(
            self.all_clips_by_motion)
        self.used_combinations = set()
        #print("")
        #print(self.all_clips_by_motion)
    
    
    def augment_motion_clips(self):
        for k, subset in self.constraints.items():
            factor = subset["motion_clip_augmentation_factor"]
            motion_types = ["poking", "push-pull", "lateral"]
            subset["motion_clips"] = {}
            for mt in motion_types:
                subset["motion_clips"][mt] = (
                    factor * subset["source_clips"][mt])
    
    
    def sort_augment_source_clips_by_motion(self, subset_name):
        subset_recipe = self.recipe[subset_name]
        subset_clip_ids = self.source_splits[subset_name]
        aug_factor = subset_recipe["motion_clip_augmentation_factor"]
        sorted_clips = {
            class_name: [] for class_name
            in subset_recipe["source_clips"]
            }
        subset_clips = self.clips.loc[
            self.clips['clip'].isin(subset_clip_ids)]
        for mt in sorted_clips.keys():
            sel = subset_clips["motion"] == mt
            sorted_clips[mt] = list(subset_clips[sel]["clip"].values)
            if mt != "static":
                sorted_clips[mt] = aug_factor * sorted_clips[mt]
                assert(
                    len(sorted_clips[mt])
                    == aug_factor*subset_recipe["source_clips"][mt])
            else:
                assert(
                    len(sorted_clips[mt])
                    == subset_recipe["source_clips"][mt])
        return sorted_clips
    
    
    def sample_dict(self, d):
        elements = []
        weights = []
        for k, v in d.items():
            elements.append(k)
            weights.append(v)
        w_sum = sum(weights)
        if w_sum <= 0:
            return None
        weights = np.array(weights, dtype=np.float32)
        weights = weights / float(w_sum)
        s = self.rng.choice(elements, size=1, replace=False, p=weights)
        return s[0]
    
    
    def assign_cable_density(self, d, composition):
        density_dict = {}
        for k, v in d.items():
            density_dict[k] = v["n_clips"]
        density = self.sample_dict(density_dict)
        assert(density is not None)
        count = self.rng.choice(
            d[density]["n_cables"], size=1, replace=False)
        count = count[0]
        assert(count is not None)
        d[density]["n_clips"] -= 1
        composition["cable_density"] = density
        composition["cable_count"] = int(count)
    
    
    def assign_cable_color_class(self, ccc, composition):
        if (composition["background_class"] == "plain_original"
                and ccc["original_cable_and_background"] > 0):
            composition["cable_color"] = "original"
            ccc["original_cable_and_background"] -= 1
        else:
            d = {
                "original": ccc["original_cable"],
                "transformed": ccc["transformed"],
                }
            composition["cable_color"] = self.sample_dict(d)
            assert(composition["cable_color"] is not None)
            if composition["cable_color"] == "original":
                ccc["original_cable"] -= 1
            else:
                ccc[composition["cable_color"]] -= 1
    
    
    def sample_general_color_transforms(self, n_total):
        transforms = []
        n_grayscale = int(0.1*n_total)
        n_invert = int(0.05*n_total)
        n_channel_shuffle = int(0.05*n_total)
        n_color_jitter = n_total - (n_grayscale + n_invert
                        + n_channel_shuffle)
        for i in range(n_grayscale):
            transforms.append(["grayscale", None])
        for i in range(n_invert):
            transforms.append(["invert", None])
        channels = [0,1,2]
        for i in range(n_channel_shuffle):
            self.rng.shuffle(channels)
            transforms.append(["channel_shuffle", channels])
        for i in range(n_color_jitter):
            params = sample_color_transform(self.rng)
            params_dict = {
                "brightness_factor":params[0],
                "contrast_factor":params[1],
                "saturation_factor":params[2],
                "hue_factor":params[3],
                }
            transforms.append(["color_jitter", params_dict])
        assert(len(transforms) == n_total)
        self.rng.shuffle(transforms)
        return transforms
    
    
    def add_color_transforms(self, compositions):
        idx_transformed_cable = []
        idx_transformed_background = []
        for i, composition in enumerate(compositions):
            if composition["cable_color"] == "transformed":
                idx_transformed_cable.append(i)
            else:
                compositions[i]["cable_transform"] = None
            if composition["background_class"] == "plain_transformed":
                idx_transformed_background.append(i)
            else:
                composition["background_transform"] = None
        n_total = len(idx_transformed_cable)
        transforms = self.sample_general_color_transforms(n_total)
        for i, t in zip(idx_transformed_cable, transforms):
            compositions[i]["cable_transform"] = t
        n_total = len(idx_transformed_background)
        transforms = self.sample_general_color_transforms(n_total)
        for i, t in zip(idx_transformed_background, transforms):
            compositions[i]["background_transform"] = t
    
    
    def assign_sampled_dict(self, d, composition, composition_key):
        sampled_value = self.sample_dict(d)
        assert(sampled_value is not None)
        composition[composition_key] = sampled_value
        d[sampled_value] -= 1
    
    
    def assign_motion_clip(self, composition, source_clips):
        # Sample one free motion clip of the given motion_type.
        # Assign it to composition.
        clip_pool = source_clips[composition["motion_type"]]
        assert(len(clip_pool) > 0)
        motion_clip = self.rng.choice(clip_pool)
        clip_pool.remove(motion_clip)
        composition["motion_clip"] = motion_clip
    
    
    def sample_combination_containing(
            self, given_config, config_pool, n_samples):
        sample = self.rng.choice(
            config_pool, size=n_samples-1, replace=False)
        sample = sample.tolist()
        while given_config in sample:
            sample = self.rng.choice(
                config_pool, size=n_samples-1, replace=False)
            sample = sample.tolist()
        sample.append(given_config)
        sample = frozenset(sample)
        assert(len(sample) == n_samples)
        return sample
    
    
    def sample_unique_combination_containing(
            self, given_config, config_pool, n_samples):
        sample = self.sample_combination_containing(
            given_config, config_pool, n_samples)
        while sample in self.used_combinations:
            sample = self.sample_combination_containing(
                given_config, config_pool, n_samples)
        return sample
    
    
    def is_sub_super_combination(self, comb):
        for c in self.used_combinations:
            if comb >= c or comb <= c:
                return True
        return False
    
    def sample_free_combination(
            self, given_config, config_pool, n_samples):
        comb = self.sample_unique_combination_containing(
            given_config, config_pool, n_samples)
        while self.is_sub_super_combination(comb):
            comb = self.sample_unique_combination_containing(
                given_config, config_pool, n_samples)
        return comb
    
    
    def assign_configurations(self, composition, all_clips):
        """Assign a combination of cable configurations to the composition.
        
        - Sample a free combination of cable_count configurations which
        includes the configuration of the motion clip and which respects
        static_cable_class.
        - If a combination of configurations is disabled, all its sub- and
        super-combinations (subsets and supersets) are also disabled.
        (Applies simply when the intersection of two sets (combinations)
        is equal to one of them.)
        - Randomly permute the configurations of the combination.
        """
        if composition["static_cable_class"] == "static_clips":
            sel = self.clips['clip'].isin(all_clips["static"])
            clip_table = self.clips.loc[sel]
            config_pool = np.unique(clip_table["configuration"])
        elif composition["static_cable_class"] == "single_images":
            config_pool = self.configurations
        else:
            raise ValueError("Unknown static_cable_class {}.".format(
                composition["static_cable_class"]))
        sel = self.clips['clip'] == composition["motion_clip"]
        given_config = self.clips.loc[sel]["configuration"].values[0]
        comb_set = self.sample_free_combination(
            given_config, config_pool, composition["cable_count"])
        comb = sorted(list(comb_set))
        self.rng.shuffle(comb)
        composition["configurations"] = comb
        composition["motion_config"] = given_config
        self.used_combinations.add(comb_set)
    
    
    def assign_clips_to_configurations(self, composition, all_clips):
        """Assign clips or images to static and moving configurations.
        
        For each static configuration, either pick a static clip or randomly
        sample any motion clip with that config. and a frame number in that
        clip.
        """
        if composition["static_cable_class"] == "static_clips":
            sel = self.clips['clip'].isin(all_clips["static"])
            clip_table = self.clips.loc[sel]
        elif composition["static_cable_class"] == "single_images":
            sel_all = None
            for mt in ["poking", "push-pull", "lateral"]:
                sel = self.clips['clip'].isin(all_clips[mt])
                if sel_all is None:
                    sel_all = sel
                else:
                    sel_all = np.logical_or(sel_all, sel)
            clip_table = self.clips.loc[sel_all]
        else:
            raise ValueError("Unknown static_cable_class {}.".format(
                composition["static_cable_class"]))
        composition["clips"] = []
        composition["clip_images"] = []
        for c in composition["configurations"]:
            if c == composition["motion_config"]:
                composition["clips"].append(composition["motion_clip"])
                composition["clip_images"].append(None)
                continue
            sel = clip_table["configuration"] == c
            options = clip_table.loc[sel]["clip"].values
            composition["clips"].append(self.rng.choice(options))
            if composition["static_cable_class"] == "static_clips":
                composition["clip_images"].append(None)
            else:
                idx = self.rng.choice(self.clip_image_indices)
                composition["clip_images"].append(int(idx))
    
    
    def add_paths_to_clips(self, composition):
        composition["clip_rgba_dirs"] = []
        for c in composition["clips"]:
            sel = self.clips['clip'] == c
            path = self.clips[sel]['rgba_dir'].values[0]
            composition["clip_rgba_dirs"].append(path)
    
    
    def assign_background_image(self, composition):
        bg_class = composition["background_class"]
        if bg_class in ["plain_original", "plain_transformed"]:
            composition["background_image_index"] = None
            composition["background_image_path"] = None
            return
        if bg_class == "clutter":
            idx = self.rng.choice(self.available_clutter_idx)
            idx = int(idx)
            composition["background_image_index"] = idx
            composition["background_image_path"] = self.bg_clutter[idx]
            self.available_clutter_idx.remove(idx)
            return
        if bg_class == "distractor":
            idx = self.rng.choice(self.available_distractors_idx)
            idx = int(idx)
            composition["background_image_index"] = idx
            composition["background_image_path"] = self.bg_distractors[idx]
            self.available_distractors_idx.remove(idx)
            return
        raise ValueError("Invalid background class: {}".format(bg_class))
    
    
    def sample_subset(self, subset_name):
        subset_constraints = self.constraints[subset_name]
        remaining_clips = self.remaining_clips_by_motion[subset_name]
        all_clips = self.all_clips_by_motion[subset_name]
        bg_class = self.sample_dict(
            subset_constraints["background_classes"])
        composition_index = 1
        while bg_class is not None:
            composition = {}
            composition["background_class"] = bg_class
            subset_constraints["background_classes"][bg_class] -= 1
            self.assign_background_image(composition)
            ccc = subset_constraints["cable_colors"]
            self.assign_cable_color_class(ccc, composition)
            self.assign_sampled_dict(
                subset_constraints["static_cable_classes"],
                composition,
                "static_cable_class")
            self.assign_sampled_dict(
                subset_constraints["motion_clips"],
                composition,
                "motion_type")
            self.assign_cable_density(
                subset_constraints["cable_densities"],
                composition)
            self.assign_motion_clip(composition, remaining_clips)
            self.assign_configurations(composition, all_clips)
            self.assign_clips_to_configurations(composition, all_clips)
            self.add_paths_to_clips(composition)
            composition["name"] = "{:04d}".format(composition_index)
            composition_index += 1
            self.compositions[subset_name].append(composition)
            bg_class = self.sample_dict(
                subset_constraints["background_classes"])
        self.add_color_transforms(self.compositions[subset_name])
    
    def sample_all(self):
        for subset_name in self.recipe:
            self.sample_subset(subset_name)
