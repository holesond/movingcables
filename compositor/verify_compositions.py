from copy import deepcopy


def get_unique_clips(subset):
    clip_set = set()
    for c in subset:
        clip_set.update(c["clips"])
    return clip_set


def test_disjoint_clips(compositions):
    clip_sets = []
    for k, subset in compositions.items():
        clip_sets.append(get_unique_clips(subset))
    for i, cs_1 in enumerate(clip_sets[0:-1]):
        for cs_2 in clip_sets[i+1:]:
            intersect = cs_1.intersection(cs_2)
            assert(not intersect)


def test_unique_backgrounds(compositions):
    bg_set = set()
    for k, subset in compositions.items():
        for c in subset:
            bg_path = c["background_image_path"]
            if bg_path is None:
                continue
            assert(bg_path not in bg_set)
            bg_set.add(bg_path)


def test_configuration_combinations(compositions):
    combinations = []
    for k, subset in compositions.items():
        for c in subset:
            comb = frozenset(c["configurations"])
            combinations.append(comb)
    for i, comb_1 in enumerate(combinations):
        for j, comb_2 in enumerate(combinations):
            if i == j:
                continue
            assert(not comb_1 >= comb_2)
            assert(not comb_1 <= comb_2)


def test_subset_attribute_counts(
        compositions, recipe, recipe_key, comp_key):
    counts = deepcopy(recipe[recipe_key])
    for k in counts:
        counts[k] = 0
    assert(counts != recipe[recipe_key])
    for c in compositions:
        counts[c[comp_key]] += 1
    assert(counts == recipe[recipe_key])


def test_subset_cable_densities(compositions, recipe):
    counts = {}
    for k in recipe["cable_densities"]:
        counts[k] = 0
    for c in compositions:
        counts[c["cable_density"]] += 1
    for k in recipe["cable_densities"]:
        assert(counts[k] == recipe["cable_densities"][k]["n_clips"])


def test_subset_cable_colors(compositions, recipe):
    counts = {}
    for k in recipe["cable_colors"]:
        counts[k] = 0
    for c in compositions:
        cc = c["cable_color"]
        if cc == "transformed":
            counts[cc] += 1
            continue
        if cc == "original":
            if (c["background_image_path"] is None and
                    c["background_transform"] is None):
                counts["original_cable_and_background"] += 1
            else:
                counts["original_cable"] += 1
            continue
        raise ValueError("Unexpected cable_color {}.".format(cc))
    assert(counts == recipe["cable_colors"])


def test_subset_moving_clips(compositions, recipe):
    counts = {}
    for k in recipe["source_clips"]:
        if k == "static":
            continue
        counts[k] = 0
    for c in compositions:
        counts[c["motion_type"]] += 1
    aug_factor = recipe["motion_clip_augmentation_factor"]
    for k in counts:
        assert(counts[k] == recipe["source_clips"][k]*aug_factor)
    

def test_subset_counts(compositions, recipe):
    attribute_keys = [
        ["background_classes", "background_class"],
        ["static_cable_classes", "static_cable_class"],
        ]
    for k in recipe:
        print("subset:", k)
        s_recipe = recipe[k]
        s_compositions = compositions[k]
        for keys in attribute_keys:
            test_subset_attribute_counts(
                s_compositions, s_recipe, keys[0], keys[1])
        test_subset_cable_densities(s_compositions, s_recipe)
        test_subset_cable_colors(s_compositions, s_recipe)
        test_subset_moving_clips(s_compositions, s_recipe)


def main(
        compositions, recipe, clips, bg_clutter, bg_distractors):
    """Test if the sampled compositions satisfy the recipe and assumptions."""
    test_disjoint_clips(compositions)
    test_unique_backgrounds(compositions)
    test_configuration_combinations(compositions)
    test_subset_counts(compositions, recipe)
