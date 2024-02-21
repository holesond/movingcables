import os
import sys
import json

import numpy as np

import show_stats



def main(attribute_name, compositions, results_dir):
    """Show dataset split performance separately for each attribute value.
    
    Arguments:
    attribute_name -- name of the dataset attribute to separate the results
    compositions -- a data structure describing all clips in the dataset split
    results_dir -- folder containing the npz result files of the dataset split
    """
    attribute_values = set()
    for clip in compositions:
        attribute_values.add(clip[attribute_name])
    attribute_values = sorted(list(attribute_values))
    for i in range(len(attribute_values)):
        if attribute_values[i].startswith("plain_"):
            attribute_values[i] = "plain_"
    attribute_values = sorted(list(set(attribute_values)))
    table = None
    table_best = None
    table_best_median = None
    for v in attribute_values:
        result_paths = []
        for clip in compositions:
            if not clip[attribute_name].startswith(v):
                continue
            path = clip["name"]+".npz"
            path = os.path.join(results_dir, path)
            result_paths.append(path)
        data = show_stats.load_data(result_paths)
        tab = show_stats.mean_table(data, column_label=v)
        tab = np.array(tab, dtype=object)
        
        tab_best = show_stats.results_at_best_iou_table(
            data, column_label=v)
        if tab_best is not None:
            tab_best = np.array(tab_best, dtype=object)
            tab_best_med = show_stats.results_at_best_iou_table(
                data, column_label=v, aggregator=np.nanmedian)
            tab_best_med = np.array(tab_best_med, dtype=object)
        if table is None:
            table = tab
            if tab_best is not None:
                table_best = tab_best
                table_best_median = tab_best_med
            continue
        table = np.concatenate((table, tab[:,1,None]), axis=1)
        if tab_best is not None:
            table_best = np.concatenate(
                (table_best, tab_best[:,1,None]), axis=1)
            table_best_median = np.concatenate(
                (table_best_median, tab_best_med[:,1,None]), axis=1)
    print("==== Means ====")
    assert(table is not None)
    show_stats.latex_stats_table(table)
    if table_best is not None:
        show_stats.latex_stats_table(table_best)
        assert(table_best_median is not None)
        print("\n==== Medians ====")
        show_stats.latex_stats_table(table_best_median)



def cmd_main():
    if len(sys.argv) != 5:
        print("Usage: python stats_by_attribute.py "
            "split_name attr_name data_config.json results/dir")
        return
    split_name = sys.argv[1]
    attribute_name = sys.argv[2]
    fn_json = os.path.expanduser(sys.argv[3])
    results_dir = os.path.expanduser(sys.argv[4])
    with open(fn_json) as f:
        compositions = json.load(f)
    main(attribute_name, compositions[split_name], results_dir)

if __name__ == "__main__":
    cmd_main()
