import os
import sys
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib



def quantity_stats(v):
    """Return several basic statistics with labels for the values in v."""
    labels = [
        "mean","median","MAD",
        "min","q0.05","q0.1","q0.2","q0.8","q0.9","q0.95","max",
        ]
    nanmad = np.nanmean(np.abs(v-np.nanmedian(v)))
    stats = [
        np.nanmean(v),np.nanmedian(v),nanmad,
        np.nanmin(v),
        np.nanquantile(v,0.05),np.nanquantile(v,0.1),
        np.nanquantile(v,0.2),np.nanquantile(v,0.8),
        np.nanquantile(v,0.9),np.nanquantile(v,0.95),
        np.nanmax(v),
        ]
    return labels, stats


def stats_table(data):
    """Return a stats table given a sequence of IoUs (data["iou"])."""
    table = []
    labels, stats = quantity_stats(data["iou"])
    first_row = ["Variable","IoU"]
    table.append(labels)
    table.append(stats)
    table = np.array(table,dtype=object)
    table = table.T
    table = table.tolist()
    table = [first_row] + table
    return table


def latex_stats_table(table):
    """Print a LaTeX table given an array of tabular data."""
    for i,row in enumerate(table):
        if i == 0:
            print(" & ".join(row) + "\\\\")
            print("\\midrule")
            continue
        numbers = ["{:.4f}".format(n) for n in row[1:]]
        s = row[0] + " & " + " & ".join(numbers) + "\\\\"
        print(s)


def load_data(path_list):
    data = {"iou": []}
    for path in path_list:
        tmp = np.load(path)
        data["iou"].append(np.nanmean(tmp["iou"]))
    return data


def main():
    if len(sys.argv) < 2:
        print("Usage: python show_stats_per_clip.py "
            "/path/to/stats/file.npz ...")
        print("")
        print("Compute the mean IoU of each given clip and show the statistics"
            "of these mean IoUs.")
        return
    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        folder = sys.argv[1]
        path_list = []
        for fn in sorted(os.listdir(folder)):
            path = os.path.join(folder, fn)
            if not os.path.isfile(path):
                continue
            path_list.append(path)
    else:
        path_list = sys.argv[1:]
    data = load_data(path_list)
    assert(data is not None)
    table = stats_table(data)
    latex_stats_table(table)
    data["iou"] = np.array(data["iou"])
    i_max = np.argmax(data["iou"])
    i_min = np.argmin(data["iou"])
    #print(data["iou"][i_min], data["iou"][i_max])
    print("Min. IoU at {}.".format(path_list[i_min]))
    print("Max. IoU at {}.".format(path_list[i_max]))
    np.set_printoptions(precision=4)
    print("sorted IoUs:")
    print(data["iou"][np.argsort(data["iou"])])
    print("clip indices sorted by IoUs:")
    print(np.argsort(data["iou"]))
    print("Top five:", np.argsort(data["iou"])[-5:])
    print("Bottom five:", np.argsort(data["iou"])[0:5])


if __name__ == "__main__":
    main()
