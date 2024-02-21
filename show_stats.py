import os
import sys
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib



def append_dict(orig, add):
    """Return a merger of two evaluation data dicts."""
    skip_arrays = set([
        "magnitude_bin_edges",
        "uncertainty_thresholds",
        "motion_thresholds",
        ])
    histogram_arrays = set([
        "epe_vs_magnitude_sum",
        "epe_vs_magnitude_count",
        ])
    for k in orig:
        if k in skip_arrays:
            continue
        if k in histogram_arrays:
            orig[k] += add[k]
            continue
        orig[k] = np.array(list(orig[k]) + list(add[k]))


def quantity_stats(v):
    """Return several basic statistics with labels for the values in v."""
    labels = ["mean","median","min","q0.05","q0.1","q0.2",
        "q0.8","q0.9","q0.95","max"]
    stats = [np.nanmean(v),np.nanmedian(v),np.nanmin(v),
        np.nanquantile(v,0.05),np.nanquantile(v,0.1),
        np.nanquantile(v,0.2),np.nanquantile(v,0.8),
        np.nanquantile(v,0.9),np.nanquantile(v,0.95),
        np.nanmax(v)]
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


def epe_at_slow_fast(data):
    """Return the EPEs at <1 and >1 pixel ground truth flow magnitude."""
    epe_vs_magnitude_sum = data["epe_vs_magnitude_sum"]
    epe_vs_magnitude_count = data["epe_vs_magnitude_count"]
    magnitude_bin_edges = data["magnitude_bin_edges"]
    assert(magnitude_bin_edges.size == epe_vs_magnitude_sum.size + 1)
    mag_bin_centers = (magnitude_bin_edges[0:-1]
        + magnitude_bin_edges[1:])*0.5
    sel_below_1 = np.nonzero(mag_bin_centers <= 1.0)
    sel_above_1 = np.nonzero(mag_bin_centers > 1.0)
    epe_gt_below_1 = (np.sum(epe_vs_magnitude_sum[sel_below_1])
        / np.sum(epe_vs_magnitude_count[sel_below_1]))
    epe_gt_above_1 = (np.sum(epe_vs_magnitude_sum[sel_above_1])
        / np.sum(epe_vs_magnitude_count[sel_above_1]))
    return epe_gt_below_1, epe_gt_above_1


def epe_at_pred_slow_fast(data):
    """Return the EPEs at <1 and >1 pixel predicted flow magnitude."""
    if "epe_pred_lt_1_sum" not in data:
        return np.nan, np.nan
    epe_pred_below_1 = (np.sum(data["epe_pred_lt_1_sum"])
        / np.sum(data["epe_pred_lt_1_count"]))
    epe_pred_above_1 = (np.sum(data["epe_pred_gt_1_sum"])
        / np.sum(data["epe_pred_gt_1_count"]))
    return epe_pred_below_1, epe_pred_above_1


def mean_table(data, column_label="Value", aggregator=np.nanmean):
    """Return a table of aggregated evaluation results.
    
    Arguments:
    data -- a dict containing the evaluation series (arrays)
    column_label -- the label (title) of the results column
        (default Value)
    aggregator -- a numpy function to aggregate each evaluation array
        (default np.nanmean)
    """
    table = []
    first_row = ["Variable", column_label]
    labels = [
        "Recall",
        "Precision",
        "IoU",
        "FP @ $\|\phi _{gt}\| \leq 1$",
        "EPE",
        "EPE @ $\|\phi _{gt}\| \leq 1$",
        "EPE @ $\|\phi _{gt}\| > 1$",
        "EPE @ $\|\phi _{p}\| \leq 1$",
        "EPE @ $\|\phi _{p}\| > 1$",
        "Ground truth moving share",
        "Ground truth static share",
        ]
    epe_gt_below_1, epe_gt_above_1 = epe_at_slow_fast(data)
    epe_pred_below_1, epe_pred_above_1 = epe_at_pred_slow_fast(data)
    stats = [
        aggregator(data["recall"]),
        aggregator(data["precision"]),
        aggregator(data["iou"]),
        aggregator(data["false_motion"]),
        np.mean(data["epe"]),
        epe_gt_below_1,
        epe_gt_above_1,
        epe_pred_below_1,
        epe_pred_above_1,
        ]
    if "N_gt_all_pixels" not in data or "N_gt_moving" not in data:
        stats.append(np.nan)
    else:
        stats.append(np.mean(data["N_gt_moving"]/data["N_gt_all_pixels"]))
    if "N_gt_all_pixels" not in data or "N_gt_static" not in data:
        stats.append(np.nan)
    else:
        stats.append(np.mean(data["N_gt_static"]/data["N_gt_all_pixels"]))
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


def show_plots(data, save_plot_data=False):
    min_samples = 10000
    
    magnitude_bin_edges = data["magnitude_bin_edges"]
    magnitude_bin_centers = (magnitude_bin_edges[0:-1]
        + magnitude_bin_edges[1:])*0.5
    s = data["epe_vs_magnitude_sum"]
    cnt = data["epe_vs_magnitude_count"]
    sel_magnitude = cnt > min_samples
    epe_vs_magnitude_mean = s[sel_magnitude]/cnt[sel_magnitude]
    magnitude_bin_centers = magnitude_bin_centers[sel_magnitude]
    
    if save_plot_data:
        np.savez_compressed(
            "epe_vs_mag_plot_latest.npz",
            magnitude=magnitude_bin_centers,
            epe=epe_vs_magnitude_mean)
    
    plt.scatter(
        magnitude_bin_centers, epe_vs_magnitude_mean, marker='.')
    plt.xlabel("True flow magnitude (pixels)")
    plt.ylabel("Mean endpoint error (pixels)")
    plt.gcf().set_size_inches(5, 4, forward=True)
    plt.tight_layout()
    plt.show()


def motion_iou_precision_recall_plot(
        m_iou, m_precision, m_recall, motion_thresholds, title):
    """Plot IoU, precision and recall as functions of motion thresholds."""
    assert(m_iou.size == motion_thresholds.size)
    assert(m_precision.size == motion_thresholds.size)
    assert(m_recall.size == motion_thresholds.size)
    colors = ['C0', 'C1', 'C2']
    labels = ['IoU', 'Precision', 'Recall']
    variables = [m_iou, m_precision, m_recall]
    if np.isinf(motion_thresholds[-1]):
        for c,l,v in zip(colors, labels, variables):
            plt.plot(motion_thresholds[:-1], v[:-1], color=c, label=l)
            plt.axline(y=v[-1], color=c, linestyle="--")
    else:
        for c,l,v in zip(colors, labels, variables):
            plt.plot(motion_thresholds, v, color=c, label=l)
    plt.xlabel("Flow magnitude threshold (pixels)")
    plt.ylabel("Mean IoU / precision / recall")
    plt.legend()
    plt.title(title)
    plt.gcf().set_size_inches(5, 4, forward=True)
    plt.tight_layout()
    plt.show()


def uncertainty_iou_precision_recall_plot(
        mu_iou, mu_precision, mu_recall,
        motion_thresholds, uncertainty_thresholds, mt, title):
    mt_idx = np.nonzero(motion_thresholds == mt)[0][0]
    colors = ['C0', 'C1', 'C2']
    labels = ['IoU', 'Precision', 'Recall']
    variables = [
        mu_iou[mt_idx,:],
        mu_precision[mt_idx,:],
        mu_recall[mt_idx,:],
        ]
    if np.isinf(uncertainty_thresholds[-1]):
        for c,l,v in zip(colors, labels, variables):
            plt.plot(
                uncertainty_thresholds[:-1], v[:-1], color=c, label=l)
            plt.axhline(y=v[-1], color=c, linestyle="--")
    else:
        for c,l,v in zip(colors, labels, variables):
            plt.plot(uncertainty_thresholds, v, color=c, label=l)
    plt.xlabel("Flow uncertainty threshold (pixels)")
    plt.ylabel("Mean IoU / precision / recall")
    plt.legend()
    plt.title(title)
    plt.gcf().set_size_inches(5, 4, forward=True)
    plt.tight_layout()
    plt.show()


def motion_uncertainty_2d(
        motion_thresholds, uncertainty_thresholds, variable, title):
    """Show a 2D plot of a variable at motion and uncertainty thresholds."""
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(6, 3, forward=True)
    ax = [ax]
    x = np.arange(-0.5, motion_thresholds.size, 1)
    y = np.arange(-0.5, uncertainty_thresholds.size, 1)
    Z = variable
    #pcm = ax[0].pcolormesh(Z.T, vmin=0, vmax=1)
    pcm = ax[0].imshow(Z.T, vmin=0, vmax=1)
        #motion_thresholds, uncertainty_thresholds, Z.T,
        #norm=matplotlib.colors.Normalize(vmin=Z.min(), vmax=Z.max()))
    ax[0].set_xticks(np.arange(0, motion_thresholds.size))
    ax[0].set_xticklabels(motion_thresholds)
    ax[0].set_yticks(np.arange(0, uncertainty_thresholds.size))
    ax[0].set_yticklabels(uncertainty_thresholds)
    ax[0].set_xlabel("Flow magnitude threshold (pixels)")
    ax[0].set_ylabel("Uncertainty threshold")
    fig.colorbar(pcm, ax=ax[0])
    #plt.gcf().set_size_inches(5, 4, forward=True)
    plt.tight_layout()
    plt.title(title)
    plt.show()


def uncertainty_precision_recall_plot(
        mu_precision, mu_recall, motion_thresholds,
        uncertainty_thresholds, title):
    """Plot precision vs. recall curves across uncertainty thresholds.
    
    Show such curves for a selection of motion thresholds (mt_selected).
    """
    mt_selected = [1.5, 2.5, 3.0, 4.0, 5.0, 6.0]
    colors = ['C{:d}'.format(i) for i in range(len(mt_selected))]
    labels = ["MT {:.1f} px".format(mt) for mt in mt_selected]
    for c,l,mt in zip(colors, labels, mt_selected):
        idx = np.nonzero(motion_thresholds == mt)[0][0]
        recall = mu_recall[idx,:]
        precision = mu_precision[idx,:]
        plt.plot(recall, precision, color=c, label=l)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.title(title)
    plt.gcf().set_size_inches(5, 4, forward=True)
    plt.tight_layout()
    plt.show()


def full_precision_recall_plot(mu_precision, mu_recall, title):
    """Plot all available precision-recall options."""
    recall = mu_recall.flatten()
    precision = mu_precision.flatten()
    idx = np.argsort(recall)
    plt.plot(recall[idx], precision[idx])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.gcf().set_size_inches(5, 4, forward=True)
    plt.tight_layout()
    plt.show()


def results_at_best_iou_table(
        data, column_label="Value", aggregator=np.nanmean):
    """Print the results at the optimal motion and uncertainty thresholds."""
    if data["mu_iou"].size == 0:
        return None
    mu_iou = aggregator(data["mu_iou"], axis=0)
    mu_precision = aggregator(data["mu_precision"], axis=0)
    mu_recall = aggregator(data["mu_recall"], axis=0)
        
    motion_thresholds = data["motion_thresholds"]
    uncertainty_thresholds = data["uncertainty_thresholds"]
    
    i_best = np.argmax(mu_iou)
    res = np.unravel_index(i_best, mu_iou.shape)
    if len(res) == 2:
        idx_motion, idx_uncertainty = res
    else:
        idx_motion = res
        idx_uncertainty = None
    mt = motion_thresholds[idx_motion]
    if idx_uncertainty is not None:
        ut = uncertainty_thresholds[idx_uncertainty]
    else:
        ut = None
    
    table = []
    first_row = ["Variable", column_label]
    labels = [
        "Recall",
        "Precision",
        "IoU",
        "$\\tau_{motion}^*$",
        ]
    stats = [
        mu_recall.flat[i_best],
        mu_precision.flat[i_best],
        mu_iou.flat[i_best],
        mt,
        ]
    if ut is not None:
        labels.append("$\\sigma_t$")
        stats.append(ut)
    table.append(labels)
    table.append(stats)
    table = np.array(table,dtype=object)
    table = table.T
    table = table.tolist()
    table = [first_row] + table
    return table


def motion_uncertainty_iou_plot(data):
    """Plot performance at multiple motion or uncertainty thresholds."""
    if data["mu_iou"].size == 0:
        return
    # frames, motion_thresholds, uncertainty_thresholds
    #print(data["mu_iou"].shape)
    mu_iou = np.nanmean(data["mu_iou"], axis=0)
    mu_precision = np.nanmean(data["mu_precision"], axis=0)
    mu_recall = np.nanmean(data["mu_recall"], axis=0)
    motion_thresholds = data["motion_thresholds"]
    uncertainty_thresholds = data["uncertainty_thresholds"]
    
    tab = results_at_best_iou_table(data)
    print("")
    print("==== Mean ====")
    latex_stats_table(tab)
    
    tab_median = results_at_best_iou_table(data, aggregator=np.nanmedian)
    print("")
    print("==== Median ====")
    latex_stats_table(tab_median)
    
    full_precision_recall_plot(mu_precision, mu_recall, "All thresholds")
    
    if len(mu_iou.shape) < 2:
        motion_iou_precision_recall_plot(
            mu_iou, mu_precision, mu_recall, motion_thresholds,
            "MaskFlownet")
        return
    
    title = "MfnProb uncertainty threshold {}".format(
        uncertainty_thresholds[-1])
    motion_iou_precision_recall_plot(
        mu_iou[...,-1], mu_precision[...,-1], mu_recall[...,-1],
        motion_thresholds, title)
    
    title = "Uncertainty threshold range [{:.1f}, {:.1f}]".format(
        uncertainty_thresholds[0], uncertainty_thresholds[-1])
    uncertainty_precision_recall_plot(
        mu_precision, mu_recall, motion_thresholds,
        uncertainty_thresholds, title)
    
    mt = 4.0
    title = "Motion threshold {:.1f} pixels".format(mt)
    uncertainty_iou_precision_recall_plot(
        mu_iou, mu_precision, mu_recall,
        motion_thresholds, uncertainty_thresholds, mt, title)
    
    motion_uncertainty_2d(
        motion_thresholds, uncertainty_thresholds, mu_iou, "IoU")
    motion_uncertainty_2d(
        motion_thresholds, uncertainty_thresholds, mu_precision,
        "Precision")
    motion_uncertainty_2d(
        motion_thresholds, uncertainty_thresholds, mu_recall, "Recall")
    
    best_iou = np.amax(mu_iou, axis=1)
    best_idx = []
    for i in range(best_iou.size):
        idx = np.amax(np.nonzero(mu_iou[i,:] == best_iou[i]))
        best_idx.append(int(idx))
    best_uncertainty = uncertainty_thresholds[best_idx]
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(6, 3, forward=True)
    ax[0].plot(motion_thresholds, best_iou)
    ax[0].set_xlabel("Flow magnitude threshold (pixels)")
    ax[0].set_ylabel("The best mean IoU")
    ax[1].plot(motion_thresholds, best_uncertainty)
    ax[1].set_xlabel("Flow magnitude threshold (pixels)")
    ax[1].set_ylabel("Optimal uncertainty threshold")
    plt.tight_layout()
    plt.show()


def load_data(path_list):
    """Load evaluation data from npz files and return them in a dict."""
    data = None
    for path in path_list:
        if data is None:
            tmp = np.load(path)
            data = {}
            for k in tmp:
                data[k] = tmp[k]
            continue
        additional = np.load(path)
        append_dict(data, additional)
    return data


def main():
    if len(sys.argv) < 2:
        print("Usage: python show_stats.py /path/to/stats/file.npz ...")
        return
    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        folder = sys.argv[1]
        path_list = []
        for fn in os.listdir(folder):
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
    print("")
    means = mean_table(data)
    latex_stats_table(means)
    motion_uncertainty_iou_plot(data)
    show_plots(data)


if __name__ == "__main__":
    main()
