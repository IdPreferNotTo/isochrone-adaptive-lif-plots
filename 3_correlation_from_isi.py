import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def k_corr(data1, data2, k):
    # Get two arbitrary data set and calculate their correlation with lag k.
    mu1 = np.mean(data1)
    mu2 = np.mean(data2)
    data1 = [x - mu1 for x in data1]
    data2 = [x - mu2 for x in data2]
    k_corr = 0
    if k == 0:
        for x, y in zip(data1, data2):
            k_corr += x * y
    else:
        for x, y in zip(data1[:-k], data2[k:]):
            k_corr += x * y
    return k_corr / len(data2[k:])

if __name__ == "__main__":
    home = os.path.expanduser("~")
    ISIs_thr = np.loadtxt(home + "/Data/isochrones/ISI_thr.dat")
    ISIs_iso = np.loadtxt(home + "/Data/isochrones/ISI_iso.dat")
    data_thr2 = np.loadtxt(home + "/Data/SCC/LIF/data/mu5.00_taua2.0_Delta2.0_taun0.000_Dn0.00e+00_Dw1.00e-03.txt")
    ISIs_thr2, x ,y, z = np.transpose(data_thr2)

    mean_ISI_thr = np.mean(ISIs_thr)
    mean_ISI_iso = np.mean(ISIs_iso)

    var_ISI_thr = np.var(ISIs_thr)
    var_ISI_iso = np.var(ISIs_iso)

    print(mean_ISI_thr, var_ISI_thr)
    print(mean_ISI_iso, var_ISI_iso, 2*0.1*mean_ISI_iso)


    k_corr_ISI_thr = []
    k_corr_ISI_iso = []
    ks = np.arange(1, 6)
    for k in ks:
        k_corr_ISI_thr.append(k_corr(ISIs_thr, ISIs_thr, k)/var_ISI_thr)
        k_corr_ISI_iso.append(k_corr(ISIs_iso, ISIs_iso, k)/var_ISI_iso)


    fig = plt.figure(tight_layout=True, figsize=(4, 6 / 2))
    gs = gs.GridSpec(1, 1)
    ax = fig.add_subplot(gs[:])
    axins = inset_axes(ax, width="50%", height="40%", loc=4)
    ax.scatter(ks, k_corr_ISI_thr, label=r"Threshold", zorder=2)
    ax.scatter(ks, k_corr_ISI_iso, label=r"Isochrone", zorder=2)
    ax.axhline(0, ls="--", c="C7", zorder=1)
    ax.set_ylim([-0.3, 0.2])
    ax.set_xlabel("$k$")
    ax.set_ylabel(r"$\rho_k$")
    ax.legend()

    axins.hist(ISIs_thr, bins=20, density=True, alpha=0.7)
    axins.hist(ISIs_iso, bins=20, density=True, alpha=0.7)
    plt.savefig(home + "/Data/isochrones/correlations_threshold_vs_isochrone.pdf", transparent=True)

    plt.show()