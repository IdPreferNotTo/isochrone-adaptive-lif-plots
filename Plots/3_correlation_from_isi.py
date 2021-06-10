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
    phase = 3*np.pi/2 #1.57, 3.14, 4.71
    D = 0.01
    #ISIs_thr = np.loadtxt(home + "/Data/isochrones/ISI_thr_D{:.1f}_phi{:.2f}.dat".format(D, phase))
    #ISIs_iso = np.loadtxt(home + "/Data/isochrones/ISI_iso_D{:.1f}_phi{:.2f}.dat".format(D, phase))

    ISIs_thr = np.loadtxt(home + "/Data/isochrones/ISI_thr_D{:.2f}_{:.2f}.dat".format(D, phase))
    data_iso = np.loadtxt(home + "/Data/isochrones/ISI_iso_D{:.2f}_{:.2f}.dat".format(D, phase))
    ISIs_iso, v_s, a_s = np.transpose(data_iso)

    mean_ISI_thr = np.mean(ISIs_thr)
    mean_ISI_iso = np.mean(ISIs_iso)

    var_ISI_thr = np.var(ISIs_thr)
    var_ISI_iso = np.var(ISIs_iso)

    print(mean_ISI_thr, var_ISI_thr)
    print(mean_ISI_iso, var_ISI_iso)


    k_corr_ISI_thr = []
    k_corr_ISI_iso = []
    ks = np.arange(1, 6)
    for k in ks:
        k_corr_ISI_thr.append(k_corr(ISIs_thr, ISIs_thr, k)/var_ISI_thr)
        k_corr_ISI_iso.append(k_corr(ISIs_iso, ISIs_iso, k)/var_ISI_iso)


    fig = plt.figure(tight_layout=True)
    gs = gs.GridSpec(1, 1)
    ax = fig.add_subplot(gs[:])
    axins1 = inset_axes(ax, width="40%", height="30%", loc=4)
    axins2 = inset_axes(ax, width="40%", height="30%", loc=2)
    ax.scatter(ks, k_corr_ISI_thr, label=r"Threshold", zorder=2)
    ax.scatter(ks, k_corr_ISI_iso, label=r"Isochrone", zorder=2)
    ax.axhline(0, ls="--", c="C7", zorder=1)
    ax.set_ylim([-0.5, 0.5])
    ax.set_xlabel("$k$")
    ax.set_ylabel(r"$\rho_k$")
    ax.legend()

    axins1.hist(ISIs_thr, bins=20, density=True, alpha=0.7)
    axins1.hist(ISIs_iso, bins=20, density=True, alpha=0.7)

    isochrone = np.loadtxt(home + "/Data/isochrones/isochrones_file_mu5.00_{:.2f}.dat".format(phase))
    axins2.scatter(v_s, a_s, s=3, c="C0")
    axins2.plot([x[0] for x in isochrone], [x[1] for x in isochrone], c="C1")
    plt.savefig(home + "/Data/isochrones/correlations_threshold_vs_isochrone_D{:.2f}_phi{:.2f}.pdf".format(D, phase), transparent=True)

    plt.show()