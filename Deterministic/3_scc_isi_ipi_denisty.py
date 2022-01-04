import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def cut_isochrone_into_branches(isochrone):
    branches = []
    branch = []
    for ele in isochrone:
        if ele[0] < 1:
            branch.append(ele)
        else:
            branch.append(ele)
            branches.append(branch.copy())
            branch.clear()
    return branches

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
    phase = 4*np.pi/2 #1.57, 3.14, 4.71
    D = 0.01
    run = 0
    mu = 2.0
    #ISIs_thr = np.loadtxt(home + "/Data/isochrones/ISI_thr_D{:.1f}_phi{:.2f}.dat".format(D, phase))
    #ISIs_iso = np.loadtxt(home + "/Data/isochrones/ISI_iso_D{:.1f}_phi{:.2f}.dat".format(D, phase))

    ISIs = np.loadtxt(home + "/CLionProjects/PhD/alif_pass_isochrone/out/ISIs_alif_mu{:.1f}_taua2.0_delta1.0_D{:.2f}_phase{:.2f}_run{:d}.dat".format(mu, D, phase, run))
    data_IPIs = np.loadtxt(home + "/CLionProjects/PhD/alif_pass_isochrone/out/IPIs_alif_mu{:.1f}_taua2.0_delta1.0_D{:.2f}_phase{:.2f}_run{:d}.dat".format(mu, D, phase, run))
    IPIs, v_pass, a_pass = np.transpose(data_IPIs)
    data_IPIs_flat = np.loadtxt(home + "/CLionProjects/PhD/alif_pass_isochrone/out/IPIs_very_flat_alif_mu{:.1f}_taua2.0_delta1.0_D{:.2f}_phase{:.2f}_run{:d}.dat".format(mu, D, phase, run))
    IPIs_flat, v_pass_flat, a_pass_flat = np.transpose(data_IPIs_flat)

    print(min(IPIs_flat), max(IPIs_flat))
    print(np.where(IPIs_flat==min(IPIs_flat)))
    print(len(IPIs_flat))
    mean_ISI = np.mean(ISIs)
    mean_IPI = np.mean(IPIs)
    mean_IPI_f = np.mean(IPIs_flat)

    var_ISI = np.var(ISIs)
    var_IPI = np.var(IPIs)
    var_IPI_f = np.var(IPIs_flat)

    CV_ISI = np.sqrt(var_ISI)/mean_ISI
    CV_IPI = np.sqrt(var_IPI)/mean_IPI
    CV_IPI_f = np.sqrt(var_IPI_f)/mean_IPI_f

    print(mean_ISI, CV_ISI)
    print(mean_IPI, CV_IPI)
    print(mean_IPI_f, CV_IPI_f)

    k_corr_ISI = []
    k_corr_IPI = []
    k_corr_IPI_f = []
    ks = np.arange(1, 6)
    for k in ks:
        k_corr_ISI.append(k_corr(ISIs, ISIs, k)/var_ISI)
        k_corr_IPI.append(k_corr(IPIs, IPIs, k)/var_IPI)
        k_corr_IPI_f.append(k_corr(IPIs_flat, IPIs_flat, k)/var_IPI_f)

    fig = plt.figure(tight_layout=True)
    gs = gs.GridSpec(1, 1)
    ax = fig.add_subplot(gs[:])
    axins1 = inset_axes(ax, width="40%", height="30%", loc=4, bbox_to_anchor=(0.0,0.1,1,1), bbox_transform=ax.transAxes)
    axins2 = inset_axes(ax, width="40%", height="30%", loc=2, bbox_to_anchor=(0.05,0,1,1), bbox_transform=ax.transAxes)
    ISI_label =  "Threshold \n ISI={:.2f} \n CV={:.2f}".format(mean_ISI, CV_ISI)
    ax.scatter(ks, k_corr_ISI, c="#482677FF", label=ISI_label, zorder=2)
    #ax.scatter(ks, k_corr_IPI, label=r"Isochrone", zorder=2)
    IPI_label =  "Isochrone \n IPI={:.2f} \n CV={:.2f}".format(mean_IPI, CV_IPI)
    ax.scatter(ks, k_corr_IPI, color="#29AF7FFF", label=IPI_label)
    IPI_f_label = "Horizontal \n IHI={:.2f} \n CV={:.2f}".format(mean_IPI_f, CV_IPI_f)
    ax.scatter(ks, k_corr_IPI_f, c="#FDE725FF", label=IPI_f_label, zorder=2)
    ax.axhline(0, ls="--", c="C7", zorder=1)
    ax.set_ylim([-0.5, 0.5])
    ax.set_xlabel("$k$")
    ax.set_ylabel(r"$\rho_k$")
    ax.set_xticks([1,2,3,4,5])
    ax.legend()

    axins1.hist(ISIs, color="#404788FF", bins=50, range=(1, 3), density=True, alpha=0.7)
    axins1.hist(IPIs, color="#3CBB75FF", bins=50, range=(1, 3), density=True, alpha=0.7)
    axins1.hist(IPIs_flat, color="#FDE725FF", bins=50, range=(1, 3), density=True, alpha=0.7)
    axins1.set_xlabel("ISI, IPI")
    axins1.set_ylabel("$P$(ISI), $P$3(IPI)")

    isochrone = np.loadtxt(home + "/CLionProjects/PhD/alif_deterministic_isochrone/out/isochrone_very_flat_mu{:.1f}_taua2.0_delta1.0_phase{:.2f}.dat".format(mu, phase))
    branches = cut_isochrone_into_branches(isochrone)
    axins2.scatter(v_pass_flat[:100], a_pass_flat[:100], s=3, c="C3")
    for branch in branches:
        axins2.plot([x[0] for x in branch], [x[1] for x in branch], c="k", ls=":")

    x, y = np.meshgrid(np.linspace(-2, 1, 10), np.linspace(0, 5, 10))
    dv = mu - x - y
    da = -y/2.
    dx = np.sqrt(dv**2 + da**2)
    strm = axins2.streamplot(x, y, dv, da, color=dx, density=0.75, linewidth=0.5, arrowsize=0.5, cmap="viridis")

    axins2.set_xlim([-2, 1])
    axins2.set_ylim([0, 5])
    axins2.set_xlabel("v")
    axins2.set_ylabel("a")


    plt.savefig(home + "/Data/isochrones/correlations_threshold_vs_isochrone_mu{:.1f}_D{:.2f}_phi{:.2f}.pdf".format(mu, D, phase), transparent=True)

    plt.show()