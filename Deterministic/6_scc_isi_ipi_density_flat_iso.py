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
    D = 0.1
    run = 0
    mu = 2.0
    #ISIs_thr = np.loadtxt(home + "/Data/isochrones/ISI_thr_D{:.1f}_phi{:.2f}.dat".format(D, phase))
    #ISIs_iso = np.loadtxt(home + "/Data/isochrones/ISI_iso_D{:.1f}_phi{:.2f}.dat".format(D, phase))

    ISIs = np.loadtxt(home + "/CLionProjects/PhD/alif_pass_isochrone/out/ISIs_alif_mu{:.1f}_taua2.0_delta1.0_D{:.2f}_phase{:.2f}_run{:d}.dat".format(mu, D, phase, run))
    data_IPIs = np.loadtxt(home + "/CLionProjects/PhD/alif_pass_isochrone/out/IPIs_alif_mu{:.1f}_taua2.0_delta1.0_D{:.2f}_phase{:.2f}_run{:d}.dat".format(mu, D, phase, run))
    IPIs, v_pass, a_pass = np.transpose(data_IPIs)

    print(min(IPIs), max(IPIs))
    print(np.where(IPIs==min(IPIs)))
    print(len(IPIs))
    mean_ISI = np.mean(ISIs)
    mean_IPI = np.mean(IPIs)

    var_ISI = np.var(ISIs)
    var_IPI = np.var(IPIs)

    CV_ISI = np.sqrt(var_ISI)/mean_ISI
    CV_IPI = np.sqrt(var_IPI)/mean_IPI

    print(mean_ISI, var_ISI)
    print(mean_IPI, var_IPI)


    k_corr_ISI = []
    k_corr_IPI = []
    ks = np.arange(1, 6)
    for k in ks:
        k_corr_ISI.append(k_corr(ISIs, ISIs, k)/var_ISI)
        k_corr_IPI.append(k_corr(IPIs, IPIs, k)/var_IPI)


    k_corr_stds = []
    for k in ks:
        k_corrs = []
        for IPI_chunk in chunks(IPIs, int(len(IPIs)/10.)):
            var_IPI = np.var(IPI_chunk)
            k_corrs.append(k_corr(IPI_chunk, IPI_chunk, k)/var_IPI)
        k_corr_std = np.std(k_corrs)
        k_corr_stds.append(k_corr_std/np.sqrt(10))

    fig = plt.figure(tight_layout=True)
    gs = gs.GridSpec(1, 1)
    ax = fig.add_subplot(gs[:])
    axins1 = inset_axes(ax, width="40%", height="30%", loc=4, bbox_to_anchor=(0.0,0.1,1,1), bbox_transform=ax.transAxes)
    axins2 = inset_axes(ax, width="40%", height="30%", loc=2, bbox_to_anchor=(0.05,0,1,1), bbox_transform=ax.transAxes)
    ISI_label =  "Threshold \n ISI={:.2f} \n CV={:.2f}".format(mean_ISI, CV_ISI) + "\n" + r"$C_V^2(1+2\sum\rho_k$) = {:.2e}".format(np.power(CV_ISI,2)*(1. + 2*sum(k_corr_ISI)))
    ax.scatter(ks, k_corr_ISI, color="#273e6e", label=ISI_label, zorder=2)
    #ax.scatter(ks, k_corr_IPI, label=r"Isochrone", zorder=2)
    IPI_label =  "Isochrone \n IPI={:.2f} \n CV={:.2f}".format(mean_IPI, CV_IPI) + "\n" + r"$C_V^2(1+2\sum\rho_k$) = {:.2e}".format(np.power(CV_IPI,2)*(1. + 2*sum(k_corr_IPI)))
    ax.errorbar(ks, k_corr_IPI, yerr=k_corr_stds, fmt='o', color="#d4c15f", label=IPI_label)
    ax.axhline(0, ls="--", c="C7", zorder=1)
    ax.set_ylim([-0.5, 0.5])
    ax.set_xlabel("$k$")
    ax.set_ylabel(r"$\rho_k$")
    ax.set_xticks([1,2,3,4,5])
    ax.legend()

    axins1.hist(ISIs, color="#273e6e", bins=50, density=True, alpha=0.7)
    axins1.hist(IPIs, color="#d4c15f", bins=50, density=True, alpha=0.7)
    axins1.set_xlabel("ISI, IPI")
    axins1.set_ylabel("$P$(ISI), $P$3(IPI)")

    isochrone = np.loadtxt(home + "/CLionProjects/PhD/alif_deterministic_isochrone/out/isochrone_mu{:.1f}_taua2.0_delta1.0_phase{:.2f}.dat".format(mu, phase))
    branches = cut_isochrone_into_branches(isochrone)
    axins2.scatter(v_pass[:100], a_pass[:100], s=3, c="C3")
    for branch in branches:
        axins2.plot([x[0] for x in branch], [x[1] for x in branch], c="k", ls=":")

    x, y = np.meshgrid(np.linspace(-2, 1, 10), np.linspace(0, 5, 10))
    dv = mu - x - y
    da = -y/2.
    dx = np.sqrt(dv**2 + da**2)
    strm = axins2.streamplot(x, y, dv, da, color=dx, density=0.75, linewidth=0.5, arrowsize=0.5, cmap="cividis")

    axins2.set_xlim([-2, 1])
    axins2.set_ylim([0, 5])
    axins2.set_xlabel("v")
    axins2.set_ylabel("a")


    plt.savefig(home + "/Data/isochrones/correlations_threshold_vs_very_flat_isochrone_mu{:.1f}_D{:.2f}_phi{:.2f}.pdf".format(mu, D, phase), transparent=True)

    plt.show()