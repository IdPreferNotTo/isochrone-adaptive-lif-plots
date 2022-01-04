import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib import rc
from matplotlib import rcParams
from scipy.stats import gaussian_kde

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


def n_chunks(lst, n):
    """Yield n successive chunks from lst of size len(lst)/n"""
    s = int(len(lst)/n)
    for i in range(n):
        yield lst[i*s:(i+1)*s]


def n_sized_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == "__main__":
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = 'Computer Modern Roman'
    rc('text', usetex=True)

    fig = plt.figure(tight_layout=True, figsize=(4, 2 * 8 / 3))
    gs = gs.GridSpec(3, 1)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax1.text(-0.2, 1.00, "(a)", size=12, transform=ax1.transAxes)
    ax2.text(-0.2, 1.00, "(b)", size=12, transform=ax2.transAxes)
    ax3.text(-0.2, 1.00, "(c)", size=12, transform=ax3.transAxes)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax1.set_xlabel("$I$")
    ax1.set_ylabel("$P(I)$")
    ax2.set_xlabel("$T$")
    ax2.set_ylabel("$P(T)$")
    ax3.set_xlabel("$I$")
    ax3.set_ylabel("$P(I)$")
    ax1.set_xlim([0, 6])
    ax2.set_xlim([0, 6])
    ax3.set_xlim([0, 6])
    ax1.set_ylim([0, 1.5])
    ax2.set_ylim([0, 1.5])
    ax3.set_ylim([0, 1.5])

    mu = 2.0
    D = 0.1
    phase = 6.28
    run = 0
    home = os.path.expanduser("~")
    folder = "/CLionProjects/PhD/alif_pass_isochrone/out/"
    file_isi = "ISIs_alif_mu{:.1f}_taua2.0_delta1.0_D{:.2f}_phase{:.2f}_run{:d}.dat".format(mu, D, phase, run)
    file_ipi = "IPIs_alif_mu{:.1f}_taua2.0_delta1.0_D{:.2f}_phase{:.2f}_run{:d}.dat".format(mu, D, phase, run)
    file_ihi = "IPIs_very_flat_alif_mu{:.1f}_taua2.0_delta1.0_D{:.2f}_phase{:.2f}_run{:d}.dat".format(mu, D, phase, run)
    ISIs = np.loadtxt(home + folder + file_isi)
    data_ipi = np.loadtxt(home + folder + file_ipi)
    data_ihi = np.loadtxt(home + folder + file_ihi)
    IPIs, v_ipi, a_ipi, bool_thr_ipi = np.transpose(data_ipi)
    IHIs, v_ihi, a_ihi, bool_thr_ihi = np.transpose(data_ihi)

    print(sum(bool_thr_ihi)/len(bool_thr_ihi))
    IPIs_1 = [ipi for ipi, thr in zip(IPIs[1:], bool_thr_ipi[1:]) if thr == 1]
    IPIs_2 = [ipi for ipi, thr in zip(IPIs[1:], bool_thr_ipi[1:]) if thr == 0]
    IHIs_1 = [ihi for ihi, thr in zip(IHIs[1:], bool_thr_ihi[1:]) if thr == 1]
    IHIs_2 = [ihi for ihi, thr in zip(IHIs[1:], bool_thr_ihi[1:]) if thr == 0]
    # The first interval does not necessarily start on the threshold, isochrone, horizontal line
    ISIs = ISIs[1:]
    IPIs = IPIs[1:]
    IHIs = IHIs[1:]
    CV_ISI = np.std(ISIs)/np.mean(ISIs)
    CV_IPI = np.std(IPIs)/np.mean(IPIs)
    CV_IHI = np.std(IHIs)/np.mean(IHIs)

    k_corrs_ISI = []
    k_corrs_IPI = []
    k_corrs_IHI = []
    var_ISI = np.var(ISIs)
    var_IPI = np.var(IPIs)
    var_IHI = np.var(IHIs)
    for k in range(1, 10):
        k_corrs_ISI.append(k_corr(ISIs, ISIs, k)/var_ISI)
        k_corrs_IPI.append(k_corr(IPIs, IPIs, k)/var_IPI)
        k_corrs_IHI.append(k_corr(IHIs, IHIs, k)/var_IHI)


    n = 50
    CVs_ISI = []
    CVs_IPI = []
    CVs_IHI = []

    for ISI_chunk in n_chunks(ISIs, n):
        CVs_ISI.append(np.std(ISI_chunk)/np.mean(ISI_chunk))
    for IPI_chunk in n_chunks(IPIs, n):
        CVs_IPI.append(np.std(IPI_chunk) / np.mean(IPI_chunk))
    for IHI_chunk in n_chunks(IHIs, n):
        CVs_IHI.append(np.std(IHI_chunk) / np.mean(IHI_chunk))
    err_CV_ISI = np.std(CVs_ISI)/np.sqrt(n)
    err_CV_IPI = np.std(CVs_IPI)/np.sqrt(n)
    err_CV_IHI = np.std(CVs_IHI)/np.sqrt(n)

    err_k_corrs_ISI = []
    err_k_corrs_IPI = []
    err_k_corrs_IHI = []
    for k in range(1, 10):
        k_ISI = []
        k_IPI = []
        k_IHI = []
        for ISI_chunk in n_chunks(ISIs, n):
            k_ISI.append(k_corr(ISI_chunk, ISI_chunk, k)/np.var(ISI_chunk))
        for IPI_chunk in n_chunks(IPIs, n):
            k_IPI.append(k_corr(IPI_chunk, IPI_chunk, k)/np.var(IPI_chunk))
        for IHI_chunk in n_chunks(IHIs, n):
            k_IHI.append(k_corr(IHI_chunk, IHI_chunk, k) / np.var(IHI_chunk))
        k_ISI_std = np.std(k_ISI)
        k_IPI_std = np.std(k_IPI)
        k_IHI_std = np.std(k_IHI)
        err_k_corrs_ISI.append(k_ISI_std / np.sqrt(n))
        err_k_corrs_IPI.append(k_IPI_std / np.sqrt(n))
        err_k_corrs_IHI.append(k_IHI_std / np.sqrt(n))

    Fano_ISI = np.power(CV_ISI, 2)*(1. + 2*sum(k_corrs_ISI))
    Fano_IPI = np.power(CV_IPI, 2)*(1. + 2*sum(k_corrs_IPI))
    Fano_IHI = np.power(CV_IHI, 2)*(1. + 2*sum(k_corrs_IHI))

    print(err_CV_ISI, err_CV_IPI, err_CV_IHI)
    print(sum(err_k_corrs_ISI), sum(err_k_corrs_IPI), sum(err_k_corrs_IHI))
    err_Fano_ISI = 2*CV_ISI*(1. + 2*sum(k_corrs_ISI))*err_CV_ISI + 2*np.power(CV_ISI, 2)*sum(err_k_corrs_ISI)
    err_Fano_IPI = 2*CV_IPI*(1. + 2*sum(k_corrs_IPI))*err_CV_IPI + 2*np.power(CV_IPI, 2)*sum(err_k_corrs_IPI)
    err_Fano_IHI = 2*CV_IHI*(1. + 2*sum(k_corrs_IHI))*err_CV_IHI + 2*np.power(CV_IHI, 2)*sum(err_k_corrs_IHI)

    ISI_label = f"$C_V={CV_ISI:.2f}$" + "\n" + rf"$F(T)={Fano_ISI:.3f} \pm{err_Fano_ISI:.3f}$"
    IPI_label = f"$C_V={CV_IPI:.2f}$" + "\n" + rf"$F(T)={Fano_IPI:.3f} \pm{err_Fano_IPI:.3f}$"
    IHI_label = f"$C_V={CV_IHI:.2f}$" + "\n" + rf"$F(T)={Fano_IHI:.3f} \pm{err_Fano_IHI:.3f}$"


    ax1.hist(ISIs, color="#404788FF", bins=100, range=(0, 5), density=True, alpha=0.5)
    density = gaussian_kde(ISIs)
    ISIs_line = np.linspace(0, 5, 100)
    density.covariance_factor = lambda: .25
    density._compute_covariance()
    ax1.plot(ISIs_line, density(ISIs_line), color="#282c54", label=ISI_label)


    hist_ipi1, bins_ipi1 = np.histogram(IPIs_1, bins=100, range=(0, 5), density=True)
    widths_ipi1 = np.diff(bins_ipi1)
    hist_ipi1 *= len(IPIs_1)/(len(IPIs_1) + len(IPIs_2))
    ax2.bar(bins_ipi1[:-1], hist_ipi1, widths_ipi1, color="#3CBB75FF", alpha=0.5)
    hist_ipi2, bins_ipi2 = np.histogram(IPIs_2, bins=100, range=(0, 5), density=True)
    widths_ipi2 = np.diff(bins_ipi2)
    hist_ipi2 *= len(IPIs_2)/(len(IPIs_1) + len(IPIs_2))
    ax2.bar(bins_ipi2[:-1], hist_ipi2, widths_ipi2, color="#3CBB75FF", alpha=0.8)
    density = gaussian_kde(IPIs)
    IPIs_line = np.linspace(0, 5, 100)
    density.covariance_factor = lambda: .1
    density._compute_covariance()
    ax2.plot(IPIs_line, density(IPIs_line), color="#298151", label=IPI_label)

    hist_ihi1, bins_ihi1 = np.histogram(IHIs_1, bins=100, range=(0, 5), density=True)
    widths_ihi1 = np.diff(bins_ihi1)
    hist_ihi1 *= len(IHIs_1)/(len(IHIs_1) + len(IHIs_2))
    ax3.bar(bins_ihi1[:-1], hist_ihi1, widths_ihi1, color="#FDE725FF", alpha=0.5)
    hist_ihi2, bins_ihi2 = np.histogram(IHIs_2, bins=100, range=(0, 5), density=True)
    widths_ihi2 = np.diff(bins_ihi2)
    hist_ihi2 *= len(IHIs_2)/(len(IHIs_1) + len(IHIs_2))
    ax3.bar(bins_ihi2[:-1], hist_ihi2, widths_ihi2, color="#FDE725FF", alpha=0.8)
    density = gaussian_kde(IHIs)
    IHIs_line = np.linspace(0, 5, 100)
    density.covariance_factor = lambda: .05
    density._compute_covariance()
    ax3.plot(IHIs_line, density(IHIs_line), color="#d4be02", label=IHI_label)

    #ax2.hist(IPIs_1, color="#3CBB75FF", bins=100, range=(0, 5), density=True, alpha=0.7, label=IPI_label)
    #ax3.hist(IHIs_1, color="#FDE725FF", bins=100, range=(0, 5), density=True, alpha=0.7, label=IHI_label)
    #ax2.hist(IPIs_2, color="#3CBB75FF", bins=100, range=(0, 5), density=True, alpha=0.7, label=IPI_label)
    #ax3.hist(IHIs_2, color="#FDE725FF", bins=100, range=(0, 5), density=True, alpha=0.7)
    ax1.legend(fancybox=False, edgecolor="k", framealpha=1.)
    ax2.legend(fancybox=False, edgecolor="k", framealpha=1.)
    ax3.legend(fancybox=False, edgecolor="k", framealpha=1.)

    plt.savefig(home + "/Data/isochrones/fig4.pdf", transparent=True)
    plt.show()