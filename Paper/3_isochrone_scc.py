import os
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import rc
from matplotlib import rcParams

class adaptive_leaky_if():
    def __init__(self, mu, tau_a, delta_a, v_r, v_t, dt):
        self.mu = mu
        self.tau_a = tau_a
        self.delta_a = delta_a
        self.v_r = v_r
        self.v_t = v_t
        self.dt = dt


    def forward(self, v, a):
        v += (self.mu - v - a) * self.dt
        a += (-a / self.tau_a) * self.dt
        spike = False
        if v > self.v_t:
            v = self.v_r
            a += self.delta_a
            spike = True
        return v, a, spike


    def forward_for_T(self, v, a, T):
        t = 0
        while (t < T):
            v, a, spike = self.forward(v, a)
            t += self.dt
        return v, a


    def backward(self, v, a):
        v -= (self.mu - v - a) * self.dt
        a -= (-a / self.tau_a) * self.dt
        reset = False
        if v < self.v_r:
            v = self.v_t
            a -= self.delta_a
            reset = True
        return v, a, reset


    def backward_for_T(self, v, a, T):
        t = T
        while (t > 0):
            v, a, spike = self.backward(v, a)
            t -= self.dt
            if a < 0 or v > v_thr:
                return None, None
        return v, a


    def limit_cycle(self):
        v: float = 0
        a: float = 0
        spikes: int = 0
        while (spikes < 100):
            v, a, spike = self.forward(v, a)
            if spike:
                spikes += 1
        t = 0
        v_s = []
        a_s = []
        t_s = []
        while (True):
            v_s.append(v)
            a_s.append(a)
            t_s.append(t)
            v, a, spike = self.forward(v, a)
            if spike:
                return [v_s, a_s, t_s]


    def period(self):
        v: float = 0
        a: float = 0
        spikes: int = 0
        while (spikes < 100):
            v, a, spike = self.forward(v, a)
            if spike:
                spikes += 1
        t = 0
        spikes = 0
        while (spikes < 100):
            v, a, spike = self.forward(v, a)
            t += dt
            if spike:
                spikes += 1
        return t / spikes


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


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
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = 'Computer Modern Roman'
    rc('text', usetex=True)
    home = os.path.expanduser("~")
    phase = 4*np.pi/2 #1.57, 3.14, 4.71
    D = 0.1
    run = 0
    mu = 2.0
    #ISIs_thr = np.loadtxt(home + "/Data/isochrones/ISI_thr_D{:.1f}_phi{:.2f}.dat".format(D, phase))
    #ISIs_iso = np.loadtxt(home + "/Data/isochrones/ISI_iso_D{:.1f}_phi{:.2f}.dat".format(D, phase))

    ISIs = np.loadtxt(home + "/CLionProjects/PhD/alif_pass_isochrone/out/ISIs_alif_mu{:.1f}_taua2.0_delta1.0_D{:.2f}_phase{:.2f}_run{:d}.dat".format(mu, D, phase, run))
    data_IPIs = np.loadtxt(home + "/CLionProjects/PhD/alif_pass_isochrone/out/IPIs_alif_mu{:.1f}_taua2.0_delta1.0_D{:.2f}_phase{:.2f}_run{:d}.dat".format(mu, D, phase, run))
    IPIs, v_pass, a_pass, nr_iso_cross = np.transpose(data_IPIs)
    data_IPIs_flat = np.loadtxt(home + "/CLionProjects/PhD/alif_pass_isochrone/out/IPIs_very_flat_alif_mu{:.1f}_taua2.0_delta1.0_D{:.2f}_phase{:.2f}_run{:d}.dat".format(mu, D, phase, run))
    IPIs_flat, v_pass_flat, a_pass_flat, nr_iso_cross_flat = np.transpose(data_IPIs_flat)

    print(min(IPIs), max(IPIs))
    print(np.where(IPIs==min(IPIs)))
    print(len(IPIs))
    mean_ISI = np.mean(ISIs)
    mean_IPI = np.mean(IPIs)
    mean_IPI_flat = np.mean(IPIs_flat)

    var_ISI = np.var(ISIs)
    var_IPI = np.var(IPIs)
    var_IPI_flat = np.var(IPIs_flat)

    CV_ISI = np.sqrt(var_ISI)/mean_ISI
    CV_IPI = np.sqrt(var_IPI)/mean_IPI
    CV_IPI_flat = np.sqrt(var_IPI_flat)/mean_IPI_flat

    k_corr_ISI = []
    k_corr_IPI = []
    k_corr_IPI_flat = []
    ks = np.arange(1, 6)
    for k in ks:
        k_corr_ISI.append(k_corr(ISIs, ISIs, k)/var_ISI)
        k_corr_IPI.append(k_corr(IPIs, IPIs, k)/var_IPI)
        k_corr_IPI_flat.append(k_corr(IPIs_flat, IPIs_flat, k)/var_IPI_flat)

    k_corr_stds = []
    for k in ks:
        k_corrs = []
        for IPI_chunk in chunks(IPIs, int(len(IPIs)/10.)):
            var_IPI = np.var(IPI_chunk)
            k_corrs.append(k_corr(IPI_chunk, IPI_chunk, k)/var_IPI)
        k_corr_std = np.std(k_corrs)
        k_corr_stds.append(k_corr_std/np.sqrt(10))

    fano_ISI = (CV_ISI**2)*(1 + 2*sum(k_corr_ISI))
    fano_IPI = (CV_IPI ** 2) * (1 + 2 * sum(k_corr_IPI))
    fano_IPI_flat = (CV_IPI_flat **2) * (1 + 2 * sum(k_corr_IPI_flat))
    print(fano_ISI, fano_IPI, fano_IPI_flat)

    fig = plt.figure(figsize=(4, 2*8/3), tight_layout=True)
    gs = gs.GridSpec(2, 1)
    ax_right = fig.add_subplot(gs[1])
    axins1 = inset_axes(ax_right, width="40%", height="30%", loc=4, bbox_to_anchor=(0.0, 0.15, 1, 1), bbox_transform=ax_right.transAxes)
    # Hide the right and top spines
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['top'].set_visible(False)

    ISI_label = "Threshold"
    ax_right.scatter(ks, k_corr_ISI, c="#404788FF", label=ISI_label, zorder=3)
    #ax.scatter(ks, k_corr_IPI, label=r"Isochrone", zorder=2)
    IPI_label = "Isochrone"
    ax_right.scatter(ks, k_corr_IPI, color="#3CBB75FF", label=IPI_label, zorder=2)
    IPI_flat_label = "Horizontal"
    ax_right.scatter(ks, k_corr_IPI_flat, color="#FDE725FF", label=IPI_flat_label, zorder=2)
    ax_right.axhline(0, ls=":", c="C7", zorder=1)
    ax_right.set_ylim([-0.5, 0.4])
    ax_right.set_xlabel("lag $k$")
    ax_right.set_ylabel(r"$\rho_k$")
    ax_right.set_xticks([1, 2, 3, 4, 5])
    ax_right.legend(fancybox=False, edgecolor="k", framealpha=1.)


    axins1.set_xlim([0,5])
    axins1.hist(ISIs, color="#404788FF", bins=50, range=(0, 5), density=True, alpha=0.7)
    axins1.hist(IPIs, color="#3CBB75FF", bins=50, range=(0, 5), density=True, alpha=0.7)
    axins1.set_xlabel("$I$, $T$")
    axins1.set_ylabel("$P(I)$, $P(T)$")


    ax_left = fig.add_subplot(gs[0])
    ax_left.spines['right'].set_visible(False)
    ax_left.spines['top'].set_visible(False)

    # Initialize Adaptive leaky IF model
    mu: float = 2.0
    tau_a: float = 2.0
    delta_a: float = 1.0

    v_res: float = 0.
    v_thr: float = 1.
    dt: float = 0.00005

    alif = adaptive_leaky_if(mu, tau_a, delta_a, v_res, v_thr, dt)
    isi = alif.period()
    print(isi)
    v_lc, a_lc, ts = alif.limit_cycle()
    print(a_lc[0], a_lc[-1])
    home = os.path.expanduser("~")
    isochrone = np.loadtxt(home + "/CLionProjects/PhD/alif_deterministic_isochrone/out/isochrone_mu{:.1f}_taua2.0_delta1.0_phase{:.2f}.dat".format(mu, phase))

    v_isos = []
    a_isos = []
    v_iso = []
    a_iso = []
    for (v1, a1) in isochrone:
        if v1 == 1:
            v_iso.append(v1)
            a_iso.append(a1)
            v_isos.append(list(v_iso))
            a_isos.append(list(a_iso))
            v_iso = []
            a_iso = []
        else:
            v_iso.append(v1)
            a_iso.append(a1)

    ax_left.set_xlabel("$v$")
    ax_left.set_ylabel("$a$")
    ax_left.plot(v_lc, a_lc, lw=1, c="k", label="Limit cycle")
    ax_left.plot(v_isos[0], a_isos[0], c="#3CBB75FF", lw=1, label="Isochrone $\phi = 2\pi $")
    for v_iso, a_iso in zip(v_isos[1:], a_isos[1:]):
        ax_left.plot(v_iso, a_iso, c="#3CBB75FF", lw=1)

    #ax_left.plot([-5, 1], [a_lc[0], a_lc[0]], color="#FDE725FF")
    #ax_left.plot([-5, 1], [a_lc[0] - delta_a, a_lc[0] - delta_a], color="#FDE725FF")
    ax_left.axvline(1, c="#404788FF")
    ax_left.axvline(0, lw=0.7, c="k", ls=":")
    ax_left.legend(fancybox=False, edgecolor="k", framealpha=1.0)
    ax_left.set_xlim([-5, 1.1])
    ax_left.set_ylim([0, 7])

    x, y = np.meshgrid(np.linspace(-5, 1, 10), np.linspace(0, 10, 10))
    dv = mu - x - y
    da = -y / tau_a
    dx = np.sqrt(dv ** 2 + da ** 2)
    strm = ax_left.streamplot(x, y, dv, da, color="C7", arrowsize=0.75, linewidth=0.3)
    #strm = ax_left.streamplot(x, y, dv, da, color=dx, arrowsize=0.75 ,linewidth=0.3, cmap="viridis")

    ax_left.text(-0.2, 1.00, "(a)", size=12, transform=ax_left.transAxes)
    ax_right.text(-0.2, 1.00, "(b)", size=12, transform=ax_right.transAxes)
    plt.savefig(home + "/Data/isochrones/fig3_b.pdf", transparent=True)
    plt.show()