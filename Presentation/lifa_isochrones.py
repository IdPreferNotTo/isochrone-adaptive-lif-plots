import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import os

import styles as st

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

if __name__ == "__main__":
    # Initialize Adaptive leaky IF model
    mu: float = 5.0
    tau_a: float = 2.0
    delta_a: float = 1.0

    v_res: float = 0.
    v_thr: float = 1.
    dt: float = 0.00005

    alif = adaptive_leaky_if(mu, tau_a, delta_a, v_res, v_thr, dt)
    isi = alif.period()
    print(isi)
    v_lc, a_lc, ts = alif.limit_cycle()

    st.set_default_plot_style()
    fig = plt.figure(tight_layout=True, figsize=(4, 3))
    gs = gs.GridSpec(1, 1)
    ax = fig.add_subplot(gs[:])
    ax.set_xlabel("$v$")
    ax.set_ylabel("$a$")
    ax.axhline(a_lc[0], c="k", ls=":")
    ax.axhline(a_lc[-1], c ="k", ls=":")
    ax.arrow(1.05, a_lc[-1], 0, a_lc[0] - a_lc[-1], fc="k", length_includes_head=True, head_width=0.03,
             head_length=0.1, lw=0.5, clip_on=False)
    ax.arrow(1.05, a_lc[0], 0, a_lc[-1] - a_lc[0], fc="k", length_includes_head=True, head_width=0.03,
             head_length=0.1, lw=0.5, clip_on=False)
    ax.text(1.1, (a_lc[0] + a_lc[-1])/2, "$\Delta$", va="center", ha="center")

    home = os.path.expanduser("~")
    phase1: float = 1 * np.pi / 2
    phase2: float = 2 * np.pi / 2
    phase3: float = 3 * np.pi / 2
    phase4: float = 4 * np.pi / 2
    phases = [phase1, phase2, phase3, phase4]
    isochrone1 = np.loadtxt(
        home + f"/CLionProjects/PhD/alif_deterministic_isochrone/out/isochrone_mu{mu:.1f}_taua2.0_delta1.0_phase{phase1:.2f}.dat")
    isochrone2 = np.loadtxt(
        home + f"/CLionProjects/PhD/alif_deterministic_isochrone/out/isochrone_mu{mu:.1f}_taua2.0_delta1.0_phase{phase2:.2f}.dat")
    isochrone3 = np.loadtxt(
        home + f"/CLionProjects/PhD/alif_deterministic_isochrone/out/isochrone_mu{mu:.1f}_taua2.0_delta1.0_phase{phase3:.2f}.dat")
    isochrone4 = np.loadtxt(
        home + f"/CLionProjects/PhD/alif_deterministic_isochrone/out/isochrone_mu{mu:.1f}_taua2.0_delta1.0_phase{phase4:.2f}.dat")
    isochrones = [isochrone1, isochrone2, isochrone3, isochrone4]

    for n, (phase, isochrone) in enumerate(zip(phases, isochrones)):
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
        ax.plot(v_isos[0], a_isos[0], c=st.colors[n], label="$\phi = {:.2f}$".format(phase))
        for v_iso, a_iso in zip(v_isos[1:], a_isos[1:]):
            ax.plot(v_iso, a_iso, c=st.colors[n])

    ax.plot(v_lc, a_lc, c="k", label="LC")
    ax.axvline(1, c="k")
    ax.axvline(0, c="k", ls="--")
    #ax.legend(fancybox=False, framealpha=1., loc=3, ncol=2)

    #v_st, a_st = np.transpose(stochstic_traj)
    #ax.scatter(v_st, a_st, c="C0", s=2, zorder=5)



    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([1, 4])
    ax.set_yticks([1, 2, 3, 4])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["$v_R$", "$v_T$"])
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


    #ax.grid()
    plt.savefig(home + "/Desktop/Presentations/Neuro MRT Phase/Figures/7_isochrones_LIFA.png",
                transparent=True)
    plt.show()