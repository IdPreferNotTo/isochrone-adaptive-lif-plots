import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

import styles as st
import functions as fc

class adaptive_leaky_if():
    def __init__(self, mu, tau_a, delta_a, D, v_r, v_t):
        self.mu = mu
        self.tau_a = tau_a
        self.delta_a = delta_a
        self.D = D
        self.v_r = v_r
        self.v_t = v_t

    def forward(self, v, a,  xi, dt):
        v += (self.mu - v - a) * dt + np.sqrt(2*D*dt)*xi
        a += (-a / self.tau_a) * dt
        spike = False
        if v > self.v_t:
            v = self.v_r
            a += self.delta_a
            spike = True
        return v, a, spike



if __name__ == "__main__":
    # Initialize Adaptive leaky IF model
    mu: float = 5.0
    tau_a: float = 2.0
    delta_a: float = 1.0
    D: float = 0.1

    v_res: float = 0.
    v_thr: float = 1.

    alif = adaptive_leaky_if(mu, tau_a, delta_a, D, v_res, v_thr)

    st.set_default_plot_style()

    fig = plt.figure(tight_layout=True, figsize=(4, 2))
    grids = gridspec.GridSpec(1, 1)
    ax1 = fig.add_subplot(grids[0])
    axis = [ax1]
    st.remove_top_right_axis(axis)

    dt = 0.001
    t = 0
    tis = []
    v = 0
    a = 0
    vs = []
    As = []

    while len(tis) < 10_000:
        t += dt
        xi = np.random.normal(0, 1)
        v, a, spike = alif.forward(v, a, xi, dt)
        vs.append(v)
        As.append(a)
        if spike == True:
            print(len(tis))
            tis.append(t)

    ISIs = []
    for t2, t1 in zip(tis[1:], tis[:-1]):
        ISIs.append(t2-t1)

    ks = np.arange(1, 6)
    var = fc.k_corr(ISIs, ISIs, 0)
    k_corrs = []
    for k in ks:
        kcorr = fc.k_corr(ISIs, ISIs, k)
        k_corrs.append(kcorr/var)
    ax1.scatter(ks, k_corrs, fc="w", ec=st.colors[1], s=20)
    ax1.axhline(0, ls=":", lw=1., c="C7")
    ax1.set_xlabel("$k$")
    ax1.set_ylabel(r"$\rho_k$")

    home = os.path.expanduser("~")
    plt.savefig(home + "/Desktop/Presentations/Neuro MRT Phase/Figures/6_correaltions_LIFA.png",
                transparent=True)
    plt.show()
