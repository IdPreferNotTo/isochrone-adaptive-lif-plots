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

    fig = plt.figure(tight_layout=True, figsize=(4, 3))
    grids = gridspec.GridSpec(2, 1)
    ax_dist1 = fig.add_subplot(grids[0])
    ax_dist2 = fig.add_subplot(grids[1])
    axis = [ax_dist1, ax_dist2]
    st.remove_top_right_axis(axis)
    home = os.path.expanduser("~")

    dt = 0.001
    t = 0
    tis = []
    v = 0
    a = 0
    vs = []
    As = []

    while len(tis) < 100_000:
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

    dI = 0.1
    I1 = 0.4
    I2 = 1.2
    hist1 = []
    hist2 = []
    for ISI_cond, ISI in zip(ISIs[:-1], ISIs[1:]):
        if ISI_cond < I1 + dI and ISI_cond > I1 - dI:
            hist1.append(ISI)
        if ISI_cond < I2 + dI and ISI_cond > I2 - dI:
            hist2.append(ISI)

    ax_dist1.hist(hist1, bins=20, density=True, color=st.colors[1])
    #ax_dist1.set_xlabel("$T_i$")
    ax_dist1.set_ylabel("$P(T_{i+1} | T_{i})$")
    ax_dist1.text(0.5, 0.85, r"$0.3 < T_{i} < 0.5$", transform=ax_dist1.transAxes)
    ax_dist1.set_xlim([0., 2.])

    ax_dist2.hist(hist2, bins=20, density=True, color=st.colors[1])
    ax_dist2.set_ylabel("$P(T_{i+1}| T_i)$")
    ax_dist2.text(0.5, 0.85, r"$1.1 < T_{i} < 1.3$", transform=ax_dist2.transAxes)
    ax_dist2.set_xlabel("$T_{i+1}$")
    ax_dist2.set_xlim([0., 2.])

    plt.savefig(home + "/Desktop/Presentations/Neuro MRT Phase/Figures/7_conditional_hist_LIFA.png",
                transparent=True)
    plt.show()
