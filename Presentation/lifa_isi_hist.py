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
    grids = gridspec.GridSpec(4, 4)
    ax_dist1 = fig.add_subplot(grids[0, 0:3])
    ax_joint = fig.add_subplot(grids[1:4, 0:3])
    ax_dist2 = fig.add_subplot(grids[1:4, 3])
    axis = [ax_dist1, ax_joint, ax_dist2]
    st.remove_top_right_axis(axis)
    home = os.path.expanduser("~")

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

    ax_dist1.hist(ISIs, bins=50, density=True, color=st.colors[1])
    #ax_dist1.set_xlabel("$T_i$")
    ax_dist1.set_ylabel("$P(T_i)$")
    ax_dist1.set_xticks([])
    ax_dist1.set_yticks([])
    ax_dist1.set_xlim([0., 2.])

    ax_dist2.hist(ISIs, bins=50, density=True, color=st.colors[1], orientation="horizontal")
    ax_dist2.set_xlabel("$P(T_{i+1})$")
    #ax_dist2.set_ylabel("$T_{i+1}$")
    ax_dist2.set_xticks([])
    ax_dist2.set_yticks([])
    ax_dist2.set_ylim([0., 2.])

    ax_joint.scatter(ISIs[0:-1], ISIs[1:], fc="w", ec=st.colors[1], s=15)
    ax_joint.set_xlabel("$T_i$")
    ax_joint.set_ylabel("$T_{i+1}$")
    ax_joint.set_xlim([0., 2.0])
    ax_joint.set_ylim([0., 2.0])
    plt.savefig(home + "/Desktop/Presentations/Neuro MRT Phase/Figures/5_hist_correaltions_LIFA.png",
                transparent=True)
    plt.show()
