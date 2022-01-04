import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

import styles as st

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
    fig = plt.figure(tight_layout=True, figsize=(4,3))
    grids = gridspec.GridSpec(3, 1)
    ax1 = fig.add_subplot(grids[0])
    ax2 = fig.add_subplot(grids[1])
    ax3 = fig.add_subplot(grids[2])
    axis = [ax1, ax2, ax3]
    st.remove_top_right_axis(axis)
    home = os.path.expanduser("~")

    dt = 0.0001
    ts = np.arange(0, 10, dt)
    tis = []
    v = 0
    a = 0
    vs = []
    As = []
    for t in ts:
        v, a, spike = alif.forward(v, a, 0, dt)

    for t in ts:
        xi = np.random.normal(0, 1)
        v, a, spike = alif.forward(v, a, xi, dt)
        vs.append(v)
        As.append(a)
        if spike == True:
            tis.append(t)

    for n, (ti, ti_next) in enumerate(zip(tis[:9], tis[1:10])):
        ax1.axvline(ti, color=st.colors[1])
        ax1.text(ti, 1.1, f"$t_{n}$")
        dt = ti_next - ti
        ax1.arrow(ti  + 0.05 * dt, 0.5 , 0.9 * dt, 0, fc="k", length_includes_head=True, head_width=0.05,
                  head_length = 0.1, lw=0.5, clip_on=False)
        ax1.arrow(ti_next - 0.05 * dt, 0.5, -0.9 * dt, 0, fc="k", length_includes_head=True, head_width=0.05,
                  head_length=0.1, lw=0.5, clip_on=False)
        ax1.text((ti + ti_next)/2, 0.55, f"$T_{n}$", ha="center")

    t_max = tis[9]
    ax1.set_xlim([0, t_max])
    ax1.set_ylim([0, 1])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['left'].set_visible(False)

    ax2.plot(ts, vs, lw=1, color=st.colors[1])
    ax2.axhline(1, lw=1, ls=":", c="C7")
    ax2.set_xlim([0, t_max])
    ax2.set_xticks([])
    ax2.set_ylabel("$v$")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["$v_R$", "$v_T$"])

    ax3.plot(ts, As, lw=1, color=st.colors[1])
    ax3.set_xlim([0, t_max])
    ax3.set_xticks([])
    ax3.set_ylabel("$a$")
    ax3.set_xlabel("$t$")
    plt.savefig(home + "/Desktop/Presentations/Neuro MRT Phase/Figures/4_timeseries_correaltions_LIFA.pdf",
                transparent=True)
    plt.show()
