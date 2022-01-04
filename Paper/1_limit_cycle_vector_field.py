import numpy as np
import matplotlib.pyplot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib import rc
from matplotlib import rcParams
import os


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
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = 'Computer Modern Roman'
    rc('text', usetex=True)

    # Initialize Adaptive leaky IF model
    mu: float = 2.0
    tau_a: float = 2.0
    delta_a: float = 1.0

    v_res: float = 0.
    v_thr: float = 1.
    dt: float = 0.00005

    alif = adaptive_leaky_if(mu, tau_a, delta_a, v_res, v_thr, dt)
    isi_det = alif.period()
    print(isi_det)
    v_lc, a_lc, ts = alif.limit_cycle()

    fig = plt.figure(tight_layout=True, figsize=(4, 2 * 8 / 3))
    x0 = 0.20
    x1 = 0.05
    y0 = 0.1
    y1 = 0.05
    hspacer2 = 0.15
    hspacer1 = 0.05
    height_t = 0.065

    height = (1 - y0 - y1 - hspacer1 - height_t -hspacer2)/2
    width = 1 - x0 - x1
    ax3 = fig.add_axes([x0, y0, width, height])
    ax2 = fig.add_axes([x0 + 0.05, y0 + height + hspacer1, width - 0.1, height_t])
    ax1 = fig.add_axes([x0, y0 + height + hspacer1 + height_t + hspacer2, width, height])

    #gs = gs.GridSpec(5, 1)
    #ax1 = fig.add_subplot(gs[0:2])
    #ax2 = fig.add_subplot(gs[2])
    #ax3 = fig.add_subplot(gs[3:5])
    ax1.set_xlabel("$v$")
    ax1.set_xlim([-0.2, 1.1])
    ax1.set_ylabel("$a$")
    ax1.set_ylim([0, 2])
    ax1.set_yticks([0, 0.5, 1, 1.5, 2])
    ax1.plot(v_lc, a_lc, c="k", label="Limit cycle")

    ax1.axvline(1, lw=0.7, c="k", ls="--")
    ax1.axvline(0, lw=0.7, c="k", ls=":")
    ax1.plot([1, 1], [a_lc[-1], a_lc[0]], c="k")
    ax1.arrow(1, a_lc[int(len(a_lc) / 2)], dx=0, dy=0.1, shape="full", fc="k", lw=0, length_includes_head=True,
              head_length=0.1, head_width=.05)
    ax1.annotate("increase", xy=(1 + 0.05, a_lc[int(len(a_lc) / 2)]), va="center", rotation=-90)

    ax1.plot([0, 1], [a_lc[0], a_lc[0]], c="k")
    ax1.arrow(0.5, a_lc[0], dx=-0.1, dy=0.0, shape="full", fc="k", lw=0, length_includes_head=True,
              head_length=.05, head_width=.1)
    ax1.annotate("reset", xy=(0.4, a_lc[0] + 0.2), va="center", backgroundcolor="white")

    x, y = np.meshgrid(np.linspace(-0.2, 1, 10), np.linspace(0, 2, 10))
    dv = mu - x - y
    da = -y / tau_a
    dx = np.sqrt(dv ** 2 + da ** 2)
    strm = ax1.streamplot(x, y, dv, da, color="C7", arrowsize=0.75, linewidth=0.3)
    # strm = ax1.streamplot(x, y, dv, da, color=dx, arrowsize=0.75, linewidth=0.3, cmap="viridis")

    # v_st, a_st = np.transpose(stochstic_traj)
    # ax.scatter(v_st, a_st, c="C0", s=2, zorder=5)
    home = os.path.expanduser("~")
    folder = "/CLionProjects/PhD/alif_pass_isochrone/out/"
    vat_stoch = np.loadtxt(home + folder + "v_a_t_alif_mu2.0_taua2.0_delta1.0_D0.10_phase6.28_run0_fig1.dat")
    t_st, v_st, a_st, idx = np.transpose(vat_stoch)

    isi = []
    a_isi = []
    t_tmp = 0
    v_s = []
    v_ss = []
    a_s = []
    a_ss = []
    for t, v, v_before, a in zip(t_st[1:], v_st[1:], v_st[:-1], a_st[1:]):
        if abs(v - v_before) > 0.5:
            isi.append(t - t_tmp - isi_det)
            a_isi.append(a - a_lc[0])
            t_tmp = t
            v_s.append(1.)
            v_s.append(1.)
            v_s.append(0.)
            a_s.append(a_s[-1])
            a_s.append(a_s[-1] + 1.)
            a_s.append(a_s[-1])
            v_ss.append(list(v_s))
            a_ss.append(list(a_s))
            v_s.clear()
            a_s.clear()
        v_s.append(v)
        a_s.append(a)

    for i, (T, a) in enumerate(zip(isi, a_isi)):
        print(i, T, a)

    init_idx = 18
    traj_idx1 = init_idx
    traj_idx2 = init_idx +1
    traj_idx3 = init_idx +2
    ax3.plot(v_ss[traj_idx1], a_ss[traj_idx1], lw=0.7, color="#404788FF")
    ax3.plot(v_ss[traj_idx2], a_ss[traj_idx2], lw=0.7, color="#3CBB75FF")
    ax3.plot(v_ss[traj_idx3], a_ss[traj_idx3], lw=0.7, color="#FDE725FF")

    t0 = 0
    t1 = isi[traj_idx1] + isi_det
    t2 = isi[traj_idx1] + isi[traj_idx2] + 2*isi_det
    t3 = isi[traj_idx1] + isi[traj_idx2] + isi[traj_idx3] + 3*isi_det

    ax2.set_xticks([t0, t1, t2, t3])
    ax2.set_xticklabels(["$t_i$", "$t_{i+1}$", "$t_{i+2}$", "$t_{i+3}$"])
    ax2.set_xlim([t0 - 0.1, t3 + 0.6])

    ax2.plot([t0, t0], [0, 1], c="k", lw=0.7)
    ax2.plot([t0 + isi_det, t0 + isi_det], [0, 1], c="k", lw=0.7, ls=":")
    ax2.text((t0 + t1) / 2, 0.8, "$I_i$", ha="center", va="center", clip_on=False)
    ax2.arrow(t0 + 0.05, 0.5, (t1 - t0) - 0.1, 0, color="#404788FF", length_includes_head=True, head_width=0.1,
              head_length=0.2, lw=0.5, clip_on=False)
    ax2.arrow(t1 - 0.05, 0.5, -(t1 - t0) + 0.1, 0, color="#404788FF", length_includes_head=True, head_width=0.1,
              head_length=0.2, lw=0.5, clip_on=False)
    ax2.arrow(t0 + isi_det - 0.05, 0.8, -(isi_det - t1 + t0) + 0.1, 0, color="#404788FF", length_includes_head=True, head_width=0.1,
              head_length=0.2, lw=0.5, clip_on=False)
    #ax2.text(t1 + (isi_det - t1 + t0) / 2, 1.2, r"$\delta I_i$", ha="center", va="center", clip_on=False)
    ax2.text(t0 + isi_det, 1.2, r"$t_i + \langle I \rangle$", ha="center", va="center", clip_on=False)

    ax2.plot([t1, t1], [0, 1], c="k", lw=0.7)
    ax2.plot([t1 + isi_det, t1 + isi_det], [0, 1], c="k", lw=0.7, ls=":")
    ax2.text(t1 + (t2 - t1) / 2, 0.8, "$I_{i+1}$", ha="center", va="center", clip_on=False)
    ax2.arrow(t1 + 0.05, 0.5, (t2 - t1) - 0.1, 0, color="#3CBB75FF", length_includes_head=True, head_width=0.1,
              head_length=0.2, lw=0.5, clip_on=False)
    ax2.arrow(t2 - 0.05, 0.5, -(t2 - t1) + 0.1, 0, color="#3CBB75FF", length_includes_head=True, head_width=0.1,
              head_length=0.2, lw=0.5, clip_on=False)
    ax2.arrow(t1 + isi_det + 0.05, 0.8, -(isi_det - t2 + t1) - 0.1, 0, color="#3CBB75FF",
              length_includes_head=True, head_width=0.1,
              head_length=0.2, lw=0.5, clip_on=False)
    #ax2.text(t2 + (isi_det - t2 + t1) / 2, 1.2, r"$\delta I_{i+1}$", ha="center", va="center", clip_on=False)
    ax2.text(t1 + isi_det, 1.2, r"$t_{i+1} + \langle I \rangle$", ha="center", va="center", clip_on=False)


    ax2.plot([t2, t2], [0, 1], c="k", lw=0.7)
    ax2.plot([t2 + isi_det, t2 + isi_det], [0, 1], c="k", lw=0.7, ls=":")
    ax2.text(t2 + (t3 - t2) / 2, 0.8, "$I_{i+2}$", ha="center", va="center", clip_on=False)
    ax2.arrow(t2 + 0.05, 0.5, (t3 - t2) - 0.1, 0, color="#FDE725FF", length_includes_head=True, head_width=0.1,
              head_length=0.2, lw=0.5, clip_on=False)
    ax2.arrow(t3 - 0.05, 0.5, -(t3 - t2) + 0.1, 0, color="#FDE725FF", length_includes_head=True, head_width=0.1,
              head_length=0.2, lw=0.5, clip_on=False)
    ax2.arrow(t2 + isi_det + 0.05, 0.8, -(isi_det - t3 + t2) - 0.1, 0, color="#FDE725FF",
              length_includes_head=True, head_width=0.1,
              head_length=0.2, lw=0.5, clip_on=False)
    #ax2.text(t3 + (isi_det - t3 + t2) / 2, 1.2, r"$\delta I_{i+2} $", ha="center", va="center", clip_on=False)
    ax2.text(t2 + isi_det, 1.2, r"$t_{i+2} + \langle I \rangle$", ha="center", va="center", clip_on=False)


    ax2.plot([t3, t3], [0, 1], c="k", lw=0.7)

    ax3.set_xlabel("$v$")
    ax3.set_xlim([-0.2, 1.1])
    ax3.set_ylabel("$a$")
    ax3.set_ylim([0, 2])
    ax3.set_yticks([0, 0.5, 1, 1.5, 2])
    ax3.axvline(1, lw=0.7, c="k", ls="--")
    ax3.axvline(0, lw=0.7, c="k", ls=":")
    ax3.plot(v_lc, a_lc, c="k", label="Limit cycle")
    ax3.arrow(1, a_ss[traj_idx1][-1] - 0.5, dx=0, dy=0.1, shape="full", fc="#404788FF", lw=0, length_includes_head=True,
              head_length=0.1, head_width=.05)
    ax3.arrow(0.5, a_ss[traj_idx1][-1], dx=-0.1, dy=0.0, shape="full", fc="#404788FF", lw=0, length_includes_head=True,
              head_length=.05, head_width=.1)

    ax3.arrow(1, a_ss[traj_idx2][-1] - 0.5, dx=0, dy=0.1, shape="full", fc="#3CBB75FF", lw=0, length_includes_head=True,
              head_length=0.1, head_width=.05)
    ax3.arrow(0.5, a_ss[traj_idx2][-1], dx=-0.1, dy=0.0, shape="full", fc="#3CBB75FF", lw=0, length_includes_head=True,
              head_length=.05, head_width=.1)

    ax3.arrow(1, a_ss[traj_idx3][-1] - 0.5, dx=0, dy=0.1, shape="full", fc="#FDE725FF", lw=0, length_includes_head=True,
              head_length=0.1, head_width=.05)
    ax3.arrow(0.5, a_ss[traj_idx3][-1], dx=-0.1, dy=0.0, shape="full", fc="#FDE725FF", lw=0, length_includes_head=True,
              head_length=.05, head_width=.1)

    # Hide the right and top spines
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_yticks([])

    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')

    ax1.text(-0.2, 1.00, "(a)", size=12, transform=ax1.transAxes)
    ax3.text(-0.2, 1.10, "(b)", size=12, transform=ax3.transAxes)
    #ax3.text(-0.2, 1.00, "C", size=12, transform=ax3.transAxes)
    plt.savefig(home + "/Data/isochrones/fig1_c.pdf")
    plt.show()
