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

if __name__ == "__main__":
    home = os.path.expanduser("~")
    phase = 4*np.pi/2 #1.57, 3.14, 4.71
    D = 0.1
    run = 0
    mu = 2.0
    #ISIs_thr = np.loadtxt(home + "/Data/isochrones/ISI_thr_D{:.1f}_phi{:.2f}.dat".format(D, phase))
    #ISIs_iso = np.loadtxt(home + "/Data/isochrones/ISI_iso_D{:.1f}_phi{:.2f}.dat".format(D, phase))
    data_IPIs = np.loadtxt(home + "/CLionProjects/PhD/alif_pass_isochrone/out/IPIs_very_flat_alif_mu{:.1f}_taua2.0_delta1.0_D{:.2f}_phase{:.2f}_run{:d}.dat".format(mu, D, phase, run))
    IPIs, v_pass, a_pass = np.transpose(data_IPIs)

    data = np.loadtxt(home + "/CLionProjects/PhD/alif_pass_isochrone/out/v_a_t_very_flat_alif_mu{:.1f}_taua2.0_delta1.0_D{:.2f}_phase{:.2f}_run{:d}.dat".format(mu, D, phase, run))

    t_ss = []
    t_s = []
    v_ss = []
    v_s = []
    a_ss = []
    a_s = []

    double_check_t = []
    for set in data:
        t_s.append(set[0])
        v_s.append(set[1])
        a_s.append(set[2])
        if set[3] == True:
            t_ss.append(t_s.copy())
            v_ss.append(v_s.copy())
            a_ss.append(a_s.copy())
            t_s.clear()
            v_s.clear()
            a_s.clear()

    for x in t_ss:
        double_check_t.append(len(x)*0.001)

    isochrone = np.loadtxt(home + "/CLionProjects/PhD/alif_deterministic_isochrone/out/isochrone_very_flat_mu{:.1f}_taua2.0_delta1.0_phase{:.2f}.dat".format(mu, phase))

    for i in range(0, 100):
        fig = plt.figure(tight_layout=True)
        grid = gs.GridSpec(1, 1)
        ax = fig.add_subplot(grid[:])
        ax.set_ylim([0, 4])
        ax.set_xlim([-1, 1])
        #axins1 = inset_axes(ax, width="40%", height="30%", loc=2)
        for v_list, a_list in zip(v_ss[i:i+1], a_ss[i:i+1]):
            ax.scatter(v_list[0], a_list[0], c="C2", zorder=5)
            ax.plot(v_list, a_list)
            ax.scatter(v_list[-1], a_list[-1], c="C3", zorder=5)
        ax.axhline(0, ls="--", c="C7", zorder=1)
        ax.set_xlabel("$v$")
        ax.set_ylabel(r"$a$")

        branches = cut_isochrone_into_branches(isochrone)
        ax.scatter(v_pass, a_pass, s=3, c="C0")
        for branch in branches:
            ax.plot([x[0] for x in branch], [x[1] for x in branch], c="k")

        x, y = np.meshgrid(np.linspace(-1, 1, 40), np.linspace(0, 4, 40))
        dv = mu - x - y
        da = -y / 2.
        dx = np.sqrt(dv ** 2 + da ** 2)
        strm = ax.streamplot(x, y, dv, da, color=dx, linewidth=0.75, cmap="cividis")

        plt.savefig(home + "/Data/isochrones/test/{:d}.png".format(i))
        plt.close()
    #axins1.hist(IPIs, bins=50, density=True, alpha=0.7)