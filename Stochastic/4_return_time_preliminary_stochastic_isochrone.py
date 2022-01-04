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
    phase = 2*np.pi/2 #1.57, 3.14, 4.71
    D = 0.5
    run = 0
    mu = 2.0


    return_times = np.loadtxt(home + "/CLionProjects/PhD/alif_stochastic_isochrone/out/stochastic_isochrone_return_times_mu{:.1f}_taua2.0_delta1.0_phase{:.2f}.dat".format(mu, phase))


    fig = plt.figure(tight_layout=True)
    grid = gs.GridSpec(1, 1)
    ax = fig.add_subplot(grid[:])
    ax.axhline(1.51237)
    ax.plot(return_times)
    #axins1 = inset_axes(ax, width="40%", height="30%", loc=2)

    #strm = ax.streamplot(x, y, dv, da, color=dx, linewidth=0.75, cmap="cividis")

    plt.savefig(home + "/Data/isochrones/stochastic_iso_return_times_test.png")
    plt.show()
    plt.close()
    #axins1.hist(IPIs, bins=50, density=True, alpha=0.7)