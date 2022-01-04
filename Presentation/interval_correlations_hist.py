import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

import styles as st

if __name__ == "__main__":
    st.set_default_plot_style()

    fig = plt.figure(tight_layout=True, figsize=(4, 3))
    grids = gridspec.GridSpec(4, 4)
    ax_dist1 = fig.add_subplot(grids[0, 0:3])
    ax_joint = fig.add_subplot(grids[1:4, 0:3])
    ax_dist2 = fig.add_subplot(grids[1:4, 3])
    axis = [ax_dist1, ax_joint, ax_dist2]
    st.remove_top_right_axis(axis)
    home = os.path.expanduser("~")

    tis = []
    for i in range(100_000):
        var = np.random.normal(0, 0.1)
        ti = i + var
        tis.append(ti)

    ISIs = []
    for t2, t1 in zip(tis[1:], tis[:-1]):
        ISIs.append(t2-t1)

    ax_dist1.hist(ISIs, bins=100, density=True, color=st.colors[1])
    #ax_dist1.set_xlabel("$T_i$")
    ax_dist1.set_ylabel("$P(T_i)$")
    ax_dist1.set_xticks([])
    ax_dist1.set_yticks([])
    ax_dist1.set_xlim([0., 2.])

    ax_dist2.hist(ISIs, bins=100, density=True, color=st.colors[1], orientation="horizontal")
    ax_dist2.set_xlabel("$P(T_{i+1})$")
    #ax_dist2.set_ylabel("$T_{i+1}$")
    ax_dist2.set_xticks([])
    ax_dist2.set_yticks([])
    ax_dist2.set_ylim([0., 2.])

    ax_joint.scatter(ISIs[0:10_000], ISIs[1:10_001], fc="w", ec=st.colors[1], s=15)
    ax_joint.set_xlabel("$T_i$")
    ax_joint.set_ylabel("$T_{i+1}$")
    ax_joint.set_xlim([0., 2.0])
    ax_joint.set_ylim([0., 2.0])
    plt.savefig(home + "/Desktop/Presentations/Neuro MRT Phase/Figures/3_hist_correaltions.png",
                transparent=True)

    plt.show()