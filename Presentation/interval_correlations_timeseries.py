import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

import styles as st

if __name__ == "__main__":
    st.set_default_plot_style()

    fig = plt.figure(tight_layout=True, figsize=(4, 2))
    grids = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(grids[0])
    axis = [ax]
    ax.spines['left'].set_visible(False)
    st.remove_top_right_axis(axis)
    home = os.path.expanduser("~")

    tis = []
    for i in range(100_000):
        var = np.random.normal(0, 0.2)
        ti = i + var
        tis.append(ti)

    ISIs = []
    for t2, t1 in zip(tis[1:], tis[:-1]):
        ISIs.append(t2-t1)

    for n, (ti, ti_next) in enumerate(zip(tis[:9], tis[1:10])):
        ax.axvline(ti, color=st.colors[1])
        ax.text(ti, 1.1, f"$t_{n}$")
        dt = ti_next - ti
        ax.arrow(ti  + 0.05 * dt, 0.5 , 0.9 * dt, 0, fc="k", length_includes_head=True, head_width=0.05,
                  head_length = 0.1, lw=0.5, clip_on=False)
        ax.arrow(ti_next - 0.05 * dt, 0.5, -0.9 * dt, 0, fc="k", length_includes_head=True, head_width=0.05,
                  head_length=0.1, lw=0.5, clip_on=False)
        ax.text((ti + ti_next)/2, 0.55, f"$T_{n}$", ha="center")

    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ax.set_xlabel("$t$")
    ax.set_yticks([])
    ax.set_ylim([0, 1.])
    plt.savefig(home + "/Desktop/Presentations/Neuro MRT Phase/Figures/2_timeseries_correaltions.png",
                transparent=True)
    plt.show()