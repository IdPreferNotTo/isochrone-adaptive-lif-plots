import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
import os


from pylab import *

cmap = cm.get_cmap('viridis', 20)    # PiYG

for i in range(cmap.N):
    viridis = cmap(i)
    # rgb2hex accepts rgb or rgba
    print(matplotlib.colors.rgb2hex(viridis))

fig = plt.figure(tight_layout=True, figsize=(4, 8 / 3))
gs = gs.GridSpec(1, 1)
ax1 = fig.add_subplot(gs[0])
ax1.plot([0, 1], [0, 0], color="#404788FF")
ax1.plot([0, 1], [1, 1], color="#3CBB75FF")
ax1.plot([0, 1], [2, 2], color="#FDE725FF")
ax1.set_ylim([-0.5, 2.5])
ax1.set_yticks([])
ax1.set_xticks([])
plt.savefig("/home/lukas/Desktop/colors.pdf")
plt.show()