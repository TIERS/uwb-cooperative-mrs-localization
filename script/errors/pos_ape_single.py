import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tikzplotlib as tikz
import seaborn as sns
import os
import matplotlib.pyplot as plt
# sns.set(style="darkgrid")
# sns.set(style="whitegrid")
# sns.set_style("white")
sns.set(style="whitegrid",font_scale=1)
import matplotlib.collections as clt
import itertools


import ptitprince as pt


folder_name = "/home/xianjia/Workspace/temp/uwb_ranging_refine_with_spatial_detection/results/single/ape/"

## get the file name of all above folders
files = [
    # "ape_tri_t0.csv",
    # "ape_tri_t1.csv",   
    # "ape_tri_t2.csv",
    # "ape_tri_t3.csv",
    # "ape_triu_t0.csv",
    # "ape_triu_t1.csv",   
    # "ape_triu_t2.csv",
    # "ape_triu_t3.csv",
    "ape_u_t0.csv",
    # "ape_u_t1.csv",   
    # "ape_u_t2.csv",
    # "ape_u_t3.csv",
    "ape_ut_t0.csv",
    # "ape_ut_t1.csv",
    # "ape_ut_t2.csv",
    # "ape_ut_t3.csv",
    "ape_utv_t0.csv",
    # "ape_utv_t1.csv",
    # "ape_utv_t2.csv",
    # "ape_utv_t3.csv",
]

pos_list = []
for inx, fs in enumerate(files):
    f_v = np.genfromtxt(os.path.join(folder_name, fs),delimiter=',')
    pos_list.append(f_v)

ape    = [pos_list[0], pos_list[1], pos_list[2]]
approaches = [
          [0]*len(pos_list[0]),
          [1]*len(pos_list[1]), 
          [2]*len(pos_list[2]), 
 ]
          
# approaches = [
#               [0]*len(pos_list[0]), 
#               [0]*len(pos_list[1]), 
#               [0]*len(pos_list[2]), 
#               [0]*len(pos_list[3]), 
#               ]       
                  

ape = list(itertools.chain.from_iterable(ape))
# robots = list(itertools.chain.from_iterable(robots))
approaches = list(itertools.chain.from_iterable(approaches))

all_gs = np.vstack([ape, approaches])
# all_gs = np.vstack([all_gs, approaches])

all_gs = np.transpose(all_gs)
print(all_gs.shape)


df = pd.DataFrame(all_gs, columns=["ape_value", "approaches" ])

# Now with the group as hue
dx = "approaches"; dy = "ape_value"; pal = "Set2"; sigma = .6; ort = "v";cut=0.0
# f, ax = plt.subplots(figsize=(8,5))
f, ax = plt.subplots()

ax=pt.RainCloud(x = dx, y = dy, data = df, palette = pal, orient=ort, bw = sigma, cut=cut, width_viol = .6,
                ax = ax, alpha = .5, jitter=0.0, dodge = True, move = .25) #pointplot = True,


plt.title("State Estimation Error of Triangulations & Particle filters\n on UWB Ranges Fused With Spatial Detection")
# plt.title("Figure P20\n  Repeated Measures Data - Example 2")
# if savefigs:
#     plt.savefig('../figs/tutorial_python/figureP20.png', bbox_inches='tight')

# plt.xlim(-1, 3)
# plt.show()

# # Major ticks every 20, minor ticks every 5
major_ticks = np.linspace(0, 0.6, 4)
minor_ticks = np.linspace(0, 0.6, 7)
print(major_ticks)

# ax.set_xticks(major_ticks)
# ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

ax.grid(which='minor', alpha=0.4)
ax.grid(which='major', alpha=0.7)

plt.grid(axis = 'y', color = 'green', linestyle = '--', linewidth = 0.5)
plt.xticks(np.array([0,1,2]), np.array(["approaches_1", "approaches_2", "approaches_3"]))

FILENAME = "raincloud_plot_ape" 
plt.savefig('./results/single/figs/img/{}.png'.format(FILENAME), bbox_inches='tight')   
tikz.save("./results/single/figs/tex/{}.tex".format(FILENAME)) 
plt.show()
# plt.yscale('log')
# plt.legend()