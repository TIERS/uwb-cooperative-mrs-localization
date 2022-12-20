import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tikzplotlib as tikz
import pandas as pd
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
   

# names = [ name for name in mcolors.CSS4_COLORS]
# colors = []
# for i in range(12):
#     colors.append(mcolors.CSS4_COLORS[names[i]])

folder_name = "/home/xianjia/Workspace/temp/uwb_ranging_refine_with_spatial_detection/ape/"

## get the file name of all above folders
files = [
    "ape_tri_t0.csv",
    "ape_tri_t1.csv",   
    "ape_tri_t2.csv",
    "ape_tri_t3.csv",
    "ape_u_t0.csv",
    "ape_u_t1.csv",
    "ape_u_t2.csv",
    "ape_u_t3.csv",
    "ape_uv_t0.csv",
    "ape_uv_t1.csv",
    "ape_uv_t2.csv",
    "ape_uv_t3.csv",
]

pos_list = []
for inx, fs in enumerate(files):
    f_v = np.genfromtxt(os.path.join(folder_name, fs),delimiter=',')
    pos_list.append(f_v)

# print(f"pos list: {len(pos_list)}")
# linestyle_ls = ['*', 'v', '^', 's', 'o', 'D', 'H', '8', 'p', '*', 'v', '^', 's', 'o', 'D', 'H', '8', 'p']
# x_cap = ["tri", "u", "uv"]
# x_height= ["r1_ape", "r3_ape", "r4_ape"]


# fig, ax = plt.subplots()


# all_gs = np.array([[],[],[]])
# all_gs = np.array([
#           [pos_list[0], pos_list[3], pos_list[6]],
#           [pos_list[1][:], pos_list[4][:], pos_list[7][:]],
#           [pos_list[2][:], pos_list[5][:], pos_list[8][:]],
#          ])


print(len(pos_list[1][:]))
ape    = [pos_list[0], pos_list[4], pos_list[8], pos_list[1][:], pos_list[5][:], pos_list[9][:], pos_list[2][:], pos_list[6][:], pos_list[10][:], pos_list[3][:], pos_list[7][:], pos_list[11][:]]
robots = [[0]*len(pos_list[0]), [0]*len(pos_list[4]), [0]*len(pos_list[8]), 
                      [1]*len(pos_list[1][:]),[1]*len(pos_list[5][:]),[1]*len(pos_list[9][:]),
                      [2]*len(pos_list[2][:]), [2]*len(pos_list[6][:]),[2]*len(pos_list[10][:]),
                      [3]*len(pos_list[3][:]), [3]*len(pos_list[7][:]),[3]*len(pos_list[11][:])]
approaches = [[0]*len(pos_list[0]), [1]*len(pos_list[4]), [2]*len(pos_list[8]), 
                      [0]*len(pos_list[1][:]),[1]*len(pos_list[5][:]),[2]*len(pos_list[9][:]),
                      [0]*len(pos_list[2][:]), [1]*len(pos_list[6][:]),[2]*len(pos_list[10][:]),
                      [0]*len(pos_list[3][:]), [1]*len(pos_list[7][:]),[2]*len(pos_list[11][:])]       


# robots = [["Robot_1"]*len(pos_list[0]), ["Robot_1"]*len(pos_list[3]), ["Robot_1"]*len(pos_list[6]), 
#                       ["Robot_2"]*len(pos_list[1][:]),["Robot_2"]*len(pos_list[4][:]),["Robot_2"]*len(pos_list[7][:]),
#                       ["Robot_3"]*len(pos_list[2][:]), ["Robot_3"]*len(pos_list[5][:]),["Robot_3"]*len(pos_list[8][:])]
# approaches = [["Triangulation"]*len(pos_list[0]), ["PF with UWB only"]*len(pos_list[3]), ["PF with UWB and Spatial Detection"]*len(pos_list[6]), 
#                       ["Triangulation"]*len(pos_list[1][:]),["PF with UWB only"]*len(pos_list[4][:]),["PF with UWB and Spatial Detection"]*len(pos_list[7][:]),
#                       ["Triangulation"]*len(pos_list[2][:]), ["PF with UWB only"]*len(pos_list[5][:]),["PF with UWB and Spatial Detection"]*len(pos_list[8][:])]                  

ape = list(itertools.chain.from_iterable(ape))
robots = list(itertools.chain.from_iterable(robots))
approaches = list(itertools.chain.from_iterable(approaches))
# gs_1 = np.array([pos_list[0], pos_list[3], pos_list[6]])
# pos_list[0], pos_list[3], pos_list[6], pos_list[1][:], pos_list[4][:], pos_list[7][:], [pos_list[2][:], pos_list[5][:], pos_list[8][:]]
# all_gs = np.array([])
all_gs = np.vstack([ape, robots])
all_gs = np.vstack([all_gs, approaches])
# print(gs_1.shape)
all_gs = np.transpose(all_gs)
print(all_gs.shape)


df = pd.DataFrame(all_gs, columns=["ape_value", "robots","approaches" ])
# print(df.head)

#adding color
# pal = "Set1"

# Now with the group as hue
dx = "robots"; dy = "ape_value"; dhue = "approaches"; pal = "Set2"; sigma = .2; ort = "v"
# f, ax = plt.subplots(figsize=(8,5))
f, ax = plt.subplots()

ax=pt.RainCloud(x = dx, y = dy, hue = dhue, data = df, palette = pal, orient=ort, bw = sigma, width_viol = .6,
                ax = ax, alpha = .5, jitter=0, dodge = True, move = .25) #pointplot = True,


plt.title("State Estimation Error of Triangulations & Particle filters\n on UWB Ranges Fused With Spatial Detection")
# plt.title("Figure P20\n  Repeated Measures Data - Example 2")
# if savefigs:
#     plt.savefig('../figs/tutorial_python/figureP20.png', bbox_inches='tight')

# plt.xlim(-1, 3)
# plt.show()

# # Major ticks every 20, minor ticks every 5
major_ticks = np.linspace(0, 1, 9)
minor_ticks = np.linspace(0, 1, 19)
print(major_ticks)

# ax.set_xticks(major_ticks)
# ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

ax.grid(which='minor', alpha=0.4)
ax.grid(which='major', alpha=0.7)

# plt.grid(axis = 'y', color = 'green', linestyle = '--', linewidth = 0.5)
plt.xticks(np.array([0,1,2,3]), np.array(["robot_1", "robot_2", "robot_3", "robot_4"]))

FILENAME = "raincloud_plot_ape" 
plt.savefig('./results/figs/img/{}.png'.format(FILENAME), bbox_inches='tight')   
tikz.save("./results/figs/tex/{}.tex".format(FILENAME)) 
plt.show()
# plt.yscale('log')
# plt.legend()