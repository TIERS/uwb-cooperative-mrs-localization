import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import tikzplotlib as tikz
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
#sns.set(style="darkgrid")
#sns.set(style="whitegrid")
#sns.set_style("white")
sns.set(style="whitegrid",font_scale=2)
import matplotlib.collections as clt


import ptitprince as pt


colors = list(np.array([
    [235, 64, 52, 80],
    [162, 52, 235, 80],
    [44,127,184, 80],
    [127,205,187, 80],
    [237,248,177, 80],
    [43,140,190, 80],
    [166,189,219, 80],
    [49,163,84, 80],
    [197,27,138, 80],
    [235, 64, 52, 80],
    [162, 52, 235, 80],
    [44,127,184, 80],
])/256)
   

# names = [ name for name in mcolors.CSS4_COLORS]
# colors = []
# for i in range(12):
#     colors.append(mcolors.CSS4_COLORS[names[i]])

folder_name = "/home/xianjia/Workspace/temp/uwb_ranging_refine_with_spatial_detection/ape/"

## get the file name of all above folders
files = [
    "ape_tri_t1.csv",
    "ape_tri_t3.csv",   
    "ape_tri_t4.csv",
    "ape_u_t1.csv",
    "ape_u_t3.csv",
    "ape_u_t4.csv",
    "ape_uv_t1.csv",
    "ape_uv_t3.csv",
    "ape_uv_t4.csv",
]

pos_list = []
for inx, fs in enumerate(files):
    f_v = np.genfromtxt(os.path.join(folder_name, fs),delimiter=',')
    pos_list.append(f_v)

print(f"pos list: {len(pos_list)}")
linestyle_ls = ['*', 'v', '^', 's', 'o', 'D', 'H', '8', 'p', '*', 'v', '^', 's', 'o', 'D', 'H', '8', 'p']
x_cap = ["tri", "u", "uv"]
x_height= ["r1_ape", "r3_ape", "r4_ape"]


fig, ax = plt.subplots()
plt.title("State Estimation Error of Triangulations & Particle filters on UWB Ranges Fused With Spatial Detection")

all_gs = [
          [pos_list[0], pos_list[3], pos_list[6]],
          [pos_list[1][:], pos_list[4][:], pos_list[7][:]],
          [pos_list[2][:], pos_list[5][:], pos_list[8][:]],
         ]

# df = pd.DataFrame(all_gs, columns = ['Column_A','Column_B','Column_C'], index = ['Item_1', 'Item_2'])

num = 3

# https://towardsdatascience.com/making-it-rain-with-raincloud-plots-496c39a2756f
for i in range(len(all_gs)):
    data = all_gs[i]
    print(data)
    # pos = [x for x in range(20*i, 20*i + num*3, 3)]
    # print(len(data), len(pos))
    pal = sns.color_palette(n_colors=1)
    ax=pt.half_violinplot(x='x', y='y', data = data, palette = pal, bw = .2, cut = 0.,
                      scale = "area", width = .6, inner = None)

    ax=sns.stripplot(x='x', y='y', data = data, palette = pal)
    # bx = ax.boxplot(data, positions = pos, notch=True, showfliers=False, autorange=True )

    # for idx, box in enumerate(bx['boxes']):
    #     # print(int(idx/3))
    #     box.set(color= colors[int(idx)], linewidth=5)
    #     ax.legend( bx["boxes"], [ "{}".format(m) for m in x_cap ], loc='upper left')
        # plt.xticks(ticks=[x for x in range(30*i, 30*i + 18, 6)],labels =["{}".format(val) for val in ["x", "y", "z"]])
# plt.ylim([-0.1, 1.0])
plt.xticks(ticks=[20 * x + 3  for x in range(3)], labels=["{}".format(x_height[i]) for i in range(len(x_height))])

# FILENAME = "real_mf_boxplot" 
# plt.savefig('{}.png'.format(FILENAME))   
# tikz.save("{}.tex".format(FILENAME)) 
plt.show()
# plt.yscale('log')
# plt.legend()