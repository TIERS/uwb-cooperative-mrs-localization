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


folder_name = "/home/xianjia/Workspace/temp/uwb_ranging_refine_with_spatial_detection/results/ape/"

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
    # "ape_u_t0.csv",
    # "ape_u_t1.csv",   
    # "ape_u_t2.csv",
    # "ape_u_t3.csv",
    # "ape_ut_t0.csv",
    # "ape_ut_t1.csv",
    # "ape_ut_t2.csv",
    # "ape_ut_t3.csv",
    # "ape_utv_t0.csv",
    'nav_u.csv',
    # "ape_utv_t1.csv",
    # "ape_utv_t2.csv",
    # "ape_utv_t3.csv",
]

pos_list = []
for inx, fs in enumerate(files):
    f_v = np.genfromtxt(os.path.join(folder_name, fs),delimiter=',')
    pos_list.append(f_v)

ape    = [pos_list[0]]
# approaches = [
#           [0]*len(pos_list[0]),
#           [1]*len(pos_list[1]), 
#           [2]*len(pos_list[2]), 
#  ]
          
# approaches = [
#               [0]*len(pos_list[0]), 
#               [0]*len(pos_list[1]), 
#               [0]*len(pos_list[2]), 
#               [0]*len(pos_list[3]), 
#               ]       
                  

# plt.boxplot(lstm_error["stacked_lstm"].values)
# plt.boxplot(lstm_error["bidirectional_lstm"].values)
import tikzplotlib as tikz

plt.clf()
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.patch.set_facecolor("lightgray")
ax.patch.set_alpha(0.4)
# plt.tight_layout()

ax.set_ylabel('Mean Square Error (m^2)')

pos=1
xlabels=[]
xticks=[]
colors = ['pink', 'lightblue', "lightgreen",'pink','lightblue', "lightgreen",'pink', 'lightblue']

# labels_legend = ["stacked_lstm", "bidirectional_lstm", "convlstm"]
labels_legend = ["PF_U"]
colors = ['pink', 'lightblue','lightgreen']

print(np.shape(ape))
# for k in range(0,total_num_uwb):
bplot = ax.boxplot(
            ape,
            vert=True,
            patch_artist=True,
            # labels=labels_legend,
            notch=0,
            sym='b+',
            whis=1,
            positions=[pos],
            widths=0.5,
            showfliers=False  
)

# pos+=3

for patch, color in zip(bplot['boxes'], colors) :
    patch.set_facecolor(color)

for i in range(1):

    bplot['medians'][i].set_color('black')
    bplot['medians'][i].set_linewidth(2)


ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
ax.set_axisbelow(True)

for j in range(1):
    xlabels.append( '{}'.format(1))
    xticks.append(pos+1)

# pos+=2

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.tick_params(axis='x', direction='out')
ax.tick_params(axis='y', length=0)    

ax.set_xticklabels(xlabels)
ax.set_xticks(xticks)    

ax.set_xlim(0,xticks[-1]+1.5)
    # ax.ylim(0,9)

hB, = ax.plot([0,0],colors[0])
# hR, = ax.plot([0,0],colors[1])
# hT, = ax.plot([0,0],colors[2])

# ax.legend((hB),(labels_legend))
# hB.set_visible(False)
# hR.set_visible(False)
# hT.set_visible(False)


FILENAME = "ape" 
plt.savefig('./results/figs/tex/nav_traj_{}.png'.format(FILENAME), bbox_inches='tight')   
tikz.save("./results/figs/tex/nav_{}.tex".format(FILENAME)) 
plt.show()



# FILENAME = "raincloud_plot_ape" 
# plt.savefig('./results/single/figs/img/{}.png'.format(FILENAME), bbox_inches='tight')   
# tikz.save("./results/single/figs/tex/{}.tex".format(FILENAME)) 
# plt.show()
# plt.yscale('log')
# plt.legend()