from math import dist
from re import M
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt 

nopoly_data = genfromtxt('data/pos_nopolyfit_1.csv', delimiter=',')
print(nopoly_data)
nopoly_mocap_pose = nopoly_data[:, [0, 1]]
print(len(nopoly_mocap_pose))
nopoly_mocap_pose = nopoly_mocap_pose[200:-200]
print(len(nopoly_mocap_pose))
nopoly_bot_pose = nopoly_data[:, [2, 3]]
print(len(nopoly_bot_pose))
nopoly_bot_pose = nopoly_bot_pose[200:-200]
print(len(nopoly_bot_pose))
nopoly_error=np.zeros(len(nopoly_mocap_pose))
nopoly_error= np.abs( np.linalg.norm(nopoly_mocap_pose-nopoly_bot_pose,axis=1))
# print(nopoly_error)
print(len(nopoly_error))

print()

poly_data = genfromtxt('data/pos_poly_1.csv', delimiter=',')
poly_mocap_pose = poly_data[:, [0, 1]]
print(len(poly_mocap_pose))
poly_mocap_pose = poly_mocap_pose[100:-100]
print(len(poly_mocap_pose))
poly_bot_pose = poly_data[:, [2, 3]]
print(len(poly_bot_pose))
poly_bot_pose = poly_bot_pose[100:-100]
print(len(poly_bot_pose))
poly_error=np.zeros(len(poly_mocap_pose))
poly_error= np.abs(np.linalg.norm(poly_mocap_pose-poly_bot_pose,axis=1))
# print(poly_error)
print(len(poly_error))

######################################################################################
# Box plot combined

plt.clf()
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.patch.set_facecolor("lightgray")
ax.patch.set_alpha(0.4)
# plt.tight_layout()

ax.set_ylabel('Error (m)')

pos=1
xlabels=[]
xticks=[]
colors = ['pink', 'lightblue', "lightgreen",'pink','lightblue', "lightgreen",'pink', 'lightblue']

labels_legend = ["without polyfit","with polyfit"]
colors = ['pink', 'lightblue','lightgreen']

# for k in range(0,total_num_uwb):
bplot = ax.boxplot(
            [nopoly_error,poly_error],
            vert=True,
            patch_artist=True,
            # labels=labels,
            notch=0,
            sym='b+',
            whis=1,
            positions=[pos,pos+1],
            widths=0.5
)

# pos+=3

for patch, color in zip(bplot['boxes'], colors) :
    patch.set_facecolor(color)

for i in range(0,2):

    bplot['medians'][i].set_color('black')
    bplot['medians'][i].set_linewidth(2)


ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
ax.set_axisbelow(True)

for j in range(0,2):
    xlabels.append( '{}'.format(1))
    xticks.append(pos+1)

pos+=2

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
hR, = ax.plot([0,0],colors[1])
# hT, = ax.plot([0,0],colors[2])

ax.legend((hB, hR),(labels_legend))
hB.set_visible(False)
hR.set_visible(False)
# hT.set_visible(False)    


# plt.ylim(0,2)
plt.savefig("boxplot.png")
plt.show()
