import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tikzplotlib as tikz

colors = list(np.array([
    [235, 64, 52, 80],
    [162, 52, 235, 80],
    [44,127,184, 80],
    [127,205,187, 80],
    [237,248,177, 80],
    [43,140,190, 80],
    [166,189,219, 80],
    [49,163,84, 80],
    [197,27,138, 80]
])/256)

pos_u   = "/home/xianjia/Workspace/temp/results/results_07102022/pos/pos_u"
pos_u_v = "/home/xianjia/Workspace/temp/results/results_07102022/pos/pos_u_v"
pos_uv  = "/home/xianjia/Workspace/temp/results/results_07102022/pos/pos_uv"

# my_data = np.genfromtxt('my_file.csv', delimiter=',')

f_u = []
for (dirpath, dirnames, filenames) in os.walk(pos_u):
    f_u.extend(filenames)

f_u_v = []
for (dirpath, dirnames, filenames) in os.walk(pos_u_v):
    f_u_v.extend(filenames)

f_uv = []
for (dirpath, dirnames, filenames) in os.walk(pos_uv):
    f_uv.extend(filenames)

d_u = []
a_u = np.zeros(shape=np.genfromtxt(os.path.join(pos_u,f_u[0]), delimiter=',').shape)
c_u = 0
for u in f_u:
    n_v = np.genfromtxt(os.path.join(pos_u,u), delimiter=',')
    if n_v.shape == a_u.shape:
        a_u += n_v
        c_u += 1
a_u /= c_u


d_u_v = []
a_u_v = np.zeros(shape=np.genfromtxt(os.path.join(pos_u_v,f_u_v[0]), delimiter=',').shape)
c_u_v = 0
for u in f_u_v:
    n_v = np.genfromtxt(os.path.join(pos_u_v,u), delimiter=',')
    if n_v.shape == a_u_v.shape:
        a_u_v += n_v
        c_u_v += 1
a_u_v /= c_u_v


d_uv = []
a_uv = np.zeros(shape=np.genfromtxt(os.path.join(pos_uv,f_uv[0]), delimiter=',').shape)
c_uv = 0
for u in f_uv:
    n_v = np.genfromtxt(os.path.join(pos_uv,u), delimiter=',')
    if n_v.shape == a_uv.shape:
        a_uv += n_v
        c_uv += 1
a_uv /= c_uv

# save the poses
# print(f"number:{c_u},{c_u_v},{c_uv}")
# np.savetxt("new_u.csv", a_u[800:-100])
# np.savetxt("new_u_v.csv", a_u_v[800:-100])
# np.savetxt("new_uv.csv", a_uv[800:-100])

print(a_u[800:-100][0:1].shape)

r1_u_e_x = np.fabs(a_u[800:-100,6] - a_u[800:-100,0])
r1_u_e_y = np.fabs(a_u[800:-100,7] - a_u[800:-100,1])
r2_u_e_x = np.fabs(a_u[800:-100,8] - a_u[800:-100,2])
r2_u_e_y = np.fabs(a_u[800:-100,9] - a_u[800:-100,3])
r3_u_e_x = np.fabs(a_u[800:-100,10] - a_u[800:-100,4])
r3_u_e_y = np.fabs(a_u[800:-100,11] - a_u[800:-100,5])

r1_u_v_e_x = np.fabs(a_u_v[800:-100,6] - a_u_v[800:-100,0])
r1_u_v_e_y = np.fabs(a_u_v[800:-100,7] - a_u_v[800:-100,1])
r2_u_v_e_x = np.fabs(a_u_v[800:-100,8] - a_u_v[800:-100,2])
r2_u_v_e_y = np.fabs(a_u_v[800:-100,9] - a_u_v[800:-100,3])
r3_u_v_e_x = np.fabs(a_u_v[800:-100,10] - a_u_v[800:-100,4])
r3_u_v_e_y = np.fabs(a_u_v[800:-100,11] - a_u_v[800:-100,5])

r1_uv_e_x = np.fabs(a_uv[800:-100,6] - a_uv[800:-100,0])
r1_uv_e_y = np.fabs(a_uv[800:-100,7] - a_uv[800:-100,1])
r2_uv_e_x = np.fabs(a_uv[800:-100,8] - a_uv[800:-100,2])
r2_uv_e_y = np.fabs(a_uv[800:-100,9] - a_uv[800:-100,3])
r3_uv_e_x = np.fabs(a_uv[800:-100,10] - a_uv[800:-100,4])
r3_uv_e_y = np.fabs(a_uv[800:-100,11] - a_uv[800:-100,5])

# color_ls = ['b+', 'r+', 'g+', 'c+', 'm', 'y', 'k', 'w', 'r'] 
color_ls =     ['b', 'r', 'g', 'c', 'm', 'b', 'r', 'g', '#1f77b4', 'b', 'r', 'g', 'c', 'm', 'b', 'r', 'g', '#1f77b4'] 
linestyle_ls = ['*', 'v', '^', 's', 'o', 'D', 'H', '8', 'p', '*', 'v', '^', 's', 'o', 'D', 'H', '8', 'p']
x_cap = ["robot01_u", "robot01_u_v", "robot01_uv", "robot02_u", "robot02_u_v", "robot02_uv", "robot03_u", "robot03_u_v", "robot03_uv"]
x_height= ["axis_x_error", "axis_y_error"]


fig, ax = plt.subplots()
plt.title("State Estimation Error of Particle filters on UWB Ranges Fused With Spatial Detection")


# all_gs = [
#           r1_u_e_x, r2_u_e_x, r3_u_e_x, 
#           r1_u_e_y, r2_u_e_y, r3_u_e_y
#          ]
# print(a_u[:][6].shape)
# print(r1_u_e_x.shape)

# bp = ax.boxplot(all_gs, patch_artist = True,
# notch ='True', vert = 0)

# plt.show()

all_gs = [
          [r1_u_e_x, r1_u_v_e_x, r1_uv_e_x, r2_u_e_x, r2_u_v_e_x, r2_uv_e_x , r3_u_e_x, r3_u_v_e_x, r3_uv_e_x], 
          [r1_u_e_y, r1_u_v_e_y, r1_uv_e_y, r2_u_e_y, r2_u_v_e_y, r2_uv_e_y , r3_u_e_y, r3_u_v_e_y, r3_uv_e_y]
         ]
num = 2

for i in range(len(all_gs)):
    data = all_gs[i]
    pos = [x for x in range(30*i, 30*i + num*9, 2)]
    print(len(data), len(pos))

    bx = ax.boxplot(data, positions = pos, notch=True, showfliers=False )
    print(f"bx size: {len(bx['boxes'])}")
    for idx, box in enumerate(bx['boxes']):
        box.set(color= colors[int(idx/(num/2))], linewidth=5)
        ax.legend( bx["boxes"], [ "{}".format(m) for m in x_cap ], loc='upper left')
        # plt.xticks(ticks=[x for x in range(30*i, 30*i + 18, 6)],labels =["{}".format(val) for val in ["x", "y", "z"]])
plt.xticks(ticks=[30 * x + 10  for x in range(2)], labels=["{}".format(x_height[i]) for i in range(len(x_height))])

# FILENAME = "real_mf_boxplot" 
# plt.savefig('{}.png'.format(FILENAME))   
# tikz.save("{}.tex".format(FILENAME)) 
plt.show()
# plt.yscale('log')
# plt.legend()