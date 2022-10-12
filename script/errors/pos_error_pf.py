import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
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
    [197,27,138, 80],
    [235, 64, 52, 80],
    [162, 52, 235, 80],
    [44,127,184, 80],
])/256)

names = [ name for name in mcolors.CSS4_COLORS]
colors = []
for i in range(12):
    colors.append(mcolors.CSS4_COLORS[names[i]])

# loal pos csv files
pos_file_list = [
    "/home/xianjia/Workspace/temp/results/results_11102022/pfilter/pos/pos_u",
    "/home/xianjia/Workspace/temp/results/results_11102022/pfilter/pos/pos_u_v",
    "/home/xianjia/Workspace/temp/results/results_11102022/pfilter/pos/pos_uv"
]

## get the file name of all above folders
files = []
for f_l in pos_file_list:
    fs = []
    for (dirpath, dirnames, filenames) in os.walk(f_l):
        fs.extend(filenames)
    files.append(fs)

pos_list = []
for inx, fs in enumerate(files):
    cnt = 0
    f_v = np.zeros(shape=np.genfromtxt(os.path.join(pos_file_list[inx], fs[0]), delimiter=',').shape)
    for f in fs:
        n_v = np.genfromtxt(os.path.join(pos_file_list[inx], f),delimiter=',')
        if n_v.shape == f_v.shape:
            f_v += n_v
            cnt += 1
    f_v /= cnt
    # print(f"shape of each file: {f_v.shape}")
    # print(f"------ cnt size fit: {cnt}")
    pos_list.append(f_v)

# save average pos of different rounds in CSVs
pos_name_list =["original_u.csv","original_u_v.csv","original_uv.csv"] 
for i in range(len(pos_name_list)):
    np.savetxt(pos_name_list[i], pos_list[i][1500:-30:,]) 


# choose only the converged part of pf
p_con_list = []
for i in range(len(pos_list)):
    p_con_list.append(pos_list[i][1500:-30, :])

# fix the translation
for inx, p in enumerate(p_con_list):
    ave = np.mean(p, axis=0)
    for i in range(6):
        p[:,i+6] = np.subtract(p[:,i+6], ave[i+6] - ave[i])

# save the translation fixed poses
pos_name_list =["new_u.csv","new_u_v.csv","new_uv.csv"] 
for i in range(len(pos_name_list)):
    np.savetxt(pos_name_list[i], p_con_list[i]) 
        
# cal the errors
# errors = []
errors_x = []
errors_y = []
for p in p_con_list:
    # error_t = []
    for i in range(6):
        if i%2 == 0:
            errors_x.append(np.fabs(p[:,i+6] - p[:, i]))
        else:
            errors_y.append(np.fabs(p[:, i+6] - p[:, i]))
        # error_t.append(np.fabs(p[:, i+6] - p[:, i]))
    # errors.append(error_t)

# color_ls = ['b+', 'r+', 'g+', 'c+', 'm', 'y', 'k', 'w', 'r'] 
color_ls =     ['b', 'r', 'g', 'c', 'm', 'b', 'r', 'g', '#1f77b4', 'b', 'r', 'g', 'c', 'm', 'b', 'r', 'g', '#1f77b4'] 
linestyle_ls = ['*', 'v', '^', 's', 'o', 'D', 'H', '8', 'p', '*', 'v', '^', 's', 'o', 'D', 'H', '8', 'p']
x_cap = ["robot01_u", "robot01_u_v", "robot01_uv", 
        "robot02_u", "robot02_u_v", "robot02_uv", 
        "robot03_u", "robot03_u_v", "robot03_uv"]
x_height= ["axis_x_error", "axis_y_error"]


fig, ax = plt.subplots()
plt.title("State Estimation Error of Particle filters on UWB Ranges Fused With Spatial Detection")

all_gs = [
          errors_x, 
          errors_y
         ]
num = 2

for i in range(len(all_gs)):
    data = all_gs[i]
    pos = [x for x in range(50*i, 50*i + num*9, 2)]
    # print(len(data), len(pos))

    bx = ax.boxplot(data, positions = pos, notch=True, showfliers=False )
    # print(f"bx size: {len(bx['boxes'])}")
    for idx, box in enumerate(bx['boxes']):
        box.set(color= colors[int(idx/3)], linewidth=5)
        ax.legend( bx["boxes"], [ "{}".format(m) for m in x_cap ], loc='upper left')
        # plt.xticks(ticks=[x for x in range(30*i, 30*i + 18, 6)],labels =["{}".format(val) for val in ["x", "y", "z"]])
plt.xticks(ticks=[50 * x + 10  for x in range(2)], labels=["{}".format(x_height[i]) for i in range(len(x_height))])

# FILENAME = "real_mf_boxplot" 
# plt.savefig('{}.png'.format(FILENAME))   
# tikz.save("{}.tex".format(FILENAME)) 
plt.show()
# plt.yscale('log')
# plt.legend()