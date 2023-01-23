from evo.tools import log
log.configure_logging(verbose=True, debug=True, silent=False)

import pprint
import numpy as np

from evo.tools import plot
import matplotlib.pyplot as plt

# temporarily override some package settings
from evo.tools.settings import SETTINGS
SETTINGS.plot_usetex = False
# temporarily override some package settings
SETTINGS.plot_figsize = [6, 6]
SETTINGS.plot_split = True
SETTINGS.plot_usetex = False

# magic plot configuration
import matplotlib.pyplot as plt


from evo.tools import file_interface
from evo.core import sync
from evo.core import metrics

from rosbags.rosbag2 import Reader as Rosbag2Reader

robot_id = 4
way_id = 1
# file = 'tri'
# file = 'u'
file = "uv"

with Rosbag2Reader("/home/xianjia/Workspace/temp/uwb_ranging_refine_with_spatial_detection/result_bags/20221011/cali_4robots_data_02/result_bag_1_0") as reader:
    if robot_id == 1:
        traj_ref = file_interface.read_bag_trajectory(reader, "/real_turtle01_pose")
        if way_id == 0: 
            traj_est = file_interface.read_bag_trajectory(reader, "/tri_turtle01_pose")
        if way_id == 1:
            traj_est = file_interface.read_bag_trajectory(reader, "/pf_turtle01_pose")
    if robot_id == 3:
        traj_ref = file_interface.read_bag_trajectory(reader, "/real_turtle03_pose")
        if way_id == 0:
            traj_est = file_interface.read_bag_trajectory(reader, "/tri_turtle03_pose")
        if way_id == 1:
            traj_est = file_interface.read_bag_trajectory(reader, "/pf_turtle03_pose")
    if robot_id == 4:
        traj_ref = file_interface.read_bag_trajectory(reader, "/real_turtle04_pose")
        if way_id == 0:
            traj_est = file_interface.read_bag_trajectory(reader, "/tri_turtle04_pose")
        if way_id == 1:
            traj_est = file_interface.read_bag_trajectory(reader, "/pf_turtle04_pose")
traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est,max_diff=0.05)
print(traj_ref)

import copy

traj_est_aligned = copy.deepcopy(traj_est)
traj_est_aligned.align(traj_ref, correct_scale=False, correct_only_scale=False)

# traj_est_aligned =  traj_est_aligned[500:]
# traj_ref         =  traj_ref[500:]
# traj_ref.reduce_to_time_range(1666597517.658, 1666597769.942)
# traj_est_aligned.reduce_to_time_range(1666597517.658, 1666597769.942)

fig = plt.figure()
traj_by_label = {
    # "estimate (not aligned)": traj_est,
    "estimate (aligned)": traj_est_aligned,
    "reference": traj_ref
}
plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
plt.show()

pose_relation = metrics.PoseRelation.translation_part
use_aligned_trajectories = True

if use_aligned_trajectories:
    data = (traj_ref, traj_est_aligned) 
else:
    data = (traj_ref, traj_est)

ape_metric = metrics.APE(pose_relation)
ape_metric.process_data(data)

ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
print(ape_stat)

ape_stats = ape_metric.get_all_statistics()
pprint.pprint(ape_stats)

np.savetxt("./ape/ape_{}_t{}.csv".format(file, robot_id), ape_metric.error)

# seconds_from_start = [t - traj_est.timestamps[0] for t in traj_est.timestamps]
# fig = plt.figure()
# # plot.error_array(fig.gca(), ape_metric.error, x_array=seconds_from_start,
# #                  statistics={s:v for s,v in ape_stats.items() if s != "sse"},
# #                  name="APE", title="APE w.r.t. " + ape_metric.pose_relation.value, xlabel="$t$ (s)")
# plt.boxplot(ape_metric.errors)
# plt.show()


# plot_mode = plot.PlotMode.xy
# fig = plt.figure()
# ax = plot.prepare_axis(fig, plot_mode)
# plot.traj(ax, plot_mode, traj_ref, '--', "gray", "reference")
# plot.traj_colormap(ax, traj_est_aligned if use_aligned_trajectories else traj_est, ape_metric.error, 
#                    plot_mode, min_map=ape_stats["min"], max_map=ape_stats["max"])
# ax.legend()
# plt.show()