from evo.tools import log
log.configure_logging(verbose=True, debug=True, silent=False)

import pprint
import numpy as np

from evo.tools import plot
import matplotlib.pyplot as plt

import tikzplotlib as tikz

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

bag_name = "20221217/rosbag2_20_06_59"
# bag_name = "20221217/rosbag2_20_09_08/"
real_poses_topics = ["/real_turtle01_pose", "/real_turtle02_pose", "/real_turtle03_pose", "/real_turtle05_pose"]
tri_poses_topics  = ["/tri_turtle01_pose", "/tri_turtle02_pose", "/tri_turtle03_pose", "/tri_turtle05_pose"]
pf_poses_topics   = ["/pf_turtle01_pose", "/pf_turtle02_pose", "/pf_turtle03_pose", "/pf_turtle05_pose"]

real_poses, uv_real_poses, tri_poses, pf_u_poses, pf_uv_poses   = [], [], [], [], []
align_tri_poses, align_pf_u_poses, align_pf_uv_poses   = [], [], []

with Rosbag2Reader(f"/home/xianjia/Workspace/temp/uwb_ranging_refine_with_spatial_detection/results/result_bags/{bag_name}/result_bag_0_0") as reader:
    for inx, topic in enumerate(real_poses_topics):
        real_poses.append(file_interface.read_bag_trajectory(reader, topic))
        # tri_poses.append(file_interface.read_bag_trajectory(reader, tri_poses_topics[inx]))
        pf_u_poses.append(file_interface.read_bag_trajectory(reader, pf_poses_topics[inx]))


with Rosbag2Reader(f"/home/xianjia/Workspace/temp/uwb_ranging_refine_with_spatial_detection/results/result_bags/{bag_name}/result_bag_1_0") as reader:
    for inx, topic in enumerate(real_poses_topics):
        uv_real_poses.append(file_interface.read_bag_trajectory(reader, real_poses_topics[inx]))
        pf_uv_poses.append(file_interface.read_bag_trajectory(reader, pf_poses_topics[inx]))


for inx in range(len(real_poses)):
    # sync
    # traj_ref_tri, traj_est_tri = sync.associate_trajectories(real_poses[inx], tri_poses[inx], max_diff=0.05)
    traj_ref_u, traj_est_u = sync.associate_trajectories(real_poses[inx], pf_u_poses[inx], max_diff=0.05)
    traj_ref_uv, traj_est_uv = sync.associate_trajectories(uv_real_poses[inx], pf_uv_poses[inx], max_diff=0.05)
    # traj_est_tri.align(traj_ref_tri, correct_scale=True, correct_only_scale=False)
    traj_est_u.align(traj_ref_u, correct_scale=True, correct_only_scale=False)
    traj_est_uv.align(traj_ref_uv, correct_scale=True, correct_only_scale=False)
    # align
    # align_tri_poses.append((traj_ref_tri, traj_est_tri))
    align_pf_u_poses.append((traj_ref_u, traj_est_u)) 
    align_pf_uv_poses.append((traj_ref_uv, traj_est_uv)) 
    # dave


# cal ape
pose_relation = metrics.PoseRelation.translation_part
use_aligned_trajectories = True

# ape_metrics_tri   = [metrics.APE(pose_relation) for _ in range(4)]
ape_metrics_pf_u  = [metrics.APE(pose_relation) for _ in range(4)]
ape_metrics_pf_uv = [metrics.APE(pose_relation) for _ in range(4)]

# for inx in range(len(align_tri_poses)):
#     if use_aligned_trajectories:
#         data = align_tri_poses[inx]
#     ape_metrics_tri[inx].process_data(data)
#     ape_stat = ape_metrics_tri[inx].get_statistic(metrics.StatisticsType.rmse)
#     print(ape_stat)
#     ape_stats = ape_metrics_tri[inx].get_all_statistics()
#     pprint.pprint(ape_stats)
#     np.savetxt("./ape/ape_tri_t{}.csv".format(inx), ape_metrics_tri[inx].error)


for inx in range(len(align_pf_u_poses)):
    if use_aligned_trajectories:
        data = align_pf_u_poses[inx]
    ape_metrics_pf_u[inx].process_data(data)
    ape_stat = ape_metrics_pf_u[inx].get_statistic(metrics.StatisticsType.rmse)
    print(ape_stat)
    ape_stats = ape_metrics_pf_u[inx].get_all_statistics()
    pprint.pprint(ape_stats)
    np.savetxt("./ape/ape_u_t{}.csv".format(inx), ape_metrics_pf_u[inx].error)


for inx in range(len(align_pf_uv_poses)):
    if use_aligned_trajectories:
        data = align_pf_uv_poses[inx]
    ape_metrics_pf_uv[inx].process_data(data)
    ape_stat = ape_metrics_pf_uv[inx].get_statistic(metrics.StatisticsType.rmse)
    print(ape_stat)
    ape_stats = ape_metrics_pf_uv[inx].get_all_statistics()
    pprint.pprint(ape_stats)
    np.savetxt("./ape/ape_uv_t{}.csv".format(inx), ape_metrics_pf_uv[inx].error)




# plot trajectories
for i in range(4):
    fig = plt.figure()
    traj_by_label = {
        # "estimate (not aligned)": traj_est,
        # "tri_robot{}".format(i): align_tri_poses[i][1],
        "pf_u_robot{}".format(i): align_pf_u_poses[i][1],
        "pf_uv_robot{}".format(i): align_pf_uv_poses[i][1],
        "groundtruth_robot{}".format(i): align_pf_u_poses[i][0]
    }
    plot.trajectories(fig, traj_by_label, plot.PlotMode.xy)

    FILENAME = "robot_{}_traj".format(i) 
    plt.savefig('{}.png'.format(FILENAME), bbox_inches='tight')   
    tikz.save("{}.tex".format(FILENAME)) 

    plt.show()

import pos_ape