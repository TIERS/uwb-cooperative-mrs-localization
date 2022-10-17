#!/usr/bin/env python
from ast import arg
from cProfile import label
import symbol
import rclpy
from rclpy.node import Node

import os 
import sys
import time
import math
import argparse
import shutil

import datetime

from std_msgs.msg               import Float64
from geometry_msgs.msg          import PoseStamped
from geometry_msgs.msg          import PoseWithCovarianceStamped
from geometry_msgs.msg          import PoseArray
from sensor_msgs.msg            import Range
from nav_msgs.msg               import Odometry
from rclpy.clock                import Clock
from rclpy.duration             import Duration
from depthai_ros_msgs.msg       import SpatialDetectionArray, SpatialDetection
from rclpy.qos                  import QoSProfile, ReliabilityPolicy, HistoryPolicy
from pfilter                    import ParticleFilter, squared_error

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

fuse_name       = ["u", "u_v", "uv"]
turtles         = ["5", "1"  , "3", "4"]
spatial_pair    = list(combinations(turtles,2))
print(spatial_pair)
[('5', '1'), ('5', '3'), ('5', '4'), ('1', '3'), ('1', '4'), ('3', '4')]
uwbs            = ["5", "7"  , "3", "4"]
uwb_pair        = [(3,7), (4,7), (2,7), (3,4), (2,3), (2,4), (7,5), (3,5),(4,5), (2,5)]
spatial_uwb     = {spatial_pair[0]: 6, spatial_pair[1]: 7, spatial_pair[2]: 8, spatial_pair[3]: 0, spatial_pair[4]: 1, spatial_pair[5]: 3}
spatial_dict    = {sp:np.array([]) for sp in spatial_pair}
print(spatial_dict)
t1_uwbs         = [0, 1, 2, 6]
t3_uwbs         = [0, 3, 4, 7] 
t4_uwbs         = [1, 3, 5, 8] 

def parse_args():
    parser = argparse.ArgumentParser(description='Options to control relative localization with only UWB, assisit with Vision, and all if vision available')
    parser.add_argument('--poses_save', type=bool, default=True, help='choose to save the estimated poses with pf')
    parser.add_argument('--images_save', type=bool, default=False, help='choose to save the images with pf')
    parser.add_argument('--fuse_group', type=int, default=0, help='0: only UWB in PF, 1: with vision replace new measurement, 2: uwb and vision together')
    parser.add_argument('--round', type=int, default=0, help='indicate which round the pf will run on a recorded data')
    args = parser.parse_args()
    return args

args = parse_args()


# Build folder to save results from different fusion combinations
if args.poses_save:
    pos_folder = "./results/pfilter/pos/pos_{}/".format(fuse_name[args.fuse_group])
    pos_file = pos_folder + 'pos_{}.csv'.format(args.round)
    if not os.path.exists(pos_folder):
        os.makedirs(pos_folder)

if args.images_save:
    images_save_path = './results/pfilter/images/images_{}/images_{}_{}/'.format(fuse_name[args.fuse_group], fuse_name[args.fuse_group], args.round)
    if not os.path.exists(images_save_path):
        os.makedirs(images_save_path)

class UWBParticleFilter(Node) :
    '''
        ROS Node that estimates relative position of two robots using odom and single uwb range.
    '''

    def __init__(self) :
        '''
            TODO Docstring
        '''

        # Init node
        super().__init__('relative_pf_rclpy')
        # Define QoS profile for odom and UWB subscribers
        self.qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.counter = 0

        # Particle filter params
        self.declare_parameters(
            namespace = '',
            parameters=[
                ("weights_sigma", 1.2),
                ("num_particles", 1500),
                ("uwb_noise", 0.05),
                ("resample_proportion", 0.01),
                ("max_pos_delay", 0.2)
            ]
        )
        
        self.weights_sigma = self.get_parameter('weights_sigma').value
        self.num_particles = self.get_parameter('num_particles').value
        self.uwb_noise = self.get_parameter('uwb_noise').value
        self.resample_proportion = self.get_parameter('resample_proportion').value


        self.get_logger().info('weights_sigma: %f, num_particles: %d, uwb_noise: %f, resample_proportion: %f'  % 
                            (self.weights_sigma,
                             self.num_particles,
                             self.uwb_noise,
                             self.resample_proportion))

        # Create filter
        self.prior_fn = lambda n: np.random.uniform(-8,2,(n,7))

        self.pf = ParticleFilter(
            prior_fn =              self.prior_fn, 
            observe_fn =            self.calc_hypothesis,  
            dynamics_fn =           self.velocity, #lambda x: x,    
            n_particles =           self.num_particles, 
            noise_fn =              self.add_noise, #lambda x: x + np.random.normal(0, noise, x.shape),
            weight_fn =             self.calc_weights, #lambda x, y : squared_error(x, y, sigma=2),
            resample_proportion =   self.resample_proportion
        )

        # all varibles 
        self.uwb_ranges             = [0.0 for _ in uwb_pair]
        self.turtles_mocaps         = [np.zeros(2) for _ in turtles]
        self.turtles_odoms          = [Odometry() for _ in turtles]
        self.last_turtles_odoms     = [Odometry() for _ in turtles]
        self.spatial_objects        = {t:np.array([]) for t in turtles}
        self.true_relative_poses    = [np.zeros(2) for _ in range(1,len(turtles))]
        self.relative_poses         = [np.zeros(2) for _ in range(1,len(turtles))]
        self.particle_odom          = np.array([0.01,0.01,0.01,0.01,0.01,0.01,0.01])

        self.get_logger().info("Subscribing to topics")
        # subscribe to uwb ranges 
        self.uwb_subs = [
            self.create_subscription(Range, "/uwb/tof/n_{}/n_{}/distance".format(p[0], p[1]), 
            self.create_uwb_ranges_cb(i),10) for i, p in enumerate(uwb_pair)]
        self.get_logger().info("{} UWB ranges received!".format(len(self.uwb_ranges)))

        # subscribe to optitrack mocap poses
        self.mocap_subs = [
            self.create_subscription(PoseStamped, "/vrpn_client_node/turtlebot{}_cap/pose".format(t), 
            self.create_mocap_pose_cb(i), 10) for i, t in enumerate(turtles)]
        self.get_logger().info("{} Mocaps poses received!".format(len(self.turtles_mocaps)))
        
        # subscribe to odometries
        self.odom_subs = [
            self.create_subscription(PoseStamped, "/cali/turtle0{}/odom".format(t), 
            self.create_odom_cb(i),10) for i, t in enumerate(turtles)]
        self.get_logger().info("{} odom poses received!".format(len(self.turtles_mocaps)))

        # subscribe to spatial detections
        self.spatial_subs = [
            self.create_subscription(PoseStamped, "/turtle0{}/color/yolov4_Spatial_detections".format(t), 
            self.create_spatial_cb(i), 10) for i, t in enumerate(turtles)]

        # pf relative poses publishers
        self.relative_pose_publishers = [self.create_publisher(PoseStamped, '/pf_turtle0{}_pose'.format(t), 10) for t in turtles[1:]]


        # Wait to get some odometry
        sys.stdout.write("Waiting for odom data...\n")
        for _ in range(100) :
            if self.turtles_odoms[0].header.stamp :
                break
            sys.stdout.write("..")
            sys.stdout.flush()
            time.sleep(0.1)
        self.get_logger().info("Odometry locked. Current odom\n")

        # Responder positions
        self.get_logger().info("UWB PF initialized. Estimating position from UWB and odom.")

        self.pos_estimation = []
        
    def create_uwb_ranges_cb(self, i):
        # print(f"uwb: {i}")
        return lambda range : self.uwb_range_cb(i, range)
        
    def uwb_range_cb(self, i, range):
        self.uwb_ranges[i] = range.range -0.32  

    def create_mocap_pose_cb(self, i):
        return lambda pos : self.mocap_pose_cb(i, pos)
        
    def mocap_pose_cb(self, i, pos):
        self.turtles_mocaps[i] = np.array([pos.pose.position.x, pos.pose.position.y])  

    def create_odom_cb(self, i):
        return lambda odom : self.odom_cb(i, odom)
        
    def odom_cb(self, i, odom):
        self.turtles_odoms[i] = odom

    def create_spatial_cb(self, i):
        return lambda detections : self.spatial_cb(i, detections)
        
    def spatial_cb(self, i, detections):
        self.spatial_objects[turtles[i]] = np.array(detections.detections)

    def relative_pose_cal(self, origin, ends, relative_poses):
        for inx, end in enumerate(ends):
            relative_poses[inx] = end - origin    

    def velocity(self, x) :
        '''
            Use Odom to update position
        '''
        xp = x + self.particle_odom
        return xp

    def add_noise(self, x) :
        '''
            Add noise to the estimations
            TODO add the noise to the measurements instead
        '''
        xp = x + np.random.normal(0, self.uwb_noise, x.shape)
        return xp

    def calc_hypothesis(self, x) :
        '''
            Given (Nx2) matrix of positions,
            create N arrays of observations (just one for now)
            T1, T3, T4, T2
            U7, U3, U4, U2
            x = [x1, y1, x2, y2, x3, y3, 0, y4]
            y = [d12, d13, d14, d23, d24, d34, d1, d2, d3, d4]
            y = [u37, u74, u72, u34, u23, u24, u75, u35, u45, u25]
                0   ,   1,   2,   3,   4,   5,   6,   7,   8,  9
            # y = [d23, d12, d13, d2, d3, d1]
            # y = [d23, d12, d13, d14, d24, d34, d2, d3, d1, d4]
            y = [u37, u74, u72, u34, u23, u24, u75, u35, u45, u25]
            # d4 = x4

        '''  
        # y = np.linalg.norm(x, axis=1)
        # print(f"y:{y.shape}")
        # print(x.shape)
        y = []

        t = datetime.datetime.now()
        
        for p in x :
            p4 = np.array([0, p[6]])
            y.append([
                np.linalg.norm(p[0:2] - p[2:4]),
                np.linalg.norm(p[0:2] - p[4:6]),
                np.linalg.norm(p[0:2] - p4),
                np.linalg.norm(p[2:4] - p[4:6]),
                np.linalg.norm(p[2:4] - p4),
                np.linalg.norm(p[4:6] - p4),
                np.linalg.norm(p[0:2]),
                np.linalg.norm(p[2:4]),
                np.linalg.norm(p[4:6]),
                np.linalg.norm(p4),
            ])
            # [('5', '1'), ('5', '3'), ('5', '4'), ('1', '3'), ('1', '4'), ('3', '4')]
            if args.fuse_group == 2:
                if spatial_dict[spatial_pair[0]].size > 0:
                    [ y.append(np.linalg.norm(p[0:2])) for _ in spatial_dict[spatial_pair[0]]]

                if spatial_dict[spatial_pair[1]].size > 0:
                    [ y.append(np.linalg.norm(p[2:4])) for _ in spatial_dict[spatial_pair[1]]]

                if spatial_dict[spatial_pair[2]].size > 0:
                    [ y.append(np.linalg.norm(p[4:6])) for _ in spatial_dict[spatial_pair[2]]]

                if spatial_dict[spatial_pair[3]].size > 0:
                    [ y.append(np.linalg.norm(p[0:2] - p[2:4])) for _ in spatial_dict[spatial_pair[3]]]

                if spatial_dict[spatial_pair[4]].size > 0:
                    [ y.append(np.linalg.norm(p[0:2] - p[4:6])) for _ in spatial_dict[spatial_pair[4]]]

                if spatial_dict[spatial_pair[5]].size > 0:
                    [ y.append(np.linalg.norm(p[2:4] - p[4:6])) for _ in spatial_dict[spatial_pair[5]]]

        tt = datetime.datetime.now() - t
        # print("Time elapsed = {}s + {}us".format(tt.seconds, tt.microseconds))

        y = np.array(y)

        # print(f"y:{y.shape}")
        return y

    def calc_weights(self, hypotheses, observations) :
        '''
            Calculate particle weights based on error
        '''
        # print(f"hypo: {hypotheses.shape}, observation: {observations.shape}")
        w = squared_error(hypotheses, observations, sigma=self.weights_sigma)
        # print(f"w: {w.shape}")
        return w
    
    def update_range_from_object_pose(self, object_end_pose_array, object_ori_pose_array):
        robots_relative_pose = np.array([-object_end_pose_array[0].position.x + object_ori_pose_array[0].position.x,
                                         -object_end_pose_array[0].position.y + object_ori_pose_array[0].position.y,
                                         -object_end_pose_array[0].position.z + object_ori_pose_array[0].position.z])
        return np.linalg.norm(robots_relative_pose)
    
    def update_filter(self) :
        '''
            Upadate particle filter
        '''
        # set measurements
        # TODO: check uwb measurements
        ## check vision measurements
        vis_meas_list = []
        if args.fuse_group == 2 or args.fuse_group == 1:
            for i, p in enumerate(spatial_pair):
                # print("the pair is {}, {}".format(p[0], p[1]))
                # print(self.spatial_objects[p[0]].size)
                if self.spatial_objects[p[0]].size > 0 and self.spatial_objects[p[1]].size > 0:
                    # print("-------------------------------")
                    for obj0 in self.spatial_objects[p[0]]:
                        for obj1 in self.spatial_objects[p[1]]:
                            vis_meas = self.update_range_from_object_pose(obj0, obj1)
                            print("vis: {}, uwb: {}".format(vis_meas, self.uwb_ranges[spatial_uwb[p]]))
                            if math.fabs(vis_meas -  self.uwb_ranges[spatial_uwb[p]]) < 0.5:
                                print("spatial and uwb ranges found")
                                spatial_dict[p].append(vis_meas)
                                vis_meas_list.append(vis_meas)
                                # vis_meas_list.append(vis_meas)
                                

        if args.fuse_group == 2 and len(vis_meas_list) > 0:
            # new_meas = np.append(self.uwb_ranges, [[self.uwb_ranges[6], self.uwb_ranges[7], self.uwb_ranges[8]]])
            new_meas = np.append(self.uwb_ranges, [vis_meas_list])
        else:
            new_meas = np.array(self.uwb_ranges)
            # new_meas = np.append([t1_u],[[t3_u],[t4_u]])
        # print(f"new_meas:{new_meas.shape}")

        # set particle odometries
        self.particle_odom[6] = 0.0
        for i in range(len(turtles[1:])):
            self.particle_odom[2*i] = self.turtles_odoms[i].pose.pose.position.x - self.last_turtles_odoms[i].pose.pose.position.x - \
                                    (self.turtles_odoms[0].pose.pose.position.x - self.last_turtles_odoms[0].pose.pose.position.x)
            self.particle_odom[2*i+1] = self.turtles_odoms[i].pose.pose.position.y - self.last_turtles_odoms[i].pose.pose.position.y - \
                                    (self.turtles_odoms[0].pose.pose.position.y - self.last_turtles_odoms[0].pose.pose.position.y)
        
        # set last turtle odoms value to be the current ones
        self.last_turtles_odoms = np.copy(self.turtles_odoms)
        
        self.pf.update(observed=new_meas)

        # self.get_logger().info("Avg. PF mean: {}, std = {}".format(self.pf.mean_state, self.pf.cov_state))
        if self.pf.cov_state[0][0] > 0.5 or self.pf.cov_state[0][1] > 0.5 :
            self.get_logger().warn("PF covariance too high with covx={} and covy={}".format(self.pf.cov_state[0], self.pf.cov_state[1]))

        # publish pf relative pose
        for i in range(len(turtles[1:])):
            relative_pose = PoseStamped()
            relative_pose.header.frame_id = "base_link"
            relative_pose.header.stamp = self.get_clock().now().to_msg()
            relative_pose.pose.position.x = self.pf.mean_state[2*i]
            relative_pose.pose.position.y = self.pf.mean_state[2*i+1]
            relative_pose.pose.position.z = 0.0
            relative_pose.pose.orientation = self.turtles_odoms[i].pose.pose.orientation
            self.relative_pose_publishers[i].publish(relative_pose)   

        # cal true or predicted relative poses
        self.relative_pose_cal(self.turtles_mocaps[0], self.turtles_mocaps[1:], self.true_relative_poses)
        pf_relative_poses = [self.pf.mean_state[0], self.pf.mean_state[1], self.pf.mean_state[2], self.pf.mean_state[3], self.pf.mean_state[4], self.pf.mean_state[5]]
        relative_poses = np.append(np.hstack(self.true_relative_poses), pf_relative_poses).tolist()
        
        # save groundtruth poses and calcuated poses to csv
        if args.poses_save: 
            self.pos_estimation.append(relative_poses)

        if args.images_save:
            self.plot_particles()

    def plot_particles(self):
        """Plot a 1D tracking result as a line graph with overlaid
        scatterplot of particles. Particles are sized according to
        normalised weight at each step.
        
            x: time values
            y: original (uncorrupted) values
            yn: noisy (observed) values
            states: dictionary return from apply_pfilter        
        """

        plt.ioff()
        plt.clf()

        symbols = ['x', 'o', '*']
        symbol_colors =['black', 'darkgray', 'lightgray']
        legends = [["T1_G", "T1_Mean", "T1_Map", "T1_Particles"],
                   ["T3_G", "T3_Mean", "T3_Map", "T3_Particles"],
                   ["T4_G", "T4_Mean", "T4_Map", "T4_Particles"]]
        for i in range(len(self.true_relative_poses)):
            plt.plot(self.true_relative_poses[i][0], self.true_relative_poses[i][1], symbols[i], c='red', label=legends[i][0])
            plt.plot(self.pf.mean_state[2*i], self.pf.mean_state[2*i+1], symbols[i], c='green', label=legends[i][1])
            plt.plot(self.pf.map_state[2*i], self.pf.map_state[2*i+1], symbols[i], c='orange', label=legends[i][2])
            plt.scatter(self.pf.transformed_particles[:,2*i], self.pf.transformed_particles[:,2*i+1], color=symbol_colors[i], label=legends[i][3]) # lightgray
        # print(f"particles shape:{self.pf.transformed_particles.shape}")
        plt.xlim(-9,9)
        plt.ylim(-9,9)

        plt.legend()
        self.counter += 1
        plt.savefig(images_save_path + "/test{}.png".format(self.counter))

    def run(self, node) :
        '''
            Create timer to update filter.
        '''
        # Reset filter
        self.pf.init_filter()

        # Set filter update timer at 10 Hz
        time.sleep(1)
        rclpy_check_rate = self.create_rate(10, self.get_clock())
        self.filter_timer = self.create_timer(1.0/20, self.update_filter)
        
        # Start calculating relative positions
        self.get_logger().info("Starting particle filter...")
        try:

            while rclpy.ok() :
                # Update objective position and publish it
                # self.get_logger().info('in loop')  
                rclpy.spin(node)           
                # rclpy_check_rate.sleep()

        except KeyboardInterrupt :
            self.get_logger().error('Keyboard Interrupt detected! Trying to stop filter node!')

        # self.filter_timer.shutdown()
    
    def __del__(self):
        # body of destructor
        self.get_logger().info("PF ends and Saving Results.")
        if args.poses_save: 
            np.savetxt(pos_file, 
            self.pos_estimation,
            delimiter =", ", 
            fmt ='% s')       


def main(args=None):
    rclpy.init(args=args)
    filter = UWBParticleFilter()
    # Reset filter
    filter.pf.init_filter()

    time.sleep(1)
    rclpy_check_rate = 10
    
    rclpy_check_rate = filter.create_rate(10, filter.get_clock())
    # Start calculating relative positions
    filter.get_logger().info("Starting particle filter...")
    try:
        try:
            while rclpy.ok() :
                # Update objective position and publish it
                # self.get_logger().info('in loop')  
                filter_timer = filter.create_timer(1/10.0, filter.update_filter)
                rclpy.spin(filter)             
                # rclpy_check_rate.sleep()
        except KeyboardInterrupt :
            filter.get_logger().error('Keyboard Interrupt detected! Trying to stop filter node!')
    except Exception as e:
        filter.destroy_node()
        filter.get_logger().info("UWB particle filter failed %r."%(e,))
    finally:
        rclpy.shutdown()
        filter.destroy_timer(filter_timer)
        filter.destroy_node()   

    

if __name__ == '__main__':
    main()


#     y.append([
#         np.linalg.norm(p[0:2] - p[2:4]),
#         np.linalg.norm(p[0:2] - p[4:6]),
#         np.linalg.norm(p[0:2] - p4),
#         np.linalg.norm(p[2:4] - p[4:6]),
#         np.linalg.norm(p[2:4] - p4),
#         np.linalg.norm(p[4:6] - p4),
#         np.linalg.norm(p[0:2]),
#         np.linalg.norm(p[2:4]),
#         np.linalg.norm(p[4:6]),
#         np.linalg.norm(p4), 
#         np.linalg.norm(p[0:2]),
#         np.linalg.norm(p[2:4]),
#         np.linalg.norm(p[4:6]),
# ])

# t1_u = []
# for t1 in t1_uwbs:
#     print(t1)
#     t1_u.append(self.uwb_ranges[t1])
# t3_u = []
# for t3 in t3_uwbs:
#     t3_u.append(self.uwb_ranges[t3])
# t4_u = []
# for t4 in t4_uwbs:
#     t4_u.append(self.uwb_ranges[t4])

# if args.fuse_group == 2 or args.fuse_group == 1:
#     if self.object_turtle01_pose_array.size != 0 and self.object_ori_pose_array.size != 0:
#         new_vision_meas = self.update_range_from_object_pose(self.object_turtle01_pose_array, self.object_ori_pose_array)
#         if math.fabs(new_vision_meas -  self.uwb75_range) < 0.5:
#             self.get_logger().info("vision: {}, uwb range: {}, truth: {}".format(new_vision_meas, self.uwb75_range, np.linalg.norm(self.true_relative_pose_turtle01)))
#             if args.fuse_group == 1:
#                 new_meas[6] = new_vision_meas
#             else:
#                 new_meas[10] = new_vision_meas
#                 # new_meas[1] = new_vision_meas

#     if self.object_turtle03_pose_array.size != 0 and self.object_ori_pose_array.size != 0:
#         new_vision_meas = self.update_range_from_object_pose(self.object_turtle03_pose_array, self.object_ori_pose_array)
#         if math.fabs(new_vision_meas -  self.uwb35_range) < 0.5:
#             self.get_logger().info("vision: {}, uwb range: {}, truth: {}".format(new_vision_meas, self.uwb75_range, np.linalg.norm(self.true_relative_pose_turtle03)))
#             if args.fuse_group == 1:
#                 new_meas[7] = new_vision_meas
#             else:
#                 new_meas[11] = new_vision_meas


#     if self.object_turtle04_pose_array.size != 0 and self.object_ori_pose_array.size != 0:
#         new_vision_meas = self.update_range_from_object_pose(self.object_turtle04_pose_array, self.object_ori_pose_array)
#         if math.fabs(new_vision_meas -  self.uwb45_range) < 0.5:
#             self.get_logger().info("vision: {}, uwb range: {}, truth: {}".format(new_vision_meas, self.uwb45_range, np.linalg.norm(self.true_relative_pose_turtle04)))
#             if args.fuse_group == 1:
#                 new_meas[8] = new_vision_meas
#             else:
#                 new_meas[12] = new_vision_meas

# # Build folder to save results from different fusion combinations
# if args.results_save:
#     fuse_name = ["u", "u_v", "uv"]
#     # err_folder = "./results/pfilter/errors/errors_{}/".format(fuse_name[args.fuse_group])
#     pos_folder = "./results/pfilter/pos/pos_{}/".format(fuse_name[args.fuse_group])
#     # range_folder = "./results/pfilter/ranges/ranges_{}/".format(fuse_name[args.fuse_group])
#     # error_file = err_folder + "error_{}.csv".format(args.round)
#     pos_file = pos_folder + 'pos_{}.csv'.format(args.round)
#     # range_file = range_folder + "range_{}.csv".format(args.round)
#     images_save_path = './results/pfilter/images/images_{}/images_{}_{}/'.format(fuse_name[args.fuse_group]), fuse_name[args.fuse_group]), args.round)
#     # if not os.path.exists(err_folder):
#     #     os.makedirs(err_folder)
#     # if not os.path.exists(pos_folder):
#     #     os.makedirs(pos_folder)
#     # if not os.path.exists(range_folder):
#     #     os.makedirs(range_folder)
#     if not os.path.exists(images_save_path):
#         os.makedirs(images_save_path)