#!/usr/bin/env python
# using the mocap to simulate the odometry data
from ast import arg
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



# rclpy.logging.set_logger_level('pf_relative_estimation', rclpy.logging.LoggingSeverity.ERROR)

def parse_args():
    parser = argparse.ArgumentParser(description='Options to control relative localization with only UWB, assisit with Vision, and all if vision available')
    parser.add_argument('--fuse_group', type=int, default=0, help='0: only UWB in PF, 1: with vision replace new measurement, 2: uwb and vision together')
    parser.add_argument('--round', type=int, default=0, help='indicate which round the pf will run on a recorded data')
    args = parser.parse_args()
    return args

args = parse_args()

if args.fuse_group == 0:
    err_folder = "./results/pfilter/errors/errors_u/"
    pos_folder = "./results/pfilter/pos/pos_u/"
    range_folder = "./results/pfilter/ranges/ranges_u/"
    error_file = err_folder + "error_{}.csv".format(args.round)
    pos_file = pos_folder + 'pos_{}.csv'.format(args.round)
    range_file = range_folder + "range_{}.csv".format(args.round)
    images_save_path = './results/pfilter/images/images_u/images_u_{}/'.format(args.round)

if args.fuse_group == 1:
    err_folder = "./results/pfilter/errors/errors_u_v/"
    pos_folder = "./results/pfilter/pos/pos_u_v/"
    range_folder = "./results/pfilter/ranges/ranges_u_v/"
    error_file = err_folder + "error_{}.csv".format(args.round)
    pos_file = pos_folder + 'pos_{}.csv'.format(args.round)
    range_file = range_folder + "range_{}.csv".format(args.round)
    images_save_path = './results/pfilter/images/images_u_v/images_u_v_{}/'.format(args.round)

if args.fuse_group == 2:
    err_folder = "./results/pfilter/errors/errors_uv/"
    pos_folder = "./results/pfilter/pos/pos_uv/"
    range_folder = "./results/pfilter/ranges/ranges_uv/"
    error_file = err_folder + "error_{}.csv".format(args.round)
    pos_file = pos_folder + 'pos_{}.csv'.format(args.round)
    range_file = range_folder + "range_{}.csv".format(args.round)
    images_save_path = './results/pfilter/images/images_uv/images_uv_{}/'.format(args.round)

if not os.path.exists(err_folder):
    os.makedirs(err_folder)

if not os.path.exists(pos_folder):
    os.makedirs(pos_folder)

if not os.path.exists(range_folder):
    os.makedirs(range_folder)

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
                ("num_particles", 1200),
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
        self.prior_fn = lambda n: np.random.uniform(-8,2,(n,6))
        print(self.prior_fn)
        self.pf = ParticleFilter(
            prior_fn =              self.prior_fn, 
            observe_fn =            self.calc_hypothesis,  
            dynamics_fn =           self.velocity, #lambda x: x,    
            n_particles =           self.num_particles, 
            noise_fn =              self.add_noise, #lambda x: x + np.random.normal(0, noise, x.shape),
            weight_fn =             self.calc_weights, #lambda x, y : squared_error(x, y, sigma=2),
            resample_proportion =   self.resample_proportion
        )
        
        # print(f"self.pf.d: {self.pf.d}")
        # print(f"n_particles: {self.pf.n_particles}")
        # print(f"particles: {self.pf.particles.shape}")
        # # print(f"mean_hypothesis: {self.pf.mean_hypothesis.shape}")
        # print(f"weights: {self.pf.weights.shape}")

        # self.odom = PoseStamped()#Odometry()
        self.last_particle_odom_turtle01 = Odometry()
        self.last_particle_odom_turtle03 = Odometry()
        self.last_particle_odom_turtle04 = Odometry()
        self.last_particle_odom_ori = Odometry()
        self.last_particle_pose_turtle01 = PoseStamped()
        self.last_particle_pose_turtle03 = PoseStamped()
        self.last_particle_pose_turtle04 = PoseStamped()
        self.last_particle_pose_ori = PoseStamped()
        self.pose_ori = PoseStamped()
        self.pose_turtle01 = PoseStamped()
        self.pose_turtle03 = PoseStamped()
        self.pose_turtle04 = PoseStamped()
        self.odom_ori = Odometry()
        self.odom_turtle01 = Odometry()
        self.odom_turtle03 = Odometry()
        self.odom_turtle04 = Odometry()
        # self.particle_odom = np.array([[0.01,0.01],[0.01,0.01], [0.01,0.01]])
        self.particle_odom = np.array([0.01,0.01,0.01,0.01,0.01,0.01])
        self.object_ori_pose_array = np.array([])
        self.object_turtle01_pose_array = np.array([])
        self.object_turtle03_pose_array = np.array([])
        self.object_turtle04_pose_array = np.array([])

        self.get_logger().info("Subscribing to topics")
        # subscribe to optitrack pose
        self.pose_ori_sub      = self.create_subscription(PoseStamped, "/vrpn_client_node/turtlebot5_cap/pose",  self.update_odom_ori_cb, 10)
        self.pose_turtle01_sub = self.create_subscription(PoseStamped, "/vrpn_client_node/turtlebot1_cap/pose",  self.update_turtle01_opti_pos_cb, 10)
        self.pose_turtle03_sub = self.create_subscription(PoseStamped, "/vrpn_client_node/turtlebot3_cap/pose",  self.update_turtle03_opti_pos_cb, 10)
        self.pose_turtle04_sub = self.create_subscription(PoseStamped, "/vrpn_client_node/turtlebot4_cap/pose",  self.update_turtle04_opti_pos_cb, 10)

        # subscriber to odometry
        self.odom_ori_sub      = self.create_subscription(Odometry, "/turtle05/odom",  self.update_odometry_ori_cb, qos_profile=self.qos)
        self.odom_turtle01_sub = self.create_subscription(Odometry, "/turtle01/odom",  self.update_odometry_turtle01_cb, qos_profile=self.qos)
        self.odom_turtle03_sub = self.create_subscription(Odometry, "/turtle03/odom",  self.update_odometry_turtle03_cb, qos_profile=self.qos)
        self.odom_turtle04_sub = self.create_subscription(Odometry, "/turtle04/odom",  self.update_odometry_turtle04_cb, qos_profile=self.qos)
        
        # subscribe to spatial detections
        self.object_ori_sub = self.create_subscription(SpatialDetectionArray, "/turtle05/color/yolov4_Spatial_detections",  self.update_object_ori_cb, 10)
        self.object_turtle01_sub = self.create_subscription(SpatialDetectionArray, "/turtle01/color/yolov4_Spatial_detections",  self.update_object_turtle01_cb, 10)
        self.object_turtle03_sub = self.create_subscription(SpatialDetectionArray, "/turtle03/color/yolov4_Spatial_detections",  self.update_object_turtle03_cb, 10)
        self.object_turtle04_sub = self.create_subscription(SpatialDetectionArray, "/turtle04/color/yolov4_Spatial_detections",  self.update_object_turtle04_cb, 10)
        self.object_turtle05_sub = self.create_subscription(SpatialDetectionArray, "/turtle05/color/yolov4_Spatial_detections",  self.update_object_turtle05_cb, 10)
    
        # subscribe to uwb ranges 
        self.uwb_34_range_sub = self.create_subscription(Range, "/uwb/tof/n_3/n_4/distance", self.update_uwb34_range_cb, 10)
        self.uwb_37_range_sub = self.create_subscription(Range, "/uwb/tof/n_3/n_7/distance", self.update_uwb37_range_cb, 10)
        self.uwb_47_range_sub = self.create_subscription(Range, "/uwb/tof/n_4/n_7/distance", self.update_uwb47_range_cb, 10)
        self.uwb_35_range_sub = self.create_subscription(Range, "/uwb/tof/n_3/n_5/distance", self.update_uwb35_range_cb, 10)
        self.uwb_45_range_sub = self.create_subscription(Range, "/uwb/tof/n_4/n_5/distance", self.update_uwb45_range_cb, 10)
        self.uwb_75_range_sub = self.create_subscription(Range, "/uwb/tof/n_7/n_5/distance", self.update_uwb75_range_cb, 10)
        self.publisher_turtle01_ = self.create_publisher(PoseStamped, '/pf_turtle01_pose', 10)
        self.publisher_turtle03_ = self.create_publisher(PoseStamped, '/pf_turtle03_pose', 10)
        self.publisher_turtle04_ = self.create_publisher(PoseStamped, '/pf_turtle04_pose', 10)

        # Wait to get some odometry
        sys.stdout.write("Waiting for odom data...")
        for _ in range(100) :
            if self.pose_ori.header.stamp and self.pose_ori.header.stamp :
                break
            sys.stdout.write("..")
            sys.stdout.flush()
            time.sleep(0.1)
        self.get_logger().info("Odometry locked. Current odom: \n{}".format(self.pose_ori.pose))

        # Responder positions
        self.get_logger().info("UWB PF initialized. Estimating position from UWB and odom.")

        # Calculate relative pose
        self.relative_pos = PoseStamped()

        self.uwb37_range = 0.0
        self.uwb34_range = 0.0
        self.uwb47_range = 0.0
        self.uwb35_range = 0.0
        self.uwb45_range = 0.0
        self.uwb75_range = 0.0
        self.true_relative_pose_turtle01 = np.array([.0,.0])
        self.true_relative_pose_turtle03 = np.array([.0,.0])
        self.true_relative_pose_turtle04 = np.array([.0,.0])
        self.pos = PoseWithCovarianceStamped()
        self.pos.header.stamp = Clock().now().to_msg()
        self.pos.pose.pose.orientation.w = 1.0

        self.ranges = []
        self.errors = []
        self.pos_estimation = []
        

    def update_odometry_ori_cb(self, odom):
        self.odom_ori = odom
        # self.get_logger().info("ori odom callback.")

    def update_odometry_turtle01_cb(self, odom):
        
        self.odom_turtle01 = odom

    def update_odometry_turtle03_cb(self, odom):
        
        self.odom_turtle03 = odom

    def update_odometry_turtle04_cb(self, odom):
        
        self.odom_turtle04 = odom

    def update_odom_ori_cb(self, pose) :
        '''
            Update pose from VIO
        '''
        self.pose_ori = pose
        # self.get_logger().info("end odom callback")

    def update_turtle01_opti_pos_cb(self, pose) :
        '''
            Update pose from VIO
        '''
        self.pose_turtle01 = pose

        end_pos = np.array([self.pose_turtle01.pose.position.x, self.pose_turtle01.pose.position.y])
        ori_pos = np.array([self.pose_ori.pose.position.x, self.pose_ori.pose.position.y]) 
        self.true_relative_pose_turtle01 = end_pos - ori_pos

        # only simulate uwb range
        # self.uwb_range = np.linalg.norm(end_pos - ori_pos) + np.random.normal(0, 0.15)
        
        # self.get_logger().info("odom end cb")

    def update_turtle03_opti_pos_cb(self, pose) :
        '''
            Update pose from VIO
        '''
        self.pose_turtle03 = pose

        end_pos = np.array([self.pose_turtle03.pose.position.x, self.pose_turtle03.pose.position.y])
        ori_pos = np.array([self.pose_ori.pose.position.x, self.pose_ori.pose.position.y]) 
        self.true_relative_pose_turtle03 = end_pos - ori_pos

        # only simulate uwb range
        # self.uwb_range = np.linalg.norm(end_pos - ori_pos) + np.random.normal(0, 0.15)
        
        # self.get_logger().info("odom end cb")

    def update_turtle04_opti_pos_cb(self, pose) :
        '''
            Update pose from VIO
        '''
        self.pose_turtle04 = pose

        end_pos = np.array([self.pose_turtle04.pose.position.x, self.pose_turtle04.pose.position.y])
        ori_pos = np.array([self.pose_ori.pose.position.x, self.pose_ori.pose.position.y]) 
        self.true_relative_pose_turtle04 = end_pos - ori_pos

        # only simulate uwb range
        # self.uwb_range = np.linalg.norm(end_pos - ori_pos) + np.random.normal(0, 0.15)
        
        # self.get_logger().info("odom end cb")

    def update_object_ori_cb(self, pose_array):
        self.object_ori_pose_array = np.array(pose_array.detections)
        # if self.object_ori_pose_array.size != 0:
            # self.get_logger().info("object ori pose size: {}".format(self.object_ori_pose_array.shape))

    def update_object_turtle01_cb(self, pose_array):
        self.object_turtle01_pose_array = np.array(pose_array.detections)

    
    def update_object_turtle03_cb(self, pose_array):
        self.object_turtle03_pose_array = np.array(pose_array.detections)

    def update_object_turtle04_cb(self, pose_array):
        self.object_turtle04_pose_array = np.array(pose_array.detections)

    def update_object_turtle05_cb(self, pose_array):
        # self.object_turtle05_pose_array = np.array(pose_array.detections)
        self.object_ori_pose_array = np.array(pose_array.detections)
        
    def update_uwb34_range_cb(self, range):
        '''
            Update range from UWB
        '''
        self.uwb34_range = range.range - 0.32


    def update_uwb37_range_cb(self, range):
        '''
            Update range from UWB
        '''
        self.uwb37_range = range.range - 0.32


    def update_uwb47_range_cb(self, range):
        '''
            Update range from UWB
        '''
        self.uwb47_range = range.range - 0.32


    def update_uwb35_range_cb(self, range):
        '''
            Update range from UWB
        '''
        self.uwb35_range = range.range - 0.32


    def update_uwb45_range_cb(self, range):
        '''
            Update range from UWB
        '''
        self.uwb45_range = range.range - 0.32


    def update_uwb75_range_cb(self, range):
        '''
            Update range from UWB
        '''
        self.uwb75_range = range.range - 0.32

    def velocity(self, x) :
        '''
            Use VIO to update position
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

            x = [x1, y1, x2, y2, x3, y3]

            y = [d23, d12, d13, d2, d3, d1]

        '''  
        # y = np.linalg.norm(x, axis=1)
        # print(f"y:{y.shape}")
        y = []

        t = datetime.datetime.now()

        for p in x :
            if args.fuse_group == 0 or args.fuse_group == 1:
                y.append([
                    np.linalg.norm(p[2:4] - p[4:6]),
                    np.linalg.norm(p[0:2] - p[2:4]),
                    np.linalg.norm(p[0:2] - p[4:6]),
                    np.linalg.norm(p[2:4]),
                    np.linalg.norm(p[4:6]),
                    np.linalg.norm(p[0:2])
                ])
            else:
                y.append([
                    np.linalg.norm(p[2:4] - p[4:6]),
                    np.linalg.norm(p[0:2] - p[2:4]),
                    np.linalg.norm(p[0:2] - p[4:6]),
                    np.linalg.norm(p[2:4]),
                    np.linalg.norm(p[4:6]),
                    np.linalg.norm(p[0:2]),
                    np.linalg.norm(p[0:2]),
                    np.linalg.norm(p[2:4]),
                    np.linalg.norm(p[4:6])
            ])

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

        plt.plot(self.true_relative_pose_turtle01[0], self.true_relative_pose_turtle01[1], 'x', c='red')
        plt.plot(self.pf.mean_state[0], self.pf.mean_state[1], 'x', c='green')
        plt.plot(self.pf.map_state[0], self.pf.map_state[1], 'x', c='orange')

        plt.plot(self.true_relative_pose_turtle03[0], self.true_relative_pose_turtle03[1], 'o', c='red')
        plt.plot(self.pf.mean_state[2], self.pf.mean_state[3], 'o', c='green')
        plt.plot(self.pf.map_state[2], self.pf.map_state[3], 'o', c='orange')

        plt.plot(self.true_relative_pose_turtle04[0], self.true_relative_pose_turtle04[1], '*', c='red')
        plt.plot(self.pf.mean_state[4], self.pf.mean_state[5], '*', c='green')
        plt.plot(self.pf.map_state[4], self.pf.map_state[5], '*', c='orange')

        plt.scatter(self.pf.transformed_particles[:,0], self.pf.transformed_particles[:,1], color="lightgray")
        plt.xlim(-9,9)
        plt.ylim(-9,9)

        # plt.legend()
        self.counter += 1
        plt.savefig(images_save_path + "/test{}.png".format(self.counter))
    
    def update_range_from_object_pose(self, object_end_pose_array, object_ori_pose_array):
        robots_relative_pose = np.array([-object_end_pose_array[0].position.x + object_ori_pose_array[0].position.x,
                                         -object_end_pose_array[0].position.y + object_ori_pose_array[0].position.y,
                                         -object_end_pose_array[0].position.z + object_ori_pose_array[0].position.z])
        return np.linalg.norm(robots_relative_pose)
    
    def update_filter(self) :
        '''
            Upadate particle filter
        '''
        # Get UWB range

        if args.fuse_group == 2:
            new_meas = np.array([self.uwb34_range, self.uwb37_range, self.uwb47_range, self.uwb35_range, self.uwb45_range, self.uwb75_range, self.uwb75_range, self.uwb35_range, self.uwb45_range])

        else:
            new_meas = np.array([self.uwb34_range, self.uwb37_range, self.uwb47_range, self.uwb35_range, self.uwb45_range, self.uwb75_range])

        # self.get_logger().info("Real dist: {}".format(np.linalg.norm(self.true_relative_pose_turtle01)))
        # self.get_logger().info("UWB34 Meas: {}， UWB37 Meas: {}, UWB47 Meas: {}".format(self.uwb34_range, self.uwb37_range, self.uwb47_range))

        # Use odometries
        self.particle_odom[0] = self.odom_turtle01.pose.pose.position.x - self.last_particle_odom_turtle01.pose.pose.position.x - (self.odom_ori.pose.pose.position.x - self.last_particle_odom_ori.pose.pose.position.x)
        self.particle_odom[1] = self.odom_turtle01.pose.pose.position.y - self.last_particle_odom_turtle01.pose.pose.position.y - (self.odom_ori.pose.pose.position.y - self.last_particle_odom_ori.pose.pose.position.y)
        self.particle_odom[2] = self.odom_turtle03.pose.pose.position.x - self.last_particle_odom_turtle03.pose.pose.position.x - (self.odom_ori.pose.pose.position.x - self.last_particle_odom_ori.pose.pose.position.x)
        self.particle_odom[3] = self.odom_turtle03.pose.pose.position.y - self.last_particle_odom_turtle03.pose.pose.position.y - (self.odom_ori.pose.pose.position.y - self.last_particle_odom_ori.pose.pose.position.y)
        self.particle_odom[4] = self.odom_turtle04.pose.pose.position.x - self.last_particle_odom_turtle04.pose.pose.position.x - (self.odom_ori.pose.pose.position.x - self.last_particle_odom_ori.pose.pose.position.x)
        self.particle_odom[5] = self.odom_turtle04.pose.pose.position.y - self.last_particle_odom_turtle04.pose.pose.position.y - (self.odom_ori.pose.pose.position.y - self.last_particle_odom_ori.pose.pose.position.y)

        # print(f"self.pf.d: {self.pf.d}")
        self.last_particle_odom_turtle01 = self.odom_turtle01
        self.last_particle_odom_turtle03 = self.odom_turtle03
        self.last_particle_odom_turtle04 = self.odom_turtle04
        self.last_particle_odom_ori = self.odom_ori


        if args.fuse_group == 2 or args.fuse_group == 1:
            if self.object_turtle01_pose_array.size != 0 and self.object_ori_pose_array.size != 0:
                new_vision_meas = self.update_range_from_object_pose(self.object_turtle01_pose_array, self.object_ori_pose_array)
                if math.fabs(new_vision_meas -  self.uwb75_range) < 0.5:
                    self.get_logger().info("vision: {}, uwb range: {}, truth: {}".format(new_vision_meas, self.uwb75_range, np.linalg.norm(self.true_relative_pose_turtle01)))
                    if args.fuse_group == 1:
                        new_meas[7] = new_vision_meas
                    else:
                        new_meas[11] = new_vision_meas
                        # new_meas[1] = new_vision_meas

            if self.object_turtle03_pose_array.size != 0 and self.object_ori_pose_array.size != 0:
                new_vision_meas = self.update_range_from_object_pose(self.object_turtle03_pose_array, self.object_ori_pose_array)
                if math.fabs(new_vision_meas -  self.uwb35_range) < 0.5:
                    self.get_logger().info("vision: {}, uwb range: {}, truth: {}".format(new_vision_meas, self.uwb75_range, np.linalg.norm(self.true_relative_pose_turtle03)))
                    if args.fuse_group == 1:
                        new_meas[9] = new_vision_meas
                    else:
                        new_meas[12] = new_vision_meas


            if self.object_turtle04_pose_array.size != 0 and self.object_ori_pose_array.size != 0:
                new_vision_meas = self.update_range_from_object_pose(self.object_turtle04_pose_array, self.object_ori_pose_array)
                if math.fabs(new_vision_meas -  self.uwb45_range) < 0.5:
                    self.get_logger().info("vision: {}, uwb range: {}, truth: {}".format(new_vision_meas, self.uwb45_range, np.linalg.norm(self.true_relative_pose_turtle04)))
                    if args.fuse_group == 1:
                        new_meas[10] = new_vision_meas
                    else:
                        new_meas[17] = new_vision_meas

        self.pf.update(observed=new_meas)
        

        # self.get_logger().info("Avg. PF mean: {}, std = {}".format(self.pf.mean_state, self.pf.cov_state))
        if self.pf.cov_state[0][0] > 0.5 or self.pf.cov_state[0][1] > 0.5 :
            self.get_logger().warn("PF covariance too high with covx={} and covy={}".format(self.pf.cov_state[0], self.pf.cov_state[1]))

        # print(np.linalg.norm(self.pf.mean_state))
        # print(np.linalg.norm(self.pf.map_state))
        # self.get_logger().info("Real relative position is {}".format(self.true_relative_pose))
        # self.get_logger().info("  -->  Estimated position is {}".format(self.pf.mean_state))
        # self.get_logger().info("  -->  Estimated position is {}\n".format(self.pf.map_state))
        # print("mean state: {}".format(self.pf.mean_state))
        self.relative_pos_turlte01 = PoseStamped()
        self.relative_pos_turtle03 = PoseStamped()
        self.relative_pos_turtle04 = PoseStamped()
        
        self.relative_pos_turlte01.header.frame_id = "base_link"
        self.relative_pos_turlte01.header.stamp = self.get_clock().now().to_msg()
        self.relative_pos_turlte01.pose.position.x = self.pf.mean_state[0]
        self.relative_pos_turlte01.pose.position.y = self.pf.mean_state[1]
        self.relative_pos_turlte01.pose.position.z = 0.0
        self.relative_pos_turlte01.pose.orientation = self.odom_turtle01.pose.pose.orientation


        self.relative_pos_turtle03.header.frame_id = "base_link"
        self.relative_pos_turtle03.header.stamp = self.get_clock().now().to_msg()
        self.relative_pos_turtle03.pose.position.x = self.pf.mean_state[2]
        self.relative_pos_turtle03.pose.position.y = self.pf.mean_state[3]
        self.relative_pos_turtle03.pose.position.z = 0.0
        self.relative_pos_turtle03.pose.orientation = self.odom_turtle01.pose.pose.orientation


        self.relative_pos_turtle04.header.frame_id = "base_link"
        self.relative_pos_turtle04.header.stamp = self.get_clock().now().to_msg()
        self.relative_pos_turtle04.pose.position.x = self.pf.mean_state[4]
        self.relative_pos_turtle04.pose.position.y = self.pf.mean_state[5]
        self.relative_pos_turtle04.pose.position.z = 0.0
        self.relative_pos_turtle04.pose.orientation = self.odom_turtle01.pose.pose.orientation

        self.publisher_turtle01_.publish(self.relative_pos_turlte01)
        self.publisher_turtle03_.publish(self.relative_pos_turtle03)
        self.publisher_turtle04_.publish(self.relative_pos_turtle04)
        
        ground_truth_turtle01 = np.linalg.norm(self.true_relative_pose_turtle01)
        ground_truth_turtle03 = np.linalg.norm(self.true_relative_pose_turtle03)
        ground_truth_turtle04 = np.linalg.norm(self.true_relative_pose_turtle04)
        ground_truth_turtle13 = np.linalg.norm(self.true_relative_pose_turtle01 - self.true_relative_pose_turtle03)
        ground_truth_turtle14 = np.linalg.norm(self.true_relative_pose_turtle01 - self.true_relative_pose_turtle04)
        ground_truth_turtle34 = np.linalg.norm(self.true_relative_pose_turtle03 - self.true_relative_pose_turtle04)
        uwb_range_estimation_turtle01 = np.linalg.norm([self.relative_pos_turlte01.pose.position.x, self.relative_pos_turlte01.pose.position.y, 0.0])
        uwb_range_estimation_turtle03 = np.linalg.norm([self.relative_pos_turtle03.pose.position.x, self.relative_pos_turtle03.pose.position.y, 0.0])
        uwb_range_estimation_turtle04 = np.linalg.norm([self.relative_pos_turtle04.pose.position.x, self.relative_pos_turtle04.pose.position.y, 0.0])
        uwb_range_estimation_turtle13 = np.linalg.norm([self.relative_pos_turlte01.pose.position.x - self.relative_pos_turtle03.pose.position.x, self.relative_pos_turlte01.pose.position.y - self.relative_pos_turtle03.pose.position.y, 0.0])
        uwb_range_estimation_turtle14 = np.linalg.norm([self.relative_pos_turlte01.pose.position.x - self.relative_pos_turtle04.pose.position.x, self.relative_pos_turlte01.pose.position.y - self.relative_pos_turtle04.pose.position.y, 0.0])
        uwb_range_estimation_turtle34 = np.linalg.norm([self.relative_pos_turtle03.pose.position.x - self.relative_pos_turtle04.pose.position.x, self.relative_pos_turtle03.pose.position.y - self.relative_pos_turtle04.pose.position.y, 0.0])
        self.ranges.append([ground_truth_turtle01, ground_truth_turtle03, ground_truth_turtle04, ground_truth_turtle13, ground_truth_turtle14, ground_truth_turtle34,
                            self.uwb75_range, self.uwb35_range, self.uwb45_range, self.uwb37_range, self.uwb47_range, self.uwb34_range,
                            uwb_range_estimation_turtle01, uwb_range_estimation_turtle03, uwb_range_estimation_turtle04, uwb_range_estimation_turtle13, uwb_range_estimation_turtle14, uwb_range_estimation_turtle34])
        # err_u_t15,err_u_t35,err_u_t45, err_u_t13,err_u_t14,err_u_t34,err_u_t15,err_u_t35,err_u_t45, err_u_t13,err_u_t14,err_u_t34 
        self.errors.append([self.uwb75_range - ground_truth_turtle01, self.uwb35_range - ground_truth_turtle03, self.uwb45_range - ground_truth_turtle04,
                            self.uwb37_range - ground_truth_turtle13, self.uwb47_range - ground_truth_turtle14, self.uwb34_range - ground_truth_turtle34,
                            uwb_range_estimation_turtle01 - ground_truth_turtle01, uwb_range_estimation_turtle03 - ground_truth_turtle03, uwb_range_estimation_turtle04 - ground_truth_turtle04,
                            uwb_range_estimation_turtle13 - ground_truth_turtle13, uwb_range_estimation_turtle14 - ground_truth_turtle14, uwb_range_estimation_turtle34 - ground_truth_turtle34])
        self.pos_estimation.append([self.true_relative_pose_turtle01[0], self.true_relative_pose_turtle01[1],
                                    self.true_relative_pose_turtle03[0], self.true_relative_pose_turtle03[1],
                                    self.true_relative_pose_turtle04[0], self.true_relative_pose_turtle04[1], 
                                    self.relative_pos_turlte01.pose.position.x, self.relative_pos_turlte01.pose.position.y,
                                    self.relative_pos_turtle03.pose.position.x, self.relative_pos_turtle03.pose.position.y,
                                    self.relative_pos_turtle04.pose.position.x, self.relative_pos_turtle04.pose.position.y])
        # if (time.perf_counter - self.plot_start) > 0.1 :
        # self.plot_particles()
        #     self.plot_start = time.perf_counter()
        # self.get_logger().info("here")

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
        np.savetxt(error_file, 
           self.errors,
        #    header="err_u_t15, err_u_t35, err_u_t45, err_u_t13, err_u_t14, err_u_t34, err_eu_t15, err_eu_t35, err_eu_t45, err_eu_t13, err_eu_t14, err_eu_t34",
           delimiter =", ", 
           fmt ='% s')

        np.savetxt(range_file, 
           self.ranges,
        #    header="range_err_u_t15, range_err_u_t35, range_err_u_t45, range_err_u_t13, range_err_u_t14, range_err_u_t34, range_err_eu_t15, range_err_eu_t35, range_err_eu_t45, range_err_eu_t13, range_err_eu_t14, range_err_eu_t34",
           delimiter =", ", 
           fmt ='% s')

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