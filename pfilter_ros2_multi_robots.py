#!/usr/bin/env python
# using the mocap to simulate the odometry data
import rclpy
from rclpy.node import Node

import os 
import sys
import time
import math
import argparse
import shutil

from std_msgs.msg               import Float64
from geometry_msgs.msg          import PoseStamped
from geometry_msgs.msg          import PoseWithCovarianceStamped
from geometry_msgs.msg          import PoseArray
from sensor_msgs.msg            import Range
from nav_msgs.msg               import Odometry
from rclpy.clock                import Clock
from rclpy.duration             import Duration
from pfilter                    import ParticleFilter, squared_error
from depthai_ros_msgs.msg       import SpatialDetectionArray, SpatialDetection
from rclpy.qos                  import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np
import matplotlib.pyplot as plt


# /turtle01/color/yolov4_Spatial_detections
# /turtle01/odom
# /turtle03/color/yolov4_Spatial_detections     
# /turtle03/odom
# /turtle04/color/yolov4_Spatial_detections
# /turtle04/odom
# /uwb/tof/n_3/n_4/distance
# /uwb/tof/n_3/n_7/distance
# /uwb/tof/n_4/n_7/distance
# /vrpn_client_node/chair_final/pose
# /vrpn_client_node/turtlebot1_cap/pose
# /vrpn_client_node/turtlebot3_cap/pose
# /vrpn_client_node/turtlebot4_cap/pose


def parse_args():
    parser = argparse.ArgumentParser(description='Options to control relative localization with only UWB, assisit with Vision, and all if vision available')
    parser.add_argument('fuse_group', type=int, help='0: only UWB in PF, 1: with vision replace new measurement, 2: uwb and vision together')
    args = parser.parse_args()
    return args

args = parse_args()
print(args.fuse_group)

error_uwb_ranges = '../errors/error_uwb.csv'

error_file_name = '../errors/error_u.csv'
images_save_path = '../images/images_u/'
pos_path = '../pos/pos_u.csv'
pos_ground = '../pos/pos_g.csv'

if args.fuse_group == 1:
    error_file_name = "../errors/error_u_v.csv"
    images_save_path = '../images/images_u_v/'
    pos_path = '../pos/pos_u_v.csv'

if args.fuse_group == 2:
    error_file_name = "../errors/error_uv.csv"
    images_save_path = '../images/images_uv/'
    pos_path = '../pos/pos_uv.csv'


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

        self.counter = 0

        # Define QoS profile for odom and UWB subscribers
        self.qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Particle filter params
        self.declare_parameters(
            namespace = '',
            parameters=[
                ("weights_sigma", 1.2),
                ("num_particles", 200),
                ("uwb_noise", 0.05),
                ("resample_proportion", 0.1),
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
        self.prior_fn = lambda n: np.random.uniform(-2,2,(n,2))
        self.pf = ParticleFilter(
            prior_fn =              self.prior_fn, 
            observe_fn =            self.calc_hypothesis,  
            dynamics_fn =           self.velocity, #lambda x: x,
            n_particles =           self.num_particles, 
            noise_fn =              self.add_noise, #lambda x: x + np.random.normal(0, noise, x.shape),
            weight_fn =             self.calc_weights, #lambda x, y : squared_error(x, y, sigma=2),
            resample_proportion =   self.resample_proportion
        )

        # self.odom = PoseStamped()#Odometry()
        self.last_particle_odom = Odometry()
        self.last_particle_odom_ori = Odometry()
        self.last_particle_pose = PoseStamped()
        self.last_particle_pose_ori = PoseStamped()
        self.pose_ori = PoseStamped()
        self.pose_end = PoseStamped()
        self.odom_ori = Odometry()
        self.odom_end = Odometry()
        self.particle_odom = np.array([0.01,0.01])
        self.object_ori_pose_array = np.array([])
        self.object_end_pose_array = np.array([])

        self.get_logger().info("Subscribing to topics")
        self.pose_ori_sub = self.create_subscription(PoseStamped, "/vrpn_client_node/turtlebot3_cap/pose",  self.update_odom_ori_cb, 10)
        self.pose_end_sub = self.create_subscription(PoseStamped, "/vrpn_client_node/turtlebot1_cap/pose",  self.update_odom_end_cb, 10)
        self.odom_end_sub = self.create_subscription(Odometry, "/turtle01/odom",  self.update_odometry_end_cb, qos_profile=self.qos)
        self.odom_ori_sub = self.create_subscription(Odometry, "/turtle03/odom",  self.update_odometry_ori_cb, qos_profile=self.qos)
        self.object_ori_sub = self.create_subscription(SpatialDetectionArray, "/turtle03/color/yolov4_Spatial_detections",  self.update_object_ori_cb, 10)
        self.object_end_sub = self.create_subscription(SpatialDetectionArray, "/turtle01/color/yolov4_Spatial_detections",  self.update_object_end_cb, 10)
        self.uwb_34_range_sub = self.create_subscription(Range, "/uwb/tof/n_3/n_4/distance", self.update_uwb34_range_cb, 10)
        self.uwb_37_range_sub = self.create_subscription(Range, "/uwb/tof/n_3/n_7/distance", self.update_uwb37_range_cb, 10)
        self.uwb_47_range_sub = self.create_subscription(Range, "/uwb/tof/n_4/n_7/distance", self.update_uwb47_range_cb, 10)
        self.publisher_ = self.create_publisher(PoseStamped, '/pf_pose', 10)


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
        self.true_relative_pose = np.array([.0,.0])
        self.pos = PoseWithCovarianceStamped()
        self.pos.header.stamp = Clock().now().to_msg()
        self.pos.pose.pose.orientation.w = 1.0

        self.errors = []
        self.errors_uwb_range = []
        self.pos_ground = []
        self.pos_estimation = []
        

    def update_odometry_ori_cb(self, odom):
        self.odom_ori = odom
        # self.get_logger().info("ori odom callback.")

    def update_odometry_end_cb(self, odom):
        
        self.odom_end = odom


    def update_odom_ori_cb(self, pose) :
        '''
            Update pose from VIO
        '''
        self.pose_ori = pose
        # self.get_logger().info("end odom callback")

    def update_odom_end_cb(self, pose) :
        '''
            Update pose from VIO
        '''
        self.pose_end = pose

        end_pos = np.array([self.pose_end.pose.position.x, self.pose_end.pose.position.y])
        ori_pos = np.array([self.pose_ori.pose.position.x, self.pose_ori.pose.position.y]) 
        self.true_relative_pose = end_pos - ori_pos

        # only simulate uwb range
        # self.uwb_range = np.linalg.norm(end_pos - ori_pos) + np.random.normal(0, 0.15)
        
        # self.get_logger().info("odom end cb")

    def update_object_ori_cb(self, pose_array):
        self.object_ori_pose_array = np.array(pose_array.detections)
        # if self.object_ori_pose_array.size != 0:
        #     self.get_logger().info("orignal objects number:{}".format(self.object_ori_pose_array.size))

    
    def update_object_end_cb(self, pose_array):
        self.object_end_pose_array = np.array(pose_array.detections)
        # if self.object_end_pose_array.size != 0:
        #     self.get_logger().info("end objects number:{}".format(self.object_end_pose_array.size))
        
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
        '''  
        # print(np.linalg.norm(x[0], axis=1))
        # print(np.linalg.norm(x[1], axis=1))    
        # print(x)       
        # y = np.array([
        #    np.linalg.norm(x[0], axis=1),
        #    np.linalg.norm(x[1], axis=1)
        # ])
        y = np.linalg.norm(x, axis=1)
        # print(y)
        return y

    def calc_weights(self, hypotheses, observations) :
        '''
            Calculate particle weights based on error
        '''
        w = squared_error(hypotheses, observations, sigma=self.weights_sigma)
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

        plt.plot(self.true_relative_pose[0], self.true_relative_pose[1], 'x', c='red')
        plt.plot(self.pf.mean_state[0], self.pf.mean_state[1], 'x', c='green')
        plt.plot(self.pf.map_state[0], self.pf.map_state[1], 'x', c='orange')

        plt.scatter(self.pf.transformed_particles[:,0], self.pf.transformed_particles[:,1], color="lightgray")
        plt.xlim(-6,6)
        plt.ylim(-6,6)

        # plt.legend()
        self.counter += 1
        plt.savefig(images_save_path + "/test{}.png".format(self.counter))
    
    def update_range_from_object_pose(self):
        robots_relative_pose = np.array([-self.object_end_pose_array[0].position.x + self.object_ori_pose_array[0].position.x,
                                         -self.object_end_pose_array[0].position.y + self.object_ori_pose_array[0].position.y,
                                         0.0])
        return np.linalg.norm(robots_relative_pose)
    
    def update_filter(self) :
        '''
            Upadate particle filter
        '''
        # Get UWB range

        # new_meas = self.uwb37_range
        # new_meas = np.array([self.uwb34_range, self.uwb37_range])
        # new_meas = np.array([self.uwb34_range, self.uwb37_range, self.uwb47_range])

        if args.fuse_group == 2:
            new_meas = np.array([self.uwb34_range, self.uwb37_range, self.uwb47_range, self.uwb34_range])
        else:
            new_meas = np.array([self.uwb34_range, self.uwb37_range, self.uwb47_range])

        self.get_logger().info("Real dist: {}".format(np.linalg.norm(self.true_relative_pose)))
        self.get_logger().info("UWB34 Meas: {}， UWB37 Meas: {}, UWB47 Meas: {}".format(self.uwb34_range, self.uwb37_range, self.uwb47_range))

        # Calculate odom from last PF uptdate
        # print(self.pose_end.pose.position.x)
        # print(self.last_particle_pose.pose.position.x)
        
        # Use optitrack
        self.particle_odom[0] = self.pose_end.pose.position.x - self.last_particle_pose.pose.position.x - (self.pose_ori.pose.position.x - self.last_particle_pose_ori.pose.position.x)
        self.particle_odom[1] = self.pose_end.pose.position.y - self.last_particle_pose.pose.position.y - (self.pose_ori.pose.position.y - self.last_particle_pose_ori.pose.position.y)

        
        # self.particle_odom[0] = self.odom_end.pose.pose.position.x - self.last_particle_odom.pose.pose.position.x - (self.odom_ori.pose.pose.position.x - self.last_particle_odom_ori.pose.pose.position.x)
        # self.particle_odom[1] = self.odom_end.pose.pose.position.y - self.last_particle_odom.pose.pose.position.y - (self.odom_ori.pose.pose.position.y - self.last_particle_odom_ori.pose.pose.position.y)

        # print("Pose end: {}".format(self.pose_end))
        # print("Pose ori: {}".format(self.pose_ori))
        # print("Odom: {}".format(self.particle_odom))
 
        # self.last_particle_odom = self.odom_end
        # self.last_particle_odom_ori = self.odom_ori

        self.last_particle_pose = self.pose_end
        self.last_particle_pose_ori = self.pose_ori

        # self.get_logger().info("vision: {}, uwb range: {}".format(new_vision_meas, self.uwb_range))

        # if args.fuse_group == 2 or args.fuse_group == 1:
        #     if self.object_end_pose_array.size != 0  and self.object_ori_pose_array.size  != 0:
        #         new_vision_meas = self.update_range_from_object_pose()
        #         if math.fabs(new_vision_meas -  self.uwb37_range) < 0.5:
        #             self.get_logger().info("vision: {}, uwb range: {}, truth: {}".format(new_vision_meas, self.uwb37_range, np.linalg.norm(self.true_relative_pose)))
        #             if args.fuse_group == 1:
        #                 new_meas = new_vision_meas
        #             else:
        #                 new_meas[3] = new_vision_meas

        self.pf.update(observed=new_meas)

        # self.get_logger().info("Avg. PF mean: {}, std = {}".format(self.pf.mean_state, self.pf.cov_state))
        if self.pf.cov_state[0][0] > 0.5 or self.pf.cov_state[0][1] > 0.5 :
            self.get_logger().warn("PF covariance too high with covx={} and covy={}".format(self.pf.cov_state[0], self.pf.cov_state[1]))

        # print(np.linalg.norm(self.pf.mean_state))
        # print(np.linalg.norm(self.pf.map_state))
        # self.get_logger().info("Real relative position is {}".format(self.true_relative_pose))
        # self.get_logger().info("  -->  Estimated position is {}".format(self.pf.mean_state))
        # self.get_logger().info("  -->  Estimated position is {}\n".format(self.pf.map_state))

        self.relative_pos = PoseStamped()
        self.relative_pos.header.frame_id = "base_link"
        self.relative_pos.header.stamp = self.get_clock().now().to_msg()
        self.relative_pos.pose.position.x = self.pf.mean_state[0]
        self.relative_pos.pose.position.y = self.pf.mean_state[1]
        self.relative_pos.pose.position.z = 0.0
        self.relative_pos.pose.orientation = self.odom_end.pose.pose.orientation
        self.publisher_.publish(self.relative_pos)
        
        ground_truth = np.linalg.norm(self.true_relative_pose)
        uwb_range_estimation = np.linalg.norm([self.relative_pos.pose.position.x, self.relative_pos.pose.position.y, 0.0])
        self.errors.append(uwb_range_estimation - ground_truth)
        self.errors_uwb_range.append(self.uwb37_range - ground_truth)
        self.pos_ground.append([self.true_relative_pose[0], self.true_relative_pose[1]])
        self.pos_estimation.append([self.true_relative_pose[0], self.true_relative_pose[1], self.relative_pos.pose.position.x, self.relative_pos.pose.position.y])
        # if (time.perf_counter - self.plot_start) > 0.1 :
        self.plot_particles()
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
        np.savetxt(error_file_name, 
           self.errors,
           delimiter =", ", 
           fmt ='% s')

        np.savetxt(error_uwb_ranges, 
           self.errors_uwb_range,
           delimiter =", ", 
           fmt ='% s')

        np.savetxt(pos_path, 
           self.pos_estimation,
           delimiter =", ", 
           fmt ='% s')       

        np.savetxt(pos_ground, 
           self.pos_ground,
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
                filter_timer = filter.create_timer(1/4.0, filter.update_filter)
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
