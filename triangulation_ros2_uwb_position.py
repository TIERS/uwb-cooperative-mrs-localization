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


# err_folder = "./results/triangulation/errors/errors_u/"
pos_folder = "./results/triangulation/pos/pos_tri/"
# range_folder = "./results/triangulation/ranges/ranges_u/"
# error_file = err_folder + "error_{}.csv".format(args.round)
pos_file = pos_folder + 'pos_{}.csv'.format(args.round)
# range_file = range_folder + "range_{}.csv".format(args.round)
# images_save_path = './results/triangulation/images/images_u/images_u_{}/'.format(args.round)


# if not os.path.exists(err_folder):
#     os.makedirs(err_folder)

if not os.path.exists(pos_folder):
    os.makedirs(pos_folder)

# if not os.path.exists(range_folder):
#     os.makedirs(range_folder)

# if not os.path.exists(images_save_path):
#     os.makedirs(images_save_path)

class UWBTriangulation(Node) :
    '''
        ROS Node that estimates relative position of two robots using odom and single uwb range.
    '''

    def __init__(self) :

        # Init node
        rclpy.init()
        self.node = rclpy.create_node('UWB_Triangulation_Positioning')
        # Define QoS profile for odom and UWB subscribers
        self.qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Calibrated UWB offset
        self.uwb_range_offset = self.node.get_parameter_or("uwb_range_offset", 0.2)
        # Max time delay for the reception of messages
        self.max_pos_delay = self.node.get_parameter_or("max_pos_delay", 0.2)
        # Minimum number for the node ID allowed
        self.uwb_min_id = self.node.get_parameter_or("uwb_min_id", 1)
        # Control printing of information

        self.pose_ori = PoseStamped()
        self.pose_turtle01 = PoseStamped()
        self.pose_turtle03 = PoseStamped()
        self.pose_turtle04 = PoseStamped()
        

        self.node.get_logger().info("Subscribing to topics")
        # subscribe to optitrack pose
        self.pose_ori_sub      = self.node.create_subscription(PoseStamped, "/vrpn_client_node/turtlebot5_cap/pose",  self.update_odom_ori_cb, 10)
        self.pose_turtle01_sub = self.node.create_subscription(PoseStamped, "/vrpn_client_node/turtlebot1_cap/pose",  self.update_turtle01_opti_pos_cb, 10)
        self.pose_turtle03_sub = self.node.create_subscription(PoseStamped, "/vrpn_client_node/turtlebot3_cap/pose",  self.update_turtle03_opti_pos_cb, 10)
        self.pose_turtle04_sub = self.node.create_subscription(PoseStamped, "/vrpn_client_node/turtlebot4_cap/pose",  self.update_turtle04_opti_pos_cb, 10)

        # subscribe to uwb ranges 
        self.uwb_34_range_sub = self.node.create_subscription(Range, "/uwb/tof/n_3/n_4/distance", self.update_uwb34_range_cb, 10)
        self.uwb_37_range_sub = self.node.create_subscription(Range, "/uwb/tof/n_3/n_7/distance", self.update_uwb37_range_cb, 10)
        self.uwb_47_range_sub = self.node.create_subscription(Range, "/uwb/tof/n_4/n_7/distance", self.update_uwb47_range_cb, 10)
        self.uwb_35_range_sub = self.node.create_subscription(Range, "/uwb/tof/n_3/n_5/distance", self.update_uwb35_range_cb, 10)
        self.uwb_45_range_sub = self.node.create_subscription(Range, "/uwb/tof/n_4/n_5/distance", self.update_uwb45_range_cb, 10)
        self.uwb_75_range_sub = self.node.create_subscription(Range, "/uwb/tof/n_7/n_5/distance", self.update_uwb75_range_cb, 10)

        self.publisher_turtle01_ = self.node.create_publisher(PoseStamped, '/pf_turtle01_pose', 10)
        self.publisher_turtle03_ = self.node.create_publisher(PoseStamped, '/pf_turtle03_pose', 10)
        self.publisher_turtle04_ = self.node.create_publisher(PoseStamped, '/pf_turtle04_pose', 10)

        # Calculate relative pose
        self.relative_pos = PoseStamped()

        self.uwb37_range = 0.0
        self.uwb34_range = 0.0
        self.uwb47_range = 0.0
        self.uwb35_range = 0.0
        self.uwb45_range = 0.0
        self.uwb75_range = 0.0
        
        self.uwb3f_range = 0.0
        self.uwb4f_range = 0.0
        self.uwb5f_range = 0.0
        self.uwb7f_range = 0.0
        
        self.true_relative_pose_turtle01 = np.array([.0,.0])
        self.true_relative_pose_turtle03 = np.array([.0,.0])
        self.true_relative_pose_turtle04 = np.array([.0,.0])
        self.true_relative_pose_fake     = np.array([2.0, 3.0])

        self.pos_estimation = []
        

    def update_odom_ori_cb(self, pose) :
        '''
            Update pose from VIO
        '''
        self.pose_ori = pose
        # self.node.get_logger().info("end odom callback")
        

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
        
        # self.node.get_logger().info("odom end cb")

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
        
        # self.node.get_logger().info("odom end cb")

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
        
        # self.node.get_logger().info("odom end cb")
        
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

    
    
    def calculate_relative_poses(self) :
        '''
            Calculates relative poses of nodes doing TOF
        '''
        # fake static robot uwb ranges
        self.uwb7f_range = np.linalg.norm(np.array([self.true_relative_pose_turtle01[0] - self.true_relative_pose_fake[0],
                                self.true_relative_pose_turtle01[1] - self.true_relative_pose_fake[1]])) + np.random.normal(0, 0.32)
        self.uwb4f_range = np.linalg.norm(np.array([self.true_relative_pose_turtle04[0] - self.true_relative_pose_fake[0],
                            self.true_relative_pose_turtle04[1] - self.true_relative_pose_fake[1]]))  + np.random.normal(0, 0.32)                           
        self.uwb3f_range = np.linalg.norm(np.array([self.true_relative_pose_turtle03[0] - self.true_relative_pose_fake[0],
                            self.true_relative_pose_turtle03[1] - self.true_relative_pose_fake[1]]))  + np.random.normal(0, 0.32)
        self.uwb5f_range = np.linalg.norm(self.true_relative_pose_fake) + np.random.normal(0, 0.32)

        # uwb_ranges = np.array([
        #     self.uwb5f_range, self.uwb7f_range, self.uwb3f_range, self.uwb4f_range,
        #     self.uwb37, self.uwb47_range, self.uwb34_range
        # ])
        positions = [np.zeros(2) for _ in range(5)] 
        positions[0] = np.array([0, 0])
        positions[1] = np.array([self.uwb5f_range, 0])

        iterative_positions = False
        try:
            arg1 = (self.uwb5f_range**2 + self.uwb75_range**2 - self.uwb7f_range**2) / (2*self.uwb5f_range*self.uwb75_range)
            if(arg1<-1.0):
                arg1 = -1.0
            elif (arg1 > 1.0):
                arg1 = 1.0
            theta = math.acos( arg1 )
            # thetas.append(theta)
            x = math.cos(theta)*self.uwb75_range
            y = math.sin(theta)*self.uwb75_range
            positions[2]=np.array([x,y]) 
 
            arg1 = (self.uwb5f_range**2 + self.uwb35_range**2 - self.uwb3f_range**2) / (2*self.uwb5f_range*self.uwb35_range)
            if(arg1<-1.0):
                arg1 = -1.0
            elif (arg1 > 1.0):
                arg1 = 1.0
            theta = math.acos( arg1 )
            # thetas.append(theta)
            x = math.cos(theta)*self.uwb35_range
            y = math.sin(theta)*self.uwb35_range
            positions[3]=np.array([x,y]) 

            arg1 = (self.uwb5f_range**2 + self.uwb45_range**2 - self.uwb4f_range**2) / (2*self.uwb5f_range*self.uwb45_range)
            if(arg1<-1.0):
                arg1 = -1.0
            elif (arg1 > 1.0):
                arg1 = 1.0
            theta = math.acos( arg1 )
            # thetas.append(theta)
            x = math.cos(theta)*self.uwb45_range
            y = math.sin(theta)*self.uwb45_range
            positions[4]=np.array([x,y]) 
            # print(positions)
            self.pos_estimation.append([self.true_relative_pose_turtle01[0], self.true_relative_pose_turtle01[1],
                            self.true_relative_pose_turtle03[0], self.true_relative_pose_turtle03[1],
                            self.true_relative_pose_turtle04[0], self.true_relative_pose_turtle04[1], 
                            positions[2][0], -positions[2][1],
                            positions[3][0], -positions[3][1],
                            positions[4][0], -positions[4][1]])

        except ValueError:
            self.node.get_logger().error("math domain error")

    def run(self) :
        '''
            Create timer to update positions.
        '''

        # Set filter update timer at 6 Hz
        time.sleep(1)
        rospy_check_rate = self.node.create_rate(10)
        # self.tof_timer = rospy.Timer(rclpy.Duration(0.2), self.calculate_relative_poses)
        self.tof_timer = self.node.create_timer(0.2, self.calculate_relative_poses)
        
        self.node.get_logger().info("Starting ToF Position Calculations...")
        try:
            rclpy.spin(self.node)
        except KeyboardInterrupt :
            self.node.get_logger().error('Keyboard Interrupt detected!')

        # self.tof_timer.shutdown()

    
    def __del__(self):
        # body of destructor
        self.node.get_logger().info("triangulation ends and Saving Results.")

        np.savetxt(pos_file, 
           self.pos_estimation,
           delimiter =", ", 
           fmt ='% s')       


def main(args=None):
    pos_cal = UWBTriangulation()
    pos_cal.run()
    

if __name__ == '__main__':
    main()
