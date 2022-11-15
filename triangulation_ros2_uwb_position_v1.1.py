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
from itertools import combinations

turtles         = ["5", "1"  , "3", "4"]
uwbs            = ["5", "7"  , "3", "4"]
uwb_pair        = [(3,7), (4,7), (2,7), (3,4), (2,3), (2,4), (7,5), (3,5),(4,5), (2,5)]

#  get parameters from terminal
def parse_args():
    parser = argparse.ArgumentParser(description='Options for triangulations to calculate the relative position of robots based on UWB rangessss')
    parser.add_argument('--poses_save', type=bool, default=False, help='choose to save the estimated poses with triangulation')
    parser.add_argument('--round', type=int, default=0, help='indicate which round the pf will run on a recorded data')
    args = parser.parse_args()
    return args

args = parse_args()

# Build folder to save results from different fusion combinations
if args.poses_save:
    pos_folder = "./results/triangulation/pos/pos_tri/"
    pos_file = pos_folder + 'pos_{}.csv'.format(args.round)
    if not os.path.exists(pos_folder):
        os.makedirs(pos_folder)

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

        # all varibles 
        self.uwb_ranges             = [0.0 for _ in uwb_pair]
        self.turtles_mocaps         = [np.zeros(2) for _ in turtles]
        self.turtles_odoms          = [Odometry() for _ in turtles]
        self.true_relative_poses    = [np.zeros(2) for _ in range(1,len(turtles))]
        self.relative_poses         = [np.zeros(2) for _ in range(1,len(turtles))]
        self.pos_estimation         = []

        self.node.get_logger().info("Subscribing to topics")
        # subscribe to uwb ranges 
        self.uwb_subs = [
            self.node.create_subscription(Range, "/uwb/tof/n_{}/n_{}/distance".format(p[0], p[1]), 
            self.create_uwb_ranges_cb(i),10) for i, p in enumerate(uwb_pair)]
        self.node.get_logger().info("{} UWB ranges received!".format(len(self.uwb_ranges)))

        # subscribe to optitrack mocap poses
        self.mocap_subs = [
            self.node.create_subscription(PoseStamped, "/vrpn_client_node/turtlebot{}_cap/pose".format(t), 
            self.create_mocap_pose_cb(i), 10) for i, t in enumerate(turtles)]
        self.node.get_logger().info("{} Mocaps poses received!".format(len(self.turtles_mocaps)))
        
        # subscribe to odometries
        self.odom_subs = [
            self.node.create_subscription(Odometry, "/cali/turtle0{}/odom".format(t), 
            self.create_odom_cb(i),10) for i, t in enumerate(turtles)]
        self.node.get_logger().info("{} odom poses received!".format(len(self.turtles_mocaps)))

        # pf relative poses publishers
        self.relative_pose_publishers = [self.node.create_publisher(PoseStamped, '/tri_turtle0{}_pose'.format(t), 10) for t in turtles[1:]]

        # Wait to get some odometry
        sys.stdout.write("Waiting for odom data...\n")
        for _ in range(100) :
            if self.turtles_odoms[0].header.stamp :
                break
            sys.stdout.write("..")
            sys.stdout.flush()
            time.sleep(0.1)
        self.node.get_logger().info("Odometry locked. Current odom\n")

        # Responder positions
        self.node.get_logger().info("UWB PF initialized. Estimating position from UWB and odom.")

        self.pos_estimation = []
        
    def create_uwb_ranges_cb(self, i):
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

    def relative_pose_cal(self, origin, ends, relative_poses):
        for inx, end in enumerate(ends):
            relative_poses[inx] = end - origin    

    
    def calculate_relative_poses(self) :
        '''
            Calculates relative poses of nodes doing TOF
        '''
        positions = [np.zeros(2) for _ in range(5)] 
        positions[0] = np.array([0, 0])
        positions[1] = np.array([self.uwb_ranges[9], 0])
        # uwb_pair   = [(3,7), (4,7), (2,7), (3,4), (2,3), (2,4), (7,5), (3,5),(4,5), (2,5)]
        try:
            arg1 = (self.uwb_ranges[9]**2 + self.uwb_ranges[6]**2 - self.uwb_ranges[2]**2) / (2*self.uwb_ranges[9]*self.uwb_ranges[6])
            theta = math.acos( arg1 )
            # thetas.append(theta)
            x = math.cos(theta)*self.uwb_ranges[6]
            y = math.sin(theta)*self.uwb_ranges[6]
            positions[2]=np.array([x,y]) 
 
            arg1 = (self.uwb_ranges[9]**2 + self.uwb_ranges[7]**2 - self.uwb_ranges[4]**2) / (2*self.uwb_ranges[9]*self.uwb_ranges[7])
            theta = math.acos( arg1 )
            # thetas.append(theta)
            x = math.cos(theta)*self.uwb_ranges[7]
            y = math.sin(theta)*self.uwb_ranges[7]
            positions[3]=np.array([x,y]) 

            arg1 = (self.uwb_ranges[9]**2 + self.uwb_ranges[8]**2 - self.uwb_ranges[5]**2) / (2*self.uwb_ranges[9]*self.uwb_ranges[8])
            theta = math.acos( arg1 )
            # thetas.append(theta)
            x = math.cos(theta)*self.uwb_ranges[8]
            y = math.sin(theta)*self.uwb_ranges[8]
            positions[4]=np.array([x,y]) 
            # print(positions)

            self.relative_pose_cal(self.turtles_mocaps[0], self.turtles_mocaps[1:], self.true_relative_poses)
            self.pos_estimation.append([self.true_relative_poses[0][0], self.true_relative_poses[0][1],
                            self.true_relative_poses[1][0], self.true_relative_poses[1][1],
                            self.true_relative_poses[2][0], self.true_relative_poses[2][1], 
                            -positions[2][1],     -positions[2][0], 
                            -positions[3][1],     -positions[3][0], 
                            -positions[4][1],     -positions[4][0], ])
            # publish pf relative pose
            for i in range(len(turtles[1:])):
                relative_pose = PoseStamped()
                relative_pose.header.frame_id = "base_link"
                relative_pose.header.stamp = self.node.get_clock().now().to_msg()
                relative_pose.pose.position.x = -positions[(i+2)][1]
                relative_pose.pose.position.y = -positions[(i+2)][0]
                relative_pose.pose.position.z = 0.0
                relative_pose.pose.orientation = self.turtles_odoms[i].pose.pose.orientation
                self.relative_pose_publishers[i].publish(relative_pose) 

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
