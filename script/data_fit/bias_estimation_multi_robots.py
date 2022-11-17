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
from depthai_ros_msgs.msg       import SpatialDetectionArray, SpatialDetection
from rclpy.qos                  import QoSProfile, ReliabilityPolicy, HistoryPolicy
from pfilter                    import ParticleFilter, squared_error
from scipy.spatial.transform    import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


turtles         = ["5", "1", "3", "4", "2"]
# [('5', '1'), ('5', '3'), ('5', '4'), ('1', '3'), ('1', '4'), ('3', '4')]
uwbs            = ["5", "7", "3", "4", "2"]
uwb_pair        = [(3,7), (4,7), (2,7), (3,4), (2,3), (2,4), (7,5), (3,5), (4,5), (2,5)]
uwb_odoms       = [(2,1), (3,1), (4,1), (2,3), (4,2), (4,3), (2,0), (2,0), (3,0), (4,0)]

class UWBBiasEstimation(Node) :
    '''
        ROS Node that estimates relative position of two robots using odom and single uwb range.
    '''
    def __init__(self) :
        # Init node
        super().__init__('uwb_bias_estimation')
        # Define QoS profile for odom and UWB subscribers
        self.qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # all varibles
        # self.count=0 
        self.uwb_ranges             = [0.0 for _ in uwb_pair]
        self.turtles_mocaps         = [np.zeros(2) for _ in turtles]
        self.turtles_odoms          = [Odometry() for _ in turtles]
        self.turtles_odoms_flag     = [False for _ in turtles]
        self.poly_coefficient       = []

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
            self.create_subscription(Odometry, "/turtle0{}/odom".format(t), 
            self.create_odom_cb(i), qos_profile=self.qos) for i, t in enumerate(turtles)]
        self.get_logger().info("{} odom poses received!".format(len(self.turtles_odoms)))
    
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
        # print("========== {} ==========".format(self.turtles_odoms_flag[i]))
        self.turtles_odoms_flag[i] = True
        self.turtles_odoms[i] = odom

    def relative_yaws(self, odom):
        r = R.from_quat([odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w])
        yaw, _, _ = r.as_euler('zxy', degrees=True)
        return yaw
        
    def bias_estimation(self):
        '''
            Bias estimation
        '''
        print(f"============{self.turtles_odoms_flag}================")
        if all(self.turtles_odoms_flag):
            print("--------------------")
            errors = []
            yaws = []
            for inx, uo in enumerate(uwb_odoms):
                mocap_range = np.linalg.norm(self.turtles_mocaps[uo[1]] - self.turtles_mocaps[uo[0]])
                errors.append(self.uwb_ranges[inx] - mocap_range)
                yaws.append(self.relative_yaws(self.turles_odoms[uo[1]] - self.relative_yaws(self.turtles_odoms[uo[0]])))
            np.savez('data/4robots_data_01.npz',        
                    uwb_range_3_7_np = errors[0],
                    uwb_range_4_7_np = errors[1] ,
                    uwb_range_2_7_np = errors[2],
                    uwb_range_3_4_np = errors[3],
                    uwb_range_2_3_np = errors[4],
                    uwb_range_2_4_np = errors[5],
                    uwb_range_7_5_np = errors[6],
                    uwb_range_3_5_np = errors[7],
                    uwb_range_4_5_np = errors[8],
                    uwb_range_2_5_np = errors[9],

                    yaw_3_7_np = yaws[0],
                    yaw_4_7_np = yaws[1],
                    yaw_2_7_np = yaws[2],
                    yaw_3_4_np = yaws[3],
                    yaw_2_3_np = yaws[4],
                    yaw_2_4_np = yaws[5],
                    yaw_7_5_np = yaws[6],
                    yaw_3_5_np = yaws[7],
                    yaw_4_5_np = yaws[8],
                    yaw_2_5_np = yaws[9],
                    # optitrack_turtle01_orientation_np = optitrack_turtle01_orientation_np,
                    # optitrack_turtle03_orientation_np = optitrack_turtle03_orientation_np,
                    # optitrack_turtle04_orientation_np = optitrack_turtle04_orientation_np,
                    # optitrack_turtle05_orientation_np = optitrack_turtle05_orientation_np,
                    # optitrack_turtle01_pose_np = optitrack_turtle01_pose_np,
                    # optitrack_turtle03_pose_np = optitrack_turtle03_pose_np,
                    # optitrack_turtle04_pose_np = optitrack_turtle04_pose_np,
                    # optitrack_turtle05_pose_np = optitrack_turtle05_pose_np
            )

            print("Data saved {}".format(self.count))
            self.count+=1
            
# turtles         = ["5", "1", "3", "4", "2"]
# # [('5', '1'), ('5', '3'), ('5', '4'), ('1', '3'), ('1', '4'), ('3', '4')]
# uwbs            = ["5", "7", "3", "4", "2"]
# uwb_pair        = [(3,7), (4,7), (2,7), (3,4), (2,3), (2,4), (7,5), (3,5), (4,5), (2,5)]
# uwb_odoms       = [(2,1), (3,1), (4,1), (2,3), (4,2), (4,3), (2,0), (2,0), (3,0), (4,0)]            


def main(args=None):
    rclpy.init(args=args)
    filter = UWBBiasEstimation()

    time.sleep(1)
    # Start calculating relative positions
    
    filter.get_logger().info("Starting UWB bias poly fit ...")
    filter_timer = filter.create_timer(1/20.0, filter.bias_estimation)
    try:
        try:
            while rclpy.ok() :
                rclpy.spin(filter)             
        except KeyboardInterrupt :
            filter.get_logger().error('Keyboard Interrupt detected! Trying to stop filter node!')
    except Exception as e:
        filter.destroy_node()
        filter.get_logger().info("UWB bias poly fit failed %r."%(e,))
    finally:
        rclpy.shutdown()
        filter.destroy_node()   

    

if __name__ == '__main__':
    main()