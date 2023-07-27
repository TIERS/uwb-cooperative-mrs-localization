#!/usr/bin/env python
# Common python libaries
import os 
import time
import math
import argparse
import numpy                    as np
import matplotlib.pyplot        as plt
import itertools
import copy

# ROS libaries
import rclpy
from rclpy.node                 import Node
from rclpy.qos                  import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg          import PoseStamped
from sensor_msgs.msg            import Range
from nav_msgs.msg               import Odometry
from depthai_ros_msgs.msg       import SpatialDetectionArray

from pfilter                    import ParticleFilter, squared_error

# from tensorflow                 import keras

from utlis  import utils

import pf_ulv

odom_transform = {
    0: ([[ 1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]],[0.0,0.0,0.0]),
    1: ([[ 1.00000000e+00, -7.58541994e-16,  5.13445841e-16],
         [ 5.62165047e-16,  1.00000000e+00,  6.07270648e-17],
         [-5.11982838e-16, -5.43282007e-17,  1.00000000e+00]], [[ 8.88178420e-14,  6.19504448e-14, -2.74780199e-15]]),
    2: ([[ 1.00000000e+00, -1.19249603e-06,  5.97508164e-07],
         [ 1.19249650e-06,  1.00000000e+00, -7.73259505e-07],
         [-5.97507242e-07,  7.73260217e-07,  1.00000000e+00]], [[ 7.78583451e-06, -9.16588829e-06, -3.91865132e-07]]),
    3: ([[ 1.00000000e+00,  6.95738684e-16,  5.84436962e-17],
         [-7.08797061e-16,  1.00000000e+00,  5.65593577e-16],
         [-5.79974105e-17, -5.61957041e-16,  1.00000000e+00]], [[-3.24185123e-14,  7.99360578e-14, -1.72084569e-15]]),
    4: ([[ 1.00000000e+00,  2.86754311e-05,  7.06831308e-07],
         [-2.86754311e-05,  1.00000000e+00, -1.37164463e-07],
         [-7.06835241e-07,  1.37144194e-07,  1.00000000e+00]], [[ 1.75782436e-04, -1.46837665e-04, -1.94010446e-06]])
}
# rectangle        
# x
# tri


class UWBRelativePositionNode(Node) :
    '''
        ROS Node that estimates relative position of two robots using odom and single uwb range.
    '''
    def __init__(self, pf_estimator, uwb_pair = [(0,1)], robot_ids=[0,1]):
        '''
            Define the parameters, publishers, and subscribers
        '''
        # Init node
        super().__init__('relative_pf_rclpy')
        # Define QoS profile for odom and UWB subscribers
        self.qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.pf_estimator = pf_estimator
        self.uwb_pairs = uwb_pair
        self.robot_ids = robot_ids
        self.uwb_ranges_dict = {p:0.0 for p in self.uwb_pairs}
        self.odom_data_dict = {id:np.array([]) for id in self.robot_ids}
        self.odom_data_aligned_dict = {id:np.array([]) for id in self.robot_ids}
        self.mocap_data_dict = {id:np.array([]) for id in self.robot_ids}
        self.spatial_dict={id:[] for id in self.robot_ids}

        self.pose_estimations = []

        self.get_logger().info("Subscribing to topics......")
        # subscribe to uwb ranges 
        self.uwb_subs = [
            self.create_subscription(Range, "/uwb/tof/n_{}/n_{}/distance".format(p[0], p[1]), 
            self.create_uwb_ranges_cb(p),qos_profile=self.qos) for i, p in enumerate(uwb_pair)]
        self.get_logger().info("{} UWB ranges subscribed!".format(len(self.uwb_ranges_dict)))

        # subscribe to optitrack mocap poses
        self.mocap_subs = [
            self.create_subscription(PoseStamped, "/vrpn_client_node/tb0{}/pose".format(i), 
            self.create_mocap_pose_cb(i), qos_profile=self.qos) for i in robot_ids]
        self.get_logger().info("{} Mocaps poses subscribed!".format(len(self.mocap_data_dict)))
        
        # subscribe to odometries
        self.odom_subs = [
            self.create_subscription(Odometry, "/turtle0{}/odom".format(i), 
            self.create_odom_cb(i), qos_profile=self.qos) for i in robot_ids]
        self.get_logger().info("{} odom poses subscribed!".format(len(self.odom_data_dict)))

        # subscribe to spatial detections
        self.spatial_subs = [
            self.create_subscription(SpatialDetectionArray, "/turtle0{}/color/yolov4_Spatial_detections".format(i), 
            self.create_spatial_cb(i), qos_profile=self.qos)  for i in robot_ids]
        self.get_logger().info("{} spatial detections subscribed!".format(len(self.spatial_dict)))

        self.get_logger().info("\\\\\\Class Initialized Done//////")
        
        
    def create_uwb_ranges_cb(self, p):
        return lambda range : self.uwb_range_cb(p, range)
        
    def uwb_range_cb(self, p, range):
        self.uwb_ranges_dict[p] = range.range

    def create_mocap_pose_cb(self, p):
        return lambda pos : self.mocap_pose_cb(p, pos)
        
    def mocap_pose_cb(self, p, pos):
        self.mocap_data_dict[p] = np.array([pos.pose.position.x, pos.pose.position.y, pos.pose.position.z, pos.pose.orientation.x, pos.pose.orientation.y, pos.pose.orientation.z, pos.pose.orientation.w])  

    def create_odom_cb(self, p):
        return lambda odom : self.odom_cb(p, odom)
        
    def odom_cb(self, p, odom):
        self.odom_data_dict[p] = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z, odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w])

    def create_spatial_cb(self, p):
        return lambda detections : self.spatial_cb(p, detections)
        
    def spatial_cb(self, p, detections):
        self.spatial_dict[p] = np.array([[det.position.x, det.position.y, det.position.y] for det in detections.detections])

    '''
        Update the data for the particle filter
            apply the odom to mocap data alignment
    '''
    def update_pf_data(self):
        # res = utils.dicts_empty([self.mocap_data_dict])
        res = utils.dict_values_empty(self.mocap_data_dict)
        # print(res)
        # the robot 0 has no odometry, so we use mocap data plus noise instead
        if res:
            # print("update odom data with mocap data......")
            self.odom_data_aligned_dict = self.mocap_data_dict
            return True
            # self.odom_data_dict[0] = self.mocap_data_dict[0] + np.random.normal(0, 0.01, 7)        
            # if utils.dict_values_empty(self.odom_data_dict):
            #     self.odom_data_aligned_dict = copy.deepcopy(self.odom_data_dict)
            #     # apply the odom to mocap data alignment
            #     for k in self.odom_data_dict:
            #         transformed_source = np.dot(self.odom_data_dict[k][0:3], np.array(odom_transform[k][0])) + np.array(odom_transform[k][1])
            #         self.odom_data_aligned_dict[k][0:3] = transformed_source
            #     return True
            # else:
            #     return False
        else:
            return False

    def update_pf_filter(self):
        # time.sleep(0.1)
        # flag = self.update_pf_data()
        if self.update_pf_data():
            # print("Updating particle filter......")
            self.pf_estimator.update_input(self.uwb_ranges_dict, self.odom_data_aligned_dict, self.spatial_dict)
            # self.get_logger().info("Updating particle filter......")
            self.pf_estimator.update_filter()
            # self.pf_estimator.plot_particles()
            print(">>>>>> robot poses saved!")
            # print("\n")

    def __del__(self):
        # self.get_logger().info(np.array(self.pose_estimations).shape)
        self.pose_estimations = self.pf_estimator.get_robot_poses()
        if len(self.pose_estimations) > 0:
            print(np.array(self.pf_estimator.get_robot_poses()).shape)
            np.savetxt("pose_estimations.csv", np.array(self.pose_estimations), delimiter=",", fmt ='% s')
            # self.get_logger().info("Pose estimations saved!")
        self.get_logger().info("Destroying UWB relative position node......")

def main(args=None):
    rclpy.init(args=args)

    robot_ids       = [0,1,2,3,4]
    uwb_pair = [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]

    estimator = pf_ulv.UWBParticleFilter(spatial_enable = False, lstm_enable = False, identical_thresh = 0.1, robot_ids = robot_ids)

    ros_node = UWBRelativePositionNode(estimator, uwb_pair = uwb_pair, 
                                        robot_ids = robot_ids)

    # filter.get_logger().info("Starting particle filter...")
    filter_timer = ros_node.create_timer(1/20.0, ros_node.update_pf_filter)

    try:
        try:
            while rclpy.ok() :
                rclpy.spin(ros_node)             
        except KeyboardInterrupt :
            ros_node.get_logger().error('Keyboard Interrupt detected! Trying to stop ros_node node!')
    except Exception as e:
        ros_node.destroy_node()
        ros_node.get_logger().info("UWB particle ros_node failed %r."%(e,))
    finally:
        rclpy.shutdown()
        ros_node.destroy_timer(filter_timer)
        ros_node.destroy_node()   

    

if __name__ == '__main__':
    main()