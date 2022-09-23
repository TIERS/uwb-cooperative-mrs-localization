
import os 
import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt


import rclpy
from rclpy.qos          import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg  import PoseStamped
from sensor_msgs.msg    import Range
from nav_msgs.msg       import Odometry



class BiasEstimation(Node) :
    '''
        ROS Node that estimates the difference between optitrack ranges, camera ranges, and uwb ranges.
    '''

    def __init__(self) :

        # Init node
        super().__init__('bias_pf_rclpy')

        # Define QoS profile for odom and UWB subscribers
        self.qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=10,
        )


        self.get_logger().info("Subscribing to odometry")
        self.pose_ori_sub = self.create_subscription(PoseStamped, "/vrpn_client_node/turtlebot1_cap/pose",  self.update_pose_ori_cb, 10)
        self.pose_end_sub = self.create_subscription(PoseStamped, "/vrpn_client_node/turtlebot3_cap/pose",  self.update_pose_end_cb, 10)
        self.odom_ori_sub = self.create_subscription(Odometry, "/turtle01/odom",  self.update_odometry_ori_cb, qos_profile=self.qos)
        self.odom_end_sub = self.create_subscription(Odometry, "/turtle03/odom",  self.update_odometry_end_cb, qos_profile=self.qos)
        self.uwb_range_sub = self.create_subscription(Range, "/uwb/tof/n_7/n_3/distance", self.update_uwb_range_cb, 10)

        # Wait to get some odometry
        sys.stdout.write("Waiting for odom data...")
        for _ in range(100) :
            if self.pose_ori.header.stamp and self.pose_ori.header.stamp :
                break
            sys.stdout.write("..")
            sys.stdout.flush()
            time.sleep(0.1)

        # Responder positions
        self.get_logger().info("Bias Estimation initialized.")

        self.optitrack_turtle01_pose = PoseStamped()
        self.optitrack_turtle03_pose = PoseStamped()

        self.uwb_range = 0.0


    def update_pose_ori_cb(self, pose) :
        '''
            Update pose from turtle01 optitrack pose
        '''
        self.optitrack_turtle01_pose = pose

    def update_pose_end_cb(self, pose) :
        '''
            Update pose from turtle03 optitrack pose
        '''
        self.optitrack_turtle03_pose = pose


    def update_uwb_range_cb(self, range):
        '''
            Update range from UWB
        '''
        self.uwb_range = range.range

        # Todo: add the code here to cal the difference.


def main(args=None):
    rclpy.init(args=args)
    filter = BiasEstimation()
    # Reset filter

    time.sleep(1)
    
    # Start calculating relative positions
    filter.get_logger().info("Starting Bia Estimation...")
    try:
        try:
            while rclpy.ok() :
                rclpy.spin(filter)             
        except KeyboardInterrupt :
            filter.get_logger().error('Keyboard Interrupt detected! Trying to stop the node!')
    except Exception as e:
        filter.destroy_node()
        filter.get_logger().info("Bias Estimation failed %r."%(e,))
    finally:
        rclpy.shutdown()
        filter.destroy_node()     

if __name__ == '__main__':
    main()
