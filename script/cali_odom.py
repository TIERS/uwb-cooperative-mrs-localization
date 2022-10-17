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
from scipy.spatial.transform import Rotation as R


class CalibrateOdom(Node) :

    def __init__(self) :
        super().__init__('cali_odom_rclpy')

        # Define QoS profile for odom and UWB subscribers
        self.qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.get_logger().info("Subscribing to topics")
        self.turtle01_cap_sub     = self.create_subscription(PoseStamped, "/vrpn_client_node/turtlebot1_cap/pose",  self.update_turtle01_cap_cb, 10)
        self.turtle03_cap_sub     = self.create_subscription(PoseStamped, "/vrpn_client_node/turtlebot3_cap/pose",  self.update_turtle03_cap_cb, 10)
        self.turtle04_cap_sub     = self.create_subscription(PoseStamped, "/vrpn_client_node/turtlebot4_cap/pose",  self.update_turtle04_cap_cb, 10)
        self.turtle05_cap_sub     = self.create_subscription(PoseStamped, "/vrpn_client_node/turtlebot5_cap/pose",  self.update_turtle05_cap_cb, 10)
        self.turtle01_odom_sub    = self.create_subscription(Odometry, "/turtle01/odom",  self.update_turtle01_odom_cb, qos_profile=self.qos)
        self.turtle03_odom_sub    = self.create_subscription(Odometry, "/turtle03/odom",  self.update_turtle03_odom_cb, qos_profile=self.qos)
        self.turtle04_odom_sub    = self.create_subscription(Odometry, "/turtle04/odom",  self.update_turtle04_odom_cb, qos_profile=self.qos)
        self.turtle01_cap_pos     = PoseStamped()
        self.turtle03_cap_pos     = PoseStamped()
        self.turtle04_cap_pos     = PoseStamped()
        self.turtle05_cap_pos     = PoseStamped()
        self.turtle01_odom        = Odometry()
        self.turtle03_odom        = Odometry()
        self.turtle04_odom        = Odometry()
        self.turtle05_odom        = Odometry()
        self.turtle01_odom_pub    = self.create_publisher(Odometry, "/cali/turtle01/odom", 10)
        self.turtle03_odom_pub    = self.create_publisher(Odometry, "/cali/turtle03/odom", 10)
        self.turtle04_odom_pub    = self.create_publisher(Odometry, "/cali/turtle04/odom", 10)
        self.turtle05_odom_pub    = self.create_publisher(Odometry, "/cali/turtle05/odom", 10)
        self.turtle01_pos         = PoseStamped()
        self.turtle03_pos         = PoseStamped()
        self.turtle04_pos         = PoseStamped()
        self.turtle05_pos         = PoseStamped()   
        self.turtle01_flag        = True
        self.turtle03_flag        = True
        self.turtle04_flag        = True
        self.turtle05_flag        = True

        self.cnt01                = 0
        self.cnt03                = 0
        self.cnt04                = 0
        self.cnt05                = 0
        # Wait to get some odometry
        sys.stdout.write("Waiting for odom data...")
        for _ in range(100) :
            if self.turtle01_odom.header.stamp and self.turtle03_odom.header.stamp :
                break
            sys.stdout.write("..")
            sys.stdout.flush()
            time.sleep(0.1)
        self.get_logger().info("Odometry locked. Current odom: \n{}".format(self.turtle03_odom.pose))

    def update_turtle01_cap_cb(self, pose):
        self.turtle01_cap_pos = pose
        if self.turtle01_flag:
            self.turtle01_pos = pose

    def update_turtle03_cap_cb(self, pose):
        self.turtle03_cap_pos = pose
        if self.turtle03_flag: 
            self.turtle03_pos = pose


    def update_turtle04_cap_cb(self, pose):
        self.turtle04_cap_pos = pose
        if self.turtle04_flag:
            self.turtle04_pos = pose

    def update_turtle05_cap_cb(self, pose):
        self.turtle05_cap_pos = pose       
        if self.turtle05_flag:
            self.turtle05_pos = pose

    def update_turtle01_odom_cb(self, odom):
        # self.turtle01_odom = odom
        self.turtle01_flag = False
        new_turtle01_odom = Odometry()
        new_turtle01_odom = odom
        new_turtle01_odom.pose.pose.position.x+= self.turtle01_pos.pose.position.x
        new_turtle01_odom.pose.pose.position.y+= self.turtle01_pos.pose.position.y
        new_turtle01_odom.pose.pose.position.z+= self.turtle01_pos.pose.position.z
        new_turtle01_odom.pose.pose.orientation = self.turtle01_pos.pose.orientation
        self.turtle01_odom_pub.publish(new_turtle01_odom)

        # turlebot 5 has no odom data recorded, so put it here.
        self.turtle05_flag = False
        new_turtle05_odom = Odometry()
        new_turtle05_odom.pose.pose.position.x += self.turtle05_pos.pose.position.x
        new_turtle05_odom.pose.pose.position.y += self.turtle05_pos.pose.position.y
        new_turtle05_odom.pose.pose.position.z += self.turtle05_pos.pose.position.z
        new_turtle05_odom.pose.pose.orientation = self.turtle05_pos.pose.orientation
        self.turtle05_odom_pub.publish(new_turtle05_odom)

    def update_turtle03_odom_cb(self, odom):
        # self.turtle03_odom = odom
        self.turtle03_flag = False
        new_turtle03_odom = Odometry()
        new_turtle03_odom = odom
        new_turtle03_odom.pose.pose.position.x += self.turtle03_pos.pose.position.x
        new_turtle03_odom.pose.pose.position.y += self.turtle03_pos.pose.position.y
        new_turtle03_odom.pose.pose.position.z += self.turtle03_pos.pose.position.z
        new_turtle03_odom.pose.pose.orientation = self.turtle03_pos.pose.orientation
        self.turtle03_odom_pub.publish(new_turtle03_odom)

    def update_turtle04_odom_cb(self, odom):
        # self.turtle04_odom = odom
        self.turtle04_flag = False
        new_turtle04_odom = Odometry()
        new_turtle04_odom = odom
        new_turtle04_odom.pose.pose.position.x  += self.turtle04_pos.pose.position.x
        new_turtle04_odom.pose.pose.position.y  += self.turtle04_pos.pose.position.y
        new_turtle04_odom.pose.pose.position.z  += self.turtle04_pos.pose.position.z
        new_turtle04_odom.pose.pose.orientation = self.turtle04_pos.pose.orientation
        self.turtle04_odom_pub.publish(new_turtle04_odom)

def main(args=None):
    rclpy.init(args=args)
    filter = CalibrateOdom()

    time.sleep(0.1)
    
    # Start calculating relative positions
    filter.get_logger().info("Starting Cali Odom...")
    try:
        try:
            while rclpy.ok() :
                rclpy.spin(filter)             
        except KeyboardInterrupt :
            filter.get_logger().error('Keyboard Interrupt detected! Trying to stop the node!')
    except Exception as e:
        filter.destroy_node()
        filter.get_logger().info("Starting Cali Odom failed %r."%(e,))
    finally:
        rclpy.shutdown()
        filter.destroy_node()   
    

if __name__ == '__main__':
    main()