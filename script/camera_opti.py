
import os 
import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt


import rclpy
from rclpy.node             import Node
from rclpy.qos              import QoSProfile,ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg      import PoseStamped
from geometry_msgs.msg      import PoseArray
from sensor_msgs.msg        import Range
from nav_msgs.msg           import Odometry
from depthai_ros_msgs.msg   import SpatialDetectionArray, SpatialDetection

class BiasEstimation(Node) :
    '''
        ROS Node that estimates the difference between optitrack ranges, camera ranges, and uwb ranges.
    '''

    def __init__(self) :

        # Init node
        super().__init__('bias_pf_rclpy')

        # Define QoS profile for odom and UWB subscribers
        self.qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )


        self.get_logger().info("Subscribing to odometry")
        self.pose_ori_sub = self.create_subscription(PoseStamped, "/vrpn_client_node/turtlebot1_cap/pose",  self.update_pose_ori_cb, 10)
        self.pose_end_sub = self.create_subscription(PoseStamped, "/vrpn_client_node/turtlebot3_cap/pose",  self.update_pose_end_cb, 10)
        self.chair_pose_sub = self.create_subscription(PoseStamped, "/vrpn_client_node/chair_final/pose",  self.update_chair_pose_cb, 10)
        # self.uwb_range_sub = self.create_subscription(Range, "/uwb/tof/n_7/n_3/distance", self.update_uwb_range_cb, 10)
        self.object_ori_sub = self.create_subscription(SpatialDetectionArray, "/turtle01/color/yolov4_Spatial_detections",  self.update_object_ori_cb, 10)
        self.object_end_sub = self.create_subscription(SpatialDetectionArray, "/turtle03/color/yolov4_Spatial_detections",  self.update_object_end_cb, 10)


        # Responder positions
        self.get_logger().info("Bias Estimation initialized.")

        self.optitrack_turtle01_pose = PoseStamped()
        self.optitrack_turtle03_pose = PoseStamped()
        self.chair_pose              = PoseStamped()
        self.object_ori_pose_array   = np.array([])
        self.object_end_pose_array   = np.array([])
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

    def update_chair_pose_cb(self, pose):
        '''
            Update pose from chair optitrack pose
        '''
        self.chair_pose = pose


    def update_object_ori_cb(self, pose_array):
        self.object_ori_pose_array = np.array(pose_array.detections)
        if self.object_ori_pose_array.size != 0 and self.object_end_pose_array.size != 0:
            cam_range_array = np.array([-self.object_end_pose_array[0].position.x + self.object_ori_pose_array[0].position.x, 
                                  -self.object_end_pose_array[0].position.y + self.object_ori_pose_array[0].position.y,
                                  0.0])
            opti_range_array = np.array([self.optitrack_turtle01_pose.pose.position.x - self.optitrack_turtle03_pose.pose.position.x,
                                   self.optitrack_turtle01_pose.pose.position.y - self.optitrack_turtle03_pose.pose.position.y,
                                   0.0])
            cam_range  = np.linalg.norm(cam_range_array)
            opti_range = np.linalg.norm(cam_range_array)
            self.get_logger().info("diff: {}".format(opti_range - cam_range))

    def update_object_end_cb(self, pose_array):
        self.object_end_pose_array = np.array(pose_array.detections)



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
