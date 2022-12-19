
from errno import EUSERS
import os 
import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
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
        # self.odom_ori_sub = self.create_subscription(Odometry, "/turtle01/odom",  self.update_odometry_ori_cb, qos_profile=self.qos)
        # self.odom_end_sub = self.create_subscription(Odometry, "/turtle03/odom",  self.update_odometry_end_cb, qos_profile=self.qos)
        self.uwb_range_sub = self.create_subscription(Range, "/uwb/tof/n_7/n_3/distance", self.update_uwb_range_cb, 10)

        # # Wait to get some odometry
        # sys.stdout.write("Waiting for odom data...")
        # for _ in range(100) :
        #     if self.pose_ori.header.stamp and self.pose_ori.header.stamp :
        #         break
        #     sys.stdout.write("..")
        #     sys.stdout.flush()
        #     time.sleep(0.1)

        # Responder positions
        self.get_logger().info("Bias Estimation initialized.")

        self.optitrack_turtle01_pose = PoseStamped()
        self.optitrack_turtle03_pose = PoseStamped()

        self.uwb_range = 0.0

        self.opti_distance = 0

        self.bias_list=[]
        self.orientation_list=[]
        self.optitrack_turtle01_orientation_list=[]
        self.optitrack_turtle03_orientation_list=[]
        self.opti_distance_list=[]
        self.uwb_range_list=[]



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

        
        print("UWB callback")

        self.bias_estimation()

    def bias_estimation(self):
        '''
            Bias estimation
        '''
        optitrack_turtle01_pose = np.array([self.optitrack_turtle01_pose.pose.position.x ,self.optitrack_turtle01_pose.pose.position.y])
        optitrack_turtle03_pose = np.array([self.optitrack_turtle03_pose.pose.position.x ,self.optitrack_turtle03_pose.pose.position.y])

        _,_,optitrack_turtle01_orientation = self.euler_from_quaternion(self.optitrack_turtle01_pose)
        _,_,optitrack_turtle03_orientation = self.euler_from_quaternion(self.optitrack_turtle03_pose)

        self.opti_distance = np.linalg.norm(optitrack_turtle03_pose-optitrack_turtle01_pose)
        
        print("Tur1 ori: {}, Tur3 ori:{}".format(np.rad2deg(optitrack_turtle01_orientation),np.rad2deg(optitrack_turtle03_orientation)))
        print("Opti distance: {}, uwb: {}".format(self.opti_distance, self.uwb_range))

        bias=self.uwb_range - self.opti_distance
        self.bias_list.append(bias)
        self.orientation_list.append(optitrack_turtle03_orientation)

        self.optitrack_turtle01_orientation_list.append(optitrack_turtle01_orientation)
        self.optitrack_turtle03_orientation_list.append(optitrack_turtle03_orientation)
        self.opti_distance_list.append(self.opti_distance)
        self.uwb_range_list.append(self.uwb_range)

        opti_distance_np = np.array(self.opti_distance_list)
        uwb_range_np = np.array(self.uwb_range_list)
        bias_np = np.array(self.bias_list)
        orientation_np = np.array(self.orientation_list)
        optitrack_turtle01_orientation_np = np.array(self.optitrack_turtle01_orientation_list)
        optitrack_turtle03_orientation_np = np.array(self.optitrack_turtle03_orientation_list)

        print("bias {}".format(bias))       

        # np.savetxt('bias_estimation.txt',(orientation_np,bias_np),delimiter=',')


        np.savez('data/bias_estimation_2robot_uwb_1-3.npz',        
                    opti_distance_np = opti_distance_np, 
                    uwb_range_np = uwb_range_np, 
                    bias_np = bias_np, 
                    orientation_np = orientation_np, 
                    optitrack_turtle01_orientation_np = optitrack_turtle01_orientation_np, 
                    optitrack_turtle03_orientation_np = optitrack_turtle03_orientation_np
                    )

    def euler_from_quaternion(self,pose):
            """
            Convert a quaternion into euler angles (roll, pitch, yaw)
            roll is rotation around x in radians (counterclockwise)
            pitch is rotation around y in radians (counterclockwise)
            yaw is rotation around z in radians (counterclockwise)
            """

            x =  pose.pose.orientation.x
            y =  pose.pose.orientation.y
            z =  pose.pose.orientation.z
            w =  pose.pose.orientation.w

            t0 = +2.0 * (w * x + y * z)
            t1 = +1.0 - 2.0 * (x * x + y * y)
            roll_x = math.atan2(t0, t1)
        
            t2 = +2.0 * (w * y - z * x)
            t2 = +1.0 if t2 > +1.0 else t2
            t2 = -1.0 if t2 < -1.0 else t2
            pitch_y = math.asin(t2)
        
            t3 = +2.0 * (w * z + x * y)
            t4 = +1.0 - 2.0 * (y * y + z * z)
            yaw_z = math.atan2(t3, t4)
        
            return roll_x, pitch_y, yaw_z # in radians


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
