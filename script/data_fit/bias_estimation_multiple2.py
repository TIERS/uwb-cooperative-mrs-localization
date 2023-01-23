
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
        self.pose_turtle01_sub = self.create_subscription(PoseStamped, "/vrpn_client_node/turtlebot1_cap/pose",  self.update_pose_turtle01_cb, 10)
        self.pose_turtle03_sub = self.create_subscription(PoseStamped, "/vrpn_client_node/turtlebot3_cap/pose",  self.update_pose_turtle03_cb, 10)
        self.pose_turtle04_sub = self.create_subscription(PoseStamped, "/vrpn_client_node/turtlebot4_cap/pose",  self.update_pose_turtle04_cb, 10)
        self.pose_turtle05_sub = self.create_subscription(PoseStamped, "/vrpn_client_node/turtlebot5_cap/pose",  self.update_pose_turtle05_cb, 10)
        # self.odom_ori_sub = self.create_subscription(Odometry, "/turtle01/odom",  self.update_odometry_ori_cb, qos_profile=self.qos)
        # self.odom_end_sub = self.create_subscription(Odometry, "/turtle03/odom",  self.update_odometry_end_cb, qos_profile=self.qos)
        self.uwb_range_5_1_sub = self.create_subscription(Range, "/uwb/tof/n_7/n_5/distance", self.uwb_range_5_1_cb, 10)
        self.uwb_range_5_3_sub = self.create_subscription(Range, "/uwb/tof/n_3/n_5/distance", self.uwb_range_5_3_cb, 10)
        self.uwb_range_5_4_sub = self.create_subscription(Range, "/uwb/tof/n_4/n_5/distance", self.uwb_range_5_4_cb, 10)

        self.uwb_range_4_3_sub = self.create_subscription(Range, "/uwb/tof/n_3/n_4/distance", self.uwb_range_4_3_cb, 10)
        self.uwb_range_4_1_sub = self.create_subscription(Range, "/uwb/tof/n_4/n_7/distance", self.uwb_range_4_1_cb, 10)
        self.uwb_range_3_1_sub = self.create_subscription(Range, "/uwb/tof/n_3/n_7/distance", self.uwb_range_3_1_cb, 10)

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
        self.optitrack_turtle04_pose = PoseStamped()
        self.optitrack_turtle05_pose = PoseStamped()

        self.uwb_range_5_1 = 0.0
        self.uwb_range_5_3 = 0.0
        self.uwb_range_5_4 = 0.0

        self.uwb_range_4_3 = 0.0
        self.uwb_range_4_1 = 0.0
        self.uwb_range_3_1 = 0.0

        # self.opti_distance_5_1 = 0
        # self.opti_distance_5_3 = 0
        # self.opti_distance_5_4 = 0

        # self.bias_5_1_list=[]
        # self.bias_5_3_list=[]
        # self.bias_5_4_list=[]

        # self.orientation_list=[]
        self.optitrack_turtle01_orientation_list=[]
        self.optitrack_turtle03_orientation_list=[]
        self.optitrack_turtle04_orientation_list=[]
        self.optitrack_turtle05_orientation_list=[]

        self.optitrack_turtle01_pose_list=[]
        self.optitrack_turtle03_pose_list=[]
        self.optitrack_turtle04_pose_list=[]
        self.optitrack_turtle05_pose_list=[]

        # self.opti_distance_5_1_list=[]
        # self.opti_distance_5_3_list=[]
        # self.opti_distance_5_4_list=[]

        self.uwb_range_5_1_list=[]
        self.uwb_range_5_3_list=[]
        self.uwb_range_5_4_list=[]

        self.uwb_range_4_1_list=[]
        self.uwb_range_4_3_list=[]
        self.uwb_range_3_1_list=[]

        self.count=0



    def update_pose_turtle01_cb(self, pose) :
        '''
            Update pose from turtle01 optitrack pose
        '''
        self.optitrack_turtle01_pose = pose

    def update_pose_turtle03_cb(self, pose) :
        '''
            Update pose from turtle03 optitrack pose
        '''
        self.optitrack_turtle03_pose = pose

    def update_pose_turtle04_cb(self, pose) :
        '''
            Update pose from turtle04 optitrack pose
        '''
        self.optitrack_turtle04_pose = pose

    def update_pose_turtle05_cb(self, pose) :
        '''
            Update pose from turtle04 optitrack pose
        '''
        self.optitrack_turtle05_pose = pose


    def uwb_range_5_1_cb(self, range):
        '''
            Update range from UWB
        '''
        self.uwb_range_5_1 = range.range

        # Todo: add the code here to cal the difference.

    def uwb_range_5_3_cb(self, range):
        '''
            Update range from UWB
        '''
        self.uwb_range_5_3 = range.range

        # Todo: add the code here to cal the difference.

    def uwb_range_5_4_cb(self, range):
        '''
            Update range from UWB
        '''
        self.uwb_range_5_4 = range.range

        # Todo: add the code here to cal the difference.

        self.bias_estimation()

    def uwb_range_4_3_cb(self, range):
        '''
            Update range from UWB
        '''
        self.uwb_range_4_3 = range.range

        # Todo: add the code here to cal the difference.

    def uwb_range_4_1_cb(self, range):
        '''
            Update range from UWB
        '''
        self.uwb_range_4_1 = range.range

        # Todo: add the code here to cal the difference.

    def uwb_range_3_1_cb(self, range):
        '''
            Update range from UWB
        '''
        self.uwb_range_3_1 = range.range

        # Todo: add the code here to cal the difference.


    def bias_estimation(self):
        '''
            Bias estimation
        '''
        optitrack_turtle01_pose = np.array([self.optitrack_turtle01_pose.pose.position.x ,self.optitrack_turtle01_pose.pose.position.y])
        optitrack_turtle03_pose = np.array([self.optitrack_turtle03_pose.pose.position.x ,self.optitrack_turtle03_pose.pose.position.y])
        optitrack_turtle04_pose = np.array([self.optitrack_turtle04_pose.pose.position.x ,self.optitrack_turtle04_pose.pose.position.y])
        optitrack_turtle05_pose = np.array([self.optitrack_turtle05_pose.pose.position.x ,self.optitrack_turtle05_pose.pose.position.y])

        _,_,optitrack_turtle01_orientation = self.euler_from_quaternion(self.optitrack_turtle01_pose)
        _,_,optitrack_turtle03_orientation = self.euler_from_quaternion(self.optitrack_turtle03_pose)
        _,_,optitrack_turtle04_orientation = self.euler_from_quaternion(self.optitrack_turtle04_pose)
        _,_,optitrack_turtle05_orientation = self.euler_from_quaternion(self.optitrack_turtle05_pose)

        # self.opti_distance_5_1 = np.linalg.norm(optitrack_turtle05_pose-optitrack_turtle01_pose)
        # self.opti_distance_5_3 = np.linalg.norm(optitrack_turtle05_pose-optitrack_turtle03_pose)
        # self.opti_distance_5_4 = np.linalg.norm(optitrack_turtle05_pose-optitrack_turtle04_pose)

        self.opti_distance_3_1 = np.linalg.norm(optitrack_turtle03_pose-optitrack_turtle01_pose)
        
        # print("Tur5 ori: {}, Tur1 ori:{}".format(np.rad2deg(optitrack_turtle05_orientation),np.rad2deg(optitrack_turtle01_orientation)))
        # print("Opti distance51: {}, uwb: {}".format(self.opti_distance_5_1, self.uwb_range_5_1))
        # print("Opti distance53: {}, uwb: {}".format(self.opti_distance_5_3, self.uwb_range_5_3))
        # print("Opti distance54: {}, uwb: {}".format(self.opti_distance_5_4, self.uwb_range_5_4))

        # bias_5_1=self.uwb_range_5_1 - self.opti_distance_5_1
        # bias_5_3=self.uwb_range_5_3 - self.opti_distance_5_3
        # bias_5_4=self.uwb_range_5_4 - self.opti_distance_5_4

        bias_3_1=self.uwb_range_3_1 - self.opti_distance_3_1

        print("Opti dist: {}, uwb dist: {}, bias: {}".format(self.opti_distance_3_1,self.uwb_range_3_1,bias_3_1))

        # self.bias_5_1_list.append(bias_5_1)
        # self.bias_5_3_list.append(bias_5_3)
        # self.bias_5_4_list.append(bias_5_4)


        # self.orientation_list.append(optitrack_turtle03_orientation)

        # self.optitrack_turtle01_orientation_list.append(optitrack_turtle01_orientation)
        # self.optitrack_turtle03_orientation_list.append(optitrack_turtle03_orientation)
        # self.optitrack_turtle04_orientation_list.append(optitrack_turtle04_orientation)
        # self.optitrack_turtle05_orientation_list.append(optitrack_turtle05_orientation)

        # self.opti_distance_5_1_list.append(self.opti_distance_5_1)
        # self.opti_distance_5_3_list.append(self.opti_distance_5_3)
        # self.opti_distance_5_4_list.append(self.opti_distance_5_4)

        self.optitrack_turtle01_orientation_list.append(optitrack_turtle01_orientation)
        self.optitrack_turtle03_orientation_list.append(optitrack_turtle03_orientation)
        self.optitrack_turtle04_orientation_list.append(optitrack_turtle04_orientation)
        self.optitrack_turtle05_orientation_list.append(optitrack_turtle05_orientation)

        self.optitrack_turtle01_pose_list.append(optitrack_turtle01_pose)
        self.optitrack_turtle03_pose_list.append(optitrack_turtle03_pose)
        self.optitrack_turtle04_pose_list.append(optitrack_turtle04_pose)
        self.optitrack_turtle05_pose_list.append(optitrack_turtle05_pose)

        self.uwb_range_5_1_list.append(self.uwb_range_5_1)
        self.uwb_range_5_3_list.append(self.uwb_range_5_3)
        self.uwb_range_5_4_list.append(self.uwb_range_5_4)

        self.uwb_range_4_3_list.append(self.uwb_range_4_3)
        self.uwb_range_4_1_list.append(self.uwb_range_4_1)
        self.uwb_range_3_1_list.append(self.uwb_range_3_1)

        # opti_distance_5_1_np = np.array(self.opti_distance_5_1_list)
        # opti_distance_5_3_np = np.array(self.opti_distance_5_3_list)
        # opti_distance_5_4_np = np.array(self.opti_distance_5_4_list)
        # bias_5_1_np = np.array(self.bias_5_1_list)
        # bias_5_3_np = np.array(self.bias_5_3_list)
        # bias_5_4_np = np.array(self.bias_5_4_list)
        # orientation_np = np.array(self.orientation_list)


        uwb_range_5_1_np = np.array(self.uwb_range_5_1_list)
        uwb_range_5_3_np = np.array(self.uwb_range_5_3_list)
        uwb_range_5_4_np = np.array(self.uwb_range_5_4_list)
        
        uwb_range_4_3_np = np.array(self.uwb_range_4_3_list)
        uwb_range_4_1_np = np.array(self.uwb_range_4_1_list)
        uwb_range_3_1_np = np.array(self.uwb_range_3_1_list)

        
        optitrack_turtle01_orientation_np = np.array(self.optitrack_turtle01_orientation_list)
        optitrack_turtle03_orientation_np = np.array(self.optitrack_turtle03_orientation_list)
        optitrack_turtle04_orientation_np = np.array(self.optitrack_turtle04_orientation_list)
        optitrack_turtle05_orientation_np = np.array(self.optitrack_turtle05_orientation_list)

        optitrack_turtle01_pose_np = np.array(self.optitrack_turtle01_pose_list)
        optitrack_turtle03_pose_np = np.array(self.optitrack_turtle03_pose_list)
        optitrack_turtle04_pose_np = np.array(self.optitrack_turtle04_pose_list)
        optitrack_turtle05_pose_np = np.array(self.optitrack_turtle05_pose_list)

        # print("bias51 {}".format(bias_5_1))       
        # print("bias53 {}".format(bias_5_3))       
        # print("bias54 {}".format(bias_5_4))       

        # np.savetxt('bias_estimation.txt',(orientation_np,bias_np),delimiter=',')

        # np.savez('data/test.npz',        
        np.savez('data/4robots_data_01.npz',        
                    uwb_range_5_1_np = uwb_range_5_1_np,
                    uwb_range_5_3_np = uwb_range_5_3_np,
                    uwb_range_5_4_np = uwb_range_5_4_np,
                    uwb_range_4_3_np = uwb_range_4_3_np,
                    uwb_range_4_1_np = uwb_range_4_1_np,
                    uwb_range_3_1_np = uwb_range_3_1_np,
                    optitrack_turtle01_orientation_np = optitrack_turtle01_orientation_np,
                    optitrack_turtle03_orientation_np = optitrack_turtle03_orientation_np,
                    optitrack_turtle04_orientation_np = optitrack_turtle04_orientation_np,
                    optitrack_turtle05_orientation_np = optitrack_turtle05_orientation_np,
                    optitrack_turtle01_pose_np = optitrack_turtle01_pose_np,
                    optitrack_turtle03_pose_np = optitrack_turtle03_pose_np,
                    optitrack_turtle04_pose_np = optitrack_turtle04_pose_np,
                    optitrack_turtle05_pose_np = optitrack_turtle05_pose_np
                    )

        print("Data saved {}".format(self.count))
        self.count+=1

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
