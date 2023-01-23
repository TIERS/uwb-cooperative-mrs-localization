
from errno import EUSERS
import os 
import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

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

        self.uwb_ranges = {}
        self.mocap_pose = {}
        self.mocap_ori  = {}
        self.total_uwb = 8
        self.dis_subscribers = []
        self.pos_subscribers = []

        self.data_save=[]

        self.uwb_pairs = [  [1,2],
                            [1,3],
                            [1,4],
                            [1,5],
                            [1,6],
                            [1,7],
                            [1,8],
                            [2,3],
                            [2,4],
                            [2,5],
                            [2,6],
                            [2,7],
                            [2,8],
                            [3,4],
                            [3,5],
                            [3,6],
                            [3,7],
                            [3,8],
                            [4,5],
                            [4,6],
                            [4,7],
                            [4,8],
                            [5,6],
                            [5,8],
                            [6,7],
                            [6,8],
                            [7,5],
                            [7,8]]

        self.turtlebot_num=[1,2,3,4,5]

        #Subscribers

        #UWB topics
        for pair in self.uwb_pairs:
            number = int('{}{}'.format(pair[0],pair[1]))
            # self.uwb_ranges[number] = []
            # self.uwb_ranges[number].append('uwb_dis{}'.format(number))
            dis_sub = self.create_subscription(Range,'/uwb/tof/n_{}/n_{}/distance'.format(pair[0],pair[1]),self.cbuwb(number), 10)
            self.dis_subscribers.append(dis_sub)

        #MOCAP positions
        for bot in self.turtlebot_num:
            # self.mocap_pose[bot] = []
            # self.mocap_ori[bot] = []
            # self.mocap_pose[k].append('mocap_pose{}'.format(k))
            pose_sub = self.create_subscription(PoseStamped,'/vrpn_client_node/tb0{}/pose'.format(bot),self.cbcap(bot), qos_profile=self.qos)
            self.pos_subscribers.append(pose_sub)

        #Timer
        meas_timer_period = 1.0/7.0 # 7hz is the slowest uwb
        self.timer_update_meas = self.create_timer(meas_timer_period, self.timer_save)
        self.check=False



    def cbuwb(self,n): # for uwb distances
        def dis_uwb(msg) :
            self.uwb_ranges[str(n)]=msg.range
            # print ("uwb {} range {}".format(n,msg.range))
        return dis_uwb

    def cbcap(self,k): # for optitrack positions
        def cap_pose(msg) :
            self.mocap_pose[k]=[msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
            self.mocap_ori[k]=msg.pose.orientation
            # print(f"mocap: {k}")
        return cap_pose

    def euler_from_quaternion(self,pose):
            """
            Convert a quaternion into euler angles (roll, pitch, yaw)
            roll is rotation around x in radians (counterclockwise)
            pitch is rotation around y in radians (counterclockwise)
            yaw is rotation around z in radians (counterclockwise)
            """

            x =  pose.x
            y =  pose.y
            z =  pose.z
            w =  pose.w

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
        
            # return roll_x, pitch_y, yaw_z # in radians
            return yaw_z

    def timer_save(self): # for updating measurements
        if self.check: 
            print('------------------------')
            # print(self.uwb_ranges)
            # print(self.mocap_pose)
            # print(self.mocap_ori)

            timestamp=self.get_clock().now().to_msg()
            timestamp=timestamp.sec+timestamp.nanosec/10e9
            for pairs in combinations(self.turtlebot_num,2):
                # print(pairs)
                mocap_range = np.linalg.norm(np.array(self.mocap_pose[pairs[0]])-np.array(self.mocap_pose[pairs[1]]))
                uwb_comb="{}{}".format(pairs[0],pairs[1])

                # #######
                # #uwb 7 is in turtlebot 1
                # #uwb 8 is in trutlebot 2
                # # print(uwb_comb)
                # uwb_comb_mod=uwb_comb.replace("1","7")
                # # print(uwb_comb)
                # if(uwb_comb_mod=="73"):
                #     uwb_comb_mod="37"
                #     # print(uwb_comb)
                # if(uwb_comb_mod=="74"):
                #     uwb_comb_mod="47"
                #     # print(uwb_comb)

                # # print(uwb_comb_mod)

                # # print(self.uwb_ranges[uwb_comb_mod],mocap_range)

                
                error=self.uwb_ranges[uwb_comb]-mocap_range
                # print(error)

                # print(self.mocap_ori[1])

                
                # print(timestamp)

                data=[  timestamp,
                        pairs[0],
                        pairs[1],
                        self.uwb_ranges[uwb_comb],
                        mocap_range,
                        self.euler_from_quaternion(self.mocap_ori[pairs[0]]),
                        self.euler_from_quaternion(self.mocap_ori[pairs[1]]),
                        error]

                self.data_save.append(data)
                
                # print(data)
            # print()
            # print(self.data_save)

            # self.data_save_np=np.array(self.data_save)
            # print(self.data_save_np)

            np.savetxt("train_newdata.csv", self.data_save,fmt=('%s'), header="timestamp,node1,node2,uwb_range,mocap_range,tb_node1_yaw,tb_node2_yaw,error",delimiter=',',comments='')

        
        
        
        self.check=True #Ignore the first timer to fill dictionaries




def main(args=None):
    rclpy.init(args=args)
    filter = BiasEstimation()
    # Reset filter

    time.sleep(1)

    
    
    # Start calculating relative positions
    filter.get_logger().info("Starting ...")
    try:
        try:
            while rclpy.ok() :
                rclpy.spin(filter)             
        except KeyboardInterrupt :
            filter.get_logger().error('Keyboard Interrupt detected! Trying to stop the node!')
    except Exception as e:
        filter.destroy_node()
        filter.get_logger().info("Starting failed %r."%(e,))
    finally:
        rclpy.shutdown()
        filter.destroy_node()     

if __name__ == '__main__':
    main()
