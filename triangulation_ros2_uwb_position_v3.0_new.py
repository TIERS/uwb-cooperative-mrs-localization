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

turtles         = ["4", "1", "2", "3", "5"]
uwbs            = ["4", "1", "2", "3", "5"]
uwb_pair        = [(4,1), (4,2), (4,3), (4,5), (1,2), (1,3), (1,5), (2,3),(2,5), (3,5)]
num_turtles     = 5

#  get parameters from terminal
def parse_args():
    parser = argparse.ArgumentParser(description='Options for triangulations to calculate the relative position of robots based on UWB rangessss')
    parser.add_argument('--poses_save', type=bool, default=True, help='choose to save the estimated poses with triangulation')
    parser.add_argument('--computation_save', type=bool, default=True, help='choose to save the computation time with triangulation')
    parser.add_argument('--round', type=int, default=0, help='indicate which round the pf will run on a recorded data')
    args = parser.parse_args()
    return args

args = parse_args()

# Build folder to save results from different fusion combinations
if args.poses_save:
    pos_folder = "./results/results_csv/triangulation/pos/pos_tri/"
    pos_file = pos_folder + 'pos_{}.csv'.format(args.round)
    if not os.path.exists(pos_folder):
        os.makedirs(pos_folder)

if args.computation_save:
    computation_save_path = "./results/results_csv/triangulation/computation/"
    computation_file = computation_save_path + 'computation_time_{}.csv'.format(args.round)
    if not os.path.exists(computation_save_path):
        os.makedirs(computation_save_path)

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
        self.computation_time       = []
        

        self.node.get_logger().info("Subscribing to topics")
        # subscribe to uwb ranges 
        self.uwb_subs = [
            self.node.create_subscription(Range, "/uwb/tof/n_{}/n_{}/distance".format(p[0], p[1]), 
            self.create_uwb_ranges_cb(i),qos_profile=self.qos) for i, p in enumerate(uwb_pair)]
        self.node.get_logger().info("{} UWB ranges received!".format(len(self.uwb_ranges)))

        # subscribe to optitrack mocap poses
        self.mocap_subs = [
            self.node.create_subscription(PoseStamped, "/vrpn_client_node/tb0{}/pose".format(t), 
            self.create_mocap_pose_cb(i), 10) for i, t in enumerate(turtles)]
        self.node.get_logger().info("{} Mocaps poses received!".format(len(self.turtles_mocaps)))
        
        # subscribe to odometries
        self.odom_subs = [
            self.node.create_subscription(Odometry, "/turtle0{}/odom".format(t), 
            self.create_odom_cb(i),qos_profile=self.qos) for i, t in enumerate(turtles)]
        self.node.get_logger().info("{} odom poses received!".format(len(self.turtles_mocaps)))

        # pf relative poses publishers
        self.relative_pose_publishers = [self.node.create_publisher(PoseStamped, '/tri_turtle0{}_pose'.format(t), 10) for t in turtles[1:]]

        # Responder positions
        self.node.get_logger().info("Triangulation initialized. Estimating position from UWB and odom.")

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

    ######################Init_poses##########################

    def positions_uwb(self, n1,n2,uwb_num,dists):
        """Calculating the positions of the points based on the different base nodes"""
        uwb_list = [*range(uwb_num)]
        thetas = []
        positions = [np.zeros(5) for _ in range(uwb_num)]
        positions_m = [np.zeros(5) for _ in range(uwb_num)]
        node1 = n1-1
        node2 = n2-1
        uwb_list.remove(node1)
        uwb_list.remove(node2)
        # print(uwb_list)
        # print(f"{node1},{node2}")
        # bases = int('{}{}'.format(n1,n2))\
        # print(f"{dists[node1][node2]},{dists[node1][idx]},{dists[idx][node2]}")
        positions[node1] = np.array([0, 0,n1,n1,n2])
        positions[node2] = np.array([dists[node1][node2], 0, n2, n1, n2]) 
        ss = True

        for idx in uwb_list:
            # print(f"{node1},{node2},{idx}")
            # print(f"{dists[node1][node2]},{dists[node1][idx]},{dists[idx][node2]}")
            arg = ( dists[node1][node2]**2 + dists[node1][idx]**2 - dists[idx][node2]**2
            ) / (2 * dists[node1][node2] * dists[node1][idx])

            if arg > 1:
                arg = 1
            if arg < -1:
                arg = -1
            theta = math.acos( arg )
            thetas.append(theta)
            x = math.cos(theta)*dists[node1][idx]
            y = math.sin(theta)*dists[node1][idx]
            y_m = math.sin(-theta)*dists[node1][idx]
            if ss:
                
                if theta != 0 and theta != math.pi:
                    positions[idx]=np.array([x,y,idx+1,n1,n2])
                    spec = idx  
                    ss = False
                else:
                    ss = True
                    positions[idx]=np.array([x,y,idx+1,n1,n2])
            else:
                d = math.dist([x,y],positions[spec][0:2]) - dists[spec][idx]
                d_m = math.dist([x,y_m],positions[spec][0:2]) - dists[spec][idx]
                
                if abs(d) < abs(d_m):
                    positions[idx]=np.array([x,y,idx+1,n1,n2])
                else:
                    positions[idx]=np.array([x,y_m,idx+1,n1,n2])
        # print("22222222222222")
        return positions

    ######################transform##########################

    def rotate(self, point,origin, angle):
        ox, oy = (origin)
        px, py = point
        qx = ox + math.cos(angle) * (px - ox) + math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (ox - px) + math.cos(angle) * (py - oy)
        return qx, qy



    def transform(self, all_positions):
        """transform points: nodes12 as the origin nodes"""
        tr_poses = []
        for i in range(len(all_positions)):
            new_pos = []
            p0 = all_positions[i][0]
            p1 = all_positions[i][1]
            angle = math.atan2(p0[1] - p1[1], p0[0] - p1[0]) + math.pi
            for j in range(len(all_positions[i])):
                x0 = all_positions[i][j][0] - p0[0]
                y0 = all_positions[i][j][1] - p0[1]
                x, y = self.rotate((x0,y0),(0,0) ,angle)
                new_pos.append([x,y,all_positions[i][j][2],all_positions[i][j][3],all_positions[i][j][4]])
            tr_poses.append(new_pos)
        return tr_poses

    ######################maj_side##########################

    def side(self, points, side_p_count, side_m_count):
            """positive or negative side based on the majority points"""
            for i in range(len(points)):
                for j in range(len(points[i])):
                    if points[i][j][1] > 0:
                        side_p_count += 1
                        self.final_pose_p.append([points[i][j][0], points[i][j][1],
                        points[i][j][2],points[i][j][3],points[i][j][4]])
                        self.final_pose_m.append([points[i][j][0], -points[i][j][1],
                        points[i][j][2],points[i][j][3],points[i][j][4]])
                    else:
                        side_m_count += 1
                        self.final_pose_p.append([points[i][j][0], -points[i][j][1],
                        points[i][j][2],points[i][j][3],points[i][j][4]])
                        self.final_pose_m.append([points[i][j][0], points[i][j][1], 
                        points[i][j][2],points[i][j][3],points[i][j][4]])    
            # if side_p_count > side_m_count:
            final = self.final_pose_p
            # else:
            #     final = final_pose_m
            return final

    ######################matching##########################
    def matching(self, avr_pose_base, positions):
        # print("matching")
        base_position=avr_pose_base
        new_positions=np.zeros(np.shape(positions))
        lineal_step=0.5
        angular_step=5 #degrees
        max_lineal=1.5
        max_angular=5
        config_error = []
        distances_err_list = []

        lineal_change=np.arange(-max_lineal,max_lineal+0.1,lineal_step)
        lineal_change=lineal_change.round(decimals=2)
        angular_change=np.arange(np.deg2rad(-max_angular),np.deg2rad(max_angular+0.01),np.deg2rad(angular_step))

        for i in range(0,len(positions)):
            
            min_configuration=np.copy(positions[i])
            min_config_error=1000

            for x in lineal_change:
        
                for y in lineal_change:
                    
                        for ang in angular_change:
                            
                            centr_point = np.average(positions[i],axis=0)
                            position_moved=np.copy(positions[i])
                            position_moved[:,0]=position_moved[:,0]+x
                            position_moved[:,1]=position_moved[:,1]+y

                            for k, point in enumerate(position_moved):
                                position_moved[k][0],position_moved[k][1] = self.rotate((point[0],point[1]),
                                (centr_point[0],centr_point[1]),ang) 

                            distances_err = np.square(self.lse_config(base_position,position_moved))
                            error_squared = distances_err.sum()

                            if(error_squared < min_config_error):
                                min_config_error=error_squared
                                min_configuration = position_moved
                                min_err_list = distances_err

            distances_err_list.append(min_err_list)
            new_positions[i]=min_configuration
            config_error.append([min_config_error,positions[i][0][3],positions[i][0][4]])

        sum_of_err,err_array = self.anomaly_base_err(config_error,len(avr_pose_base),distances_err_list)

        return(new_positions, sum_of_err,err_array)

    ######################lse_config##########################

    def lse_config(self, base_position,positions):

        """computes de least squared error between two configuration"""

        base_position=base_position[:,:2]
        positions=positions[:,:2]
        distances=np.zeros(len(base_position))
        distances=np.linalg.norm(base_position-positions,axis=1)

        return distances
    # #####################remove_sus_bases####################

    def remove_sus_bases(self, poses,index):
        j = 0

        while j<len(poses):
            if poses[j][0][3]==index or poses[j][0][4]==index:
                poses = np.delete(poses,j,axis=0)
                
                j = j
            else:
                j += 1
        return poses

    ######################variance############################

    def var(self, points):
        poses_arr = np.array(points)
        reshaped = np.reshape(poses_arr,(len(points)*len(points[0]),5))
        variance = []
        dists = []
        node_points = []
        for nd in range(len(points[0])):
            for i in range(len(reshaped)):
                if reshaped[i][2] == nd+1:
                    node_points.append(reshaped[i])
        reshaped_p = np.reshape(node_points,(len(points[0]),len(points),5))
        avr = np.mean(reshaped_p, axis=1)
        for v in range(len(reshaped_p)):
            dists = np.linalg.norm(reshaped_p[v][:,:2] - avr[v][:2], axis=1)
            variance.append(np.mean(dists))
        variance_arr = np.array(variance)
        return variance_arr


    def calculate_relative_poses(self) :
        '''
            Calculates relative poses of nodes doing TOF
        '''
        start = time.time_ns() / (10 ** 9)
        positions = [np.zeros(2) for _ in range(num_turtles)] 
        positions[0] = np.array([0, 0])
        # positions[1] = np.array([self.uwb_ranges[9], 0])
        uwb_num = 5
        # time = 1
        all_positions = []
        self.final_pose_p = []
        self.final_pose_m = []
        side_p_count = 0
        side_m_count = 0
        init_imp = -100
        # print("******")
        dists = np.zeros((uwb_num+1,uwb_num+1))
        # 7,3,4,5,2
        # 4,1,2,3,5
        dists[0][1],dists[1][0]=self.uwb_ranges[0],self.uwb_ranges[0]
        dists[0][2],dists[2][0]=self.uwb_ranges[1],self.uwb_ranges[1]
        dists[0][3],dists[3][0]=self.uwb_ranges[2],self.uwb_ranges[2]
        dists[0][4],dists[4][0]=self.uwb_ranges[3],self.uwb_ranges[3]
        dists[1][2],dists[2][1]=self.uwb_ranges[4],self.uwb_ranges[4]
        dists[1][3],dists[3][1]=self.uwb_ranges[5],self.uwb_ranges[5]
        dists[1][4],dists[4][1]=self.uwb_ranges[6],self.uwb_ranges[6]
        dists[2][3],dists[3][2]=self.uwb_ranges[7],self.uwb_ranges[7]
        dists[2][4],dists[4][2]=self.uwb_ranges[8],self.uwb_ranges[8]
        dists[3][4],dists[4][3]=self.uwb_ranges[9],self.uwb_ranges[9]
        # uwb_pair        = [(4,1), (4,2), (4,3), (4,5), (1,2), (1,3), (1,5), (2,3),(2,5), (3,5)]
        # n1 = [1,1,1,1,2,2,2,3,3,4]
        # n2 = [2,3,4,5,3,4,5,4,5,5]
        n1 = [1]
        n2 = [2]
        # uwb_pair   = [(3,7), (4,7), (2,7), (3,4), (2,3), (2,4), (7,5), (3,5),(4,5), (2,5)]
        try:
            # print("11111111111111111111111111111")
            for (nd1, nd2) in zip(n1,n2):
                # print(f"{nd1},{nd2}")
                positions = self.positions_uwb(nd1, nd2, uwb_num, dists)
                all_positions.append(positions)
            # print(f"{all_positions}")
            transformed_poses = self.transform(all_positions)
            # print("-------------------")
        
            sided_poses = self.side(transformed_poses,side_p_count, side_m_count)
            # print(f"{sided_poses}")
            pos_arr = np.array(sided_poses)
            www = np.array(transformed_poses)

            reshaped_positions = np.reshape(pos_arr,(len(n1),uwb_num,5))


            # avr_pose_base = reshaped_positions[0]
            avr_pose_base = np.mean(reshaped_positions, axis=0)
            # print(f"avr_pose_base:{avr_pose_base}")
            # new_positions,err,array_err = self.matching(avr_pose_base,reshaped_positions)

            # sus_base_nodes = [i+1 for i,j in enumerate(err) if j> np.mean(err)*1.5]
            # print(f"new_positions:{new_positions}")

            # avg_position_all = np.mean(new_positions, axis=0)
            # print(len(self.turtles_mocaps))
            # print(avr_pose_base.shape)

            self.relative_pose_cal(self.turtles_mocaps[0], self.turtles_mocaps[1:], self.true_relative_poses)
            # print(self.true_relative_poses)
            real_tmp, esti_tmp = [], []
            for inx in range(len(self.turtles_mocaps) - 1):
                real_tmp.append(self.true_relative_poses[inx][0])
                real_tmp.append(self.true_relative_poses[inx][1])
                esti_tmp.append(avr_pose_base[inx+1][0])
                esti_tmp.append(avr_pose_base[inx+1][1])
            print(real_tmp+esti_tmp)
            self.pos_estimation.append(real_tmp+esti_tmp)
            # self.pos_estimation.append([self.true_relative_poses[0][0], self.true_relative_poses[0][1],
            #                 self.true_relative_poses[1][0], self.true_relative_poses[1][1],
            #                 self.true_relative_poses[2][0], self.true_relative_poses[2][1], 
            #                 -avr_pose_base[0][0],     -avr_pose_base[0][1], 
            #                 -avr_pose_base[1][0],     -avr_pose_base[1][1], 
            #                 -avr_pose_base[2][0],     -avr_pose_base[2][1], 
            #                 -avr_pose_base[3][0],     -avr_pose_base[3][1],
            #                 -avr_pose_base[4][0],     -avr_pose_base[4][1], 
            #                  ])
            # print(new_positions)
            # publish pf relative pose
            for i in range(len(turtles[1:])):
                relative_pose = PoseStamped()
                relative_pose.header.frame_id = "base_link"
                relative_pose.header.stamp = self.node.get_clock().now().to_msg()
                relative_pose.pose.position.x = avr_pose_base[i+1][0]
                relative_pose.pose.position.y = avr_pose_base[i+1][1]
                # print(f"{-avr_pose_base[i+1][0]},{-avr_pose_base[i+1][1]}"
                relative_pose.pose.position.z = 0.0
                relative_pose.pose.orientation = self.turtles_odoms[i].pose.pose.orientation
                self.relative_pose_publishers[i].publish(relative_pose) 

        except ValueError:
            self.node.get_logger().error("math domain error")
        end = time.time_ns() / (10 ** 9)
        self.computation_time.append(end - start)

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
        self.node.get_logger().info("triangulation ends and Saving Results. Got {} poses and {} computation recordings.".format(
                                len(self.pos_estimation), len(self.computation_time)))
        if args.poses_save:
            np.savetxt(pos_file, 
                self.pos_estimation,
                delimiter =", ", 
                fmt ='% s')     

        if args.computation_save: 
            np.savetxt(computation_file, 
                self.computation_time,
                delimiter =", ", 
                fmt ='% s')        


def main(args=None):
    pos_cal = UWBTriangulation()
    pos_cal.run()
    

if __name__ == '__main__':
    main()
