#!/usr/bin/env python

import time
import math
import numpy
import random
import argparse
import rclpy
from rclpy.node import Node
from rclpy.qos          import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg       import Odometry

from geometry_msgs.msg  import Twist
from geometry_msgs.msg  import PoseStamped

class TurtleControl(Node) :

    def __init__(self) :

        # Init node
        super().__init__('turtle_01_control')
        # Define QoS profile for odom and UWB subscribers
        self.qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Predefined trajectory
        self.trajectory = numpy.array([
                [2.9988980293273926, 4.696284294128418], 
                [1.0675207376480103, 4.670303821563721], 
                [1.0273232460021973 , 6.421392440795898], 
                [ 3.035672187805176, 6.444340229034424],
            ])
        self.objective_idx = 0
        self.yaw = 0

        self.pos = numpy.zeros(2)

        # Params
        self.dist_threshold = 0.2
        self.angular_threshold = 0.15
        self.angular_speed = 0.5
        self.linear_speed = 1
    
        # UWB Subscriber
        self.pos_sub = self.create_subscription(PoseStamped, "/vrpn_client_node/turtlebot1_cap/pose",  self.pos, 10)
        # self.vio_sub = rospy.Subscriber("/t265/odom/sample", Odometry, self.odom_cb)

        # Velocity Publisher
        self.vel_pub = self.create_publisher(Twist, "/turtle01/cmd_vel", 10)

    def orientation_to_euler(self, orientation):
        orientation_q = orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, yaw) = euler_from_quaternion (orientation_list)
        return yaw

    def odom_cb(self, msg) :
        '''
            Mainly used for orientation from VIO
        '''
        self.yaw = self.orientation_to_euler(msg.pose.pose.orientation)

    def pos_cb(self, msg) :
        '''
            Update robot's position
        '''
        self.pos[0] = msg.pose.position.x
        self.pos[1] = msg.pose.position.y
        self.yaw = self.orientation_to_euler(msg.pose.pose.orientation)


    def control(self):
        '''
            Controls predefined trajectory based on UWB
        '''

        vel = Twist()

        if numpy.linalg.norm(self.pos - self.trajectory[self.objective_idx]) < self.dist_threshold :
            self.objective_idx = (self.objective_idx + 1) % int(len(self.trajectory))

        theta = math.atan2(
            self.trajectory[self.objective_idx][1] - self.pos[1],
            self.trajectory[self.objective_idx][0] - self.pos[0]
        )

        print("We want to go angle {}, now at {}".format(theta, self.yaw))

        if abs(self.yaw - theta) > self.angular_threshold :
            
            if self.yaw < theta :
                vel.angular.z = self.angular_speed
            else :
                vel.angular.z = -self.angular_speed
            
        else :

            if abs(self.yaw - theta) > 3*self.angular_threshold :
                if self.yaw < theta :
                    vel.angular.z = self.angular_speed
                else :
                    vel.angular.z = -self.angular_speed

            vel.linear.x = self.linear_speed

        print("We are at {} moving to {} with vels {},{}".format(self.pos, self.trajectory[self.objective_idx], vel.linear.x, vel.angular.z))
        
        self.vel_pub.publish(vel)

def main(args=None):
    rclpy.init(args=args)
    controller = TurtleControl()
    # controller.run()
    

    time.sleep(1)
    # Start calculating relative positions
    controller.get_logger().info("Starting...")
    try:
        try:
            while rclpy.ok() :
                rclpy.spin(controller)             
        except KeyboardInterrupt :
            controller.get_logger().error('Keyboard Interrupt detected! Trying to stop controller node!')
    except Exception as e:
        controller.destroy_node()
        controller.get_logger().info("Controller failed %r."%(e,))
    finally:
        rclpy.shutdown()
        controller.destroy_node()   

    

if __name__ == '__main__':
    main()
