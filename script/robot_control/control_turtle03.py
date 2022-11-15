#!/usr/bin/env python

import time
import math
import numpy
import numpy as np
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
        super().__init__('turtle_03_control')
        # Define QoS profile for odom and UWB subscribers
        self.qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # # Predefined trajectory
        # self.trajectory = numpy.array([
        #         [ 3.035672187805176, 6.444340229034424],
        #         [1.0273232460021973 , 6.421392440795898], 
        #         [1.0675207376480103, 4.670303821563721], 
        #         [2.9988980293273926, 4.696284294128418], 
        #     ])

        # self.trajectory = numpy.array([
        #     [6.086298942565918 , 0.5701131820678711  ],
        #     [4.156103610992432  , 0.5414536595344543 ],
        #     [4.133383274078369  , 1.1638691425323486 ],
        #     [6.156064033508301 , 1.1792606115341187  ],
        #     [6.055030345916748 , 1.7517801523208618  ],
        #     [4.1496124267578125 , 1.6548444032669067 ],
        #     [4.111688613891602 , 2.274160385131836  ],
        #     [6.140777587890625 , 2.3119454383850098  ],
        #     ])

        self.trajectory = numpy.array([
            [6.086298942565918 , 0.5701131820678711  ],
            
            [4.133383274078369  , 1.1638691425323486 ],

            [4.156103610992432  , 0.5414536595344543 ],

            [6.156064033508301 , 1.1792606115341187  ],
            
            [4.1496124267578125 , 1.6548444032669067 ],

            [6.055030345916748 , 1.7517801523208618  ],

            [4.111688613891602 , 2.274160385131836  ],

            [6.140777587890625 , 2.3119454383850098  ],
            ])
        
        self.objective_idx = 0
        self.yaw = 0

        self.pos = numpy.zeros(2)

        # Params
        self.dist_threshold = 0.2
        self.angular_threshold = 0.20
        self.angular_speed = 0.5
        self.linear_speed = 0.8
    
        # UWB Subscriber
        self.pos_sub = self.create_subscription(PoseStamped, "/vrpn_client_node/turtlebot3_cap/pose",  self.pos_cb, 10)
        # self.vio_sub = rospy.Subscriber("/t265/odom/sample", Odometry, self.odom_cb)

        # Velocity Publisher
        self.vel_pub = self.create_publisher(Twist, "/turtle03/cmd_vel", 10)

    def euler_from_quaternion(self, quaternion):
        """
        Converts quaternion (w in last place) to euler roll, pitch, yaw
        quaternion = [x, y, z, w]
        Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
        """
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        # yaw = (yaw % 2*numpy.pi - numpy.pi)

        return roll, pitch, yaw
        

    def orientation_to_euler(self, orientation):
        orientation_q = orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, yaw) = self.euler_from_quaternion(orientation)
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
        self.yaw = self.orientation_to_euler(msg.pose.orientation)

    def control(self) :
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

        if abs(self.yaw - theta) > self.angular_threshold and (abs((self.yaw - theta)) % 2*numpy.pi) > self.angular_threshold:
            
            # if self.yaw < theta :
            vel.angular.z = self.angular_speed
            # else :
                # vel.angular.z = -self.angular_speed
            
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
    controller_timer = controller.create_timer(1/10.0, controller.control)

    # time.sleep(1)
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
        controller_timer.destroy()
        controller.destroy_node()   

    

if __name__ == '__main__':
    main()
