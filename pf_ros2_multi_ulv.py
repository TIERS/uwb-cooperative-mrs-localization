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

from tensorflow                 import keras

from utlis                      import utils

fake_odom       = True
# with_polyfit    = True
fuse_name       = ["u", "uv"]
turtles         = ["4", "1", "2"  , "3", "5"]
spatial_pair    = list(itertools.combinations(turtles,2))
# print(spatial_pair)
# [('4', '1'), ('4', '2'), ('4', '3'), ('4', '5'), ('1', '2'), ('1', '3'), ('1', '5'), ('2', '3'), ('2', '5'), ('3', '5')]
uwbs            = ["4", "1", "2"  , "3", "5"]
uwb_pair        = [(4,1), (4,2), (4,3), (4,5), (1,2), (1,3), (1,5), (2,3), (2,5), (3,5)]
spatial_uwb     = {spatial_pair[i]: i for i in range(len(uwb_pair))}
spatial_dict    = {sp:[] for sp in spatial_pair}
uwb_turtles     = [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
# uwb_odoms       = [(2,1), (3,1), (0,1), (2,3), (0,2), (0,3), (1,0), (2,0), (3,0), (1,0)]


#  get parameters from terminal
def parse_args():
    parser = argparse.ArgumentParser(description='Options to control relative localization with only UWB, assisit with Vision, and all if vision available')
    parser.add_argument('--poses_pub', type=bool, default=True, help='choose to publish the estimated poses with pf')
    parser.add_argument('--poses_save', type=bool, default=True, help='choose to save the estimated poses with pf')
    parser.add_argument('--images_save', type=bool, default=False, help='choose to save the images with pf')
    parser.add_argument('--computation_save', type=bool, default=True, help='choose to save the computation time with pf')
    parser.add_argument('--fuse_group', type=int, default=0, help='0: only UWB in PF, 1: uwb and vision together')
    parser.add_argument('--with_model', type=utils.str2bool, default=False, help=' choose to model the uwb error or not')
    parser.add_argument('--round', type=int, default=0, help='indicate which round the pf will run on a recorded data')
    args = parser.parse_args()
    return args

args = parse_args()

# Build folder to save results from different fusion combinations
if args.poses_save:
    pos_folder = "./results/results_csv/pfilter/pos/pos_{}/".format(fuse_name[args.fuse_group])
    pos_file = pos_folder + 'pos_{}.csv'.format(args.round)
    if not os.path.exists(pos_folder):
        os.makedirs(pos_folder)

if args.images_save:
    images_save_path = './results/results_csv/pfilter/images/images_{}/images_{}_{}/'.format(fuse_name[args.fuse_group], fuse_name[args.fuse_group], args.round)
    if not os.path.exists(images_save_path):
        os.makedirs(images_save_path)

if args.computation_save:
    computation_save_path = "./results/results_csv/pfilter/computation/computation_{}/".format(fuse_name[args.fuse_group])
    computation_file = computation_save_path + 'computation_time_{}.csv'.format(args.round)
    if not os.path.exists(computation_save_path):
        os.makedirs(computation_save_path)

class UWBParticleFilter(Node) :
    '''
        ROS Node that estimates relative position of two robots using odom and single uwb range.
    '''
    def __init__(self) :
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

        # Particle filter params
        self.declare_parameters(
            namespace = '',
            parameters=[
                ("weights_sigma", 1.2),
                ("num_particles", 400),
                ("uwb_noise", 0.10),
                ("resample_proportion", 0.01),
                ("max_pos_delay", 0.2)
            ]
        )
        
        self.weights_sigma = self.get_parameter('weights_sigma').value
        self.num_particles = self.get_parameter('num_particles').value
        self.uwb_noise = self.get_parameter('uwb_noise').value
        self.resample_proportion = self.get_parameter('resample_proportion').value

        self.get_logger().info('weights_sigma: %f, num_particles: %d, uwb_noise: %f, resample_proportion: %f'  % 
                            (self.weights_sigma,
                             self.num_particles,
                             self.uwb_noise,
                             self.resample_proportion))

        # all varibles 
        self.vision_flag            = False
        self.num_vision_found       = 0
        self.num_vision_used        = 0
        self.num_states             = 8
        self.counter                = 0
        self.uwb_ranges             = [0.0 for _ in uwb_pair]
        self.turtles_mocaps         = [np.zeros(6) for _ in turtles]
        self.turtles_odoms_flag     = [False for _ in turtles]
        self.turtles_odoms          = [Odometry() for _ in turtles]
        self.last_turtles_odoms     = [Odometry() for _ in turtles]
        self.spatial_objects        = {t:np.array([]) for t in turtles}
        self.true_relative_poses    = [np.zeros(2) for _ in range(1,len(turtles))]
        self.relative_poses         = [np.zeros(2) for _ in range(1,len(turtles))]
        self.particle_odom          = np.array([0.001]*self.num_states)
        self.prior_init             = np.array([0.001]*self.num_states)
        self.pf_init_flag           = False
        self.fake_odom              = [np.zeros(2) for _ in turtles]
        self.fake_last_odom         = [np.zeros(2) for _ in turtles]
        self.vis_meas_list          = []
        self.vis_thresh             = 0.10
        self.sp_temp                = []
        self.computation_time       = []
        self.poly_coefficient       = [ 2.30932370e-13,  1.03347377e-11, -9.03676014e-08,  2.61712111e-05,  -2.07631167e-03,  2.15006000e-01]
        if args.with_model:
            self.models                 = [keras.models.load_model('/home/xianjia/Workspace/temp/lstm_ws/lstm_uwb_{}'.format(inx)) for inx in range(len(uwb_pair))]
            self.lstm_inputs            = [[] for _ in uwb_pair]
            self.n_steps                = 30
            self.uwb_lstm_ranges        = []
            self.uwb_real               = []
            self.uwb_inputs             = []
            time.sleep(1.0)

        self.get_logger().info("Subscribing to topics......")
        # subscribe to uwb ranges 
        self.uwb_subs = [
            self.create_subscription(Range, "/uwb/tof/n_{}/n_{}/distance".format(p[0], p[1]), 
            self.create_uwb_ranges_cb(i),qos_profile=self.qos) for i, p in enumerate(uwb_pair)]
        self.get_logger().info("{} UWB ranges subscribed!".format(len(self.uwb_ranges)))

        # subscribe to optitrack mocap poses
        self.mocap_subs = [
            self.create_subscription(PoseStamped, "/vrpn_client_node/tb0{}/pose".format(t), 
            self.create_mocap_pose_cb(i), qos_profile=self.qos) for i, t in enumerate(turtles)]
        self.get_logger().info("{} Mocaps poses subscribed!".format(len(self.turtles_mocaps)))
        
        # subscribe to odometries
        self.odom_subs = [
            self.create_subscription(Odometry, "/turtle0{}/odom".format(t), 
            self.create_odom_cb(i), qos_profile=self.qos) for i, t in enumerate(turtles)]
        self.get_logger().info("{} odom poses subscribed!".format(len(self.turtles_odoms)))

        # subscribe to spatial detections
        self.spatial_subs = [
            self.create_subscription(SpatialDetectionArray, "/turtle0{}/color/yolov4_Spatial_detections".format(t), 
            self.create_spatial_cb(i), qos_profile=self.qos) for i, t in enumerate(turtles)]
        self.get_logger().info("{} spatial detections subscribed!".format(len(self.spatial_objects)))

        timer_period = 1/20.0
        # self.pub_timer = self.create_timer(timer_period, self.pub_timer_callback)

        # pf relative poses publishers
        self.real_pose_publishers = [self.create_publisher(PoseStamped, '/real_turtle0{}_pose'.format(t), qos_profile=self.qos) for t in turtles]
        self.relative_pose_publishers = [self.create_publisher(PoseStamped, '/pf_turtle0{}_pose'.format(t), qos_profile=self.qos) for t in turtles[1:]]

        self.fake_odom_publishers = [self.create_publisher(Odometry, '/fake_t0{}_odom'.format(t), 10) for t in turtles]

        self.pos_estimation = []

        time.sleep(1.0)
        self.get_logger().info("\\\\\\Class Initialized Done//////")
        
        
    def create_uwb_ranges_cb(self, i):
        return lambda range : self.uwb_range_cb(i, range)
        
    def uwb_range_cb(self, i, range):
        self.uwb_ranges[i] = range.range
        # self.uwb_inputs = self.cal_lstm_input()
        if args.with_model:
            node1_mocap = self.turtles_mocaps[uwb_turtles[i][0]] 
            node2_mocap = self.turtles_mocaps[uwb_turtles[i][1]]
            node1_yaws = utils.euler_from_quaternion(np.array([node1_mocap[2], node1_mocap[3], node1_mocap[4],node1_mocap[5]]))
            node2_yaws = utils.euler_from_quaternion(np.array([node2_mocap[2], node2_mocap[3], node2_mocap[4],node2_mocap[5]]))
            self.lstm_inputs[i].append([self.uwb_ranges[i], node1_yaws, node2_yaws])
            if len(self.lstm_inputs[i]) > self.n_steps:
                lstm_input_arr = np.array(self.lstm_inputs[i][-self.n_steps:])
                bia = self.models[i].predict(np.reshape(lstm_input_arr,(1, self.n_steps, 3)), verbose = 0)
                self.uwb_ranges[i] = self.uwb_ranges[i] - bia[0]
            else:
                self.uwb_ranges[i] = self.uwb_ranges[i] - 0.32
        else:
            self.uwb_ranges[i] = self.uwb_ranges[i] - 0.32

    def create_mocap_pose_cb(self, i):
        return lambda pos : self.mocap_pose_cb(i, pos)
        
    def mocap_pose_cb(self, i, pos):
        self.turtles_mocaps[i] = np.array([pos.pose.position.x, pos.pose.position.y, pos.pose.orientation.x, pos.pose.orientation.y, pos.pose.orientation.z, pos.pose.orientation.w])  
        true_relative_pos = pos
        true_relative_pos.header.stamp = self.get_clock().now().to_msg()
        true_relative_pos.pose.position.x =  pos.pose.position.x - self.turtles_mocaps[0][0]
        true_relative_pos.pose.position.y =  pos.pose.position.y - self.turtles_mocaps[0][1]
        true_relative_pos.pose.position.z = 0.0
        self.real_pose_publishers[i].publish(true_relative_pos)

    def create_odom_cb(self, i):
        return lambda odom : self.odom_cb(i, odom)
        
    def odom_cb(self, i, odom):
        self.turtles_odoms_flag[i] = True
        self.turtles_odoms[i] = odom

    def create_spatial_cb(self, i):
        return lambda detections : self.spatial_cb(i, detections)
        
    def spatial_cb(self, i, detections):
        self.spatial_objects[turtles[i]] = np.array(detections.detections)

    def relative_pose_cal(self, origin, ends, relative_poses):
        for inx, end in enumerate(ends):
            relative_poses[inx] = end[0:2] - origin[0:2]   
    
    def update_range_from_object_pose(self, object_end_pose, object_ori_pose):
        robots_relative_pose = np.array([-object_end_pose.position.x + object_ori_pose.position.x,
                                         -object_end_pose.position.y + object_ori_pose.position.y,
                                         -object_end_pose.position.z + object_ori_pose.position.z])
        return np.linalg.norm(robots_relative_pose)
 
    def velocity(self, x) :
        '''
            Use Odom to update position
        '''
        xp = x + self.particle_odom
        return xp

    def add_noise(self, x) :
        '''
            Add noise to the estimations
            TODO add the noise to the measurements instead
        '''
        xp = x + np.random.normal(0, self.uwb_noise, x.shape)
        return xp

    def calc_hypothesis(self, x) :
        '''
            Given (Nx2) matrix of positions,
            create N arrays of observations (just one for now)
        '''  
        # start = time.time_ns()/10**9
        y = [] 
        temp = np.array([x[:,0:2], x[:,2:4], x[:,4:6], x[:,6:8],
                         x[:,2:4] - x[:,0:2], x[:,4:6] - x[:,0:2], x[:,6:8] - x[:,0:2],
                         x[:,4:6] - x[:,2:4], x[:,6:8] - x[:,2:4],
                         x[:,6:8] - x[:,4:6]
        ])
        y = np.linalg.norm(temp, axis=2)
        sp_temp = copy.deepcopy(self.sp_temp)
        if args.fuse_group == 1 and len(self.vis_meas_list) > 0:
            tmp = []
            for sp in range(len(sp_temp)):
                for _ in sp_temp[sp]:
                    tmp.append(temp[sp][:,0])
                    tmp.append(temp[sp][:,1])
            y = np.concatenate((y, np.array(tmp)), axis=0)
            self.vision_flag = False
        # print(f"hypothesis: {time.time_ns()/10**9 - start}")
        return np.transpose(y)

    def calc_weights(self, hypotheses, observations) :
        '''
            Calculate particle weights based on error
        '''
        return squared_error(hypotheses, observations, sigma=self.weights_sigma)
    

    def pf_filter_init(self):
        init_mocaps = np.array(self.turtles_mocaps)
        self.prior_init = init_mocaps[1:, 0:2]- np.array([self.turtles_mocaps[0][0], self.turtles_mocaps[0][1]])
        self.prior_fn = lambda n: self.prior_init.flatten() + np.random.normal(0,0.2,(n,self.num_states)) #np.random.uniform(-8,8,(n,8))+self.odoms_init

        self.pf = ParticleFilter(
            prior_fn =              self.prior_fn, 
            observe_fn =            self.calc_hypothesis,  
            dynamics_fn =           self.velocity, 
            n_particles =           self.num_particles, 
            noise_fn =              self.add_noise, 
            weight_fn =             self.calc_weights,
            resample_proportion =   self.resample_proportion
        )

        self.pf.init_filter()
        self.pf_init_flag = True

    def update_particle_odom(self):
        if fake_odom:
            for i in range(1, len(turtles)):
                self.particle_odom[2*(i-1)] = self.fake_odom[i][0] - self.fake_last_odom[i][0] - \
                                        (self.fake_odom[0][0] - self.fake_last_odom[0][0])
                self.particle_odom[2*(i-1)+1] = self.fake_odom[i][1] - self.fake_last_odom[i][1] - \
                                        (self.fake_odom[0][1] - self.fake_last_odom[0][1])
            self.fake_last_odom = np.copy(self.fake_odom)
        else:
            for i in range(1, len(turtles)):
                self.particle_odom[2*(i-1)] = (self.turtles_odoms[i].pose.pose.position.x - self.last_turtles_odoms[i].pose.pose.position.x - \
                                        (self.turtles_odoms[0].pose.pose.position.x - self.last_turtles_odoms[0].pose.pose.position.x))
                self.particle_odom[2*(i-1)+1] = (self.turtles_odoms[i].pose.pose.position.y - self.last_turtles_odoms[i].pose.pose.position.y - \
                                        (self.turtles_odoms[0].pose.pose.position.y - self.last_turtles_odoms[0].pose.pose.position.y))
            self.last_turtles_odoms = np.copy(self.turtles_odoms)

    def relative_poses_pub(self):
        # publish pf relative pose
        for i in range(len(turtles[1:])):
            relative_pose = PoseStamped()
            relative_pose.header.frame_id = "base_link"
            relative_pose.header.stamp = self.get_clock().now().to_msg()
            relative_pose.pose.position.x = self.pf.mean_state[2*i]
            relative_pose.pose.position.y = self.pf.mean_state[2*i+1]
            relative_pose.pose.position.z = 0.0
            relative_pose.pose.orientation = self.turtles_odoms[i].pose.pose.orientation
            self.relative_pose_publishers[i].publish(relative_pose)   

    def relative_poses_save(self):
        self.relative_pose_cal(self.turtles_mocaps[0], self.turtles_mocaps[1:], self.true_relative_poses)
        pf_relative_poses = [self.pf.mean_state[i] for i in range(self.num_states)]
        relative_poses = np.append(np.hstack(self.true_relative_poses), pf_relative_poses).tolist()
        # save groundtruth poses and calcuated poses to csv
        self.pos_estimation.append(relative_poses)

    def plot_particles(self):
        """Plot a 1D tracking result as a line graph with overlaid
        scatterplot of particles. Particles are sized according to
        normalised weight at each step.
            x: time values
            y: original (uncorrupted) values
            yn: noisy (observed) values
            states: dictionary return from apply_pfilter        
        """

        plt.ioff()
        plt.clf()

        symbols = ['x', 'o', '*']
        symbol_colors =['black', 'darkgray', 'lightgray']
        legends = [["T1_G", "T1_Mean", "T1_Map", "T1_Particles"],
                   ["T3_G", "T3_Mean", "T3_Map", "T3_Particles"],
                   ["T4_G", "T4_Mean", "T4_Map", "T4_Particles"]]
        for i in range(len(self.true_relative_poses)):
            plt.plot(self.true_relative_poses[i][0], self.true_relative_poses[i][1], symbols[i], c='red', label=legends[i][0])
            plt.plot(self.pf.mean_state[2*i], self.pf.mean_state[2*i+1], symbols[i], c='green', label=legends[i][1])
            plt.plot(self.pf.map_state[2*i], self.pf.map_state[2*i+1], symbols[i], c='orange', label=legends[i][2])
            plt.scatter(self.pf.transformed_particles[:,2*i], self.pf.transformed_particles[:,2*i+1], color=symbol_colors[i], label=legends[i][3]) # lightgray
        # print(f"particles shape:{self.pf.transformed_particles.shape}")
        plt.xlim(-9,9)
        plt.ylim(-9,9)

        plt.legend()
        self.counter += 1
        plt.savefig(images_save_path + "/test{}.png".format(self.counter))

    def cal_lstm_input(self):
        node1_mocap = [self.turtles_mocaps[ut[0]] for ut in uwb_turtles]
        node2_mocap = [self.turtles_mocaps[ut[1]] for ut in uwb_turtles]
        node1_yaws = [utils.euler_from_quaternion(np.array([mo[2], mo[3], mo[4],mo[5]])) for mo in node1_mocap]
        node2_yaws = [utils.euler_from_quaternion(np.array([mo[2], mo[3], mo[4],mo[5]])) for mo in node2_mocap]
        lstm_input_arrs = []
        for inx in range(len(self.uwb_ranges)):
            self.lstm_inputs[inx].append([self.uwb_ranges[inx], node1_yaws[inx], node2_yaws[inx]])
            if len(self.lstm_inputs[inx]) > self.n_steps:
                lstm_input_arr = np.array(self.lstm_inputs[inx][-self.n_steps:])
                lstm_input_arrs.append(lstm_input_arr)
            else:
                lstm_input_arrs.append([])
        return np.array(lstm_input_arrs)
    
    def fake_odom_fun(self):
        # self.get_logger().info("Set Fake Odom.")
        for t, t_cap in enumerate(self.turtles_mocaps):
            mean, std = 0.0, 0.02
            self.fake_odom[t][0] = t_cap[0] + np.random.normal(mean, std)
            self.fake_odom[t][1] = t_cap[1] + np.random.normal(mean, std)
            temp_odom = Odometry()
            temp_odom.pose.pose.position.x = self.fake_odom[t][0]
            temp_odom.pose.pose.position.y = self.fake_odom[t][1]
            temp_odom.pose.pose.position.z = 0.0
            # self.fake_odom_publishers[t].publish(temp_odom)

    def update_vision_measurements(self):
        uwb_ranges = copy.deepcopy(self.uwb_ranges)
        self.vis_meas_list.clear()
        if args.fuse_group == 1:
            for p in spatial_pair:
                spatial_dict[p].clear()
                if self.spatial_objects[p[0]].size > 0 and self.spatial_objects[p[1]].size > 0:
                    for obj in itertools.product(self.spatial_objects[p[0]], self.spatial_objects[p[1]]):
                            vis_meas = self.update_range_from_object_pose(obj[0], obj[1])
                            if math.fabs(vis_meas -  uwb_ranges[spatial_uwb[p]]) < self.vis_thresh:
                                self.num_vision_found+=1
                                if not self.vision_flag:
                                    spatial_dict[p].extend([[obj[0], obj[1]]])
                                    # vis_meas_list.append(vis_meas)
                                    self.vis_meas_list.append(obj[1].position.x - obj[0].position.x)
                                    self.vis_meas_list.append(obj[1].position.y - obj[0].position.y)
                                
    def update_vis_meas(self):
        self.vision_flag = True
        vis_meas_list = copy.deepcopy(self.vis_meas_list)
        self.sp_temp = [spatial_dict[key] for key in spatial_dict]
        self.num_vision_used += 1
        return vis_meas_list

    def get_measurements(self, uwb_ranges):
        new_meas = np.array(uwb_ranges,dtype=float)
        if args.fuse_group == 1 and len(self.vis_meas_list) > 0:
            vis_meas_list = self.update_vis_meas()
            new_meas = np.append(uwb_ranges, [vis_meas_list])
        return new_meas

    def update_lstm_uwb(self):
        self.uwb_inputs = self.cal_lstm_input()
        self.uwb_lstm_ranges = []
        for inx, u_input in enumerate(self.uwb_inputs):
            if u_input.size == 0:
                self.uwb_lstm_ranges.append(self.uwb_ranges[inx] - 0.32)
                # self.uwb_lstm_ranges =  [ur - 0.32 for ur in self.uwb_ranges]
            else:
                # start = time.time_ns()/10**9
                uwb_bia =  self.models[inx].predict(np.reshape(u_input, (1, self.n_steps, 3)), verbose = 0)             
                # print(f"prediction time: {time.time_ns()/10**9 - start}")
                self.uwb_lstm_ranges.append(self.uwb_ranges[inx] - uwb_bia[0])
            
    def update_filter(self) : # This is the main process of the PF.
        '''
            Upadate particle filter
        '''
        start = time.time_ns() / (10 ** 9) # for calculating the processing frequency
        if all(self.turtles_odoms_flag[1:]) and  not self.pf_init_flag:  # Check if receive all the odom data and if PF not initialized, then initialize
            self.pf_filter_init()
            self.get_logger().info("UWB PF initialized. Estimating position from UWB and odom.")

        if self.pf_init_flag: # If PF initialized
            if fake_odom:
                self.fake_odom_fun() # Update odom 

            # # check uwb measurements with lstm model or not, there is another thread (timer) updating the lstm corrected uwb ranges
            # if args.with_model > 0:
            #     # print('///////////// using lstm model ////////////////')
            #     uwb_ranges = copy.deepcopy(self.uwb_lstm_ranges)
            # else:
            #     uwb_ranges = [ur - 0.32 for ur in self.uwb_ranges]

            # uwb_ranges = [ur for ur in self.uwb_ranges]
            uwb_ranges = copy.deepcopy(self.uwb_ranges)
            
            new_meas = self.get_measurements(uwb_ranges) # check if vision needed, if yes, add vision. otherwise, use only the uwb measurements

            self.update_particle_odom() # Update the particle odom

            self.pf.update(observed=new_meas)

            for i in range(self.num_states) :
                if self.pf.cov_state[i][0] > 0.3 or self.pf.cov_state[i][1] > 0.3 :
                    self.get_logger().warn("PF covariance too high for Turtle {} with covx={} and covy={}".format(i/2, self.pf.cov_state[i][0], self.pf.cov_state[i][1]))

            if args.poses_pub:
                self.relative_poses_pub()

            if args.poses_save: 
                self.relative_poses_save()

            if args.images_save:
                self.plot_particles()
        end = time.time_ns() / (10 ** 9)
        self.computation_time.append(end - start)
        # print(f"pf time: {end - start}")

    def pub_timer_callback(self):
        if args.poses_pub and self.pf_init_flag:
            self.relative_poses_pub()

    def __del__(self):
        # body of destructor
        self.get_logger().info("PF ends and Saving Results. And in the process,{} vision found {} vision used".format(self.num_vision_found, self.num_vision_used))
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
    rclpy.init(args=args)
    filter = UWBParticleFilter()

    time.sleep(1)
    # Start calculating relative positions
    
    filter.get_logger().info("Starting particle filter...")
    filter_timer = filter.create_timer(1/5.0, filter.update_filter)
    # if args.with_model:
    #     uwb_timer = filter.create_timer(1/5.0, filter.update_lstm_uwb)
    vis_timer = filter.create_timer(1/15.0, filter.update_vision_measurements)

    # pose_pub_timer = filter.create_timer(1/6.0, filter.pose_pub)
    try:
        try:
            while rclpy.ok() :
                rclpy.spin(filter)             
                # rclpy_check_rate.sleep()
        except KeyboardInterrupt :
            filter.get_logger().error('Keyboard Interrupt detected! Trying to stop filter node!')
    except Exception as e:
        filter.destroy_node()
        filter.get_logger().info("UWB particle filter failed %r."%(e,))
    finally:
        rclpy.shutdown()
        filter.destroy_timer(filter_timer)
        filter.destroy_timer(vis_timer)
        # filter.destroy_timer(uwb_timer)
        # filter.destroy_timer(pose_pub_timer)
        filter.destroy_node()   

    

if __name__ == '__main__':
    main()