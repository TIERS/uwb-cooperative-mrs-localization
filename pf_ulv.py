#!/usr/bin/env python
import numpy as np
from itertools import chain

from utlis import utils
from pfilter import ParticleFilter, squared_error, gaussian_noise

from tensorflow import keras

class UWBParticleFilter():
    def __init__(self, spatial_enable = False, lstm_enable = False, identical_thresh = 0.1, robot_ids = [0,1,2]):
        self.init_roi = (-8, 8)
        self.num_particles = 50
        self.num_states = 2*len(robot_ids)
        self.esti_noise = 0.05
        self.weights_sigma = 1.2
        self.resample_proportion = 0.01
        self.pf_init_flag = False
        self.data_ready_flag = False
        self.first_iter_flag = True

        self.spatial_enable = spatial_enable
        self.lstm_enable = lstm_enable
        self.identical_thresh = identical_thresh
        self.robot_ids = robot_ids

        self.robot_poses = []
        self.odom_trans = []
        self.odom_trans_prev = []
        if self.spatial_enable:
            self.cooperative_spatial_dict = {}

        if self.lstm_enable:
            self.lstm_steps = 30
            self.lstm_input_dict = {}
            self.models = {}

    '''
        Initialize particle filter
    '''
    def pf_filter_init(self):

        self.prior_fn = lambda n:np.random.uniform(self.init_roi[0],self.init_roi[1], (n,self.num_states))

        self.pf = ParticleFilter(
            prior_fn =              self.prior_fn, 
            observe_fn =            self.calc_hypothesis,  
            dynamics_fn =           self.motion_model, 
            n_particles =           self.num_particles, 
            noise_fn =              self.add_noise, 
            weight_fn =             self.calc_weights,
            resample_proportion =   self.resample_proportion
        )

        self.pf.init_filter()
        self.pf_init_flag = True



    '''
        Update particles' states with odometry data
    '''
    def motion_model(self, x):
        return x + self.particle_odom

    '''
        add normal noise to estimation
    '''
    def add_noise(self, x) :
        return x + np.random.normal(0, self.esti_noise, x.shape)

    '''
        update the hypothesis based on the particle filter states
    '''
    def calc_hypothesis(self, x) :
        hypo = np.linalg.norm(np.array([x[:,0:2]]), axis=2)
        if self.spatial_enable:
            for key in self.cooperative_spatial_dict:
                sp0 = self.cooperative_spatial_dict[key] - x[:,0:2]
                sp1 = self.cooperative_spatial_dict[key] - x[:,0:2]
                rp = sp1 - sp0
                hypo.append(rp[:,0])
                hypo.append(rp[:,1])
        return np.transpose(hypo) 

    '''
        Calculate particle weights based on squared_error
    '''
    def calc_weights(self, hypotheses, observations) :        
        return squared_error(hypotheses, observations, sigma=self.weights_sigma)


    '''
        Update the particle filters based on the odometry data movement info
    '''    
    def updata_particle_odom(self):
        self.odom_trans = list(chain.from_iterable([[val[0],val[1]] for val in self.odom_data]))
        if not self.odom_trans_prev:
            self.odom_trans_prev = self.odom_trans
        self.particle_odom = [x - y for x, y in zip(self.odom_trans, self.odom_trans_prev)]
        self.odom_data_prev = self.odom_trans

    def update_robots_poses(self):
        for id in self.robot_ids:
            self.robot_poses.append([self.pf.mean_state[2*id], self.pf.mean_state[2*id+1]])

    def update_input(self, uwb_data, odom_data):
        self.uwb_dict = uwb_data
        self.odom_data = odom_data
        self.data_ready_flag = True

    def update_input(self, uwb_data, odom_data, spatial_data):
        self.uwb_dict = uwb_data
        self.odom_data = odom_data
        self.spatial_dict = spatial_data
        self.data_ready_flag = True

    # function to check if the detection is identical for each robot
    def identical_detection(self, det, p):
        dp0 = det - p[0]
        dp1 = det - p[1]
        return np.linalg.norm(dp1 - dp0) < self.identical_thresh
    

    # loop all the detections and robot pairs to check if there is identical detection, if yes, add to the spatial_dict
    def detection_iterate(self, dets, rpairs, poses, thresh, spatial_dict):
        for det in dets:
            for p in rpairs:
                if self.identical_detection(det, poses[p[0]], poses[p[1]], thresh):
                    return spatial_dict[p].append(det)

    # loop all the detections and robot pairs to check if there is identical detection, if yes, add to the spatial_dict
    def detection_iterate(self, robot_ids, spatial_dict):
        for id in spatial_dict:
            if len(spatial_dict[id]) > 0:
                for det in spatial_dict[id]:
                    for p in robot_ids:
                        if p != id:
                            if self.identical_detection(det, (self.robot_poses[id], self.robot_poses[p])):
                                self.cooperative_spatial_dict[(id, p)].append(det)

    def set_lstm_modes(self, model_paths):
        for key in self.uwb_dict:
            self.models[key] = keras.models.load_model(model_paths[key])

    def set_lstm_input(self):
        for key in self.uwb_dict:
            self.lstm_input_dict[key].append([self.uwb_dict[key], self.odom_data[key[0]][2], self.odom_data[key[1]][2]])

        if len(list(self.lstm_input_dict.values())[0]) > self.lstm_steps:
            for key in self.uwb_dict:
                lstm_input_arr = np.array(self.lstm_input_dict[key][-self.lstm_steps:])
                bia = self.models[key].predict(np.reshape(lstm_input_arr,(1, self.lstm_steps, 3)), verbose = 0)
                uwb_prev = self.uwb_dict[key]
                self.uwb_dict[key] = uwb_prev - bia[0]

    def set_observation(self):
        self.observation = [] 
        for key in self.uwb_dict:
            self.observation.append(self.uwb_dict[key])
        if self.spatial_enable:
            for key in self.cooperative_spatial_dict:
                sp0 = self.cooperative_spatial_dict[key] - self.robot_poses[key[0]]
                sp1 = self.cooperative_spatial_dict[key] - self.robot_poses[key[1]]
                rp = sp1 - sp0
                self.observation.append(rp[0])
                self.observation.append(rp[1])

    def get_robot_poses(self):
        return self.robot_poses

    '''
        Update the particle filter
    '''
    def update_filter(self):
        if self.data_ready_flag and not self.pf_init_flag:
            self.pf_filter_init()
            
        if self.pf_filter_init:
            self.updata_particle_odom()
            if self.lstm_enable:
                self.set_lstm_input()
            if not self.first_iter_flag and self.spatial_enable: 
                self.detection_iterate(self.robot_ids, self.spatial_dict)
            self.set_observation()
            self.pf.update(observed=self.observation)
            self.update_robots_poses()
            self.first_iter_flag = False


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import csv

# Function to parse the list from a string representation
def parse_list(cell_value):
    return ast.literal_eval(cell_value)

all_topics = [
    '/uwb/tof/n_4/n_1/distance', '/uwb/tof/n_4/n_2/distance', '/uwb/tof/n_4/n_3/distance',
    '/uwb/tof/n_4/n_5/distance', '/uwb/tof/n_1/n_2/distance', '/uwb/tof/n_1/n_3/distance',
    '/uwb/tof/n_1/n_5/distance', '/uwb/tof/n_2/n_3/distance', '/uwb/tof/n_2/n_5/distance',
    '/uwb/tof/n_3/n_5/distance', '/vrpn_client_node/tb01/pose', '/vrpn_client_node/tb02/pose',
    '/vrpn_client_node/tb03/pose', '/vrpn_client_node/tb05/pose', '/turtle01/odom',
    '/turtle02/odom', '/turtle03/odom', '/turtle05/odom'
]

# topics needs to be aligned:
l = {'/turtle01/odom': parse_list, '/turtle02/odom': parse_list, '/turtle03/odom': parse_list, '/turtle05/odom': parse_list,
     '/vrpn_client_node/tb01/pose': parse_list, '/vrpn_client_node/tb02/pose': parse_list, '/vrpn_client_node/tb03/pose': parse_list,'/vrpn_client_node/tb05/pose': parse_list}

if __name__ == "__main__":
    # TODO0: save the aligned trajctories to a csv file and copy it back to the original data csv file 
    # TODO1: read the csv file and loop the data to update the particle filter; 
        # TODO1.1).only uwb and odometry data
        # TODO1.2).uwb, odometry and lstm corrected ranges <not working yet, should be a quick check>
        # TODO1.3).uwb, odometry, lstm corrected ranges and spatial detections <not working yet, should be a quick check>
    # TODO2: save the results to a csv file
    # TODO3: plot the results and save it into a .tex file

    # Loop the data to update the particle filter
    # The data should be in a format of dictionary: 
    #       uwb distance (uwb and robot share the same index): [(robot_i, robot_j), distance]
    #       odometry data: [[x0,y0,x1,y1,x2,y2]
    #       spatial detections: [robot_i, detections]

    # define the data variables,fake data to test
    robot_ids = [0,1,2,3,4]
    uwb_pairs = [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    uwb_ranges_dict = {pair:np.array([0.1, 0.2]) for pair in uwb_pairs}
    odom_data = [np.array([0.1, 0.1, 0.1]), np.array([0.2,0.2,0.2]), np.array([0.4,0.4,0.4])]
    spatial_dict = {0:[np.array([1,2]), np.array([3,4])], 1:[np.array([2,3]), np.array([4,5])], 2:[np.array([3,4]), np.array([5,6])]}
    
    # read data
    data = pd.read_csv('data.csv', converters=l).to_numpy()
    print(data.shape)


    # intialize the particle filter
    uwb_pf = UWBParticleFilter(spatial_enable=False, lstm_enable=False, robot_ids = [0,1,2])  
    # models_path = {0: 'models/robot0.h5', 1: 'models/robot1.h5', 2: 'models/robot2.h5'}
    # uwb_pf.set_lstm_models(models_path) 

    for i in range(10):
        uwb_pf.update_input(uwb_ranges_dict, odom_data, spatial_dict)
        uwb_pf.update_filter()
        print(uwb_pf.get_robot_poses())

