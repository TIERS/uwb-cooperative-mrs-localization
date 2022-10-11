# uwb_ranging_refine_with_spatial_detection
This repo contains code for refine the Ultral-Wideband ranging with the detected spatial information of an object seen by multiple robots at the same time.

## Dependency
<!-- TODO: add installation approaches -->
ros2 galactic
pfilter
numpyls
depthai_ros_msgs
matplotlib

## Data
### Recorded ros2 bags from 2022/09/23. 

2robots_move_one_static: turtlebot4 and turtlebot1 was moving a circle while turltebot3 static. 

3robots_moving_circles: turltebot1, turltebot3, and turtlebot4 were all moving a circle at the same time and observed the chair from time to time.

### Recorded ros2 bags from 2022/09/28.
3robots_moving_1_static_01: turtlebot1, turtlebot3, and turtlebot4 were moving in a circle while turtlebot 5 static. 2 chairs were added during the recording as objects.

### Recorded ros2 bags from 2022/10/04.
4robots_data_01: turtlebot1, turtlebot3, and turtlebot4 were moving in different circles while turtlebot 5 static. 2 chairs were added during the recording as objects. 

cali_4robots_data_01: The odometry calibrated version based on the above one.

Topics can be seen as follows.
```
# spatial detection results. turtle05 can always see one object.
/turtle01/color/yolov4_Spatial_detections
/turtle03/color/yolov4_Spatial_detections
/turtle04/color/yolov4_Spatial_detections
/turtle05/color/yolov4_Spatial_detections
# odometry data
/turtle01/odom
/turtle03/odom
/turtle04/odom
/turtle05/odom
# uwb range data
/uwb/tof/n_3/n_4/distance
/uwb/tof/n_3/n_5/distance
/uwb/tof/n_3/n_7/distance
/uwb/tof/n_4/n_5/distance
/uwb/tof/n_4/n_7/distance
/uwb/tof/n_7/n_5/distance
# optitrack position data
/vrpn_client_node/chair2/pose
/vrpn_client_node/chair_final/pose
/vrpn_client_node/turtlebot1_cap/pose
/vrpn_client_node/turtlebot3_cap/pose
/vrpn_client_node/turtlebot4_cap/pose
/vrpn_client_node/turtlebot5_cap/pose
```

## Run
### Positioning
#### Run Once
currently, if you only want to run one round of each filter,  mainly run the code in 
```
pfilter_ros2_uwb_position_multi_robots.py --fuse_group 0 --round 0
```
Arguments meaning(currently not used):
```
--fuse_group 
0: only uwb
1: uwb or vision
2: uwb and vision

--round  # this indicates how many rounds that your filter will run for each fusing group.
```

After running this code, the images of particles will be saved in 
```
../images/
```
The groudtruth and estimated relative pose will be in 
```
../pos/
```
The errors will be in 
```
../errors/
```
#### Run In a Loop
Run a script that can generate all the results of all rounds of different fusing group.
```
python script/run_filter.py
```


### Calibration
#### odom 
The odom has translations compared with its global position. So we need to calibrate it and republish the topics to:

```
/cali/turtle01/odom
/cali/turtle03/odom
/cali/turtle04/odom
/cali/turtle05/odom
```
currently mainly run the code in 
```
python script/cali_odom.py
```
#### uwb

currently mainly run the following. First run 
```
python script/bias_estimation.py
```
while running a rosbag. It will then save the information in

```
/data/bias_estimation.npz
```
Then run the script

```
python script/plot_bias.py
```
It will save the images in

```
/images/bias_estimation.png
```



#### stereo camera
<!-- FIXME: bias are big, needs to check the code -->
currently mainly run the code in 
```
script/camera_opti.py
```

### Results visualization
<!-- TODO: violin plot or rainbow plot -->
currently mainly run the code in 
```
script/errors/
```

#### Trajectory estimation
Best results for now.
##### Triangulation state estimation


##### Centralized PF state estimation
###### Trajectories
only UWB ranges             |  UWB ranges or spatial detection          |  UWB ranges and spatial detection
:-------------------------:|:-------------------------: |:-------------------------:
![](./demos/centralized_pf_u.png)  |  ![](./demos/centralized_pf_u_v.png)|![](./demos/centralized_pf_uv.png)

##### Federated PF state estimation
###### Errors
![](./demos/centralized_pf_xy_error.png)

## TODO
- [ ] add polyfit uwb bia value to the data input.
- [ ] run single pf in each robot.
- [ ] have a centeralized pf. 
- [ ] add feedback data from centeralized pf to local pf.
- [ ] improve visualization of results
- [x] add triangulation code and results.
