# uwb_ranging_refine_with_spatial_detection
This repo contains code for refine the Ultral-Wideband ranging with the detected spatial information of an object seen by multiple robots at the same time.

## Dependency

ros2 galactic
pfilter

## Data
### Recorded ros2 bags from 2022/09/23. 

2robots_move_one_static: turtlebot4 and turtlebot1 was moving a circle while turltebot3 static. 

3robots_moving_circles: turltebot1, turltebot3, and turtlebot4 were all moving a circle at the same time and observed the chair from time to time.

### Recorded ros2 bags from 2022/09/28.
3robots_moving_1_static_01: turtlebot1, turtlebot3, and turtlebot4 were moving in a circle while turtlebot 5 static. 2 chairs were added during the recording as objects. Topics can be seen as follows.
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
currently mainly run the code in 
```
pfilter_ros2_multi_robots.py
```
### Calibrate odom
The odom has translations compared with its global position. So we need to calibrate it and republish the topics to:
```
/cali/turtle01/odom
/cali/turtle03/odom
/cali/turtle04/odom
/cali/turtle05/odom
```
currently mainly run the code in 
```
script/cali_odom.py
```

### Results Visualization
currently mainly run the code in 
```
script/errors/
```
