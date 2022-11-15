from audioop import bias
import matplotlib.pyplot as plt
import numpy as np

# x = np.random.normal(170, 10, 250)
rosbag="4robots_data_01"

npzfile = np.load('data/{}.npz'.format(rosbag))

# for k in npzfile.files:
#     print (k)

uwb_range_5_1_np = npzfile['uwb_range_5_1_np']
uwb_range_5_3_np = npzfile['uwb_range_5_3_np']
uwb_range_5_4_np = npzfile['uwb_range_5_4_np']
uwb_range_4_3_np = npzfile['uwb_range_4_3_np']
uwb_range_4_1_np = npzfile['uwb_range_4_1_np']
uwb_range_3_1_np = npzfile['uwb_range_3_1_np']
optitrack_turtle01_orientation_np = npzfile['optitrack_turtle01_orientation_np']
optitrack_turtle03_orientation_np = npzfile['optitrack_turtle03_orientation_np']
optitrack_turtle04_orientation_np = npzfile['optitrack_turtle04_orientation_np']
optitrack_turtle05_orientation_np = npzfile['optitrack_turtle05_orientation_np']
optitrack_turtle01_pose_np = npzfile['optitrack_turtle01_pose_np']
optitrack_turtle03_pose_np = npzfile['optitrack_turtle03_pose_np']
optitrack_turtle04_pose_np = npzfile['optitrack_turtle04_pose_np']
optitrack_turtle05_pose_np = npzfile['optitrack_turtle05_pose_np']

opti_distance_5_1_np = np.linalg.norm(optitrack_turtle05_pose_np-optitrack_turtle01_pose_np,axis=1)
opti_distance_5_3_np = np.linalg.norm(optitrack_turtle05_pose_np-optitrack_turtle03_pose_np,axis=1)
opti_distance_5_4_np = np.linalg.norm(optitrack_turtle05_pose_np-optitrack_turtle04_pose_np,axis=1)

opti_distance_4_3_np= np.linalg.norm(optitrack_turtle04_pose_np-optitrack_turtle03_pose_np,axis=1)
opti_distance_4_1_np = np.linalg.norm(optitrack_turtle04_pose_np-optitrack_turtle01_pose_np,axis=1)
opti_distance_3_1_np = np.linalg.norm(optitrack_turtle03_pose_np-optitrack_turtle01_pose_np,axis=1)


bias_5_1_np=uwb_range_5_1_np - opti_distance_5_1_np
bias_5_3_np=uwb_range_5_3_np - opti_distance_5_3_np
bias_5_4_np=uwb_range_5_4_np - opti_distance_5_4_np

bias_4_3_np=uwb_range_4_3_np - opti_distance_4_3_np
bias_4_1_np=uwb_range_4_1_np - opti_distance_4_1_np
bias_3_1_np=uwb_range_3_1_np - opti_distance_3_1_np

# print(len(uwb_range_4_1_np))
# print(len(opti_distance_4_1_np))

# for i in range(0,10):
# for i in range(0,len(bias_3_1_np)):
    # print(opti_distance_3_1_np[i], uwb_range_3_1_np[i],bias_3_1_np[i])

# orientation_5_1_np = np.rad2deg(optitrack_turtle01_orientation_np-optitrack_turtle05_orientation_np)
# orientation_5_3_np = np.rad2deg(optitrack_turtle03_orientation_np-optitrack_turtle05_orientation_np)
# orientation_5_4_np = np.rad2deg(optitrack_turtle04_orientation_np-optitrack_turtle05_orientation_np)

orientation_5_1_np = np.rad2deg(optitrack_turtle01_orientation_np-optitrack_turtle05_orientation_np)
orientation_5_3_np = np.rad2deg(optitrack_turtle03_orientation_np-optitrack_turtle05_orientation_np)
orientation_5_4_np = np.rad2deg(optitrack_turtle04_orientation_np-optitrack_turtle05_orientation_np)

orientation_4_3_np = np.rad2deg(optitrack_turtle03_orientation_np-optitrack_turtle04_orientation_np)
orientation_4_1_np = np.rad2deg(optitrack_turtle01_orientation_np-optitrack_turtle04_orientation_np)
orientation_3_1_np = np.rad2deg(optitrack_turtle01_orientation_np-optitrack_turtle03_orientation_np)

# for i in range(0,len(optitrack_turtle05_orientation_np)):
    # print("{} {}".format(np.rad2deg(optitrack_turtle05_orientation_np[i]),np.rad2deg(optitrack_turtle04_orientation_np[i])))

# for i,value in enumerate(orientation_5_1_np):
#     if value<0:        
#         orientation_5_1_np[i]=360+value

# for i,value in enumerate(orientation_5_3_np):
#     if value<0:        
#         orientation_5_3_np[i]=360+value

# for i,value in enumerate(orientation_5_4_np):
#     if value<0:        
#         orientation_5_4_np[i]=360+value


# print(orientation_np)
# print(bias_np)
#############################################################################################
print("Turtlebot 5")
average_5_1 = np.average(bias_5_1_np)
average_5_3 = np.average(bias_5_3_np)
average_5_4 = np.average(bias_5_4_np)
print("Error average51: {}".format(average_5_1))
print("Error average53: {}".format(average_5_3))
print("Error average54: {}".format(average_5_4))

std_5_1 = np.std(bias_5_1_np, ddof=1)
std_5_3 = np.std(bias_5_3_np, ddof=1)
std_5_4 = np.std(bias_5_4_np, ddof=1)
print("Standard deviation51: {}".format(std_5_1))
print("Standard deviation53: {}".format(std_5_3))
print("Standard deviation54: {}".format(std_5_4))

plt.scatter(orientation_5_1_np,bias_5_1_np,label="5-1")
plt.scatter(orientation_5_3_np,bias_5_3_np,label="5-3")
plt.scatter(orientation_5_4_np,bias_5_4_np,label="5-4")
plt.legend()
plt.ylim(-0.2,2)
plt.savefig('images/{}_node5.png'.format(rosbag)) 
plt.show()

###############################################################################################
print("Turtlebot 4")
average_4_3 = np.average(bias_4_3_np)
average_4_1 = np.average(bias_4_1_np)
average_5_4 = np.average(bias_5_4_np)
print("Error average43: {}".format(average_4_3))
print("Error average41: {}".format(average_4_1))
print("Error average54: {}".format(average_5_4))

std_4_3 = np.std(bias_4_3_np, ddof=1)
std_4_1 = np.std(bias_4_1_np, ddof=1)
std_5_4 = np.std(bias_5_4_np, ddof=1)
print("Standard deviation43: {}".format(std_4_3))
print("Standard deviation41: {}".format(std_4_1))
print("Standard deviation54: {}".format(std_5_4))

plt.scatter(orientation_4_3_np,bias_4_3_np,label="4-3")
plt.scatter(orientation_4_1_np,bias_4_1_np,label="4-1")
plt.scatter(orientation_5_4_np,bias_5_4_np,label="5-4")
plt.legend()
plt.ylim(-0.2,2)
plt.savefig('images/{}_4.png'.format(rosbag)) 
plt.show()

###############################################################################################
print("Turtlebot 3")
average_4_3 = np.average(bias_4_3_np)
average_3_1 = np.average(bias_3_1_np)
average_5_3 = np.average(bias_5_3_np)
print("Error average43: {}".format(average_4_3))
print("Error average31: {}".format(average_3_1))
print("Error average53: {}".format(average_5_3))

std_4_3 = np.std(bias_4_3_np, ddof=1)
std_3_1 = np.std(bias_3_1_np, ddof=1)
std_5_3 = np.std(bias_5_3_np, ddof=1)
print("Standard deviation43: {}".format(std_4_3))
print("Standard deviation31: {}".format(std_3_1))
print("Standard deviation53: {}".format(std_5_3))

plt.scatter(orientation_4_3_np,bias_4_3_np,label="4-3")
plt.scatter(orientation_3_1_np,bias_3_1_np,label="3-1")
plt.scatter(orientation_5_3_np,bias_5_3_np,label="5-3")
plt.legend()
plt.ylim(-0.2,2)
plt.savefig('images/{}_3.png'.format(rosbag)) 
plt.show()

# ###############################################################################################

# plt.scatter(orientation_5_1_np,bias_5_1_np,label="5-1")
# plt.scatter(orientation_5_3_np,bias_5_3_np,label="5-3")
# plt.scatter(orientation_5_4_np,bias_5_4_np,label="5-4")
# plt.legend()
# plt.ylim(-0.2,1)
# plt.show()

#####Polinomial fit for 54

bias_5_4_np[abs(bias_5_4_np) > 3 ] = average_5_4

data=np.stack((orientation_5_4_np,bias_5_4_np),axis=-1)
ind=np.argsort(data[:,0])
data_sorted=data[ind]

p=np.polyfit(data_sorted[:,0],data_sorted[:,1],5)
print("Coeeficient values: {}".format(p))
predict = np.poly1d(p)
x_test = 15
print("\nGiven x_test value is: ", x_test)
y_pred = predict(x_test)
print("\nPredicted value of y_pred for given x_test is: ", y_pred)


x = np.arange(min(data_sorted[:,0]), max(data_sorted[:,0]), 1)
y = predict(x)

plt.scatter(orientation_5_4_np,bias_5_4_np)
plt.plot(x,predict(x),color="red")
plt.ylim(-0.2,2)
plt.savefig('images/{}_fit54.png'.format(rosbag))
plt.show()