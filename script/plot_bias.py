from audioop import bias
import matplotlib.pyplot as plt
import numpy as np

# x = np.random.normal(170, 10, 250)
npzfile = np.load('data/bias_estimation_2robot_uwb.npz')

opti_distance_np = npzfile['opti_distance_np'] 
uwb_range_np = npzfile['uwb_range_np']
bias_np = npzfile['bias_np']
orientation_np = npzfile['orientation_np']
optitrack_turtle01_orientation_np = npzfile['optitrack_turtle01_orientation_np']
optitrack_turtle03_orientation_np = npzfile['optitrack_turtle03_orientation_np']



orientation_np = np.rad2deg(optitrack_turtle01_orientation_np-optitrack_turtle03_orientation_np)

for i,value in enumerate(orientation_np):
    if value<0:        
        orientation_np[i]=360+value


# print(orientation_np)
# print(bias_np)

average = np.average(bias_np)
print("Error average: {}".format(average))

std = np.std(bias_np, ddof=1)
print("Standard deviation: {}".format(std))

plt.scatter(orientation_np,bias_np)
plt.savefig('data/bias_estimation_2robot_uwb.png')
plt.show()