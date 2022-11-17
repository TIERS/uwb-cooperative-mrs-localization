from audioop import bias
import matplotlib.pyplot as plt
import numpy as np

# from scipy.optimize import curve_fit
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression

#Maybe check this later
#https://www.askpython.com/python/examples/polynomial-regression-in-python

npzfile = np.load('data/bias_estimation_2robots_uwb.npz')

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
plt.savefig('images/bias_estimation_2robot_uwb.png')
plt.show()

data=np.stack((orientation_np,bias_np),axis=-1)
ind=np.argsort(data[:,0])
data_sorted=data[ind]


p=np.polyfit(data_sorted[:,0],data_sorted[:,1],5)
print("Coeeficient values: {}".format(p))
predict = np.poly1d(p)
x_test = 15
print("\nGiven x_test value is: ", x_test)
y_pred = predict(x_test)
print("\nPredicted value of y_pred for given x_test is: ", y_pred)


x = np.arange(0, 360, 1)
y = predict(x)

plt.scatter(orientation_np,bias_np)
plt.plot(x,predict(x),color="red")
plt.savefig('images/bias_estimation_2robot_uwb_fit.png')
plt.show()
