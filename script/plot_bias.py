import matplotlib.pyplot as plt
import numpy as np

# x = np.random.normal(170, 10, 250)
npzfile = np.load('bias_estimation.npz')
orientation_np=npzfile['orientation_np'] 
bias_np=npzfile['bias_np'] 



orientation_np = np.rad2deg(orientation_np)

# print(orientation_np)
# print(bias_np)


plt.scatter(orientation_np,bias_np)
plt.show() 