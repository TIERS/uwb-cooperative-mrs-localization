from cProfile import label
from distutils.log import error
import numpy as np
import matplotlib.pyplot as plt


# pos_u = np.loadtxt('./pos/pos_u.csv')
# pos_u_v = np.loadtxt('./pos/pos_u_v.csv')
# pos_uv = np.loadtxt('./pos/pos_uv.csv')

errors_uwb = np.loadtxt('./errors/error_uwb.csv')
errors_uwb_1 = np.loadtxt('./errors/error_u_3.csv')
errors_uwb_2 = np.loadtxt('./errors/error_u_13.csv')
errors_uwb_3 = np.loadtxt('./errors/error_u_134.csv')
errors_uv = np.loadtxt('./errors/error_uv.csv')
# errors_u = np.loadtxt('./errors/error_u.csv')
# errors_u_v = np.loadtxt('./errors/error_u_v.csv')
# errors_uv = np.loadtxt('./errors/error_uv.csv')


plt.style.use('seaborn-whitegrid')

# fig = plt.figure()
# ax = plt.axes()

print(errors_uwb.size)
print(errors_uwb_1.size)
print(errors_uwb_2.size)
print(errors_uwb_3.size)

errors = [np.absolute(errors_uwb[10:errors_uwb.size - 5]), np.absolute(errors_uwb_1[10:errors_uwb_1.size - 5]), 
          np.absolute(errors_uwb_2[10: errors_uwb_2.size - 5]), np.absolute(errors_uwb_3[10:errors_uwb_3.size - 5]),
          np.absolute(errors_uv[10:errors_uv.size - 5])]

# plt.boxplot(errors)

fig = plt.figure(figsize =(10, 7))
ax = fig.add_subplot(111)

# Creating axes instance
bp = ax.boxplot(errors, patch_artist = True,
    notch ='True', vert = 0)

colors = ['#0000FF', '#00FF00',
  '#FFFF00', '#FF00FF']

for patch, color in zip(bp['boxes'], colors):
 patch.set_facecolor(color)

# changing color and linewidth of
# whiskers
for whisker in bp['whiskers']:
 whisker.set(color ='#8B008B',
    linewidth = 1.5,
    linestyle =":")

# changing color and linewidth of
# caps
for cap in bp['caps']:
 cap.set(color ='#8B008B',
   linewidth = 2)

# changing color and linewidth of
# medians
for median in bp['medians']:
 median.set(color ='red',
   linewidth = 3)

# changing style of fliers
for flier in bp['fliers']:
 flier.set(marker ='D',
   color ='#e7298a',
   alpha = 0.5)
 
# x-axis labels
ax.set_yticklabels(['real uwb ranges', '1 uwb ranges meas',
     '2 uwb ranges meas','3 uwb range meas', '3 uwb + 1 vis'])

# Adding title
plt.title("uwb range error box plot")

# Removing top axes and right axes
# ticks
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
 
# show plot
plt.show()



# x = np.arange(errors_uwb.shape[0])
# plt.plot(x, errors_uwb, label='only uwb ranges')

# y = np.arange(errors_u.shape[0])
# plt.plot(y, errors_u, label='uwb ranges with pf')

# z = np.arange(errors_u_v.shape[0])
# plt.plot(z, errors_u_v, label='uwb ranges integrating spatial (one meas) with pf')

# j = np.arange(errors_uv.shape[0])
# plt.plot(j, errors_uv, label='uwb ranges integrating spatial (two meas) with pf')

# plt.legend()
# plt.show()
# print(errors[0])