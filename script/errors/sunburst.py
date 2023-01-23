import plotly.express as px
import matplotlib.pyplot as plt
import tikzplotlib as tikz
data = dict(
    character=["Eve", "Cain", "Seth", "Enos", "Noam", "Abel", "Awan", "Enoch", "Azura"],
    parent=["Jetson Nano", "Eve", "Eve", "Seth", "Seth", "Eve", "Eve", "Awan", "Eve" ],
    value=[10, 14, 12, 10, 2, 6, 6, 4, 4])

fig = px.sunburst(
    data,
    names='character',
    parents='parent',
    values='value',
)
fig.show()


FILENAME = "sunburst" 
plt.savefig('{}.png'.format(FILENAME), bbox_inches='tight')   
tikz.save("{}.tex".format(FILENAME)) 
# plt.show()

# % Jetson Nano, ARMv8, 1.5GHZ \\
# % PF: 23.2, 442
# % Tri: 21.3, 99
# % spatial: 331M

# % Rasperry Pi, 1.8Ghz\\
# % PF: 23.2, 83
# % Tri:22.4, 66
# % turtlebot4\_diagnostics: 12.8, 53
# % rp\_lidar: 0.5, 14
# % turtlebot4\_node: 1.2, 16

# % PC:
# % PF: 2.8, 390
# % Tri: 2.9, 69
