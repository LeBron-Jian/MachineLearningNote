from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(111, projection ='3d')

# Make data

u = np.linspace(0, 2 * np.pi, 100)

v = np.linspace(0, np.pi, 100)

x = 10 * np.outer(np.cos(u), np.sin(v))

y = 10 * np.outer(np.sin(u), np.sin(v))

z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the surface

ax1.plot_surface(x, y, z, color='b')

plt.show()