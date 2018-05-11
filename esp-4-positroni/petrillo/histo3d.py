from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt
import numpy as np
import sys
import lab4

filename = sys.argv[1]

ch1, ch2, ch3, c3 = lab4.loadtxt(filename, unpack=True, usecols=(0,1,2,9))
c3 = c3 > 500

fig = plt.figure('histo3d')
fig.clf()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(ch1[c3], ch2[c3], ch3[c3], color='black', marker='o')

ax.set_xlabel('ch1')
ax.set_ylabel('ch2')
ax.set_zlabel('ch3')

ax.set_xlim(220, 250)
ax.set_ylim(220, 250)
ax.set_zlim(200,300)



fig.show()
