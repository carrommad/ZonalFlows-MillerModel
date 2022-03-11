import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import h5py

x = np.linspace(-3, 3, 256)
y = np.linspace(-3, 3, 256)
X, Y = np.meshgrid(x, y)

fig = plt.figure()

n = 5

color = iter(cm.summer(np.linspace(0, 1, n)))

for id in range(n):
    Z = id + np.sinc(np.sqrt(X ** 2 + Y ** 2))

    c = next(color)
    
    ax = fig.gca(projection = '3d')
    surf = ax.plot_surface(X, Y, Z, label='$\delta$ = '+str(id), color=c)

    #this is to solve the legend error (error : 'AttributeError: 'Poly3DCollection' object has no attribute '_edgecolors2d'')
    surf._facecolors2d=surf._facecolor3d
    surf._edgecolors2d=surf._edgecolor3d

ax.set_xlabel('$\epsilon$', fontsize=20, rotation=150)
ax.set_ylabel('$\kappa$')
ax.set_zlabel('$\chi$', fontsize=30, rotation=60)
ax.legend() 

plt.show()


# create hdf5 data structure
with h5py.File('name.h5','w') as hdf:   
    hdf.create_dataset('dataset1', data=x)
    hdf.create_dataset('dataset2', data=y)
