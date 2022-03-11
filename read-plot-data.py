import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib.font_manager as font_manager

import h5py

# read from a data set
filename = 'data2022.03.11-13.16.37'
hdf = h5py.File('data/'+filename+'.h5','r')

ls = list(hdf.keys())
print('List of datasets in this file: \n', ls)

temp = hdf.get('Earr')
Earr = np.array(temp)

temp = hdf.get('Karr')
Karr = np.array(temp)

temp = hdf.get('Darr')
Darr = np.array(temp)

temp = hdf.get('Chi')
Chi  = np.array(temp)

Narr = len(Earr)
print('Narr = ', Narr)

print('Shape of dataset1: \n', Chi.shape)
hdf.close()


# plot figures
fig = plt.figure()
font = {'family' : 'serif',  
        'color'  : 'black',  
        'weight' : 'normal',  
        'size'   : 22,  
        }
font_prop = font_manager.FontProperties(size=20)

X, Y = np.meshgrid(Earr,Karr)

color = iter(cm.summer(np.linspace(0,1, Narr)))

ax = plt.axes(projection ='3d')

for id_da_plot in range(Narr):
    
    Z = Chi[:,:,id_da_plot]

    c = next(color)
    
    surf = ax.plot_surface(X, Y, Z, label='$\delta$ = '+str(Darr[id_da_plot]), color=c)

    #this is to solve the legend/label error in 3d surface plots (error : 'AttributeError: 'Poly3DCollection' object has no attribute '_edgecolors2d'')
    surf._facecolors2d=surf._facecolor3d
    surf._edgecolors2d=surf._edgecolor3d

ax.set_xlabel('$\epsilon$')
ax.set_ylabel('$\kappa$')
ax.set_zlabel('$\chi$')
ax.legend() 

#plt.savefig('parametric-study.eps', format='eps')
#plt.savefig('figures/fig-chi_'+filename+'.png', dpi=300)
plt.show()