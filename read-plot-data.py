import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib.font_manager as font_manager


# To load an array
# load array
Data = np.loadtxt('data.npy')
# print the array
#print(data)


fig = plt.figure()
font = {'family' : 'serif',  
        'color'  : 'black',  
        'weight' : 'normal',  
        'size'   : 22,  
        }
font_prop = font_manager.FontProperties(size=20)

X, Y = np.meshgrid(Earr,Karr)

color = iter(cm.summer(np.linspace(0, 1, Narr)))

ax = plt.axes(projection ='3d')

for id_da_plot in range(Narr):
    
    Z = Data[:,:,id_da_plot]

    c = next(color)
    
    surf = ax.plot_surface(X, Y, Z, label='$\delta$ = '+str(Darr[id_da_plot]), color=c)

    #this is to solve the legend/label error in 3d surface plots (error : 'AttributeError: 'Poly3DCollection' object has no attribute '_edgecolors2d'')
    surf._facecolors2d=surf._facecolor3d
    surf._edgecolors2d=surf._edgecolor3d

ax.set_xlabel('$\epsilon$')
ax.set_ylabel('$\kappa$')
ax.set_zlabel('$\chi$')
ax.legend() 

plt.show()