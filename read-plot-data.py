import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib.font_manager as font_manager

import h5py

# read from a data set
filename = 'data_2022-03-11_13-45-15'
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


## plot figures
font = {'family' : 'serif',  
        'color'  : 'black',  
        'weight' : 'normal',  
        'size'   : 22,  
        }
font_prop = font_manager.FontProperties(size=20)

# plot 4D-info via \epsilon-\kappa surfaces on \chi for multiple \delta
plt.figure(1)

X, Y = np.meshgrid(Earr,Karr)

color = iter(cm.summer(np.linspace(0,1, Narr)))

ax = plt.axes(projection ='3d')

step_da_plot = 3

for id_da_plot in range(0,Narr,step_da_plot):
    
    Z = 1./Chi[:,:,id_da_plot]

    c = next(color)
    
    surf = ax.plot_surface(X, Y, Z, label='$\delta$ = '+str(round(Darr[id_da_plot],1)), color=c)
    # surf = ax.plot_surface(X, Y, Z, color=c)

    #this is to solve the legend/label error in 3d surface plots (error : 'AttributeError: 'Poly3DCollection' object has no attribute '_edgecolors2d'')
    surf._facecolors2d=surf._facecolor3d
    surf._edgecolors2d=surf._edgecolor3d

ax.set_xlabel('$\epsilon$')
ax.set_ylabel('$\kappa$')
ax.set_zlabel('$\chi^{-1}$')
ax.set_title('($\epsilon,\kappa$)-surfaces of $\chi^{-1}$ for different values of $\delta$')
ax.legend() 

#plt.savefig('parametric-study.eps', format='eps')
#plt.savefig('figures/fig-chi_'+filename+'.png', dpi=300)


# plot \epsilon vs. \chi for different \kappa fixing \delta at mean value

step_ka_plot = int(Narr/5)

for id_ka_plot in range(0,Narr,step_ka_plot):

    
    Z = 1./Chi[:,id_ka_plot,int(Narr/2)]
    
    plt.figure(2)
    plt.plot(Earr, Z, label='$\kappa$ = '+str(round(Karr[id_ka_plot],1)))

    plt.xlabel('$\epsilon$',fontdict=font)
    plt.ylabel('$\chi^{-1}}$',fontdict=font)
    plt.title('$\chi^{-1}$ vs. $\epsilon$ for fixed $\delta$ = ' + str(round(Darr[int(Narr/2)],1)))
    plt.grid(True)
    plt.legend()

# plot \epsilon vs. \chi for different \delta fixing \kappa at mean value

step_da_plot = int(Narr/5)

for id_da_plot in range(0,Narr,step_da_plot):

    
    Z = 1./Chi[:,int(Narr/2),id_da_plot]
    
    plt.figure(3)
    plt.plot(Earr, Z, label='$\delta$ = '+str(round(Darr[id_da_plot],1)))

    plt.xlabel('$\epsilon$',fontdict=font)
    plt.ylabel('$\chi^{-1}}$',fontdict=font)
    plt.title('$\chi^{-1}$ vs. $\epsilon$ for fixed $\kappa$ = ' + str(round(Karr[int(Narr/2)],1)))
    plt.grid(True)
    plt.legend()

plt.show()