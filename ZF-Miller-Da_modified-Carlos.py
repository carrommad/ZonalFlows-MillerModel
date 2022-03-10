#!/usr/bin/python
#coding=utf-8

#-------------------
# 09-03-2022 Carlos 
# Packages.
#-------------------
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import scipy.integrate as integrate
import scipy.special as special
from scipy.optimize import minimize
from scipy.optimize import newton_krylov
import scipy.misc as misc
import scipy.linalg as linalg
from scipy.interpolate import splev, splrep,splint
import os
import time
import matplotlib.font_manager as font_manager


#t0=time.time()
#t1=time.time()
#print (t1-t0)


#-------------------
# 09-03-2022 Carlos 
# Description of the code.
#-------------------
#this script describes the local Miller equilibrium

##################################
# TO DO:
# 1) Parametric study of (3-loop in the 3 variables):
# - eps
# - ka
# - da
#
# 2) Possible update/enhacement to code:
#  - Introduce small wavelength effects.
##################################

#-------------------
# 09-03-2022 Carlos 
# Different sets of parameters (parameters bast Mast shot 8500 from 'collisionality scaling of the electron heat flux'):
# - Set 1
# - Set 2
# - Set 3
# Value to choose parameter set with a boolean door:
param_set = 1
#-------------------

#-------------------
# 09-03-2022 Carlos 
# Parameter set 1
#-------------------
if param_set == 1:
    
    #r is normalized as epsilon=r/R_{0}, R_0 = 1
    #eps=0.65/1.46
    eps=1/3.0
    #elongation kappa
    ka=2.0
    #triangularity delta
    #da=0.22
    #shafranov shift, usually we take is as a*eps, with a an \mathcal{O}(1) constant,
    dp=0
    #s_{\delta} - (dependent on parameter 'da')
    #sd=0.16*0.65/np.sqrt(1-da**2)
    #d=arcsin delta  - (dependent on parameter 'da')
    #d=np.arcsin(da)
    #s_{\kappa}=(r/\kappa)\partial_{r}\kappa
    sk=0.40*0.65/ka
    #safety factor
    q=1.9
    #magnetic shear
    s=1.8
    #alpha=-q^2 R_{0}\partial_{r}\beta
    al=0.12*q**2/eps

#-------------------
# 09-03-2022 Carlos 
# Parameter set 2
#-------------------
elif param_set == 2:
    
    ##r is normalized as epsilon=r/R_{0}, R_0 = 1
    eps=1/3.17
    ##elongation kappa
    ka=1.66
    ##triangularity delta
    da=0.416
    ##shafranov shift, usually we take is as a*eps, with a an \mathcal{O}(1) constant,
    dp=0.345
    ##s_{\delta}
    sd=1.37
    ##d=arcsin delta
    d=np.arcsin(da)
    ##s_{\kappa}=(r/\kappa)\partial_{r}\kappa
    sk=0.7
    ##safety factor
    q=3.03
    ##magnetic shear
    s=1.00
    ##alpha=-q^2 R_{0}\partial_{r}\beta
    al=1.0

#-------------------
# 09-03-2022 Carlos 
# Parameter set 3
#-------------------
elif param_set == 3:
    
    ##r is normalized as epsilon=r/R_{0}, R_0 = 1
    eps=0.1
    ##elongation kappa
    ka=1.8
    ##triangularity delta
    da=0.0
    ##shafranov shift, usually we take is as a*eps, with a an \mathcal{O}(1) constant,
    dp=0.0
    ##s_{\delta}
    sd=0.0
    ##d=arcsin delta
    d=np.arcsin(da)
    ##s_{\kappa}=(r/\kappa)\partial_{r}\kappa
    sk=0.0
    ##safety factor
    q=1.4
    ##magnetic shear
    s=1.0
    ##alpha=-q^2 R_{0}\partial_{r}\beta
    al=0.0


#-------------------
# 09-03-2022 Carlos 
# Start of the code.
#-------------------

#partial_{\theta} l
#-------------------
# 07-03-2022 Carlos 
# Eq. (2.107) of the Gyro Technical Guide using Eq. (34) of Miller (1998) for R and Z, 
# with the normalization of r proposed above by Haotian: R_0, r = eps = r/R_0.
#-------------------
def DTL(theta,eps,ka,da,dp,sd,sk,q,s,al):
    d=np.arcsin(da)
    xi=theta+d*np.sin(theta)
    temp=eps*np.sqrt( (1.0+d*np.cos(theta))**2*(np.sin(xi))**2+ka**2*(np.cos(theta))**2 )
    return temp

#\mathcal{J}_r (Jacobian)
#-------------------
# 07-03-2022 Carlos 
# Eq. (2.108) of the Gyro Technical Guide using Eq. (34) of Miller (1998) for R and Z.
#-------------------
def J(theta,eps,ka,da,dp,sd,sk,q,s,al):
    d=np.arcsin(da)
    #sd=0.16*0.65/np.sqrt(1-da**2)
    xi=theta+d*np.sin(theta)
    temp=ka*eps*(1.0+eps*np.cos(xi))*( np.cos(theta)*(np.cos(xi)+dp-sd*np.sin(theta)*np.sin(xi))+(1.0+sk)*(1.0+d*np.cos(theta))*np.sin(xi)*np.sin(theta) )
    return temp

#|\nabla r|^{2}
#-------------------
# 07-03-2022 Carlos 
# Square of Eq. (2.109) of the Gyro Technical Guide using Eq. (34) of Miller (1998) for R, 
# along with previous Eqs.(2.107-8) (some quantities are already simplified, e.g. eps^2, R^2 = (1.0+eps*np.cos(xi))^2).
#-------------------
def NR2(theta,eps,ka,da,dp,sd,sk,q,s,al):
    d=np.arcsin(da)
    #sd=0.16*0.65/np.sqrt(1-da**2)
    xi=theta+d*np.sin(theta)
    #-------------------
    # 07-03-2022 Carlos 
    # Summands' order modified from original fersion so as to coincide with previous definition of \partial_{\theta} l.
    #-------------------
    f1=(1.0+d*np.cos(theta))**2*(np.sin(xi))**2+ka**2*(np.cos(theta))**2
    f2=ka**2*( np.cos(theta)*(np.cos(xi)+dp-sd*np.sin(theta)*np.sin(xi))+(1.0+sk)*(1.0+d*np.cos(theta))*np.sin(xi)*np.sin(theta) )**2
    return f1/f2

#\partial_{\theta}|\nabla r|^{2}
def DTNR2(theta,eps,ka,da,dp,sd,sk,q,s,al):
    d=np.arcsin(da)
    #sd=0.16*0.65/np.sqrt(1-da**2)
    xi=theta+d*np.sin(theta)
    f11=-2*ka**2*np.cos(theta)*np.sin(theta)+2.0*(1+d*np.cos(theta))*np.sin(xi)*(-d*np.sin(theta)*np.sin(xi)+(1+d*np.cos(theta))**2*np.cos(xi))
    f12=ka**2*( np.cos(theta)*(np.cos(xi)+dp-sd*np.sin(theta)*np.sin(xi))+(1.0+sk)*(1.0+d*np.cos(theta))*np.sin(xi)*np.sin(theta) )**2
    f1=f11/f12
    f21=2*ka*( ka**2*(np.cos(theta))**2+(1.0+d*np.cos(theta))**2*(np.sin(xi))**2 )*( -np.sin(theta)*(np.cos(xi)+dp-sd*np.sin(theta)*np.sin(xi))+np.cos(theta)*(-np.sin(xi)*(1+d*np.cos(theta))-sd*np.cos(theta)*np.sin(xi)-sd*np.sin(theta)*np.cos(xi)*(1+d*np.cos(theta))  )-(1+sk)*d*(np.sin(theta))**2*np.sin(xi)+(1+sk)*(1+d*np.cos(theta))**2*np.cos(xi)*np.sin(theta)+(1+sk)*(1+d*np.cos(theta))*np.sin(xi)*np.cos(theta) )
    f22=ka**3*( np.cos(theta)*(np.cos(xi)+dp-sd*np.sin(theta)*np.sin(xi))+(1.0+sk)*(1.0+d*np.cos(theta))*np.sin(xi)*np.sin(theta) )**3
    f2=f21/f22
    return f1-f2

#R
#-------------------
# 07-03-2022 Carlos 
# Normalization of R proposed above by Haotian: R_0 = 1, r = eps = r/R_0.
#-------------------
def R(theta,eps,ka,da,dp,sd,sk,q,s,al):
    d=np.arcsin(da)
    return 1+eps*np.cos(theta+d*np.sin(theta))

#partial_{theta}R
def DTR(theta,eps,ka,da,dp,sd,sk,q,s,al):
    d=np.arcsin(da)
    xi=theta+d*np.sin(theta)
    temp=-eps*(1+d*np.cos(theta))*np.sin(xi)
    return temp

#partial_{theta}Z
def DTZ(theta,eps,ka,da,dp,sd,sk,q,s,al):
    temp=ka*eps*np.cos(theta)
    return temp

#partial^{2}_{theta}R
def DDTR(theta,eps,ka,da,dp,sd,sk,q,s,al):
    d=np.arcsin(da)
    xi=theta+d*np.sin(theta)
    temp=-eps*(1+d*np.cos(theta))**2*np.cos(xi)+eps*d*np.sin(theta)*np.sin(xi)
    return temp

#partial^{2}_{theta}Z
def DDTZ(theta,eps,ka,da,dp,sd,sk,q,s,al):
    temp=-ka*eps*np.sin(theta)
    return temp

#r_{c}
#-------------------
# 07-03-2022 Carlos 
# Eq. (2.110) of the Gyro Technical Guide.
#-------------------
def RC(theta,eps,ka,da,dp,sd,sk,q,s,al):
    f1=DTL(theta,eps,ka,da,dp,sd,sk,q,s,al)
    return f1**3/(DTR(theta,eps,ka,da,dp,sd,sk,q,s,al)*DDTZ(theta,eps,ka,da,dp,sd,sk,q,s,al)-DTZ(theta,eps,ka,da,dp,sd,sk,q,s,al)*DDTR(theta,eps,ka,da,dp,sd,sk,q,s,al))

#sin u
#-------------------
# 07-03-2022 Carlos 
# Eq. (3) of Miller (1998) canceling the time differential.
# - It's correct, we are following Gyro Technical Guide, the u is different than that of Miller (1998): check Eqs.(2.52-3).
#-------------------
def SINU(theta,eps,ka,da,dp,sd,sk,q,s,al):
    return -DTR(theta,eps,ka,da,dp,sd,sk,q,s,al)/DTL(theta,eps,ka,da,dp,sd,sk,q,s,al)

#cos u
#-------------------
# 07-03-2022 Carlos 
# Eq. (2) of Miller (1998) canceling the time differential.
# - It's correct, we are following Gyro Technical Guide, the u is different than that of Miller (1998): check Eqs.(2.52-3).
#-------------------
def COSU(theta,eps,ka,da,dp,sd,sk,q,s,al):
    return DTZ(theta,eps,ka,da,dp,sd,sk,q,s,al)/DTL(theta,eps,ka,da,dp,sd,sk,q,s,al)

#-------------------
# 07-03-2022 Carlos 
# Loop in parameter: triangularity.
#-------------------
Narr=20
#Qarr=np.linspace(0.5,5.5,Narr)
Darr=np.linspace(-0.5,0.5,Narr)
Data=np.zeros(Narr)
for j in range(Narr):
    #q=Qarr[j]
    da=Darr[j]
    d=np.arcsin(da)
    #-------------------
    # 07-03-2022 Carlos 
    # Why don't we define this parameter at the beginning with the rest of the variables?
    #-------------------
    sd=0.16*0.65/np.sqrt(1-da**2)
    #I/B_unit
    #-------------------
    # 07-03-2022 Carlos 
    # Eq. (2.111) of Gyro Technical Guide where we have changed the notation: I := f(r).
    # - The part "lambda x:" indiciates to the function "integrate.quad" which is the integration variable.
    #-------------------
    I=2*np.pi*eps/integrate.quad(lambda x: DTL(x,eps,ka,da,dp,sd,sk,q,s,al)/np.sqrt(NR2(x,eps,ka,da,dp,sd,sk,q,s,al))/(1+eps*np.cos(x+d*np.sin(x))),0,2*np.pi)[0]

    #B_t
    #-------------------
    # 07-03-2022 Carlos 
    # Eq. (2.112) of Gyro Technical Guide.
    # - Why don't we use I as input parameter?
    #-------------------
    def BT(theta,eps,ka,da,dp,sd,sk,q,s,al):
        d=np.arcsin(da)
        return I/(1+eps*np.cos(theta+d*np.sin(theta)))

    #partial_{\theta} B_{t}
    def DTBT(theta,eps,ka,da,dp,sd,sk,q,s,al):
        return -I/(R(theta,eps,ka,da,dp,sd,sk,q,s,al))**2*DTR(theta,eps,ka,da,dp,sd,sk,q,s,al)

    #B_p
    #-------------------
    # 07-03-2022 Carlos 
    # Eq. (2.113) of Gyro Technical Guide.
    #-------------------
    def BP(theta,eps,ka,da,dp,sd,sk,q,s,al):
        d=np.arcsin(da)
        return eps/(1+eps*np.cos(theta+d*np.sin(theta)))*np.sqrt(NR2(theta,eps,ka,da,dp,sd,sk,q,s,al))/q

    #partial_{\theta} B_{p}
    def DTBP(theta,eps,ka,da,dp,sd,sk,q,s,al):
        return -eps*DTR(theta,eps,ka,da,dp,sd,sk,q,s,al)/(R(theta,eps,ka,da,dp,sd,sk,q,s,al))**2*np.sqrt(NR2(theta,eps,ka,da,dp,sd,sk,q,s,al))/q+0.5*eps*DTNR2(theta,eps,ka,da,dp,sd,sk,q,s,al)/(R(theta,eps,ka,da,dp,sd,sk,q,s,al)*q*np.sqrt(NR2(theta,eps,ka,da,dp,sd,sk,q,s,al)))

    #B
    #-------------------
    # 07-03-2022 Carlos 
    # Eq. (2.114) of Gyro Technical Guide.
    #-------------------
    def B(theta,eps,ka,da,dp,sd,sk,q,s,al):
        return np.sqrt((BT(theta,eps,ka,da,dp,sd,sk,q,s,al))**2+(BP(theta,eps,ka,da,dp,sd,sk,q,s,al))**2)

    #partial_{theta}B
    def DTB(theta,eps,ka,da,dp,sd,sk,q,s,al):
        return (BP(theta,eps,ka,da,dp,sd,sk,q,s,al)*DTBP(theta,eps,ka,da,dp,sd,sk,q,s,al)+BT(theta,eps,ka,da,dp,sd,sk,q,s,al)*DTBT(theta,eps,ka,da,dp,sd,sk,q,s,al))/B(theta,eps,ka,da,dp,sd,sk,q,s,al)

    #gsin
    #-------------------
    # 07-03-2022 Carlos 
    # Eq. (2.115) of Gyro Technical Guide.
    #-------------------
    def GSIN(theta,eps,ka,da,dp,sd,sk,q,s,al):
        temp=BT(theta,eps,ka,da,dp,sd,sk,q,s,al)*DTB(theta,eps,ka,da,dp,sd,sk,q,s,al)/(B(theta,eps,ka,da,dp,sd,sk,q,s,al))**2/DTL(theta,eps,ka,da,dp,sd,sk,q,s,al)
        return temp

    #gcos1
    #-------------------
    # 07-03-2022 Carlos 
    # Eq. (2.117) of Gyro Technical Guide.
    #-------------------
    def GCOS1(theta,eps,ka,da,dp,sd,sk,q,s,al):
        temp=(BT(theta,eps,ka,da,dp,sd,sk,q,s,al)/B(theta,eps,ka,da,dp,sd,sk,q,s,al))**2*COSU(theta,eps,ka,da,dp,sd,sk,q,s,al)/R(theta,eps,ka,da,dp,sd,sk,q,s,al)+(BP(theta,eps,ka,da,dp,sd,sk,q,s,al)/B(theta,eps,ka,da,dp,sd,sk,q,s,al))**2/RC(theta,eps,ka,da,dp,sd,sk,q,s,al)
        return temp

    #gcos2
    #-------------------
    # 07-03-2022 Carlos 
    # Eq. (2.118) of Gyro Technical Guide.
    #-------------------
    def GCOS2(theta,eps,ka,da,dp,sd,sk,q,s,al):
        #nonzero in finite beta limit:      
        temp=-0.5*np.sqrt(NR2(theta,eps,ka,da,dp,sd,sk,q,s,al))/(B(theta,eps,ka,da,dp,sd,sk,q,s,al))**2*al/q**2
        #zero in low beta limit:
        #temp=0
        return temp

    #gcos
    #-------------------
    # 07-03-2022 Carlos 
    # Eq. (2.116) of Gyro Technical Guide.
    #-------------------
    def GCOS(theta,eps,ka,da,dp,sd,sk,q,s,al):
        return GCOS1(theta,eps,ka,da,dp,sd,sk,q,s,al)+GCOS2(theta,eps,ka,da,dp,sd,sk,q,s,al)

    #usin
    #-------------------
    # 07-03-2022 Carlos 
    # Eq. (2.119) of Gyro Technical Guide.
    #-------------------
    def USIN(theta,eps,ka,da,dp,sd,sk,q,s,al):
        return SINU(theta,eps,ka,da,dp,sd,sk,q,s,al)

    #ucos
    #-------------------
    # 07-03-2022 Carlos 
    # Eq. (2.120) of Gyro Technical Guide.
    #-------------------
    def UCOS(theta,eps,ka,da,dp,sd,sk,q,s,al):
        return BT(theta,eps,ka,da,dp,sd,sk,q,s,al)/B(theta,eps,ka,da,dp,sd,sk,q,s,al)*COSU(theta,eps,ka,da,dp,sd,sk,q,s,al)

    #E1
    #-------------------
    # 07-03-2022 Carlos 
    # Eq. (2.121) of Gyro Technical Guide.
    #-------------------
    # - exact expression, time consuming
    def E1E(theta,eps,ka,da,dp,sd,sk,q,s,al):
        temf=lambda x: 2*DTL(x,eps,ka,da,dp,sd,sk,q,s,al)/(R(x,eps,ka,da,dp,sd,sk,q,s,al)*np.sqrt(NR2(x,eps,ka,da,dp,sd,sk,q,s,al)))*BT(x,eps,ka,da,dp,sd,sk,q,s,al)/BP(x,eps,ka,da,dp,sd,sk,q,s,al)*(eps/RC(x,eps,ka,da,dp,sd,sk,q,s,al)-eps/R(x,eps,ka,da,dp,sd,sk,q,s,al)*COSU(x,eps,ka,da,dp,sd,sk,q,s,al))
        return integrate.quad(temf, 0,theta,limit=2000)[0]

    #construct the Fourier representation of exact E1, valid to 1e-5


    # - the coefficient of linear component
    #-------------------
    # 07-03-2022 Carlos 
    # Take E1 at the extremes (0,2*\Pi) and compute the slope.
    #-------------------
    e1linear=(E1E(2*np.pi,eps,ka,da,dp,sd,sk,q,s,al)-E1E(0,eps,ka,da,dp,sd,sk,q,s,al))/(2*np.pi)

    # - oscillatory component
    #number of sample point in theta
    Ntheta=100
    Theta=np.linspace(0,2*np.pi, Ntheta)
    Datafe1=np.zeros(Ntheta)
    for i in range(Ntheta):
        #-------------------
        # 07-03-2022 Carlos 
        # Deviation from the straight line connecting E1 at 0 and at 2*\Pi,
        # hence "oscillatory component" above.
        #-------------------
        Datafe1[i]=E1E(Theta[i],eps,ka,da,dp,sd,sk,q,s,al)-e1linear*Theta[i]

    #-------------------
    # 07-03-2022 Carlos 
    # Given the set of data points (x[i], y[i]), "splrep" determines a smooth spline approximation,
    # where the separation between x points is not constant.
    #-------------------
    sple1=splrep(Theta,Datafe1)

    #-------------------
    # 07-03-2022 Carlos 
    # To avoid computing the integral that defines E1 for each \theta, we have decomposed it as:
    # - A linear component "temp1"
    # - An oscillatory component "temp2"
    #-------------------
    def E1(theta,eps,ka,da,dp,sd,sk,q,s,al):
        #linear component
        temp1=e1linear*theta
        #oscillating components retaind
        temp2=splev(theta%(2*np.pi),sple1)
        return temp1+temp2

    #E2
    #-------------------
    # 07-03-2022 Carlos 
    # Eq. (2.122) of Gyro Technical Guide. We have 2 versions:
    # - The exact, and time-consuming, definition 
    # - And an analogous "linear + oscillatory" decomposition, as in E1, has been made below.
    #-------------------
    def E2E(theta,eps,ka,da,dp,sd,sk,q,s,al):
        temf=lambda x: DTL(x,eps,ka,da,dp,sd,sk,q,s,al)/(R(x,eps,ka,da,dp,sd,sk,q,s,al)*np.sqrt(NR2(x,eps,ka,da,dp,sd,sk,q,s,al)))*(B(x,eps,ka,da,dp,sd,sk,q,s,al)/BP(x,eps,ka,da,dp,sd,sk,q,s,al))**2
        return integrate.quad(temf, 0,theta,limit=2000)[0]

    # - the coefficient of linear component
    e2linear=(E2E(2*np.pi,eps,ka,da,dp,sd,sk,q,s,al)-E2E(0,eps,ka,da,dp,sd,sk,q,s,al))/(2*np.pi)

    # - oscillatory component
    Datafe2=np.zeros(Ntheta)
    for i in range(Ntheta):
        Datafe2[i]=E2E(Theta[i],eps,ka,da,dp,sd,sk,q,s,al)-e2linear*Theta[i]
    sple2=splrep(Theta,Datafe2)
    def E2(theta,eps,ka,da,dp,sd,sk,q,s,al):
        #linear component
        temp1=e2linear*theta
        #oscillating components retaind
        temp2=splev(theta%(2*np.pi),sple2)
        return temp1+temp2

    #E3
    #-------------------
    # 07-03-2022 Carlos 
    # Eq. (2.123) of Gyro Technical Guide. We have 2 versions:
    # - The exact, and time-consuming, definition 
    # - And an analogous "linear + oscillatory" decomposition, as in E1, has been made below.
    #-------------------
    def E3E(theta,eps,ka,da,dp,sd,sk,q,s,al):
        temf=lambda x: 0.5*DTL(x,eps,ka,da,dp,sd,sk,q,s,al)/R(x,eps,ka,da,dp,sd,sk,q,s,al)*BT(x,eps,ka,da,dp,sd,sk,q,s,al)/(BP(x,eps,ka,da,dp,sd,sk,q,s,al))**3
        return integrate.quad(temf, 0,theta,limit=2000)[0]

    # - the coefficient of linear component
    e3linear=(E3E(2*np.pi,eps,ka,da,dp,sd,sk,q,s,al)-E3E(0,eps,ka,da,dp,sd,sk,q,s,al))/(2*np.pi)

    # - oscillatory component
    Datafe3=np.zeros(Ntheta)
    for i in range(Ntheta):
        Datafe3[i]=E3E(Theta[i],eps,ka,da,dp,sd,sk,q,s,al)-e3linear*Theta[i]
    sple3=splrep(Theta,Datafe3)
    def E3(theta,eps,ka,da,dp,sd,sk,q,s,al):
        #linear component
        temp1=e3linear*theta
        #oscillating components retaind
        temp2=splev(theta%(2*np.pi),sple3)
        return temp1+temp2

    #f^{*}
    #-------------------
    # 07-03-2022 Carlos 
    # Eq. (2.124) of Gyro Technical Guide.
    # - Careful: this variable is following the EXACT definitions of Ei (e.g. E1, E2, E3).
    #-------------------
    fs=(2*np.pi*q*s/eps-E1E(2*np.pi,eps,ka,da,dp,sd,sk,q,s,al)/eps+al/q**2*E3E(2*np.pi,eps,ka,da,dp,sd,sk,q,s,al))/E2E(2*np.pi,eps,ka,da,dp,sd,sk,q,s,al)

    #Theta
    #-------------------
    # 07-03-2022 Carlos 
    # Eq. (2.125) of Gyro Technical Guide.
    # - Careful: this variable is following the APPROXIMATE definitions of Ei (e.g. E1, E2, E3).
    #-------------------
    def THETA(theta,eps,ka,da,dp,sd,sk,q,s,al):
        temp=R(theta,eps,ka,da,dp,sd,sk,q,s,al)*BP(theta,eps,ka,da,dp,sd,sk,q,s,al)*np.sqrt(NR2(theta,eps,ka,da,dp,sd,sk,q,s,al))/B(theta,eps,ka,da,dp,sd,sk,q,s,al)*(E1(theta,eps,ka,da,dp,sd,sk,q,s,al)/eps+fs*E2(theta,eps,ka,da,dp,sd,sk,q,s,al)-al/q**2*E3(theta,eps,ka,da,dp,sd,sk,q,s,al))
        return temp

    #G_{q}
    #-------------------
    # 07-03-2022 Carlos 
    # Eq. (2.126) of Gyro Technical Guide.
    #-------------------
    def GQ(theta,eps,ka,da,dp,sd,sk,q,s,al):
        temp=eps*B(theta,eps,ka,da,dp,sd,sk,q,s,al)/(q*R(theta,eps,ka,da,dp,sd,sk,q,s,al)*BP(theta,eps,ka,da,dp,sd,sk,q,s,al))
        return temp

    #G_{theta}
    #-------------------
    # 07-03-2022 Carlos 
    # Eq. (2.127) of Gyro Technical Guide.
    #-------------------
    def GT(theta,eps,ka,da,dp,sd,sk,q,s,al):
        temp=B(theta,eps,ka,da,dp,sd,sk,q,s,al)*R(theta,eps,ka,da,dp,sd,sk,q,s,al)/(eps*np.sqrt(NR2(theta,eps,ka,da,dp,sd,sk,q,s,al)))*DTL(theta,eps,ka,da,dp,sd,sk,q,s,al)
        return temp

    #k_{\perp}^{2}/k_{\theta}^{2}
    #-------------------
    # 10-03-2022 Carlos 
    # - Not necessary for zonal flow studies.
    #-------------------
    #def KP2(theta,eps,ka,da,dp,sd,sk,q,s,al):
    #    gq=GQ(theta,eps,ka,da,dp,sd,sk,q,s,al)
    #    th=THETA(theta,eps,ka,da,dp,sd,sk,q,s,al)
    #    return gq**2*(1+th**2)

    #g
    #-------------------
    # 09-03-2022 Carlos 
    # This is a factor of the first line of Eq. (2.128) of Gyro Technical Guide.
    # - Why is "GCOS2" multiplied by 0.5? Due to the finite beta effect.
    # - Not necessary for zonal flow studies.
    #-------------------
    #def G(theta,eps,ka,da,dp,sd,sk,q,s,al):
    #    return GCOS1(theta,eps,ka,da,dp,sd,sk,q,s,al)+0.5*GCOS2(theta,eps,ka,da,dp,sd,sk,q,s,al)+THETA(theta,eps,ka,da,dp,sd,sk,q,s,al)*GSIN(theta,eps,ka,da,dp,sd,sk,q,s,al)

    #(B G_{\theta}/k_{\perp}^{2})partial_{\theta}(k_{\perp}^{2}/(B G_{\theta}))
    #-------------------
    # 10-03-2022 Carlos 
    # - Not necessary for zonal flow studies.
    #-------------------
    #def Q1(theta,eps,ka,da,dp,sd,sk,q,s,al):
    #    tempf=lambda x: np.log(KP2(x,eps,ka,da,dp,sd,sk,q,s,al)/(B(x,eps,ka,da,dp,sd,sk,q,s,al)*GT(x,eps,ka,da,dp,sd,sk,q,s,al)))
    #    return misc.derivative(tempf,theta,dx=1e-4,order=3)
        #tempf=lambda x: KP2(x,eps,ka,da,dp,sd,sk,q,s,al)/(B(x,eps,ka,da,dp,sd,sk,q,s,al)*GT(x,eps,ka,da,dp,sd,sk,q,s,al))
        #return misc.derivative(tempf,theta,dx=1e-4,order=3)*B(theta,eps,ka,da,dp,sd,sk,q,s,al)*GT(theta,eps,ka,da,dp,sd,sk,q,s,al)/KP2(theta,eps,ka,da,dp,sd,sk,q,s,al)
    #Q1=np.vectorize(Q1)

    #define Q0
    #-------------------
    # 10-03-2022 Carlos 
    # - Not necessary for zonal flow studies.
    #-------------------
    #def Q0(theta,eps,ka,da,dp,sd,sk,q,s,al):
    #    gq=GQ(theta,eps,ka,da,dp,sd,sk,q,s,al)
    #    gt=GT(theta,eps,ka,da,dp,sd,sk,q,s,al)
    #    g=G(theta,eps,ka,da,dp,sd,sk,q,s,al)
    #    return  al*g*gq*gt**2/(KP2(theta,eps,ka,da,dp,sd,sk,q,s,al)*B(theta,eps,ka,da,dp,sd,sk,q,s,al))
        
    #define QOME
    #-------------------
    # 10-03-2022 Carlos 
    # - Not necessary for zonal flow studies.
    #-------------------
    #def QOME(theta,eps,ka,da,dp,sd,sk,q,s,al):
    #    gt=GT(theta,eps,ka,da,dp,sd,sk,q,s,al)
    #    b=B(theta,eps,ka,da,dp,sd,sk,q,s,al)
    #    return  gt**2/b**2

    #define chi
    #-------------------
    # 09-03-2022 Carlos 
    # What is \chi?
    # - Dielectric susceptibility: \chi = 1 + 1.6 * q^2 / \eps^{1/2}
    #-------------------
    def CHI(eps,ka,da,dp,sd,sk,q,s,al):
        av1=integrate.quad(lambda x: GT(x,eps,ka,da,dp,sd,sk,q,s,al)/B(x,eps,ka,da,dp,sd,sk,q,s,al),0,2*np.pi)[0]
        av2=integrate.quad(lambda x: GT(x,eps,ka,da,dp,sd,sk,q,s,al)/(B(x,eps,ka,da,dp,sd,sk,q,s,al))**3,0,2*np.pi)[0]
        av3=integrate.quad(lambda x: GT(x,eps,ka,da,dp,sd,sk,q,s,al)*B(x,eps,ka,da,dp,sd,sk,q,s,al)/NR2(x,eps,ka,da,dp,sd,sk,q,s,al),0,2*np.pi)[0]
        temp1=av2/av1
        Lambda=np.linspace(0,1/B(np.pi,eps,ka,da,dp,sd,sk,q,s,al),50)
        Datatemp=np.zeros(len(Lambda))
        for i in range(len(Datatemp)):
            Datatemp[i]=integrate.quad(lambda x: B(x,eps,ka,da,dp,sd,sk,q,s,al)/BP(x,eps,ka,da,dp,sd,sk,q,s,al)*DTL(x,eps,ka,da,dp,sd,sk,q,s,al)/np.sqrt(1-Lambda[i]*B(x,eps,ka,da,dp,sd,sk,q,s,al)),-np.pi+1e-4,np.pi-1e-4)[0]
        sp=splrep(Lambda,1/Datatemp)
        av4=splint(0,1/B(np.pi,eps,ka,da,dp,sd,sk,q,s,al),sp)
        temp2=1.5*q*av1*av4

        temp=(q*I/eps)**2*av3/av1*(temp1-temp2)

        return 1+temp

#-------------------
# 09-03-2022 Carlos 
# Loop in safety factor.
#-------------------
#Qarr=np.linspace(0.5,5.5,20)
#Data=np.zeros(len(Qarr))
#for i in range(len(Data)):
    Data[j]=CHI(eps,ka,Darr[j],dp,0.16*0.65/np.sqrt(1-Darr[j]**2),sk,q,s,al)

#-------------------
# 09-03-2022 Carlos 
# Plot of safety factor (q) vs. dielectric susceptibility or polarization (1+1.635*Qarr**2/np.sqrt(eps)???)
#-------------------
fig1_bool = 0
if fig1_bool:
    plt.figure(1)
    font = {'family' : 'serif',  
            'color'  : 'black',  
            'weight' : 'normal',  
            'size'   : 22,  
            }
    font_prop = font_manager.FontProperties(size=20)

    plt.plot(Qarr,Data, '-',linewidth=2.5, label=r'$\epsilon=%g$'%eps)
    plt.plot(Qarr,1+1.635*Qarr**2/np.sqrt(eps), '--',linewidth=2.5, label=r'$\epsilon=%g$'%eps)

    plt.xlabel(r'$q$',fontdict=font)
    plt.ylabel(r'$\chi_{i}$',fontdict=font)
    plt.title(r'Polarization: Ion-acoustic')
    plt.grid(True)
    plt.legend()
    plt.show()

#-------------------
# 09-03-2022 Carlos 
# - Plot of triangularity (\delta) vs. dielectric susceptibility or polarization (\chi_{i})
#-------------------
fig2_bool = 1
if fig2_bool:
    plt.figure(2)
    font = {'family' : 'serif',  
            'color'  : 'black',  
            'weight' : 'normal',  
            'size'   : 22,  
            }
    font_prop = font_manager.FontProperties(size=20)

    plt.plot(Darr,1/Data, '-',linewidth=2.5, label=r'$\epsilon=%g$'%eps)
    #plt.plot(Qarr,1+1.635*Qarr**2/np.sqrt(eps), '--',linewidth=2.5, label=r'$\epsilon=%g$'%eps)

    plt.xlabel(r'$\delta$',fontdict=font)
    plt.ylabel(r'$\chi^{-1}_{i}$',fontdict=font)
    #plt.title(r'Polarization: Ion-acoustic')
    plt.grid(True)
    plt.legend()
    plt.show()