from pylab import arange,zeros,exp
from numpy import sign

# Define grid
Npts = 100
Xmin = -1.
Xmax = 1.
DX = (Xmax-Xmin)/Npts
# Place grid points at cell centers
Xgrid = arange(Xmin+DX/2.,Xmax,DX)

# Time step
CFL = 0.5
DT_step = CFL*DX
T_end = 5.
DT_obs = 0.25

# Initial conditions
def GetInitialData():
    res = zeros([Npts])
    for i in range(0,Npts):
        res[i] = exp(-16.*Xgrid[i]**2.)
    return res

def minmod(a,b,c):
    if (a-b)*(b-c) <= 0.:
        return 0.
    else:
        return sign(b-a)*min(abs(b-a),abs(c-a))

def ReconstructFromCenterToFaces(U_center):
    # Simplest reconstruction
    U_L = zeros([Npts])
    U_R = zeros([Npts])
    for i in range(0,Npts):
        i0 = (i-1)%Npts
        i1 = i
        i2 = (i+1)%Npts
        i3 = (i+2)%Npts
        U_L[i] = U_center[i1]+0.5*minmod(U_center[i0],U_center[i1],U_center[i2])
        U_R[i] = U_center[i2]-0.5*minmod(U_center[i1],U_center[i2],U_center[i3])
            
    for i in range(0,Npts):
        U_L[i]=U_center[i]
        if i<Npts-1:
            U_R[i] = U_center[i+1]
        else:
            U_R[i] = U_center[0]
    return U_L,U_R

def ComputedtU(U_center):
    # Reconstruct from cell centers to cell faces
    U_L,U_R = ReconstructFromCenterToFaces(U_center)
    F_L = zeros([Npts])
    F_R = zeros([Npts])
    for i in range(0,Npts):
        F_L[i] = U_L[i]**2./2.
        F_R[i] = U_R[i]**2./2.
        
    # Riemann solver
    F_face = zeros([Npts])
    for i in range(0,Npts):
        charspeed = abs(U_L[i])
        if abs(U_R[i])>charspeed:
            charspeed = abs(U_R[i])
        F_face[i] = 0.5*(F_L[i]+F_R[i])-0.5*charspeed*(U_R[i]-U_L[i])
    
    # Derivative of F
    dtU = zeros([Npts])
    dtU[0] = -(F_face[0]-F_face[Npts-1])/DX
    for i in range(1,Npts):
        dtU[i] = -(F_face[i]-F_face[i-1])/DX
    return dtU
            
def TakeTimeStep(U_center,DT):
    dtU = ComputedtU(U_center)
    U_half = U_center+DT/2.*dtU
    dtU = ComputedtU(U_half)
    U_center = U_center + DT*dtU
    return U_center

# Initial conditions
Time = 0.
U_center = GetInitialData()
LastTObs = 0.

# Visualize model
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

while(Time<T_end):
    U_center = TakeTimeStep(U_center,DT_step)
    Time = Time+DT_step
    if(Time>=LastTObs+DT_obs):
        # Observe results
        mA = 0.3+0.7*(Time/T_end)
        plt.plot(Xgrid,U_center,linestyle='-',color='black',alpha=mA)
        LastTObs=LastTObs+DT_obs
