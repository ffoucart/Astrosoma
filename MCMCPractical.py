from pylab import pi,arange,zeros,sin,cos,exp,log,sqrt,log10
from numpy import random as rn
from numpy.linalg import norm

# Define Min/max frequency and number of bins [Units: Hz]
Freq_Low = 10.
Freq_High = 1000.
Nbins_f = 200

# Logarithmically spaced frequency bins
logFreq_Low = log10(Freq_Low)
logFreq_High = log10(Freq_High)
logBinWidth = (logFreq_High-logFreq_Low)/(Nbins_f)
LogFreq_BinBottom = arange(logFreq_Low,logFreq_High,logBinWidth)
LogFreq_BinTop = LogFreq_BinBottom + 0.01
Freq_BinBottom = zeros(Nbins_f)
Freq_BinCenter = zeros(Nbins_f)
Freq_BinTop = zeros(Nbins_f)
Freq_BinWidth = zeros(Nbins_f)
Freq_BinBottom[0] = 10.**LogFreq_BinBottom[0]
for i in range(1,Nbins_f):
    Freq_BinBottom[i] = 10.**LogFreq_BinBottom[i]
    Freq_BinTop[i-1] = Freq_BinBottom[i]
    Freq_BinWidth[i-1] = Freq_BinTop[i-1]-Freq_BinBottom[i-1]
    Freq_BinCenter[i-1] = 0.5*(Freq_BinTop[i-1]+Freq_BinBottom[i-1])
Freq_BinTop[Nbins_f-1] = 10.**LogFreq_BinTop[Nbins_f-1]
Freq_BinWidth[Nbins_f-1] = Freq_BinTop[Nbins_f-1]-Freq_BinBottom[Nbins_f-1]
Freq_BinCenter[Nbins_f-1] = 0.5*(Freq_BinTop[Nbins_f-1]+Freq_BinBottom[Nbins_f-1])

# Amplitude of LIGO noise at given frequency
Noise_Ampl = zeros(Nbins_f)
for i in range(0,Nbins_f):
    if(Freq_BinCenter[i]<100.):
        Noise_Ampl[i]=10.**(-23.+1.5*(2.-log10(Freq_BinCenter[i])))
    else:
        if(Freq_BinCenter[i]<400.):
            Noise_Ampl[i]=1.e-23
        else:
            Noise_Ampl[i]=10.**(-23.+0.3*(log10(Freq_BinCenter[i])-log10(400.))/(3.-log10(400.)))

# Phenomenological waveform model
# Returns real and imaginary part of the Fourier transform of the GW signal
# M = m1+m2 = Total mass of binary [Units: Solar masses]
# eta = m1*m2/M**2. = symmetric mass ratio [dimensionless]
# d = distance to the binary [Units : 1 ~ 1.5km]
# t0 = Time shift [Units : s]
# phi0 = Phase shift [radian]
def PhenomGW(M,eta,d,t0,phi0):
    f_merge = (0.29740*eta**2.+0.044810*eta+0.095560)/(pi*M)
    f_ring = (0.59411*eta**2.+0.089794*eta+0.19111)/(pi*M)
    sigma_ph = (0.50801*eta**2.+0.077515*eta+0.022369)/(pi*M)
    f_cut = (0.84845*eta**2.+0.12848*eta+0.27299)/(pi*M)
    w_ph = pi*sigma_ph/2.*(f_merge/f_ring)**(2./3)
    C = 4.96e-6*(M**5./(f_merge)**7.)**(1./6)/d/pi**(2./3)*(5.*eta/6.)**0.5
    Psi = (0.17516*eta**2.+0.079483*eta-0.072390)
    Waveform = zeros([2,Nbins_f])
    for i in range(0,Nbins_f):
        A_GW = 0.
        f_GW = Freq_BinCenter[i]*4.96e-6
        if(f_GW<f_merge):
            A_GW = C*(f_merge/f_GW)**(7./6.)
        else:
            if(f_GW<f_ring):
                A_GW = C*(f_merge/f_GW)**(2./3.)
            else:
                if(f_GW<f_cut):
                    A_GW=C*w_ph*sigma_ph/2./pi/((f_GW-f_ring)**2.+sigma_ph**2./4.)
        Phi_GW = phi0+2.*pi*t0*f_GW/4.96e-6 + Psi/eta*(pi*M*f_GW)**(-5./3.)
        Waveform[0,i] = A_GW*cos(Phi_GW)
        Waveform[1,i] = A_GW*sin(Phi_GW)        
    return Waveform


# Create fake random LIGO noise in Fourier space
# [Not the correct way to create LIGO noise, but will work for our
# purpose here]
def GetNoise():
    Noise = zeros([2,Nbins_f])
    for i in range(0,Nbins_f):
        A_n = rn.randn()*Noise_Ampl[i]/sqrt(2)
        Phi_n = rn.rand()*2.*pi
        Noise[0,i] = A_n*cos(Phi_n)
        Noise[1,i] = A_n*sin(Phi_n)
    return Noise

# Compute likelihood of ModelGW given SignalGW
def GetLIGOLogLikelihood(ModelGW,SignalGW):
    mA = (ModelGW-SignalGW)
    sFreq = sqrt(Freq_BinWidth)
    mA[0,:] = mA[0,:]/Noise_Ampl[:]*sFreq[:]
    mA[1,:] = mA[1,:]/Noise_Ampl[:]*sFreq[:]
    mL=norm(mA)**2.
    return -2.*mL

# Compute SNR of SignalGW
def GetLIGOSNR(SignalGW):
    mA = 1.*SignalGW
    sFreq = sqrt(Freq_BinWidth)
    mA[0,:] = mA[0,:]/Noise_Ampl[:]*sFreq[:]
    mA[1,:] = mA[1,:]/Noise_Ampl[:]*sFreq[:]
    mRho = 2.*norm(mA)
    return mRho
