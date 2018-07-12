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

# Compute ln(likelihood) of ModelGW given SignalGW
# We return the log of the likelihood, as the likelihood itself
# is often a very small number : ln(L) ~ -1000.
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


########### Example of a solution ###########

# Run chains at multiple temperatures,search for M only
M1 = 28.
M2 = 32.
Mt = M1+M2
Eta = (M1*M2)/Mt**2.
d = 1e22
print("SNR = ",GetLIGOSNR(PhenomGW(Mt,Eta,d,0,0)))
Signal = PhenomGW(Mt,Eta,d,0,0)+GetNoise()
Model = PhenomGW(Mt,Eta,d,0,0)
Ltrue = GetLIGOLogLikelihood(Model,Signal)
print("Likelihood of true signal is ",Ltrue)

# Temperature of chains
Tchain = ([1,4,10,30,50])
Nc = len(Tchain)

# Prior : 
# Total mass in [40,80], with uniform probability within that interval
Mmin = 40.
Mmax = 80.

# Initialize chains (Mass M_c, Likelihood L_c, Width of random jumps Sigma_c)
M_c = zeros([Nc])
L_c = zeros([Nc])
Sigma_c = zeros([Nc])
for i in range(0,Nc):
    M_mcmc = Mmin+rn.rand()*(Mmax-Mmin)
    Model_mcmc = PhenomGW(M_mcmc,Eta,d,0,0)
    M_c[i]= M_mcmc
    L_c[i]= GetLIGOLogLikelihood(Model_mcmc,Signal)
    # Guess for initial sigma
    Sigma_c[i] = 2.5
    
Ndraw_mcmc = 1
Trace=[[M_c[0],L_c[0]]]

Naccept = 0
Nreject = 0
Nbound = 0

while Ndraw_mcmc < 100000.:
    if(Ndraw_mcmc%1000==0):
        print(Ndraw_mcmc,Naccept,Nreject,Nbound)
    # Loop over chains
    for i in range(0,Nc):
        M_mcmc = M_c[i]
        L_mcmc = L_c[i]
        # Draw new values of M
        newM = M_mcmc + rn.randn()*Sigma_c[i]
        # Reject out of bounds points
        if (newM<Mmin or newM>Mmax):
            Sigma_c[i] = Sigma_c[i]*0.5
            if(i==0):
                Nbound = Nbound+1
            continue
            
        newModel = PhenomGW(newM,Eta,d,0,0)
        newL = GetLIGOLogLikelihood(newModel,Signal)
        # Accept / Reject condition
        if rn.rand()**Tchain[i] <= exp(newL-L_mcmc):
            # Accept new draw
            M_c[i] = newM
            L_c[i] = newL
            # We only keep track of the chain with T=1
            if(i==0):
                Trace.append([newM,newL])
                Naccept = Naccept+1
            Sigma_c[i] = Sigma_c[i]*8.
            # Do not allow Sigma >~ Mmax-Mmin
            if(Sigma_c[i]>20.):
                Sigma_c[i]=20.
        else:
            # Reject new draw
            Sigma_c = Sigma_c*0.5
            if(i==0):
                Nreject = Nreject+1
        
    Ndraw_mcmc = Ndraw_mcmc+1
    
    # Chain exchange, if needed
    for i in range(0,Nc):
        for j in range(i+1,Nc):
            if(L_c[j]>L_c[i]*rn.rand()**(Tchain[i]*Tchain[j]/(Tchain[j]-Tchain[i]))):
                temp = M_c[j]
                M_c[j] = M_c[i]
                M_c[i] = temp
                temp = L_c[j]
                L_c[j] = L_c[i]
                L_c[i] = temp
                temp = Sigma_c[j]
                Sigma_c[j] = Sigma_c[i]
                Sigma_c[i] = temp

##################  Post processing ############
# Visualize trace and compare to true likelihood
Mmin = 59.9
Mmax = 60.1
N1 = 100
dM = (Mmax-Mmin)/(N1-1.)

binnedTrace =zeros([N1])
binnedL =zeros([N1])
TotalL = 0.
Mtab =zeros([N1])
for i in range(0,N1):
    Mtab[i] = Mmin + (i+0.5)*dM

Ntrace = len(Trace)
print("We have ",Ntrace," results from MCMC")
# Only keep 50% of the trace
StartTrace = int(0.5*Ntrace)
for i in range(StartTrace,Ntrace):
    m = Trace[i][0]
    l = Trace[i][1]
    ii = int((m-Mmin)/dM)
    if(ii>=0 and ii<N1):
        binnedTrace[ii] = binnedTrace[ii]+1.
        
for i in range(0,N1):
    M = Mmin + i*dM
    Model = PhenomGW(M,Eta,d,0,0)
    # Use log(L') = log(L/Ltrue) so that the largest likelihood is ~1 
    L = GetLIGOLogLikelihood(Model,Signal)-Ltrue
    binnedL[i] = exp(1.*L)
    TotalL = TotalL + binnedL[i]

# Normalize luminosity and output of MCMC     
for i in range(0,N1):
    binnedL[i] = binnedL[i]/TotalL
    binnedTrace[i] = binnedTrace[i]/(Ntrace-StartTrace)
    
plt.plot(Mtab,binnedTrace,color='black')
plt.plot(Mtab,binnedL,color='red')
