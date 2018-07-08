# Astrosoma

In this practical session, we will implement a Markov-Chain Monte-Carlo algorithm to extract the parameters
of a binary system, given its gravitational wave signal.

To get started, include as a library or add to your python notebook the code provided in MCMCPractical.py.
This code includes functions and variables useful for the rest of the problem set.

* Nbins_f : number of frequency bins used in this analysis (we will use GW signals expressed in frequency space,
as a vector of Nbins_f values computed at specified frequencies). This value can be modified as desired.

* Freq_Low, Freq_High : lowest and highest frequencies used in the analysis

* Freq_BinCenter, Freq_BinWidth : Central value and width of the Nbins_f frequency bins.

* PhenomGW(M,eta,d,t0,phi0)
This function outputs the real and imaginary part of the Fourier transform of the gravitational wave signal,
at the specified set of frequencies. The output of the function is a matrix of dimensions [2,Nbins_f].
Assuming that the masses of the two merging objects are m1 and m2, the input variables are
M = m1+m2 : Total mass of the binary
eta = m1*m2/M^2 : Symmetric mass ratio
d : distance between the observer and the source
t0,phi0 : arbitary time and phase shifts that can be applied to the waveform

* GetNoise()
Obtain a realization of the (fake) LIGO noise in Fourier space. As PhenomGW, this outputs a [2,Nbins_f] matrix with the real and imaginary parts of the noise at the Nbins_f chosen frequencies

* GetLIGOLikelihood(ModelGW,SignalGW)
Returns the likelihood of the input model ModelGW given the input signal SignalGW.
Both input variables are [2,Nbins_f] matrices, while the output is a real number.

* GetLIGOSNR(SignalGW)
Returns the signal-to-noise ratio of the input signal SignalGW.