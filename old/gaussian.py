
#Import required packages
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import signal




sigma=[]
mu=[]
alpha=[]
N=[]
akN=[]
K=3
for k in range(0,K):
	sig_k=.1
	mu_k=k
	alpha_k=k/K
	x = np.random.normal(mu_k, sig_k, 1000)
	N_k=(1/(sig_k*np.sqrt(2*np.pi)))*np.exp(-(1/2)*((x-mu_k)/sig_k)**2)

	sigma.append(sig_k)
	mu.append(mu_k)
	alpha.append(alpha_k)
	N.append(N_k)

	akN.append(alpha_k*N_k)

P=np.sum(akN)
print(P)

posterior_prob=N*alpha_k/P



