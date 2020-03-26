
import numpy as np
import random
import math


def GaussianNormal(Sigma, x, mu):

	exponent=-.5*np.transpose(x-mu)*np.linalg.inv(Sigma)*(x-mu)
	print(exponent)

	print(np.linalg.det(Sigma))

	p=(1/math.sqrt((2*math.pi())**3*np.linalg.det(Sigma)))#*math.exp(exponent)

	return p



def GaussianMixtureModel(K, dataset):
	N=len(dataset)

	# Compute starting values
	Sigma=[]
	alpha=[]
	mu=[]
	for k in range (0,K):
		sig_b=random.randint(0,255)
		sig_g=random.randint(0,255)
		sig_r=random.randint(0,255)

		Sigma_k = np.array([[sig_r**2, sig_r*sig_g, sig_r*sig_b],
			  				[sig_r*sig_g, sig_g**2, sig_g*sig_b],
			  				[sig_r*sig_b, sig_g*sig_b, sig_b**2]])

		Sigma.append(Sigma_k)

		alpha_k=1/K
		alpha.append(alpha_k)

		mu_b=random.randint(0,int(math.sqrt(255)))
		mu_g=random.randint(0,int(math.sqrt(255)))
		mu_r=random.randint(0,int(math.sqrt(255)))

		mu_k=[mu_b, mu_g, mu_r]
		mu.append(mu_k)
		


	print(Sigma[k])
	print(alpha)
	print(mu[k])
	print(dataset[0])


	# Compute probabilities for weights
	p=[]
	for i in range (0,N):
		for k in range(0,K):
			p_k=GaussianNormal(Sigma[k], dataset[i], mu[k])
			p.append(p_k*alpha[k])

	print(p)

	# Compute the weights
	for k in range(0,K):
		w_ik=p[k]/sum(p)






	# Compute new values
	while convergence>conv:
		for k in range(0,K):
			N_k=sum(weights)
			alpha_k=N_k/N
			mu_k=(1/N_k)*sum(weights*x_i)

# return mu, Sigma, alpha

# N is the length of our dataset (number of pixels in every frame for that color bouy)
# Compute starting values
	# K random sigma_g, sigma_r, sigma_b (to get a positive definite Sigma) - 0-255

	# Sigma_k = [[sig_r_k**2, sig_r_k*sig_g_k, sig_r_k*sig_b_k],
			  # [sig_r_k*sig_g_k, sig_g_k**2, sig_g_k*sig_b_k],
			  # [sig_r_k*sig_b_k, sig_g_k*sig_b, sig_b_k**2]]

	# Sigma=[Sigma_1...Sigma_K]

	# K random mu (0-255) 3xK array (one for each channel)

	# K random alpha (sum of all alpha=1). Start->alpha_k=1/K
	# alpha_1 = rand (0,1)
	# alpha_2=rand(0,1-alpha_1)
	# alpha_3=1-(alpha_2+alpha_1)


# Compute weights

	# compute all weights using starting values (loop through all points and all K)
	# probabilities determined via gaussian equation
	# use equation from EM notes bottom of page1

# Compute new values
	# alpha_k_new=sum of weights / N
	# mu_new_k = 1/N_k sum weights*xi

# Iteratively solve
	# Check for convergence. exit when sufficiently converged



if __name__ == '__main__':
	dataset=np.array([[15,16,17],
					[7,8,9],
					[11,12,13],
					[14,15,16],
					[19,20,21],
					[6,5,3]])
	K=4
	GaussianMixtureModel(K,dataset)


