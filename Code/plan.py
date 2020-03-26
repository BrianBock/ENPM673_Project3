
import numpy as np
import random
import math
from get_data import *


def GaussianNormal(Sigma, x, mu):

    A = np.asarray(x-mu)
    Sig_inv = np.linalg.inv(Sigma)
    At = np.transpose(A)

    exponent=-.5*np.linalg.multi_dot((At,Sig_inv,A))
    denom = math.sqrt((2*math.pi**3)*np.linalg.det(Sigma))

    p= (1/denom)*math.exp(exponent)

    return p



def GaussianMixtureModel(K, dataset):
    N=len(dataset[0])

    # Compute starting values
    Sigma=[]
    alpha=[]
    mu=[]

    var_b=np.var(dataset[0,:])
    var_g=np.var(dataset[1,:])
    var_r=np.var(dataset[2,:])

    mu_b = np.mean(dataset[0,:])
    mu_g = np.mean(dataset[1,:])
    mu_r = np.mean(dataset[2,:])
 
    sig_rg, sig_rb, sig_gb = 0,0,0   
    for i in range(N):
        sig_rg += (dataset[2,i]-mu_r)*(dataset[1,i]-mu_g)*(1/N)
        sig_rb += (dataset[2,i]-mu_r)*(dataset[0,i]-mu_b)*(1/N)
        sig_gb += (dataset[1,i]-mu_g)*(dataset[0,i]-mu_b)*(1/N)

    Sigma_k = np.array([[var_r, sig_rg, sig_rb],
                        [sig_rg, var_g, sig_gb],
                        [sig_rb, sig_gb, var_b]])

    for k in range (0,K):
        Sigma.append(Sigma_k)

        alpha_k=1/K
        alpha.append(alpha_k)

        ind=random.randint(0,N)
        mu_b = dataset[0,ind]
        mu_g = dataset[1,ind]
        mu_r = dataset[2,ind]

        mu_k=np.array([[mu_r], [mu_g], [mu_b]])
        mu.append(mu_k)
        
    log_like = 0
    diff = 100
    thresh = 1
    while diff>thresh:
        prev = log_like
        # Compute weights
        w_ik = np.zeros((N,K)) 
        for i in range (0,N):
            p = []
            p_sum = 0
            for k in range(0,K):
                x_i = np.array([[dataset[2,i]],[dataset[1,i]],[dataset[0,i]]])
                p_k=GaussianNormal(Sigma[k], x_i, mu[k])
                p.append(p_k)
                p_sum += p_k*alpha[k]
            for k in range(K):
                if p_sum == 0:
                    w_ik[i,k] = 0
                else:
                    w_ik[i,k] = (p[k]*alpha[k])/p_sum 

        # Compute new values
        mu = []
        Sigma = []
        alpha = []
        for k in range(0,K):
            N_k=sum(w_ik[:,k])
            
            alpha_k=N_k/N
            alpha.append(alpha_k)

            sum_arr = np.zeros((3,1))
            for i in range(N):
                x_i = np.array([[dataset[2,i]],[dataset[1,i]],[dataset[0,i]]])
                sum_arr = sum_arr+ w_ik[i,k]*x_i

            mu_k = (1/N_k)*sum_arr
            mu.append(mu_k)

            sum_arr = np.zeros((3,3))
            for i in range(N):
                x_i = np.array([[dataset[2,i]],[dataset[1,i]],[dataset[0,i]]])
                A = w_ik[i,k]*np.dot((x_i-mu_k),np.transpose(x_i-mu_k))
                sum_arr = sum_arr + A

            Sigma_k = (1/N_k)*sum_arr
            Sigma.append(Sigma_k)

        # Compute Log Likelihood with new values
        log_like = 0
        for i in range(N):
            sum_term = 0
            for k in range(K):
                x_i = np.array([[dataset[2,i]],[dataset[1,i]],[dataset[0,i]]])
                p_k=GaussianNormal(Sigma[k], x_i, mu[k])
                sum_term+=alpha[k]*p_k
            log_like += np.log(sum_term)

        # print(log_like)

        diff = abs(log_like-prev)

        print(diff)

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

    bouy_colors = ['yellow','orange','green']
    colorspace = 'HSV' #HSV or BGR
    training_data = {}
    testing_data = {}
    for color in bouy_colors:
        train_path = 'Training Data/'+color
        test_path = 'Testing Data/'+color

        train_data= generate_dataset(train_path,colorspace)
        test_data= generate_dataset(test_path,colorspace)

        training_data[color] = train_data
        testing_data[color] = test_data

    K=4

    dataset = training_data['yellow']
    GaussianMixtureModel(K,dataset)


