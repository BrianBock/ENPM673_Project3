
import numpy as np
import random
import math
import os
from get_data import *


def GaussianNormal(Sigma, x, mu):

    A = np.asarray(x-mu)
    Sig_inv = np.linalg.inv(Sigma)
    At = np.transpose(A)

    exponent=-.5*np.linalg.multi_dot((At,Sig_inv,A))
    denom = math.sqrt((2*math.pi**3)*np.linalg.det(Sigma))

    p= (1/denom)*math.exp(exponent)

    return p


def writeGMM(Sigma, mu, alpha, path):
    if os.path.exists(path):
        os.remove(path)
    # Create the file
    GMM_output=open(path,"a+")

    # Write Sigma
    np.savez(path,Sigma=Sigma,mu=mu,alpha=alpha)



def GaussianMixtureModel(K, dataset, thresh, path):
    path+="/GMMoutput.npz"
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
        
    diff = 100
    iter_count = 0
    while diff>thresh:
        if iter_count == 0:
            prev = 0
        else:
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

        diff = abs(log_like-prev)
        iter_count += 1

        print(diff)

    writeGMM(Sigma, mu, alpha, path)
    return Sigma, mu, alpha


def test_score(Sigma,mu,alpha,dataset):
    N = len(dataset[0])

    count = 0
    thresh = 1*10**-6
    for i in range(N):
        prob_sum = 0
        p=[]
        for k in range(K):
            x_i = np.array([[dataset[2,i]],[dataset[1,i]],[dataset[0,i]]])
            p_k = GaussianNormal(Sigma[k], x_i, mu[k])
            prob_sum += p_k*alpha[k]
            p.append(p_k*alpha[k])
        print(prob_sum)
        # for k in range(0,K):
        #     prob=p_k/prob_sum
        #     print(prob)
        if prob_sum >= thresh:
            print("True")
            count += 1
            # break

    return (count/N *100)



def readGMM(path):
    path+="/GMMoutput.npz"
    if os.path.exists(path):
        print("File exists")
        GMM_file=open(path,"r")
        
        with np.load(path) as data:
            Sigma = data['Sigma']
            mu = data['mu']
            alpha = data['alpha']

        newGMM=False

    else:
        print("Cannot read '"+path+"'. Will compute new GMM values.")
        newGMM = True
        Sigma, mu, alpha = None, None, None

    return newGMM, Sigma, mu, alpha





if __name__ == '__main__':

    bouy_colors = ['yellow','orange','green']
    # bouy_colors = ['orange']#,'orange','green']
    colorspace = 'HSV' #HSV or BGR
    training_data = {}
    testing_data = {}

    K=4
    thresh = 2

    mean_images = {}

    Theta = {}


    for color in bouy_colors:
        train_path = 'Training Data/'+color
        test_path = 'Testing Data/'+color

        train_data= generate_dataset(train_path,colorspace)
        test_data= generate_dataset(test_path,colorspace)

        training_data[color] = train_data
        testing_data[color] = test_data


        newGMM=False

        if not newGMM:
            # Check if the right files exist. If they don't, toggle newGMM=True
            print("Checking if the GMM outputs exist. If they don't, I'll need to compute them.")
            newGMM, Sigma, mu, alpha=readGMM(train_path)

        if newGMM:
            print("Generating new GMM values...")
            print(color)
            dataset = training_data[color]
            Sigma, mu, alpha = GaussianMixtureModel(K,dataset,thresh,train_path)

        

        mu_colors = np.zeros((500,500,3),np.uint8)

        r_starts = [(0,0),(250,0),(0,250),(250,250)]
        for k in range(K):
            print('\n\nK = '+str(k+1))
            print('Sigma')
            print(Sigma[k])
            print('mu')
            print(mu[k])
            print('alpha')
            print(alpha[k])

            r = mu[k][0]
            g = mu[k][1]
            b = mu[k][2]
            x,y = r_starts[k]
            start = (x,y)
            end = (x+250,y+250)

            cv2.rectangle(mu_colors,start,end,(int(b),int(g),int(r)),-1)

        mean_images[color] = mu_colors

        Theta[color] = [Sigma,mu,alpha]

    for color in bouy_colors:
        cv2.imshow(color+'mean colors (Press any key to continue)',mean_images[color])
    
    cv2.waitKey(0)

    dataset = testing_data['orange']
    Sigma = Theta['green'][0]
    mu = Theta['green'][1]
    alpha = Theta['green'][2]
    score = test_score(Sigma,mu,alpha,dataset)
    print('\n'+str(score))


