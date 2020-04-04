
import numpy as np
from scipy.stats import multivariate_normal
import random
import math
import os
from datetime import datetime
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



def GaussianMixtureModel(K, dataset, thresh, path,f):
    path+="/GMMoutput.npz"
    N= dataset.shape[1]
    start_time=datetime.now()
    # Compute starting values
    Sigma=[]
    alpha=[]
    mu=[]

    # ch_b = dataset[0,:]
    # ch_g = dataset[1,:]
    # ch_r = dataset[2,:]

    # var_b=np.var(ch_b)
    # var_g=np.var(ch_g)
    # var_r=np.var(ch_r)

    # mu_b = np.mean(ch_b)
    # mu_g = np.mean(ch_g)
    # mu_r = np.mean(ch_r)
 
    Sigma_k = np.cov(dataset,bias=True)

    # sig_rg, sig_rb, sig_gb = 0,0,0   
    # for i in range(N):
    #     sig_rg += (ch_r-mu_r)*(ch_g-mu_g)*(1/N)
    #     sig_rb += (ch_r-mu_r)*(ch_b-mu_b)*(1/N)
    #     sig_gb += (ch_g-mu_g)*(ch_b-mu_b)*(1/N)

    # Sigma_k = np.array([[var_r, sig_rg, sig_rb],
    #                     [sig_rg, var_g, sig_gb],
    #                     [sig_rb, sig_gb, var_b]])

    for k in range (0,K):
        Sigma.append(Sigma_k)

        alpha_k=1/K
        alpha.append(alpha_k)

        ind=random.randint(0,N)
        mu_b = dataset[0,ind]
        mu_g = dataset[1,ind]
        mu_r = dataset[2,ind]

        mu_k=np.array([[mu_b], [mu_g], [mu_r]])
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
        # for i in range (0,N):
        #     p = []
        #     p_sum = 0
        #     for k in range(0,K):
        #         x_i = np.array([[dataset[0,i]],[dataset[1,i]],[dataset[2,i]]])
        #         p_k=GaussianNormal(Sigma[k], x_i, mu[k])
        #         p.append(p_k)
        #         p_sum += p_k*alpha[k]

        p_sum = np.zeros(N)
        p_k = []
        for k in range(K):
            p = multivariate_normal.pdf(np.transpose(dataset),mean=np.transpose(mu[k])[0],cov = Sigma[k],allow_singular=True)
            p_k.append(p)
            p_sum += p

        for i in range(N):
            for k in range(K):
                p_sum_i = p_sum[i]
                if p_sum_i == 0:
                    w_ik[i,k] = 0
                else:
                    p_ik = p_k[k][i]
                    w_ik[i,k] = (p_ik*alpha[k])/p_sum_i 

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
                x_i = np.array([[dataset[0,i]],[dataset[1,i]],[dataset[2,i]]])
                sum_arr = sum_arr+ w_ik[i,k]*x_i

            mu_k = (1/N_k)*sum_arr
            mu.append(mu_k)

            sum_arr = np.zeros((3,3))
            for i in range(N):
                x_i = np.array([[dataset[0,i]],[dataset[1,i]],[dataset[2,i]]])
                A = w_ik[i,k]*np.dot((x_i-mu_k),np.transpose(x_i-mu_k))
                sum_arr = sum_arr + A

            Sigma_k = (1/N_k)*sum_arr
            Sigma.append(Sigma_k)

        # Compute Log Likelihood with new values
        # log_like = 0
        # for i in range(N):
        #     sum_term = 0
        #     for k in range(K):
        #         x_i = np.array([[dataset[0,i]],[dataset[1,i]],[dataset[2,i]]])
        #         p_k=GaussianNormal(Sigma[k], x_i, mu[k])
        #         sum_term+=alpha[k]*p_k
        #     log_like += np.log(sum_term)
        
        log_like = 0
        for k in range(K):
            log_like_k = sum(multivariate_normal.logpdf(np.transpose(dataset),mean=np.transpose(mu[k])[0],cov=Sigma[k],allow_singular=True))
            log_like += log_like_k


        diff = abs(log_like-prev)
        iter_count += 1

        print(diff)
        end_time=datetime.now()
        f.write(str(diff)+","+str(end_time)+"\n")


    writeGMM(Sigma, mu, alpha, path)
    
    return Sigma, mu, alpha


def determine_threshold(Sigma,mu,alpha,test_data,percentage):
    N = len(test_data[0])

    probs = []
    
    for i in range(N):
        prob_sum = 0

        for k in range(K):
            x_i = np.array([[test_data[0,i]],[test_data[1,i]],[test_data[2,i]]])
            p_k = GaussianNormal(Sigma[k], x_i, mu[k])
            prob_sum += p_k*alpha[k]

        probs.append(prob_sum)
    
    # Sort list and find value that will fit percentage of data 
    probs.sort(reverse=True)

    min_prob = probs[int(N*percentage)]


    return min_prob


def test_score(Sigma,mu,alpha,test_data,threshold):
    N = len(test_data[0])
    
    count = 0
    for i in range(N):
        prob_sum = 0

        for k in range(K):
            x_i = np.array([[test_data[0,i]],[test_data[1,i]],[test_data[2,i]]])
            p_k = GaussianNormal(Sigma[k], x_i, mu[k])
            prob_sum += p_k*alpha[k]

        if prob_sum >= threshold:
            count+=1

    return count/N*100



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

    
    # bouy_colors = ['yellow','orange','green']
    bouy_colors = ['orange']
  
    colorspace = 'BGR' #HSV or BGR
    training_data = {}
    testing_data = {}

    K = 8
    diff_thresh = 5
    newGMM=True

    mean_images = {}

    Theta = {}

    # Smaller Sample/
    for color in bouy_colors:
        f=open(str(color)+"diff_times.txt","a+")
        train_path = 'Training Data/'+color
        test_path = 'Testing Data/'+color

        train_data= generate_dataset(train_path,colorspace)
        test_data= generate_dataset(test_path,colorspace)

        training_data[color] = train_data
        testing_data[color] = test_data

        print('Shape of train data array for ' + color + ': ',end = '')
        print(train_data.shape[1])
        print('Shape of test_data data array for ' + color + ': ',end = '')
        print(test_data.shape[1])


        need_confirm=True
        while need_confirm:
            if not newGMM:
                # Check if the right files exist. If they don't, toggle newGMM=True
                print("Checking if the GMM outputs exist. If they don't, I'll need to compute them.")
                newGMM, Sigma, mu, alpha=readGMM(train_path)

            if newGMM:
                confirm=input("This will overwrite any exisiting GMM values. Are you sure you want to continue? Press 'y' to confirm or 'n' to abort: ")
                if confirm.lower() == 'y' or confirm.lower() == 'yes':
                    need_confirm=False
                    print("Generating new GMM values...")
                    print(color)
                    dataset = training_data[color]
                    start_time=datetime.now()
                    Sigma, mu, alpha = GaussianMixtureModel(K,dataset,diff_thresh,train_path,f)
                    end_time=datetime.now()
                    print("Finished in "+str(end_time-start_time)+" (hours:min:sec)")

                if confirm.lower()=='n' or confirm.lower() == 'no':
                    print("No new GMM values will be computed. Please note that 'newGMM' is still set to 'True' in gaussian_mixture_model.py. If you don't want any new GMM values computed, you'll need to toggle this to False before you run this program again. Exiting")
                    need_confirm=False
                    exit()
                else:
                    print("I don't understand what you tried to enter. Please try again.")




        # mu_colors = np.zeros((500,500,3),np.uint8)

        # r_starts = [(0,0),(250,0),(0,250),(250,250)]
        for k in range(K):
            print('\n\nK = '+str(k+1))
            print('Sigma')
            print(Sigma[k])
            print('mu')
            print(mu[k])
            print('alpha')
            print(alpha[k])

        #     b = mu[k][0]
        #     g = mu[k][1]
        #     r = mu[k][2]
        #     x,y = r_starts[k]
        #     start = (x,y)
        #     end = (x+250,y+250)

        #     cv2.rectangle(mu_colors,start,end,(int(b),int(g),int(r)),-1)

        # mean_images[color] = mu_colors

        Theta[color] = [Sigma,mu,alpha]

    # for color in bouy_colors:
    #     cv2.imshow(color+' mean colors (Press any key to continue)',mean_images[color])
    
    # cv2.waitKey(0)

    # test_data = testing_data['green']
    # Sigma = Theta['green'][0]
    # mu = Theta['green'][1]
    # alpha = Theta['green'][2]
    # thresh = determine_threshold(Sigma,mu,alpha,test_data,.9)
    # print()
    # print(thresh)

    # test_data = testing_data['orange']
    # score = test_score(Sigma,mu,alpha,test_data,thresh)
    # print('The score is: '+str(score))



