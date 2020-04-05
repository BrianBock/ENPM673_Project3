from scipy.stats import multivariate_normal
import math
from get_data import *

# We have n data points that each have 3 channels (B,G,R) which are numbers between 0 and 255


# def GaussianNormal(Sigma, x, mu):

#     A = np.asarray(x-mu)
#     Sig_inv = np.linalg.inv(Sigma)
#     At = np.transpose(A)

#     exponent=-.5*np.linalg.multi_dot((At,Sig_inv,A))
#     denom = math.sqrt((2*math.pi**3)*np.linalg.det(Sigma))

#     p= (1/denom)*math.exp(exponent)

#     return p



colorspace = 'BGR' #HSV or BGR

buoy_colors = ['orange','green','yellow']
clusters_per_buoy = 2

x = []
mu = []
Sigma = []
pi = []

K = len(buoy_colors)*clusters_per_buoy

# Configure data set and initialze starting values
for color in buoy_colors: 
    path = 'Smaller Sample/Training Data/' + color
    data = generate_dataset(path,colorspace)
    g = data[1,:]
    r = data[2,:]

    # Generate full dataset as python list of numpy arrays of green and red channel for each data_point
    for g_ch,r_ch in zip(g,r):
        x_i = np.array([g_ch,r_ch])
        x.append(x_i)

    split = len(g)//clusters_per_buoy
    for i in range(clusters_per_buoy):
        # Initialize mu_k with appropriate number of clusters per buoy
        start = i*split
        end = (i+1)*split
        if i == clusters_per_buoy-1:
            mu.append(np.array([np.mean(g[start:]),np.mean(r[start:])]))
        else:
            mu.append(np.array([np.mean(g[start:end]),np.mean(r[start:end])]))
        
        # Initialize Sigma_k as identity matrix
        Sigma.append(np.identity(2))

        # Initialize mixture weights as 1/K
        pi.append(1/K)

x = np.asarray(x)

print("Starting E step")
print(pi)
# E Step 
    # Calculate the responsibility (r_ik) of each data point in each of the K clusters
    # Sum of r_ik over K = 1
    # To calculate r_ik:
        # Find the N_k for all clusters (calculate the guassian norm using the mu and Sigma for that cluster)
        # r_ik = pi_k*N_k/sum(pi_k*N_k)

n = len(x)
r = np.zeros((n,K))
for i in range(n):
    print(x[i])
    N = []
    denom = 0
    for k in range(K):
        N_k = multivariate_normal.pdf(x[i],mean=mu[k],cov = Sigma[k])
        # N_k=GaussianNormal(Sigma[k],x[i],mu[k])
        # print(N_k)
        N.append(N_k)
        denom += pi[k]*N_k
    for k in range(K):
        # print(denom)
        r[i,k] = pi[k]*N[k]/denom

print("Starting M step")
# M Step
    # Calculate new values for mu, Sigma and pi for each cluster using the responsibilities from the E step
    # mu_k = sum(r_ik*x_i)/sum(r_ik)
    # Sigma_k = sum(r_ik*(x_i-mu_k)*(x_i-mu_k)')/sum(r_ik)
    # pi_k = 1/n*sum(r_ik)

for k in range(K):
    r_k = r[:,k]
    print(r_k)
    mu[k] = sum(r_k.dot(x))/sum(r_k)
    print(mu[k])

    Sigma_k = np.zeros((2,2))
    for i in range(n):
        a = x[i]-mu[k] # 2xn matrix
        Sigma_k += r_k[i]*(a.dot(np.transpose(a))) # 2x2 matrix

    Sigma_k /= np.sum(r_k)

    Sigma[k] = Sigma_k

    pi_k=(1/n)*sum(r_k)
   

    # Calculate Log Likelihood
        # sum_i(ln(sum_k(pi_k*N_k)))
        # Stop when log likelihood stops changing dramatically 
a=0
b=0
for i in range(0,n):
    for k in range (0,K):
        a+=pi[k]*N[k]
    b+=a



# We are modeling the points as a mixture of K gaussians 

# Each gaussian has a corresponding mixture weight pi_k, which sum to 1

# Each gaussian is defined by a vector of mu_k and covariance matrix Sigma_k

# Try to find K=6 clusters (2 for each color buoy)
# Steps for EM
#     Initialize all mu_k,Sigma_k, and pi_k
#         Set all Sigma_k equal to the identity matrix
#         For each buoy color, split the data in half and find the mean of each half,
#         use these as the starting mean for that cluster
#         Initialize all pi_k as 1/K
#     E Step 
#         Calculate the responsibility (r_ik) of each data point in each of the K clusters
#         Sum of r_ik over K = 1
#         To calculate r_ik:
#             Find the N_k for all clusters (calculate the guassian norm using the mu and Sigma for that cluster)
#             r_ik = pi_k*N_k/sum(pi_k*N_k)
#     M Step
#         Calculate new values for mu, Sigma and pi for each cluster using the responsibilities from the E step
#         mu_k = sum(r_ik*x_i)/sum(r_ik)
#         Sigma_k = sum(r_ik*(x_i-mu_k)*(x_i-mu_k)')/sum(r_ik)
#         pi_k = 1/n*sum(r_ik)
#     Calculate Log Likelihood
#         sum_i(ln(sum_k(pi_k*N_k)))
#         Stop when log likelihood stops changing dramatically 

# Result of EM will be mu, Sigma and pi for all 6 clusters
# We then need to label each cluster according to a bouy color



def plots():

    colorspace = 'BGR' #HSV or BGR

    bouy_colors = ['orange','green','yellow']

    fig,((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,figsize= (12,8))

    for color in bouy_colors: 
        path = 'Smaller Sample/Training Data/' + color
        data = generate_dataset(path,colorspace)
        b = data[0,:]
        g = data[1,:]
        r = data[2,:]
        ax1.scatter(b,r,edgecolors = color,facecolors='none')
        ax2.scatter(g,r,edgecolors = color,facecolors='none')
        ax3.scatter(g,b,edgecolors = color,facecolors='none')

        ax4.hist(b,color=color,alpha=.2)
        ax5.hist(g,color=color,alpha=.2)
        ax6.hist(r,color=color,alpha=.2)


    plt.sca(ax1)
    plt.xlabel("Blue Channel")
    plt.ylabel("Red Channel")

    plt.sca(ax2)
    plt.xlabel("Green Channel")
    plt.ylabel("Red Channel")

    plt.sca(ax3)
    plt.xlabel("Green Channel")
    plt.ylabel("Blue Channel")

    plt.sca(ax4)
    plt.title('B Channel')
    plt.xlabel('Intensity')
    plt.ylabel('Num pixels')

    plt.sca(ax5)
    plt.title('G Channel')
    plt.xlabel('Intensity')
    plt.ylabel('Num pixels')

    plt.sca(ax6)
    plt.title('R Channel')
    plt.xlabel('Intensity')
    plt.ylabel('Num pixels')

    plt.show()
