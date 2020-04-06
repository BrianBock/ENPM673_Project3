from scipy.stats import multivariate_normal
import math
from matplotlib.patches import Ellipse
from get_data import *

def writeGMM(Sigma, mu, pi):
    path = "EMoutput.npz"
    if os.path.exists(path):
        os.remove(path)
    # Create the file
    GMM_output=open(path,"a+")

    # Write Sigma
    np.savez(path,Sigma=Sigma,mu=mu,pi=pi)


def readGMM():
    path = "EMoutput.npz"
    if os.path.exists(path):
        print("File exists")
        GMM_file=open(path,"r")
        
        with np.load(path) as data:
            Sigma = data['Sigma']
            mu = data['mu']
            pi = data['pi']

        newGMM=False

    else:
        print("Cannot read '"+path+"' ")
        exit()

    return Sigma, mu, pi


def initialize_data(buoy_colors,clusters_per_buoy):
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
            Sigma.append(np.identity(2)*50)

            # Initialize mixture weights as 1/K
            pi.append(1/K)

    x = np.asarray(x)

    return x,mu,Sigma,pi


def EM(x,mu,Sigma,pi,K):
    n = len(x)
    r = np.zeros((n,K))

    count = 0
    converge_thresh = 5

    while True:
        # print("Starting E step")
        for k in range(K):
            if k == 0:
                N = multivariate_normal.pdf(x,mean=mu[k],cov = Sigma[k]).reshape(1,n)
            else:
                N = np.append(N,multivariate_normal.pdf(x,mean=mu[k],cov = Sigma[k]).reshape(1,n),axis=0)

        if count != 0:
            prev = log_like
        
        log_like = 0

        for i in range(n):
            denom = 0
            for k in range(K):
                denom += pi[k]*N[k,i]
            for k in range(K):
                r[i,k] = pi[k]*N[k,i]/denom

            log_like += np.log(denom)

        if count != 0:
            print('Change in log-likelihood:',end=' ')
            print(log_like - prev)
            if log_like - prev < converge_thresh:
                break

        for k in range(K):
            r_k = r[:,k]
            mu[k] = np.sum(np.array([r_k*x[:,0],r_k*x[:,1]]),axis=1)/np.sum(r_k)

            numerator = np.zeros((2,2))
            mu_k = mu[k].reshape(2,1)

            for i in range(n):
                a = x[i].reshape(2,1)-mu_k
                numerator += r_k[i]*a*a.T
            
            Sigma[k] = numerator/np.sum(r_k)

            pi[k] = 1/n*np.sum(r_k)

        count += 1

        writeGMM(Sigma, mu, pi)

    return mu,Sigma,pi


def plots(colorspace,bouy_colors):
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





if __name__ == '__main__':
    colorspace = 'BGR' #HSV or BGR

    buoy_colors = ['orange','green','yellow']
    clusters_per_buoy = 5
    K = len(buoy_colors)*clusters_per_buoy

    newEM = False

    if newEM:
        x,init_mu,init_Sigma,init_pi = initialize_data(buoy_colors,clusters_per_buoy)
        mu,Sigma,pi = EM(x,init_mu,init_Sigma,init_pi,K)

    else:
        Sigma, mu, pi = readGMM()

    patches = []
    fig, ax = plt.subplots()
    for k in range(K):
        center = (mu[k][0],mu[k][1])
        w,v = np.linalg.eig(Sigma[k])
        width = 2*math.sqrt(5.991*w[0])
        height = 2*math.sqrt(5.991*w[1])
        angle = np.rad2deg(math.atan2(max(v[0]),max(v[1])))

        ellipse = Ellipse(center, width, height,angle=angle,color=buoy_colors[k%3],alpha = .4)
        ax.add_patch(ellipse)

    for color in buoy_colors: 
        path = 'Smaller Sample/Training Data/' + color
        data = generate_dataset(path,colorspace)
        g = data[1,:]
        r = data[2,:]
        ax.scatter(g,r,edgecolors = color,facecolors='none')

    plt.xlim((0,255))
    plt.ylim((0,255))
    plt.show()


