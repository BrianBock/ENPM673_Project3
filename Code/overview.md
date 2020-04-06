We are modeling the points as a mixture of K gaussians 

Each gaussian has a corresponding mixture weight pi_k, which sum to 1

Each gaussian is defined by a vector of mu_k and covariance matrix Sigma_k

Try to find K=6 clusters (2 for each color buoy)
Steps for EM
    Initialize all mu_k,Sigma_k, and pi_k
        Set all Sigma_k equal to the identity matrix
        For each buoy color, split the data in half and find the mean of each half,
        use these as the starting mean for that cluster
        Initialize all pi_k as 1/K
    E Step 
        Calculate the responsibility (r_ik) of each data point in each of the K clusters
        Sum of r_ik over K = 1
        To calculate r_ik:
            Find the N_k for all clusters (calculate the guassian norm using the mu and Sigma for that cluster)
            r_ik = pi_k*N_k/sum(pi_k*N_k)
    M Step
        Calculate new values for mu, Sigma and pi for each cluster using the responsibilities from the E step
        mu_k = sum(r_ik*x_i)/sum(r_ik)
        Sigma_k = sum(r_ik*(x_i-mu_k)*(x_i-mu_k)')/sum(r_ik)
        pi_k = 1/n*sum(r_ik)
    Calculate Log Likelihood
        sum_i(ln(sum_k(pi_k*N_k)))
        Stop when log likelihood stops changing dramatically


Result of EM will be mu, Sigma and pi for all 6 clusters
We then need to label each cluster according to a bouy color