# window1=signal.gaussian(100, std=10)
# window2=signal.gaussian(100, std=5)

# window3=window1+window2

# plt.plot(window1)
# plt.plot(window2)
# plt.plot(window3)

# mu, sigma = 0, 0.1 # mean and standard deviation
# x = np.random.normal(mu, sigma, 1000)

# N=len(x)

# mu_ml=(1/N)*np.sum(x)
# sig_x=[]
# for xn in x:
# 	sig_x.append((xn-mu_ml)*(xn-mu_ml))

# Sigma_ml=(1/N)*np.sum(sig_x)

# print(Sigma_ml)

# alpha_k_new=N_k/N





#


# s 

# newmu=mu+.25
# a = np.random.normal(newmu, sigma, 1000)

# b=a+s

# count, bins, ignored = plt.hist(s, 90, density=True)
# plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')

# count, bins, ignored = plt.hist(a, 30, density=True)
# plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - newmu)**2 / (2 * sigma**2) ),linewidth=2, color='r')

# count, bins, ignored = plt.hist(b, 30, density=True)
# plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - newmu)**2 / (2 * sigma**2) ),linewidth=2, color='r')


# plt.show()