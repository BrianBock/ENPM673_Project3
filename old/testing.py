#Import required packages
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# import statistics as stats
import cv2


def getImage(color,count):
	path = 'Training/'+color+'/'+str(count)+'.png'
	image=cv2.imread(path)

	if image is None:
		print("Unable to import image '"+path+"'. Quitting.")
		exit()
	return image





def training(color):
	count=0
	image=getImage(color,count)

	(B,G,R)=cv2.split(image.astype("float"))

	# print(B,G,R)

	sig_b=np.std(B) #stdev in Blue channel
	sig_g=np.std(G) #stdev in Green channel
	sig_r=np.std(R) #stdev in Red channel

	mu_b=np.mean(B)
	mu_g=np.mean(G)
	mu_r=np.mean(R)

	# print(sig_b,sig_g,sig_r)

	covariance=np.array([[sig_r**2, sig_r*sig_g, sig_r*sig_b],
						  [sig_r*sig_g, sig_g**2, sig_g*sig_b],
						  [sig_r*sig_b, sig_g*sig_b, sig_b**2]])

	print(covariance)

	p_orange=(1/math.sqrt(2*math.pi))




training("orange")


