import numpy as np

x=np.array([[227],[232]])
mu=np.array([[197.35349],[235.2249]])

rik=.0027154

a=rik*(x-mu)*np.transpose(x-mu)
print(a)