# Brian Bock
# ENPM809T
# HW 3

# Import required packages
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime as dtime
import datetime
import time

filepath="diff_times.txt"

# Import the data
dataFile = open(filepath,"r")

data=[]
diff_list=[]
time_s_list=[]

for i,line in enumerate(dataFile):
	line=line[:-1] #lines end with "\n", remove
	diff,time=line.split(',')
	diff_list.append(float(diff))
	# print(diff)
	# timestamp=time[11:]
	
	timestamp=dtime.strptime(time, "%Y-%m-%d %H:%M:%S.%f")#convert the strings to datetime objects
	# print(timestamp)
	time_s  = timestamp.timestamp()-1585952995.141063
	# print(time_s)
	time_s_list.append(time_s)

	data.append(timestamp)
	# print(mytime[i])

delta_time=[]

for i in range (0,len(data)-1):
	delta=data[i+1]-data[i]
	delta_ms  = delta / datetime.timedelta(milliseconds=1)
	delta_time.append(delta_ms)
	#print(delta_ms)


# plt.hist(data,20,ec='black')
plt.plot(time_s_list,diff_list,"red")
plt.xlabel('Time (s)')
plt.ylabel('Difference')
plt.title("Processing Time per Iteration")
plt.yscale('log')
# plt.xscale('log')
plt.grid('both')
# print(statistics.mean(delta_time))

# # Create a new figure
# plt.figure()
# # Create a list of frames (1-len(data))
# framecount=list(range(1, len(data)))
# plt.plot(framecount,delta_time,'.-')
# plt.ylabel('Frame')
# plt.xlabel('Processing Time (ms)')
# plt.title("Processing Time (No Object in Frame)")

# #plt.legend()
plt.show()






