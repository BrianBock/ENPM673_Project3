import matplotlib.pyplot as plt
from EM import *

# fig,((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,figsize= (12,8))
buoy_colors = ['yellow','orange','green']
colorspace = 'BGR' #HSV or BGR
# fig,((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,figsize= (12,8))

# fig1, ax1=plt.subplots()
# fig2, ax2=plt.subplots()
# fig3, ax3=plt.subplots()
fig4, ax4=plt.subplots()
fig5, ax5=plt.subplots()
fig6, ax6=plt.subplots()

for color in buoy_colors: 
    path = 'Training Data/' + color
    data = generate_dataset(path,colorspace)
    b = data[0,:]
    g = data[1,:]
    r = data[2,:]

    # ax1.scatter(b,r,edgecolors = color,facecolors='none')
    # ax2.scatter(g,r,edgecolors = color,facecolors='none')
    # ax3.scatter(g,b,edgecolors = color,facecolors='none')

    ax4.hist(b,color=color,density=True,alpha=.5,ec=color)
    ax5.hist(g,color=color,density=True,alpha=.5,ec=color)
    ax6.hist(r,color=color,density=True,alpha=.5,ec=color)


# plt.sca(ax1)
# plt.title('Image Data in BR Channels')
# plt.xlabel("Blue Channel")
# plt.ylabel("Red Channel")

# plt.sca(ax2)
# plt.title('Image Data in GR Channels')
# plt.xlabel("Green Channel")
# plt.ylabel("Red Channel")

# plt.sca(ax3)
# plt.title('Image Data in GB Channels')
# plt.xlabel("Green Channel")
# plt.ylabel("Blue Channel")

plt.sca(ax4)
plt.title('B Channel')
plt.xlabel('Intensity')
plt.ylabel('Normalized pixels')

plt.sca(ax5)
plt.title('G Channel')
plt.xlabel('Intensity')
plt.ylabel('Normalized pixels')

plt.sca(ax6)
plt.title('R Channel')
plt.xlabel('Intensity')
plt.ylabel('Normalized pixels')

plt.show()








# import matplotlib.pyplot as plt
# from EM import *

# # fig,((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,figsize= (12,8))
# buoy_colors = ['yellow','orange','green']
# colorspace = 'HSV' #HSV or BGR
# # fig,((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,figsize= (12,8))

# fig1, ax1=plt.subplots()
# fig2, ax2=plt.subplots()
# fig3, ax3=plt.subplots()
# # fig4, ax4=plt.subplots()
# # fig5, ax5=plt.subplots()
# # fig6, ax6=plt.subplots()

# for color in buoy_colors: 
#     path = 'Training Data/' + color
#     data = generate_dataset(path,colorspace)
#     b = data[0,:]
#     g = data[1,:]
#     r = data[2,:]

#     ax1.scatter(b,r,edgecolors = color,facecolors='none')
#     ax2.scatter(g,r,edgecolors = color,facecolors='none')
#     ax3.scatter(g,b,edgecolors = color,facecolors='none')

#     # ax4.hist(b,color=color,alpha=.5,ec=color)
#     # ax5.hist(g,color=color,alpha=.5,ec=color)
#     # ax6.hist(r,color=color,alpha=.5,ec=color)


# plt.sca(ax1)
# plt.title('Image Data in HV Channels')
# plt.xlabel("Hue Channel")
# plt.ylabel("Value Channel")

# plt.sca(ax2)
# plt.title('Image Data in SV Channels')
# plt.xlabel("Saturation Channel")
# plt.ylabel("Value Channel")

# plt.sca(ax3)
# plt.title('Image Data in SH Channels')
# plt.xlabel("Saturation Channel")
# plt.ylabel("Hue Channel")

# # plt.sca(ax4)
# # plt.title('B Channel')
# # plt.xlabel('Intensity')
# # plt.ylabel('Num pixels')

# # plt.sca(ax5)
# # plt.title('G Channel')
# # plt.xlabel('Intensity')
# # plt.ylabel('Num pixels')

# # plt.sca(ax6)
# # plt.title('R Channel')
# # plt.xlabel('Intensity')
# # plt.ylabel('Num pixels')

# plt.show()
