from EM import *

def readGMM(color):
    path = 'EMoutput' + color + '.npz'
    if os.path.exists(path):
        print("File exists")
        GMM_file=open(path,"r")
        
        with np.load(path) as data:
            Sigma = data['Sigma']
            mu = data['mu']
            pi = data['pi']

    else:
        print("Cannot read '"+path+"' ")
        exit()

    return Sigma, mu, pi



            
buoy_colors = ['orange','green','yellow']

Theta = {}
for color in buoy_colors:
    Sigma, mu, pi = readGMM(color)
    Theta[color] = {'Sigma':Sigma,'mu':mu,'pi':pi}

# print(Theta)
filename = '../media/detectbuoy.avi'
input_video = cv2.VideoCapture(filename)

input_video.set(1,20)


for num in range(25):
    print("Frame "+str(num))
    ret, frame = input_video.read()

    h,w = frame.shape[:2]

    new_frame=np.zeros([h,w,3],np.uint8)

    g = frame[:,:,1].flatten()
    r = frame[:,:,2].flatten()

    x = []
    for g_ch,r_ch in zip(g,r):
        x_i = np.array([g_ch,r_ch])
        x.append(x_i)
    
    x = np.asarray(x)

    probs = {}
    for color in buoy_colors:
        # print(color)
        Sigma = Theta[color]['Sigma']
        mu = Theta[color]['mu']
        pi = Theta[color]['pi']
        
        K = len(mu)
        p = np.zeros((1,len(x)))
        for k in range(K):
            p += multivariate_normal.pdf(x,mean=mu[k],cov = Sigma[k])*pi[k]

        probs[color] = p.T

    for i in range(len(x)):
        pixel_p = []
        for color in buoy_colors:
            pixel_p.append(probs[color][i])
        
        # if any(val > 1*10**-4 for val in pixel_p):
        max_ind = pixel_p.index(max(pixel_p))

        row = i//w
        column = i%w

        # if max_ind == 0:
        #     frame[row,column] = (14,127,255)
        # elif max_ind == 1:
        #     frame[row,column] = (118,183,56)
        # elif max_ind == 2:
        #     frame[row,column] = (132,241,238)

        if max_ind == 0 and pixel_p[max_ind] > 1*10**-4:
            new_frame[row,column] = (14,127,255)
        elif max_ind == 1 and pixel_p[max_ind] > 2*10**-4:
            new_frame[row,column] = (96,215,30)
        elif max_ind == 2 and pixel_p[max_ind] > 1*10**-4:
            new_frame[row,column] = (77,245,255)

        # print(max_ind)
    combined_frame = np.concatenate((frame, new_frame), axis=1)
    if num==0:
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        output_video = 'output.mp4'
        fps_out = 10

        if os.path.exists(output_video):
            os.remove(output_video)
        out_vid = cv2.VideoWriter(output_video, fourcc, fps_out, (combined_frame.shape[1],combined_frame.shape[0]))
    # cv2.imshow('Frame',numpy_horizontal_concat)
    # cv2.waitKey(0)
    # # if the user presses 'q' release the video which will exit the loop
    # if cv2.waitKey(1) == ord('q'):
    #     video.release()

    # print('Writing to video. Please Wait.')
    # cv2.imshow("Frame",combined_frame)
    # cv2.waitKey(0)
    out_vid.write(combined_frame)
out_vid.release()

