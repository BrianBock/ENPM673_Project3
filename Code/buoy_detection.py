from EM import *
            
buoy_colors = ['orange','green','yellow']
training_channels = {'orange':(1,2),'green':(0,1),'yellow':(1,2)}

Theta = {}
for color in buoy_colors:
    Sigma, mu, pi = readGMM(color)
    Theta[color] = {'Sigma':Sigma,'mu':mu,'pi':pi}

train_percent = .6
K = 3
threshs = determineThesholds(Theta,buoy_colors,K,train_percent,training_channels)

print(threshs)

filename = '../media/detectbuoy.avi'
input_video = cv2.VideoCapture(filename)

start_frame = 20
input_video.set(1,start_frame)

print('Writing to video. Please Wait.')
count = 0

while input_video.isOpened():
    print("Frame "+str(count+start_frame))
    
    ret, frame = input_video.read()

    if not ret:
        break

    h,w = frame.shape[:2]

    segmented_frames = {}
    for color in buoy_colors:
        segmented_frames[color] = np.zeros([h,w,3],np.uint8)

    probs = {}

    for color in buoy_colors:
        Sigma = Theta[color]['Sigma']
        mu = Theta[color]['mu']
        pi = Theta[color]['pi']
        
        ch1 = frame[:,:,training_channels[color][0]].flatten()
        ch2 = frame[:,:,training_channels[color][1]].flatten()

        x = []
        for ch_x,ch_y in zip(ch1,ch2):
            x_i = np.array([ch_x,ch_y])
            x.append(x_i)
    
        x = np.asarray(x)

        K = len(mu)
        p = np.zeros((1,len(x)))
        for k in range(K):
            p += multivariate_normal.pdf(x,mean=mu[k],cov = Sigma[k])*pi[k]

        probs[color] = p.T

    for i in range(len(x)):
        pixel_p = []
        
        for color in buoy_colors:
            pixel_p.append(probs[color][i])
        
        max_ind = pixel_p.index(max(pixel_p))

        row = i//w
        column = i%w

        if max_ind == 0 and pixel_p[max_ind] > threshs['orange']:
            segmented_frames['orange'][row,column] = (14,127,255)
        elif max_ind == 1 and pixel_p[max_ind] > threshs['green']:
            segmented_frames['green'][row,column] = (96,215,30)
        elif max_ind == 2 and pixel_p[max_ind] > threshs['yellow']:
            segmented_frames['yellow'][row,column] = (77,245,255)

    all_colors = np.zeros([h,w,3],np.uint8)

    for color in buoy_colors:
        all_colors = cv2.bitwise_or(all_colors,segmented_frames[color])

    combined_frame = np.concatenate((frame, all_colors), axis=1)
    if count == 0:
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        output_video = 'output.mp4'
        fps_out = 5

        if os.path.exists(output_video):
            os.remove(output_video)
        out_vid = cv2.VideoWriter(output_video, fourcc, fps_out, (combined_frame.shape[1],combined_frame.shape[0]))

    cv2.imshow("Frame",combined_frame)
    # if the user presses 'q' release the video which will exit the loop
    if cv2.waitKey(1) == ord('q'):
        input_video.release()
 
    out_vid.write(combined_frame)

    count += 1


out_vid.release()

