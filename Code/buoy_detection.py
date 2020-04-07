from EM import *
import imutils

def addContours(image,segmented_frames,buoy_colors):
    blank_image=np.zeros((image.shape[0],image.shape[1],3),np.uint8)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, bin_image = cv2.threshold(grey, 1, 255, cv2.THRESH_BINARY) 
    # cv2.imshow("bin",bin_image)
    # cv2.waitKey(0)

    blurred_image=cv2.GaussianBlur(bin_image,(5,5),0)
    # Find the contours of the threshed image
    cnts=cv2.findContours(blurred_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts)>0:
        # Draw the contours on the image
        # sort the contours to include only the three largest
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:8]
        # contour_image=cv2.drawContours(image, cnts, -1, (0,0,255), 2)
        # cv2.imshow("bin",contour_image)
        # cv2.waitKey(0)
        # exit()

        diff=[]

       # Color (B,G,R)
        # color = (255, 255, 255) 
        # Line thickness of -1 = filled in 
        thickness = -1
        for contour in cnts:


            # c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            # print(radius)
            A=math.pi*radius**2
            # print(A)
            A_contour=cv2.contourArea(contour)
            diff.append(abs(A-A_contour)/A)

        # print(diff)
        thresh=.4
        circle_inds=[]
        for i,val in enumerate(diff):
            if val<thresh:
                circle_inds.append(i)

        for ind in circle_inds:
            buoy_contour=cnts[ind]
            ((x, y), radius) = cv2.minEnclosingCircle(buoy_contour)
            A=math.pi*radius**2
            print(A)
            count=[]
            if A>500:
                for color in buoy_colors:
                    buoy_region=segmented_frames[color][int(y-radius):int(y+radius),int(x-radius):int(x+radius),:]
                    grey = cv2.cvtColor(buoy_region, cv2.COLOR_BGR2GRAY)
                    count.append(len(np.nonzero(grey)[0]))
                print(count)
                max_ind = count.index(max(count))

                bgr_colors={'orange':(14,127,255),'green':(96,215,30),'yellow':(77,245,255)}

                ring_color=bgr_colors[buoy_colors[max_ind]]

                image = cv2.circle(blank_image, (int(x),int(y)), int(radius), ring_color, thickness)

    return image







            
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

start_frame = 0
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

    contour_frame=addContours(all_colors,segmented_frames,buoy_colors)
    # cv2.imshow("Image",all_colors)
    # cv2.waitKey(0)


    combined_frame = np.concatenate((frame, contour_frame), axis=1)
    

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

