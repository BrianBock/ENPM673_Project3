import cv2 
import numpy as np
import os


if __name__ == '__main__':
    filename = '../media/detectbuoy.avi'
    video = cv2.VideoCapture(filename)
    tot_frames = int(video.get(7))
    count = 0
    colors = ['yellow','orange','green']
    bgr = {'yellow':(52,232,235),'orange':(52,137,235),'green':(52,235,86)}
    for color in colors:
        path = '../Training/'+color
        print(path)
        if os.path.exists(path):

            for img in os.listdir(path):
                if img.endswith('.png'):
                    os.remove(path+'/'+img)
        else:
            os.mkdir(path)


    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    while video.isOpened():
        ret, frame = video.read()
        
        if ret:
            h,w,ch = frame.shape
            frames = {}
            frame_w_text = frame.copy() 
            text = 'Frame ' + str(count+1) + ' of ' + str(tot_frames)
            cv2.putText(frame_w_text,text,(w-200,h-10),font,1,(0,0,0),2,cv2.LINE_AA)
            cv2.putText(frame_w_text,text,(w-200,h-10),font,1,(255,255,255),1,cv2.LINE_AA)
            bouy_count = 0

            for color in colors:
                color_frame = frame_w_text.copy()
                cv2.rectangle(color_frame,(0,h-40),(40,h),bgr[color],-1)
                frames[color]=color_frame

            for color in colors:
                clicked = False
                roi = []
                text1 = 'Draw a rectangle arround the ' + color + ' bouy'
                text2 = 'If the bouy is not present press ESC'
                
                cv2.putText(frames[color],text1,(10,20),font,1,(0,0,0),2,cv2.LINE_AA)
                cv2.putText(frames[color],text1,(10,20),font,1,(255,255,255),1,cv2.LINE_AA)
                cv2.putText(frames[color],text2,(10,40),font,1,(0,0,0),2,cv2.LINE_AA)
                cv2.putText(frames[color],text2,(10,40),font,1,(255,255,255),1,cv2.LINE_AA)
                

                cv2.imshow('Frame',frames[color])
                (x,y,w,h) = cv2.selectROI("Frame", frames[color], fromCenter=False,showCrosshair=False)

                if w == 0 or h == 0:
                    continue

                else:
                    bouy_count += 1
                    bouy = frame[y:y+h,x:x+w]
                    mask = np.zeros(bouy.shape[:2],np.uint8)
                    cv2.ellipse(mask,(w//2,h//2),(w//2,h//2),0,0,360,255,-1)
                    bouy = cv2.bitwise_and(bouy,bouy,mask = mask)
                    # cv2.imshow('Bouy',bouy) 
                    # cv2.waitkey(0)
                    # cv2.destroyWindow('Bouy')
                    path = '../Training/'+color+'/'+str(count)+'.png'
                    cv2.imwrite(path,bouy)
                    
            if bouy_count == 0:
                video.release()
                cv2.destroyAllWindows()

            count +=1
        else:
            video.release()
