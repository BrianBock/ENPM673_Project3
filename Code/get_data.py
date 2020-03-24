import cv2 
import numpy 

def click_event(event, x, y, flags, param):
    global clicked

    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        clicked = True

if __name__ == '__main__':
    filename = '../media/detectbuoy.avi'
    video = cv2.VideoCapture(filename)
    tot_frames = int(video.get(7))
    count = 0
    colors = ['yellow','orange','green']
    font = cv2.FONT_HERSHEY_SIMPLEX 
    while video.isOpened():
        ret, frame = video.read()
        h,w,ch = frame.shape
        frames = {}
        for color in colors:
            frames[color] = frame.copy()
        if ret:
            for color in colors:
                clicked = False
                text1 = 'Click on the ' + color + ' bouy'
                text2 = 'If the bouy is not present press ESC'
                text3 = 'Frame ' + str(count) + ' of ' + str(tot_frames)
                cv2.putText(frames[color],text1,(10,20),font,.6,(255,255,255),1,cv2.LINE_AA)
                cv2.putText(frames[color],text2,(10,40),font,.6,(255,255,255),1,cv2.LINE_AA)
                cv2.putText(frames[color],text3,(10,h-10),font,.6,(255,255,255),1,cv2.LINE_AA)
                cv2.imshow('Frame',frames[color])
                cv2.setMouseCallback("Frame", click_event)
                
                while clicked == False:
                    key = cv2.waitKey(1)
                    if key == 27:
                        print("ESC was pressed")
                        break

                    elif key == ord('q'):
                        video.release()
                        break
                if key == ord('q'):
                    break

            count +=1
        else:
            video.release()
