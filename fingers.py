import cv2
import numpy as np
import copy
import math

#Constant values needed for some of the functions used
y_end = 0.85
x_end = 0.45
threshold_value = 10


#This opens the webcam(0) and captures the infront frame
cap = cv2.VideoCapture(0)           
bgsub = cv2.createBackgroundSubtractorMOG2(0,50,detectShadows=False)
kernel_3 = np.ones((3,3),np.uint8)
kernel_2 = np.ones ((2,2),np.uint8)


#This function removes the still background and captures the fore-ground part (basically things in motion)
def removeBG(frame):
    rm = bgsub.apply(frame)
    blur = cv2.bilateralFilter(rm,9,75,75)
    eroded = cv2.erode(blur, kernel_2, iterations=1)
    fg = cv2.morphologyEx(blur,cv2.MORPH_OPEN,kernel_2)
    result = cv2.bitwise_and(frame, frame,mask=fg)
    return result


#The crop function crops the top right part of the whole captured image where a hand is likely to be seen
def crop(image):
    image = image[0:int(y_end * image.shape[0]),int(x_end * image.shape[1]) : image.shape[1]]
    return image


#This function is to select the largest seen contour(hand) among all detected
def largest(contours) :
    contour=np.array([[23,144]])
    max_area = 0
    for cnt in contours :
        area = cv2.contourArea(cnt)
        if area > max_area :
            contour = cnt
    
    return contour


#Mathematical calculation to count the fingers
def countfingers(cnt) :
    count = 0
    hull = cv2.convexHull(cnt,returnPoints=False)
    defects = cv2.convexityDefects(cnt,hull)
    if type(defects) != type(None):
        for i in range(defects.shape[0]):
            start,end,far,_ = defects[i][0]
            start = tuple(cnt[start][0])
            end = tuple(cnt[end][0])
            far = tuple(cnt[far][0])
            
            a = math.sqrt((end[0]-start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((end[0]-far[0])**2 + (end[1] - far[1])**2)
            c = math.sqrt((far[0]-start[0])**2 + (far[1] - start[1])**2)
            angle = math.acos((b**2 + c**2 - a**2) / (2*b*c) )
            if angle <= (1./2.) *math.pi :
                count+=1
    return count


#Main program which captures the frame infinitely for calculation until 'q' is pressed on the keyboard
while True :
    
    ret, frame = cap.read()
    if ret :
        frame = cv2.flip(frame,1)
        result = removeBG(frame)
        work_image = crop(result)
        
        gray = cv2.cvtColor(work_image , cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (45,45), 0)
        ret,threshold = cv2.threshold (gray, threshold_value, 255, cv2.THRESH_BINARY )
        res = cv2.dilate (threshold , kernel_2, iterations=1)
        res_copy = copy.deepcopy(res)
        cv2.imshow('res',res)
        
        image, contours, hierarchy = cv2.findContours(gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE )
        length = len (contours)
        contour = largest(contours)
        #contour = contours[0]
        
        black = np.zeros(res.shape,np.uint8)

        hull = cv2.convexHull(contour)
        cv2.drawContours(black,[contour],0,(255,0,0),3)
        cv2.drawContours(black,[hull],0,(255,0,0),3)
        fingers = countfingers(contour)
        cv2.imshow('black', black)
        cv2.putText(black,"str(fingers)",(int(0.25*gray.shape[0]) , int(0.2*gray.shape[1])),cv2.FONT_HERSHEY_SIMPLEX,4,(255,0,0),3)
        print(fingers)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

