# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2

#Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

#defining a function that will do the detections
#draw rectangle to detect faces

#we need grayscale version of image and also the original because
#returned image will not be grayscale
#this function works only on single image one by one
def detect(gray,frame):
    #---so we need the co-ordinates of the rectangle that will detect the face
    #---x,y,w,h (x and y are coordinates of upper left corner, w is width of rectangle)
    #(h is height of rectangle)
    #---detectMultiScale takes the image in b&w, ScaleFactor how much image will be reduced, 1.3 times
    #or we can say how much size of filter will be increased
    #---minNeighbours define how many neighbours need to be accepted for a zone of pixel to be accepted
    #1.3 and 5 is good combo to detect faces in webcam
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    
    #--we will create a for loop that will iterate through faces and for each of the faces detected
    #--we will create a rectangle and in that we will detect eyes
    for (x,y,w,h) in faces:
        #(x,y)-> upperleft corner
        #(x+w,y+h)->lowerright corner
        #---rectangle(original image, co-ord of upperleft,co-ord of upperright,color,thickness)
        cv2.rectangle(frame,(x,y),(x+w,y+h), (255,0,0), 2)
        #so now we will detect the eyes inside the face frame and not the whole image
        #to save computation time
        #*--*Note we need specify the area both for grayscale and colored image
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        #so eyes now contain the info of rectangle's x,y,w,h
        eyes = eye_cascade.detectMultiScale(roi_gray,1.2,15)
        smile = smile_cascade.detectMultiScale(roi_gray,1.9,30)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(110,255,0),2)
        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(255,255,0),2)
            cv2.putText(frame,"Smile",(sx+150,sy+120),cv2.FONT_HERSHEY_TRIPLEX,1,(0,255,255),2)
    
    return frame

#Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0) #0 for internal webcam #1 for external webcam
while(1):
    #we need to get the last frame coming from video_capture 
    #and video_capture.read returns two arguments and 2nd argument is last frame we need
    _, frame = video_capture.read()
    #so we need the gray version
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas = detect(gray,frame)
    cv2.imshow('Video',canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_capture.release() #to turn off webcam
cv2.destroyAllWindows()
        
    