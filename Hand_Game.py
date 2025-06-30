import cv2
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp 
import numpy as np
from cvzone import putTextRect
import random
import time


# Webcam

cap = cv2.VideoCapture(0)
cap.set (3,1280)
cap.set (4,720)


# HAnd detector 

detector = HandDetector(detectionCon=0.8 , maxHands=1)

x = [300,245,200,170,145,130,112,103,93,87,80,75,70,67,62,59,57]
y = [20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
#Ax**2+Bx+c
coff = np.polyfit(x,y ,2)

cx , cy = 250 , 250 
color = (255,0,255)
color_black  = (0,0,0)
color_white  = (255,255,255)
counter      = 0
score = 0
timestart = time.time()
totaltime = 31
# LOOP
while True :
    success , img = cap.read()
    img = cv2.flip(img , 1)
    hands ,img = detector.findHands(img  , draw = False)

    if time.time()-timestart < totaltime :
    
        if hands :
            lmList  = hands[0]['lmList']
            x,y,h,w = hands[0]['bbox']  
            x1 , y1 = lmList[5][:2]
            x2 , y2 = lmList[17][:2]
    
            distance = ((y2-y1)**2 + (x2-x1)**2)**0.5
            A , B , C = coff
            distanceCM = A*distance**2 + B*distance + C
    
    
            if distanceCM < 40 : 
                if  x < cx < x + w and y < cy < y + h :
                    counter = 1
            cv2.rectangle( img , (x,y) , (x + w  , y + h ), (255 , 0 ,255) ,3)
            putTextRect(img,f'{int(distanceCM)} cm' , (x + 5,y - 10) )
                    
        if counter :
            counter += 1 
            color  = (0 , 255 , 0)
            if counter == 3 :
                cx , cy  = random.randint(100 , 1100)  , random.randint(100 , 600)
                color = (255 , 0 , 255)
                score +=1
                counter = 0 
        
    
    
        #draw Buttons
        cv2.circle(img , (cx ,cy) , 30  , color, cv2.FILLED )
        cv2.circle(img , (cx ,cy) , 10  , color_white , cv2.FILLED )
        cv2.circle(img , (cx ,cy) , 20  , color_white , 2          )
        cv2.circle(img , (cx ,cy) , 30  , color_black , 2          )
    
    
        # Game HUD
    
        putTextRect(img , f'TIME : {int(totaltime -( time.time()-timestart))} ', (1000 , 75) ,  scale = 3 )
        putTextRect(img , f'SCORE : {str(score).zfill(2)} ', (60 , 75) ,  scale = 3 )
    else :

         putTextRect(img , 'Game Over', (400,400) , scale = 5 , offset=5 , thickness=7)
         putTextRect(img , f'Your Score : {score}', (450,500) , scale = 3 , offset=20 )
         putTextRect(img , 'Press R to Restart', (400,625) , scale = 3 , offset=20 )
         putTextRect(img , 'Press Q to Quit', (450,700) , scale = 3 , offset=20 )












    
    cv2.imshow('Image' , img)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('r'):
        timestart = time.time()
        score = 0
    elif key == ord('q') : 
        break
cap.release()
cv2.destroyAllWindows()