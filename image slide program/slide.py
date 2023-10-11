import cv2
import os
from cvzone.HandTrackingModule import HandDetector

#variable for camera 
width,height=720,480
folderPath = "pImages"

#calling path 
pathImages = sorted(os.listdir(folderPath),key=len)
#print(pathImages)

#variable
imgNumber = 0
hs,ws = 200, 250
buttonPressed =False
bcounter=0
Bdelay =30

#hand detector 
detector = HandDetector(detectionCon=0.8, maxHands=1)


#camera acces line
cap = cv2.VideoCapture(0)
cap.set(3,width)
cap.set(4,height)



while True:
    #importing the presenting file
    success, img =cap.read()
    img= cv2.flip(img,1)
    pathFullImage=os.path.join(folderPath,pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)
    
    hands, img= detector.findHands(img)  
 
    if hands and buttonPressed is False:
       hand = hands[0]
       fingers =detector.fingersUp(hand)
       #print(fingers)

       #left moving gesture 
       if fingers == [1,0,0,0,0]:
         print("left")
         buttonPressed = True
         if imgNumber>0:
          imgNumber -=1

        #right moving gesture
       if fingers == [0,0,0,0,1]:
          print("right")
          buttonPressed=True
          if imgNumber<len(pathImages)-1:
           imgNumber +=1
       
       
 #button pressed changes
    if buttonPressed:
     bcounter +=1
     if bcounter> Bdelay:
       bcounter=0
       buttonPressed =False 

       

      
    
       
       

    #adding camera to slide
    imgSmall =cv2.resize(img,(ws,hs))
    h,w,_=imgCurrent.shape
    imgCurrent[0:hs,w-ws:w]=imgSmall

    # use only in demo ,cv2.imshow("Image",img)
    cv2.imshow("Slides",imgCurrent)


    key =cv2.waitKey(1)
    if key== ord('`'):
     break

