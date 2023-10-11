import cv2
import numpy as np
import HandTrackingModule as htm
import autopy
 

Wcam, Hcam = 640, 480
frameR = 100 # Frame Reduction
smoothening = 20

 
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
 
cap = cv2.VideoCapture(0)
cap.set(3, Wcam)
cap.set(4, Hcam)
detector = htm.handDetector(maxHands=1,detectionCon=0.8)
wScr, hScr = autopy.screen.size()
# print(wScr, hScr)
 
while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # print(x1, y1, x2, y2)
    
    # 3. Check which fingers are up
    fingers = detector.fingersUp()
    # print(fingers)
    cv2.rectangle(img, (frameR, frameR), (Wcam - frameR, Hcam - frameR),
    (255, 0, 255), 2)
    # 4. Only Index Finger : Moving Mode
    if fingers[1] == 1 and fingers[2] == 1:
        # 5. Convert Coordinates
        x3 = np.interp(x1, (frameR, Wcam - frameR), (0, wScr))
        y3 = np.interp(y1, (frameR, Hcam - frameR), (0, hScr))
        # 6. Smoothen Values
        clocX = plocX + (x3 - plocX) / smoothening
        clocY = plocY + (y3 - plocY) / smoothening
    
        # 7. Move Mouse
        autopy.mouse.move(wScr - x3, y3)
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        plocX, plocY = clocX, clocY
        
    # 8. Both Index and middle fingers are up : Clicking Mode
    if fingers[1] == 1 and fingers[2] == 1:
        # 9. Find distance between fingers
        length, img, lineInfo = detector.findDistance(8, 12, img)
        print(length)
        # 10. Click mouse if distance short
        if length < 30:
            cv2.circle(img, (lineInfo[4], lineInfo[5]),
            15, (0, 255, 0), cv2.FILLED)
            autopy.mouse.click()
    
  
    # 12. Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)