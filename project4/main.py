import time
import handtrackingmodule as htm
import numpy as np
import autopy
import cv2

ptime  =  0
wcam , hcam = 640 , 480
wsrc , hsrc = autopy.screen.size()
framer = 100
smoothenign = 5

plocx , plocy = 0,0
clocx , clocy = 0,0
cap = cv2.VideoCapture(0)
cap.set(3 , wcam)
cap.set(4 , hcam)

detector = htm.handDetector(maxHands=1)

while True:

    success , img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img)

    if len(lmlist)!=0:
        x1,y1 = lmlist[8][1:]
        x2,y2 = lmlist[12][1:]
        fingers = detector.handup()
        cv2.rectangle(img , (framer , framer) , (wcam-framer , hcam-framer) , (255,0,255) , 2)

        if fingers[1]==1 and fingers[2]==0:
            x3 = np.interp(x1 , (framer,wcam-framer) , (0,wsrc))    
            y3 = np.interp(y1 , (framer,hcam-framer) , (0,hsrc))    
            clocx = plocx+(x3-plocx)/smoothenign
            clocy = plocy+(y3-plocy)/smoothenign
            autopy.mouse.move(wsrc-clocx,clocy)
            cv2.circle(img , (x1,y1) , 15 , (255,0,255) , cv2.FILLED)
            plocx,plocy = clocx,clocy

        if fingers[1] and fingers[2]:
            length , img , lineinfo =  detector.finddistance(8,12,img)   
            # print(length)
            if length<25:
                cv2.circle(img , (lineinfo[4] , lineinfo[5]) , 15 , (0,255,0) , cv2.FILLED)
                autopy.mouse.click()
                
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(img , str(int(fps)) , (20 ,50) , cv2.FONT_HERSHEY_PLAIN , 3 , (255,0,0) , 3)
    cv2.imshow("Image" , img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break