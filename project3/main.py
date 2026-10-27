import handtrackingmodule as htm
import os
import cv2
import time
import numpy as np


brushthickness  = 15


folderPath = "Header"
mylist = os.listdir(folderPath)
print(mylist)
overlaylist = []

for impath in mylist:
    iamge = cv2.imread(f'{folderPath}/{impath}')
    overlaylist.append(iamge)

header = overlaylist[0]
drawcolor =  (0 , 0 ,255)

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector = htm.handDetector(detectionCon=0.75)
xp , yp = 0 , 0
ptime = 0
imgcanvas = np.zeros((720 , 1280 , 3) , np.uint8)
while True:
    success , img = cap.read()
    img = cv2.flip(img , 1)

    img = detector.findHands(img)
    lmlsit = detector.findPosition(img , draw=False)

    if len(lmlsit)!=0:
        x1,y1  = lmlsit[8][1:]
        x2,y2  = lmlsit[12][1:]
        fingers = detector.handup()
        # print(fingers)

        if fingers[1] and fingers[2]:
            # print("selection mode")
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlaylist[0]
                    drawcolor = (0,0,255)
                elif 500 < x1 < 750:
                    header = overlaylist[1]
                    drawcolor = (255,0,0)
                elif 800 < x1 < 950:
                    header = overlaylist[2]
                    drawcolor = (0,255,0)
                elif 1050 < x1 < 1200:
                    header = overlaylist[3]
                    drawcolor = (0,0,0)
                    brushthickness = 30
            cv2.rectangle(img , (x1,y1-25)  , (x2,y2+25) , drawcolor , cv2.FILLED)



        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1,y1) , 15 , drawcolor , cv2.FILLED)
            # print("drawing mode")

            if xp == 0 and yp == 0:
                xp , yp = x1,y1
            cv2.line(img , (xp , yp) , (x1,y1) , drawcolor , brushthickness)
            cv2.line(imgcanvas , (xp , yp) , (x1,y1) , drawcolor , brushthickness)

            xp , yp = x1,y1


    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime        
    cv2.putText(img , str(int(fps)) , (20,100)  , cv2.FONT_HERSHEY_PLAIN, 4 , (255,0,255) , 2)
    img[0:125 , 0:1280] = header
    img = cv2.addWeighted(img , 0.5 , imgcanvas , 0.5 ,0)
    cv2.imshow("Image" , img)
    cv2.imshow("Canvas" , imgcanvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break