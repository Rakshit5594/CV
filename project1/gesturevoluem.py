import cv2
import time
import numpy as np
import math
import handtrackingmodule as htm
from pycaw.pycaw import AudioUtilities , IAudioEndpointVolume
from ctypes import cast , POINTER
from comtypes import CLSCTX_ALL

wcam , hcam = 640 , 480
cap = cv2.VideoCapture(0)
cap.set(3,wcam)
cap.set(4,hcam)
ptime = 0

device = AudioUtilities.GetSpeakers()
interface = device.Activate(IAudioEndpointVolume._iid_ , CLSCTX_ALL , None)
volume = cast(interface , POINTER(IAudioEndpointVolume))
volrange = volume.GetVolumeRange()

minvol = volrange[0]
maxvol = volrange[1]


vol = 0
volbar = 400
detector = htm.handDetector(detectionCon=0.75)
while True:
    success , img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img , draw=False)
    if len(lmlist)!=0:
        # print(lmlist[4] , lmlist[8])

        x,y = lmlist[4][1] , lmlist[4][2]
        x2,y2 = lmlist[8][1] , lmlist[8][2]
        cx,cy = (x+x2)//2 , (y+y2)//2
        length = math.hypot(x2-x , y2-y)
        # print(length)

        cv2.circle(img , (x,y) , 15 , (255,0,0) , cv2.FILLED)
        cv2.circle(img , (x2,y2) , 15 , (255,0,0) , cv2.FILLED)
        cv2.circle(img , (cx,cy) , 15 , (255,0,0) , cv2.FILLED)
        cv2.line(img , (x,y) , (x2,y2) , (255,0,232) , 3)

        # hand range min: 15 , max: 160
        # vol range = -65 , 0
        vol = np.interp(length , [15,160] , [minvol , maxvol])
        volbar = np.interp(length , [50,300] , [400 , 150])
        volume.SetMasterVolumeLevel(vol, None)

        if length<20:
            cv2.circle(img , (cx,cy) , 15 , (255,0,255) , cv2.FILLED)

    cv2.rectangle(img , (50,150) , (85,400) , (0,255,0) , 3)        
    cv2.rectangle(img , (50,int(volbar)) , (85,400) , (0,255,0) , cv2.FILLED)        

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(img , f'FPS: {int(fps)}' , (30,50) , cv2.FONT_HERSHEY_COMPLEX , 1 ,(255,0,0) , 3)
    cv2.imshow("Image" , img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break