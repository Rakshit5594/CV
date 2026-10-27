import cv2
import os
import time
import handtrackingmodule as htm

wcam , hcam = 640 , 480


cap = cv2.VideoCapture(0)
cap.set(3 , wcam)
cap.set(4 , hcam)
ptime = 0
##############################################################
# code for displaying images over the existing web cam image

# folderpath = "FingerImages"
# mylist = os.listdir(folderpath)
# print(mylist)
# overlaylist = []
# for impath in mylist:
#     image = cv2.imread(f'{folderpath}/{impath}')
#     overlaylist.append(image)
##############################################

detector = htm.handDetector(detectionCon=0.75)
tipids = [4,8,12,16,20]
while True:
    success , img = cap.read()

    # h,w,c = overlaylist[0].shape
    # img[0:h , 0:w] = overlaylist[0]

    img = detector.findHands(img)
    lmlist = detector.findPosition(img , draw=False)
    # print(lmlist)
    
    if len(lmlist)!=0:
        fingers = []
        if lmlist[tipids[0]][1] > lmlist[tipids[0]-1][1]:
                # print("index finger open")
                fingers.append(1)

        else:    
                fingers.append(0)   

        for id in range(1,5):
            if lmlist[tipids[id]][2] < lmlist[tipids[id]-2][2]:
                # print("index finger open")
                fingers.append(1)

            else:    
                fingers.append(0)    

        # print(fingers)
        totalfingers = fingers.count(1)
        # print(totalfingers)
        cv2.rectangle(img , (20,255) , (170 , 425) , (0,255,0) , cv2.FILLED)
        cv2.putText(img , str(totalfingers) , (45,375) , cv2.FONT_HERSHEY_PLAIN , 10 , (255,0,0) , 25)
    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime
    cv2.putText(img , f'FPS: {int(fps)}' , (400,70) , cv2.FONT_HERSHEY_PLAIN , 1 ,(255,0,0) , 3)

    

    
    cv2.imshow("Image" , img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break