import cv2
import mediapipe as mp
import time

mpdraw = mp.solutions.drawing_utils
mppose = mp.solutions.pose
pose = mppose.Pose()
cap = cv2.VideoCapture(0)
ptime = 0
while True:
    success , img  = cap.read()
    imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mpdraw.draw_landmarks(img , results.pose_landmarks , mppose.POSE_CONNECTIONS)
        for id , lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            print(id,lm)
            cx,cy = int(lm.x*w) , int(lm.y*h )
            cv2.circle(img , (cx,cy) , 10 , (255,0,255) , cv2.FILLED)
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime  =  ctime
    cv2.putText(img , str(int(fps)) , (70,50) , cv2.FONT_HERSHEY_PLAIN , 3 , (255 , 0 ,0) , 3)

    cv2.imshow("Image" ,   img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break