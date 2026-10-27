import cv2
import mediapipe as mp
import time

mpdraw =  mp.solutions.drawing_utils
mpfacemesh = mp.solutions.face_mesh 

facemesh =  mpfacemesh.FaceMesh(max_num_faces = 2)
drawspec = mpdraw.DrawingSpec(thickness  = 1 , circle_radius = 1)
ptime = 0
cap = cv2.VideoCapture(0)
while True:
    success , img  = cap.read()
    imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    results = facemesh.process(imgRGB)

    if results.multi_face_landmarks:
        for facelandmarks in results.multi_face_landmarks:
            mpdraw.draw_landmarks(img , facelandmarks , mpfacemesh.FACEMESH_TESSELATION , drawspec , drawspec)

            for id,lm in enumerate(facelandmarks.landmark):
                ih , iw , ic = img.shape
                x,y = int(lm.x*iw) , int(lm.y*ih)
                print(id ,x,y)

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(img , f'FPS: {int(fps)}' , (20,70) , cv2.FONT_HERSHEY_PLAIN , 3 , (255,0,134) , 2 )    
    cv2.imshow("image" , img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break