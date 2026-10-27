import cv2
import mediapipe as mp
import time
cap = cv2.VideoCapture(0)
ptime = 0

mpfacedetection = mp.solutions.face_detection
mpdraw = mp.solutions.drawing_utils
facedetection = mpfacedetection.FaceDetection(0.75)

while True:
    success , img = cap.read()

    imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    results = facedetection.process(imgRGB)

    if results.detections:
        for id , detection in enumerate(results.detections):
            # print(id , detection)
            # mpdraw.draw_detection(img , detection)
            # print(detection.location_data.relative_bounding_box)
            bboxc = detection.location_data.relative_bounding_box
            ih,iw,ic = img.shape
            bbox = int(bboxc.xmin*iw) , int(bboxc.ymin*ih) , int(bboxc.width*iw) , int(bboxc.height*ih)
            cv2.rectangle(img , bbox , (255,0,255) , 2)
            cv2.putText(img , f'{int(detection.score[0]*100)}%' , (bbox[0],bbox[1]-20) , cv2.FONT_HERSHEY_PLAIN , 2 , (255,0,255) , 2 )

    
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime=ctime
    cv2.putText(img , f'FPS: {int(fps)}' , (20,70) , cv2.FONT_HERSHEY_PLAIN , 3 , (255,0,134) , 2 )
    cv2.imshow("image" , img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

