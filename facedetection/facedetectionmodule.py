import cv2
import mediapipe as mp
import time


class FaceDetector():
    def __init__(self , mindetectioncon = 0.5):
        self.mindetectioncon = mindetectioncon
        self.mpfacedetection = mp.solutions.face_detection
        self.mpdraw = mp.solutions.drawing_utils
        self.facedetection = self.mpfacedetection.FaceDetection(mindetectioncon)

    def findFaces(self , img , draw = True):


        imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        self.results = self.facedetection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id , detection in enumerate(self.results.detections):
               
                bboxc = detection.location_data.relative_bounding_box
                ih,iw,ic = img.shape
                bbox = int(bboxc.xmin*iw) , int(bboxc.ymin*ih) , int(bboxc.width*iw) , int(bboxc.height*ih)
                bboxs.append([id , bbox , detection.score])
                cv2.rectangle(img , bbox , (255,0,255) , 2)
                cv2.putText(img , f'{int(detection.score[0]*100)}%' , (bbox[0],bbox[1]-20) , cv2.FONT_HERSHEY_PLAIN , 2 , (255,0,255) , 2 )

        return img , bboxs
    
def main():
    cap = cv2.VideoCapture(0)
    ptime = 0   
    detector = FaceDetector()
    while True:
        success , img = cap.read()
        img,bboxs = detector.findFaces(img)    
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime=ctime
        cv2.putText(img , f'FPS: {int(fps)}' , (20,70) , cv2.FONT_HERSHEY_PLAIN , 3 , (255,0,134) , 2 )
        cv2.imshow("image" , img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
if __name__ == '__main__':
    main()