import cv2
import mediapipe as mp
import time


class FaceMeshdetector():
    def __init__(self , staticMode = False , maxFaces = 2 , minDetectioncon = 0.5 , minTrackingcon = 0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectioncon = minDetectioncon
        self.minTrackingcon = minTrackingcon
        self.mpdraw =  mp.solutions.drawing_utils
        self.mpfacemesh = mp.solutions.face_mesh 
        self.facemesh =  self.mpfacemesh.FaceMesh(max_num_faces = 2)
        self.drawspec = self.mpdraw.DrawingSpec(thickness  = 1 , circle_radius = 1)

    def findFaceMesh(self  , img , draw = True):
            self.imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
            self.results = self.facemesh.process(self.imgRGB)
            faces = []
            if self.results.multi_face_landmarks:
                
                for facelandmarks in self.results.multi_face_landmarks:
                        if draw:
                            self.mpdraw.draw_landmarks(img , facelandmarks , self.mpfacemesh.FACEMESH_TESSELATION , self.drawspec , self.drawspec)
                        face = []
                        for id,lm in enumerate(facelandmarks.landmark):
                            ih , iw , ic = img.shape
                            x,y = int(lm.x*iw) , int(lm.y*ih)
                            #print(id ,x,y)
                            face.append([x,y])
                        faces.append(face)    
            return img , faces

   
def main():
    ptime = 0
    cap = cv2.VideoCapture(0)
    detector = FaceMeshdetector()
    while True:
        success , img  = cap.read()
        img , faces= detector.findFaceMesh(img)
        if len(faces)!= 0:
             print(faces[0])
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime   
        cv2.putText(img , f'FPS: {int(fps)}' , (20,70) , cv2.FONT_HERSHEY_PLAIN , 3 , (255,0,134) , 2 )    
        cv2.imshow("image" , img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
