import cv2
import mediapipe as mp
import math

class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            model_complexity=1,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

        self.tipids = [4,8,12,16,20]

        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        if img is None:
            return img  # return unprocessed if frame is empty

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []

        if self.results and self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]

                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return self.lmList
    
    def handup(self):
        fingers = []
        if self.lmList[self.tipids[0]][1] < self.lmList[self.tipids[0]-1][1]:
                # print("index finger open")
                fingers.append(1)

        else:    
                fingers.append(0)   

        for id in range(1,5):
            if self.lmList[self.tipids[id]][2] < self.lmList[self.tipids[id]-2][2]:
                # print("index finger open")
                fingers.append(1)

            else:    
                fingers.append(0)  


        return fingers

    def finddistance(self , p1,p2,img , draw = True , r = 15 , t=3):
        x1,y1 = self.lmList[p1][1:]        
        x2,y2 = self.lmList[p2][1:]        
        cx , cy = (x1+x2)//2 , (y1+y2)//2

        if draw:
            cv2.line(img , (x1,y1) , (x2,y2) , (255,0,255))
            cv2.circle(img , (x1,y1) , r , (255,0,255) , cv2.FILLED)      
            cv2.circle(img , (x2,y2) , r , (255,0,255) , cv2.FILLED)      
            cv2.circle(img , (cx,cy) , r , (0,0,255) , cv2.FILLED)      
        length = math.hypot(x2-x1 , y2-y1)

        return length , img , [x1 , y1 , x2 , y2 , cx , cy]
    
    