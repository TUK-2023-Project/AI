#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pillow


# In[1]:


import cv2
import mediapipe as mp
import numpy as np
import time, os
from PIL import ImageFont, ImageDraw, Image


# In[ ]:


max_num_hands=1
gesture={
    0:'a',1:'b',2:'c',3:'d',4:'e',5:'f',6:'g',7:'h',
    8:'i',9:'j',10:'k',11:'l',12:'m',13:'n'
}
rps_gesture={
    0:'ㄱ',1:'ㄴ',2:'ㄷ',3:'ㄹ',4:'ㅁ',5:'ㅂ',6:'ㅅ',7:'ㅇ',
    8:'ㅈ',9:'ㅊ',10:'ㅋ',11:'ㅌ',12:'ㅍ',13:'ㅎ', 25:'done',26:'spacing',27:'clear'
}
# rps_gesture={
#     0:'a',1:'b',2:'c',3:'d',4:'e',5:'f',6:'g',7:'h',
#     8:'i',9:'j',10:'k',11:'l',12:'m',13:'n', 25:'done',26:'spacing',27:'clear'
# }

#MediaPipe hands model
mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils
hands=mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

#Gesture recognition model
file = np.genfromtxt('dataSet.txt', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(0)

#한국어 인식함수
def myPutText(src,text,pos,font_size,font_color):
    img_pil=Image.fromarray(src)
    draw=ImageDraw.Draw(img_pil)
    font=ImageFont.truetype('한국기계연구원_bold.ttf',font_size)
    draw.text(pos,text,font=font,fill=font_color)
    return np.array(img_pil)
    
img=np.zeros((480,640,3),dtype=np.uint8)
FONT_SIZE=30
COLOR=(255,255,255) #흰색

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 점과 점 사이의 뺄셈 연산을 통해 많은 벡터 만들어냄
    if result.multi_hand_landmarks:
        for res in result.multi_hand_landmarks: #여러 개의 손을 인식할 수 있으므로, for문 반복
            joint = np.zeros((21, 3)) 
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])

            # Draw gesture result
            if idx in rps_gesture.keys():
                img=myPutText(img,rps_gesture[idx],(20, 50),FONT_SIZE,(255,255,255))
#                 cv2.putText(img, text=rps_gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,font=font, fontScale=5, color=(255, 255, 255), thickness=2)
                

            # Other gestures
            # cv2.putText(img, text=gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('text', img)
    if cv2.waitKey(1) == ord('q'):
        break


# In[ ]:





# In[ ]:




