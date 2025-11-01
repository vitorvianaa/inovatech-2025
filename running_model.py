import cv2
import mediapipe as mp
import os
from dotenv import load_dotenv
from keras.models import load_model
import numpy as np

load_dotenv()
model = load_model('first_model.h5')
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

dataset = []

labels = ['A', 'B', 'C', 'D', 'E']

while True:
    response, frame = cap.read()
    if not response:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    #     for hand_landmarks in result.multi_hand_landmarks:
    #             landmarks_list = []
    #             for lm in hand_landmarks.landmark:
    #                 landmarks_list.extend([lm.x, lm.y, lm.z])
                
    #             X = np.array([landmarks_list])
    #             prediction = model.predict(X)[0]
    #             index_label = np.argmax(prediction)
    #             accuracy = prediction[index_label]
                
    #             if accuracy < 0.7:
    #                 print('nada detectado')
    #             else:
    #                 print(f'Result do model: {labels[index_label]}')
    #             continue            
    # print('nada detectado')
    
    cv2.imshow("MostraMao", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:
        break
    elif key == 13:  # Enter para capturar
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks_list = []
                for lm in hand_landmarks.landmark:
                    landmarks_list.extend([lm.x, lm.y, lm.z])
                
                X = np.array([landmarks_list])
                prediction = model.predict(X)[0]
                print('prediction: ', prediction)
                index_label = np.argmax(prediction)
                accuracy = prediction[index_label]
                print(f'accuracy: {accuracy}')  
                print(f'Result do model: {labels[index_label]}')   

cap.release()
cv2.destroyAllWindows()


