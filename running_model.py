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

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'U', 'V', 'W', 'X', 'Y', 'Z']

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


                # Pega o top 3
                top_indices = np.argsort(prediction)[::-1][:3]
                top_values = prediction[top_indices]
                top_labels = [labels[i] for i in top_indices]

                # Mostra a diferença entre as mais prováveis
                print("Top 3 predições:")
                for lbl, val in zip(top_labels, top_values):
                    print(f"{lbl}: {val*100:.2f}%")
                
                # filtrando por trashold
                threshold = 0.8
                margem = 1e-8
                indices = np.where(prediction + margem >= threshold)[0]
                
                # colocando do maior p menor
                indices = indices[np.argsort(prediction[indices])[::-1]]
                top_filtered = [(labels[i], float(prediction[i])) for i in indices]
                
                print(f'top filtered: {top_filtered}')
                # printando na tela
                if len(top_filtered) == 0:
                    print('Sem classificacao')
                elif len(top_filtered) == 1:
                    print(f'Classificacao do modelo: {top_filtered[0][0]}. Acuracia: ({top_filtered[0][1]}%)')
                else:
                    suggestions = " ou ".join([f"{lbl} ({conf}%)" for lbl, conf in top_filtered])
                    print(f'voce quis dizer: {suggestions} ?')
              
               

cap.release()
cv2.destroyAllWindows()


