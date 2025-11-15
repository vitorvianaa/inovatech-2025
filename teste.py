import cv2
import mediapipe as mp
from save_data import save_data
import os
import time
from dotenv import load_dotenv
load_dotenv()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

dataset = []

current_label = os.getenv('CURRENT_LABEL')
TARGET = 500
delay = 0.05

print(f"Iniciando captura automática da letra {current_label}...")

while True:
    if len(dataset) >= TARGET:
        print("Coleta concluída!")
        break

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        hand_landmarks = result.multi_hand_landmarks[0]
        landmarks_list = []
        for lm in hand_landmarks.landmark:
            landmarks_list.extend([lm.x, lm.y, lm.z])

        sample = {
            'landmarks': landmarks_list,
            'label': current_label
        }
        dataset.append(sample)

        cv2.putText(frame, f"Capturando... {len(dataset)}/{TARGET}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        time.sleep(delay)
    else:
        cv2.putText(frame, "Mostre a mão para capturar...",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Captura Automática", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        print("Cancelado pelo usuário.")
        break

cap.release()
cv2.destroyAllWindows()

print(f"{len(dataset)} amostras capturadas para a letra {current_label}")
save_data(dataset=dataset, label=current_label)
