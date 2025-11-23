import cv2
import mediapipe as mp
from save_data import save_data
import os
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
                sample = {
                    'landmarks': landmarks_list,
                    'label': current_label
                }
                dataset.append(sample)
                print(f"Amostra capturada: {sample['label']} com {len(landmarks_list)} valores")

    if len(dataset) == 500:
        print('dados coletados...')
        break   

print(f'dataset letra {current_label}: ', dataset)
print(f'{len(dataset)} amostras capturadas da letra {current_label}')
cap.release()
cv2.destroyAllWindows()

save_data(dataset=dataset, label=current_label)
