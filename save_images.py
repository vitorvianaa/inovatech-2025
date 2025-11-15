import cv2
import mediapipe as mp
import os
import time
from dotenv import load_dotenv

load_dotenv()
current_label = os.getenv('CURRENT_LABEL')
output_dir = f"dataset/{current_label}"

os.makedirs(output_dir, exist_ok=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

print(f"[INFO] Coletando imagens para a classe '{current_label}'. Pressione ENTER para capturar, ESC para sair.")
count_images = 0
while True:
        
    if count_images == 400:
        break
    
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # get bounding box
            h, w, _ = frame.shape
            x_values = [lm.x for lm in hand_landmarks.landmark]
            y_values = [lm.y for lm in hand_landmarks.landmark]
            xmin, xmax = int(min(x_values) * w), int(max(x_values) * w)
            ymin, ymax = int(min(y_values) * h), int(max(y_values) * h)

            # enlarging image
            margin = 20
            xmin = max(xmin - margin, 0)
            ymin = max(ymin - margin, 0)
            xmax = min(xmax + margin, w)
            ymax = min(ymax + margin, h)

            # show rectangle
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        
            key = cv2.waitKey(1)
            if key == 13:  # ENTER
                hand_crop = frame[ymin:ymax, xmin:xmax]
                filename = os.path.join(output_dir, f"{time.time()}.jpg")
                cv2.imwrite(filename, hand_crop)
                print(f"[+] Imagem salva em {filename}")
                count_images += 1
                print(f'Quantidade de imagens capturadas: {count_images}')

            elif key == 27:  # ESC
                cap.release()
                cv2.destroyAllWindows()
                exit()

    cv2.imshow("Captura de Mão", frame)
    